"""Run nested grouped Bayesian tuning on the frozen PathSnap training matrix."""

import argparse
from dataclasses import asdict
import csv
import fcntl
from hashlib import sha256
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from src.alignment.core import AlignmentPolicy
from src.models.__main__ import DIFFICULT, _slice_errors, peak_mib, versions
from src.models.dataset import CANDIDATE_CONFIG, build_dataset
from src.models.evaluation import evaluate
from src.models.inference import FEATURES, SEED, build_model, positive_scores

from .artifacts import (bind_directory, file_hash, fingerprint, freeze_model,
                        read_json, read_record, save_record, score_checkpoint, write_json)
from .report import render_reports
from .search import SAMPLER, SPACE, search_study, validate_params
from .validation import MODEL, fit_model, grouped_folds, score_fold


POLICY = "91b3c862ebe5388235c7a574ed4ed46a5a37782ac872eb9273abda850317e758"
BASELINE_REPORTS = Path("reports/phase3")
BASELINE_ARTIFACTS = Path("artifacts/models/phase3")


def records(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fold_assignments(data, groups):
    return [dict(study=study, fold=index, role=role, trip_id=data.trips[code])
            for study, folds in groups.items() for index, fold in enumerate(folds, 1)
            for role, rows in (("fit", fold.fit), ("validation", fold.valid))
            for code in np.unique(data.trip_codes[rows])]


def source_hash():
    paths = [p for module in ("tuning", "models", "alignment", "candidates", "preprocessing")
             for p in (Path(__file__).resolve().parents[1] / module).glob("*.py")
             if p.name != "report.py"]
    return fingerprint({f"{p.parent.name}/{p.name}": file_hash(p) for p in sorted(paths)})


def check_outputs(args):
    report, artifact = args.report_dir.resolve(), args.artifact_dir.resolve()
    protected = (args.data_dir.resolve(), args.alignment_dir.resolve(),
                 Path("data").resolve(), *(Path("reports", name).resolve() for name in
                                           ("eda", "phase2a", "phase2b", "phase3")),
                 BASELINE_ARTIFACTS.resolve(), Path("src").resolve(), Path(".git").resolve())

    def overlaps(a, b):
        return a.is_relative_to(b) or b.is_relative_to(a)

    if overlaps(report, artifact) or any(overlaps(path, p) for path in (report, artifact) for p in protected):
        raise ValueError("Use separate output directories outside source data and earlier phase outputs")
    local_root = Path.cwd().resolve()
    if artifact.is_relative_to(local_root) and not artifact.is_relative_to(Path("artifacts/models").resolve()):
        raise ValueError("Local models, records and checkpoints must stay under artifacts/models")


def phase3_reference(data, policy, baseline, assignment_rows, args):
    manifest = read_json(BASELINE_REPORTS / "experiment_manifest.json")
    actual = dict(train_digest=data.digest(), features=list(FEATURES),
                  candidate_config=asdict(CANDIDATE_CONFIG), policy_id=policy.policy_id,
                  training_trips=list(data.trips))
    if any(manifest.get(key) != value for key, value in actual.items()):
        raise ValueError("Training dataset does not match the frozen Phase 3 contract")
    if any(manifest["model_parameters"][MODEL][key] != value for key, value in baseline.items()):
        raise ValueError("Baseline pipeline does not match the frozen Phase 3 hyperparameters")
    current_phase3_code = sha256(b"".join(
        p.read_bytes() for p in sorted(Path("src/models").glob("*.py")))).hexdigest()
    frozen_folds = {(int(row["fold"]), row["trip_id"]) for row in records(BASELINE_REPORTS / "fold_assignments.csv")}
    current = {(row["fold"], row["trip_id"]) for row in assignment_rows
               if row["study"] == "outer" and row["role"] == "validation"}
    local_manifest = BASELINE_ARTIFACTS / "manifest.json"
    if (current == frozen_folds and manifest["code_sha256"] == current_phase3_code
            and manifest["versions"] == versions()
            and manifest["workers"] == args.workers and manifest["batch_size"] == args.batch_size
            and local_manifest.exists() and read_json(local_manifest) == manifest
            and all((BASELINE_ARTIFACTS / "cv_checkpoints" / f"{MODEL}_fold{i}.npz").is_file()
                    for i in range(1, args.outer_folds + 1))):
        return "phase3_verified_checkpoints"
    return "recomputed_same_outer_folds"


def fitted_fold(data, params, fold, args):
    scores, fit_s, predict_s = score_fold(data, params, fold, args.workers, args.batch_size)
    return scores, dict(fit_s=fit_s, predict_s=predict_s, rss_mib=peak_mib())


def fixed_fold(data, baseline, fold, index, args, origin):
    if origin != "phase3_verified_checkpoints":
        return fitted_fold(data, baseline, fold, args)
    path = BASELINE_ARTIFACTS / "cv_checkpoints" / f"{MODEL}_fold{index}.npz"
    with np.load(path, allow_pickle=False) as bundle:
        scores, times = bundle["scores"], bundle["elapsed"]
    reported = next(r for r in records(BASELINE_REPORTS / "cross_validation_metrics.csv")
                    if r["model"] == MODEL and int(r["fold"]) == index)
    metrics = evaluate(data, scores, fold.valid)[0]
    if any(metrics[k] != float(reported[k]) for k in metrics):
        raise ValueError(f"Phase 3 baseline score mismatch for fold {index}")
    return scores, dict(fit_s=float(times[0]), predict_s=float(times[1]), rss_mib=float(times[2]))


def bootstrap(per_trip, seed):
    observed = np.array([row["eligible"] for row in per_trip])
    differences = np.array([row["tuned_correct"] - row["baseline_correct"] for row in per_trip])
    count = len(observed)
    samples = np.random.default_rng(seed).integers(0, count, size=(10_000, count))
    draws = differences[samples].sum(axis=1) / observed[samples].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(lower=float(low), upper=float(high), resamples=len(draws), unit="trip")


def paired_reports(data, baseline_scores, tuned_scores):
    baseline, base_trips, base_ranks = evaluate(data, baseline_scores)
    tuned, tuned_trips, tuned_ranks = evaluate(data, tuned_scores)
    if not np.array_equal(base_ranks[0], tuned_ranks[0]) or not np.array_equal(base_ranks[1], tuned_ranks[1]):
        raise ValueError("Paired candidate groups differ")
    per_trip = [dict(trip_id=a["trip_id"], eligible=a["eligible"], baseline_correct=a["correct"],
                     tuned_correct=b["correct"], baseline_top1=a["top1"], tuned_top1=b["top1"],
                     top1_difference=b["top1"] - a["top1"])
                for a, b in zip(base_trips, tuned_trips, strict=True)]
    slices = [dict(slice=a["slice"], eligible=a["eligible"], baseline_top1=a["top1"], tuned_top1=b["top1"],
                   difference=None if not a["eligible"] else b["top1"] - a["top1"])
              for a, b in zip(_slice_errors(data, base_ranks), _slice_errors(data, tuned_ranks), strict=True)]
    nearest = evaluate(data, -data.X[:, FEATURES.index("distance_rank")], probabilities=False)[0]
    return baseline, tuned, nearest, per_trip, slices


def run_training(train, policy, args):
    start = perf_counter()
    outer = grouped_folds(train, n_splits=args.outer_folds)
    groups = {"outer": outer, **{f"outer_{i:02d}": grouped_folds(train, fold.fit, args.inner_folds)
                               for i, fold in enumerate(outer, 1)},
              "final": grouped_folds(train, n_splits=args.final_folds)}
    assignments = fold_assignments(train, groups)
    baseline_config = build_model(MODEL).named_steps["classifier"].get_params()
    baseline = {name: baseline_config[name] for name in SPACE}
    validate_params(baseline)
    origin = phase3_reference(train, policy, baseline, assignments, args)
    inputs = ["experiment_manifest.json", "fold_assignments.csv", "cross_validation_metrics.csv",
              "model_comparison.csv", "student_test_metrics.csv", "training_runtime.csv"]
    manifest = dict(schema=1, purpose="phase4_nested_grouped_classifier_tuning",
                    train_digest=train.digest(), training_trips=list(train.trips), training_coverage=train.coverage,
                    policy_id=policy.policy_id, candidate_config=asdict(CANDIDATE_CONFIG), features=list(FEATURES),
                    study_fold_sha256=fingerprint(assignments), source_sha256=source_hash(),
                    phase3_report_sha256={name: file_hash(BASELINE_REPORTS / name) for name in inputs},
                    search_space={name: list(spec) for name, spec in SPACE.items()}, baseline=baseline_config,
                    baseline_source=origin, sampler=SAMPLER, objective="pooled_observation_top1_proxy_agreement",
                    seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                    final_folds=args.final_folds, outer_trials=args.outer_trials, final_trials=args.final_trials,
                    workers=args.workers, batch_size=args.batch_size, versions=versions())
    directory = args.artifact_dir
    bind_directory(directory, manifest)
    prior_report = args.report_dir / "optimization_manifest.json"
    if prior_report.exists() and read_json(prior_report) != manifest:
        raise ValueError("Report directory is bound to a different experiment")
    manifest_id = fingerprint(manifest)
    studies = directory / "studies"
    checkpoints = directory / "outer_checkpoints"
    all_trials, outer_metrics, runtimes, outer_selection = [], [], [], []
    baseline_scores = np.full(len(train.y), np.nan)
    tuned_scores = baseline_scores.copy()
    for index, fold in enumerate(outer, 1):
        name = f"outer_{index:02d}"
        history, best = search_study(train, name, groups[name], baseline, args.outer_trials, args, studies, manifest_id)
        all_trials.extend(history)
        outer_selection.append(dict(study=name, trial=best["number"], params=best["params"],
                                    inner_top1=best["metrics"]["top1"]))
        for model, params, destination in (("fixed", baseline, baseline_scores), ("tuned", best["params"], tuned_scores)):
            compute = (lambda: fixed_fold(train, baseline, fold, index, args, origin)) if model == "fixed" else (
                lambda: fitted_fold(train, params, fold, args))
            scores, timing = score_checkpoint(checkpoints / f"{name}-{model}.npz",
                fingerprint(dict(manifest=manifest_id, outer=index, params=params, model=model)),
                len(fold.valid), compute)
            destination[fold.valid] = scores
            metrics = evaluate(train, scores, fold.valid)[0]
            outer_metrics.append(dict(fold=index, model=model, trips=len(np.unique(train.trip_codes[fold.valid])),
                                      **metrics))
            runtimes.append(dict(stage="outer", study=name, model=model, **timing))
            print(f"{name} {model} outer Top-1={metrics['top1']:.4%}", flush=True)
    if not np.isfinite(tuned_scores).all() or not np.isfinite(baseline_scores).all():
        raise ValueError("Outer score coverage is incomplete")
    fixed, tuned, nearest, per_trip, slices = paired_reports(train, baseline_scores, tuned_scores)
    interval = bootstrap(per_trip, args.seed)
    if origin == "phase3_verified_checkpoints":
        published = next(row for row in records(BASELINE_REPORTS / "model_comparison.csv") if row["model"] == MODEL)
        if any(value != float(published[name]) for name, value in fixed.items()):
            raise ValueError("Pooled Phase 3 baseline differs from frozen published metrics")
    history, final = search_study(train, "final", groups["final"], baseline, args.final_trials, args, studies, manifest_id)
    all_trials.extend(history)
    selection = dict(final_trial=final["number"], final_params=final["params"], final_cv=final["metrics"],
                     outer=outer_selection, source="full_training_grouped_cv",
                     tie_break=["pooled Top-1 descending", "trip-macro Top-1 descending",
                                "candidate log loss ascending", "canonical hyperparameters ascending"],
                     recommendation="phase4" if (interval["lower"] > 0 and
                                                  tuned["trip_macro_top1"] >= fixed["trip_macro_top1"]) else "phase3",
                     recommendation_rule="Positive lower paired-trip interval and preserved trip-macro Top-1",
                     nested_top1_difference=tuned["top1"] - fixed["top1"], paired_trip_interval=interval)
    write_json(directory / "model_selection.json", selection)

    def fit_final():
        begin = perf_counter()
        model = fit_model(train, final["params"], slice(None), args.workers)
        return model, perf_counter() - begin

    model, frozen = freeze_model(directory, manifest, selection, fit_final, train.X[:1024], args)
    print(f"Final training-only parameters frozen: trial {final['number'] + 1}, {frozen['sha256'][:12]}", flush=True)
    return dict(manifest=manifest, selection=selection, frozen=frozen, model=model, all_trials=all_trials,
                outer_metrics=outer_metrics, runtimes=runtimes, assignments=assignments, fixed=fixed,
                tuned=tuned, nearest=nearest, per_trip=per_trip, slices=slices,
                difficult=[row for row in per_trip if row["trip_id"] in DIFFICULT],
                session_training_s=perf_counter() - start)


def student_evaluation(result, policy, args, data_view):
    path = args.artifact_dir / "student_evaluation.json"
    contract = fingerprint(dict(model=result["frozen"]["sha256"], policy=policy.policy_id,
                                phase3_student_sha256=result["manifest"]["phase3_report_sha256"]["student_test_metrics.csv"]))
    if path.exists():
        return read_record(path, contract)
    assert (args.artifact_dir / "frozen_model.json").is_file()
    (data_view / "student_test").symlink_to(args.data_dir.absolute() / "student_test", target_is_directory=True)
    print("Scoring the frozen model on student testing", flush=True)
    student = build_dataset(data_view, "student_test", policy)
    start = perf_counter()
    scores = positive_scores(result["model"], student.X, args.batch_size, args.workers)
    predict_s = perf_counter() - start
    metrics, trips, ranks = evaluate(student, scores)
    reference = next(r for r in records(BASELINE_REPORTS / "student_test_metrics.csv") if r["model"] == MODEL)
    if (metrics["eligible"], metrics["candidate_rows"]) != (int(reference["eligible"]), int(reference["candidate_rows"])):
        raise ValueError("Student-testing counts differ from the frozen Phase 3 baseline")
    score = dict(metrics=metrics, per_trip=trips, slices=_slice_errors(student, ranks),
                 baseline={key: float(value) for key, value in reference.items() if key != "model"},
                 student_digest=student.digest(), predict_s=predict_s,
                 candidate_misses=sum(row["misses"] for row in student.coverage),
                 qualified=sum(row["qualified"] for row in student.coverage),
                 original=sum(row["original"] for row in student.coverage), rss_mib=peak_mib())
    save_record(path, contract, score)
    return score


def run(args):
    check_outputs(args)
    start = perf_counter()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    with (args.artifact_dir / "run.lock").open("a") as lock, tempfile.TemporaryDirectory(prefix="pathsnap-phase4-data-") as view:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = AlignmentPolicy(**read_json(args.policy))
        if policy.policy_id != POLICY:
            raise ValueError("Phase 4 requires the frozen Phase 2B alignment policy")
        data_view = Path(view)
        (data_view / "train").symlink_to(args.data_dir.absolute() / "train", target_is_directory=True)
        print("Building frozen training data", flush=True)
        train = build_dataset(data_view, "train", policy, args.alignment_dir)
        result = run_training(train, policy, args)
        recorded_student = (args.artifact_dir / "student_evaluation.json").is_file()
        result["student"] = (student_evaluation(result, policy, args, data_view)
                             if recorded_student or not args.training_only else None)
        result["session_wall_s"] = perf_counter() - start
        result["session_rss_mib"] = peak_mib()
        render_reports(result, args.report_dir)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--alignment-dir", type=Path, default=Path("data/alignments"))
    parser.add_argument("--policy", type=Path, default=Path("reports/phase2b/train/policy.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/models/phase4"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase4"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--final-folds", type=int, default=5)
    parser.add_argument("--outer-trials", type=int, default=20)
    parser.add_argument("--final-trials", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    if min(args.outer_folds, args.inner_folds, args.final_folds) < 2:
        parser.error("All fold counts must be at least two")
    if min(args.outer_trials, args.final_trials, args.workers, args.batch_size) < 1 or args.seed < 0:
        parser.error("Trial budgets, workers and batch size must be positive, and seed must be non-negative")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Phase 4 stopped: {exc}\n")


if __name__ == "__main__":
    main()
