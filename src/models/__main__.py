"""Full Phase 3 experiment with leakage-safe selection and resumable local fits."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from src.alignment.core import AlignmentPolicy

from .dataset import build_dataset, CANDIDATE_CONFIG
from .evaluation import evaluate, make_folds, observation_metrics, selection_key
from .inference import FEATURES, MODELS, SEED, build_model, load_model, positive_scores
from .report import render_reports


DIFFICULT = ("00000026", "00000083", "00000013", "00000019", "00000054", "00000069")


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


def versions():
    return dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                scipy=scipy.__version__, joblib=joblib.__version__, threadpoolctl=threadpoolctl.__version__)


def fit_pipeline(dataset, name, indices, workers):
    if dataset.partition != "train":
        raise ValueError("Only training rows may fit a model or preprocessor")
    model = build_model(name, workers)
    with threadpool_limits(workers):
        model.fit(dataset.X[indices], dataset.y[indices])
    if name == "random_forest":
        # Thread completion order can change floating-point summation and upset
        # exact ties between candidates. Fit trees in parallel, then score in order.
        model.set_params(classifier__n_jobs=1)
    return model


def _fold_fit(train, name, fold, fitting, validation, args, cache):
    file = cache / f"{name}_fold{fold}.npz"
    if file.exists():
        with np.load(file, allow_pickle=False) as saved:
            scores, elapsed = saved["scores"], saved["elapsed"]
        if scores.shape != (len(validation),) or not (
            np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        ):
            raise ValueError(f"Invalid fold checkpoint: {file}")
    else:
        start = perf_counter()
        model = fit_pipeline(train, name, fitting, args.workers)
        fitted = perf_counter()
        scores = positive_scores(model, train.X[validation], args.batch_size, args.workers)
        end = perf_counter()
        elapsed = np.array([fitted-start, end-fitted, peak_mib()])
        temp = file.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez(stream, scores=scores, elapsed=elapsed)
        temp.replace(file)
    return scores, dict(fit_s=float(elapsed[0]), predict_s=float(elapsed[1]),
                        predict_rows_per_s=len(validation)/float(elapsed[1]),
                        process_peak_rss_mib=float(elapsed[2]))


def _slice_errors(dataset, rankings):
    trips, obs, ranks, sizes = rankings
    X = dataset.X
    group_starts = np.flatnonzero(np.r_[True, (dataset.trip_codes[1:] != dataset.trip_codes[:-1])
                                        | (dataset.observation_ids[1:] != dataset.observation_ids[:-1])])
    column = FEATURES.index
    distances = X[:, column("distance_m")]
    lengths = X[:, column("arc_length_m")]
    duplicate_like = np.zeros(len(group_starts), dtype=bool)
    reverse_like = duplicate_like.copy()
    near_parallel_like = duplicate_like.copy()
    # Observable feature signatures only. Equal lengths alone do not establish geometry.
    multi = np.flatnonzero(sizes >= 2)
    first = group_starts[multi]
    second = first + 1
    tied = np.abs(distances[first] - distances[second]) <= 0.05
    same_length = np.abs(lengths[first] - lengths[second]) <= 0.05
    fractions = X[:, column("projection_fraction")]
    headings = X[:, column("heading_difference_deg")]
    hf, hs = headings[first], headings[second]
    same_fraction = np.abs(fractions[first] - fractions[second]) <= 0.01
    flip_fraction = np.abs(fractions[first] + fractions[second] - 1) <= 0.01
    duplicate_like[multi] = tied & same_length & same_fraction & (np.abs(hf - hs) <= 1)
    reverse_like[multi] = tied & same_length & flip_fraction & (np.abs(hf + hs - 180) <= 1)
    near_parallel_like[multi] = ((np.abs(distances[first] - distances[second]) <= 3)
                                 & (np.abs(hf - hs) <= 10) & ~duplicate_like[multi])
    ties = np.zeros(len(ranks), dtype=bool)
    ties[multi] = tied
    masks = {
        "distance_tie_within_0.05m": ties,
        "reverse_geometry_signature_nearest_two": reverse_like,
        "duplicate_geometry_signature_nearest_two": duplicate_like,
        "near_parallel_signature_nearest_two": near_parallel_like,
        "heading_unavailable_on_nearest": X[group_starts, column("heading_difference_valid")] == 0,
        "speed_unavailable": X[group_starts, column("speed_valid")] == 0,
        "gap_over_30s": X[group_starts, column("delta_t_s")] > 30,
        "four_or_more_candidates": sizes >= 4,
    }
    # Data is emitted in trip/observation order, as are the ranks returned by evaluate.
    assert np.array_equal(trips, dataset.trip_codes[group_starts])
    assert np.array_equal(obs, dataset.observation_ids[group_starts])
    return [dict(slice=name, **observation_metrics(ranks[mask])) for name, mask in masks.items()]


def run(args):
    raw_root = args.data_dir.resolve()
    for output in (args.report_dir, args.artifact_dir):
        if any(output.resolve().is_relative_to(root) for root in (raw_root, Path("data/raw").resolve())):
            raise ValueError("Outputs must be outside raw data")
        output.mkdir(parents=True, exist_ok=True)
    policy = AlignmentPolicy(**json.loads(args.policy.read_text(encoding="utf-8")))
    if policy.policy_id != "91b3c862ebe5388235c7a574ed4ed46a5a37782ac872eb9273abda850317e758":
        raise ValueError("Phase 3 requires the frozen Phase 2B training policy")
    print("Verifying and joining training candidates and proxy targets", flush=True)
    train = build_dataset(args.data_dir, "train", policy, args.alignment_dir)
    folds = make_folds(train)
    cache = args.artifact_dir / "cv_checkpoints"
    cache.mkdir(exist_ok=True)
    parameters = {name: build_model(name, args.workers).named_steps["classifier"].get_params()
                  for name in MODELS}
    code_digest = sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(file.read_bytes())
    contract = dict(schema=1, train_digest=train.digest(), code_sha256=code_digest.hexdigest(),
                    model_parameters=parameters, prediction_overrides={"random_forest": {"n_jobs": 1}},
                    features=list(FEATURES),
                    candidate_config=asdict(CANDIDATE_CONFIG), policy_id=policy.policy_id, seed=SEED,
                    versions=versions(), workers=args.workers, batch_size=args.batch_size,
                    training_trips=list(train.trips), training_counts={
                        key: sum(row[key] for row in train.coverage)
                        for key in ("original", "qualified", "eligible", "misses", "unqualified",
                                    "candidate_rows", "positives", "negatives")})
    manifest = args.artifact_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Artifact directory was created with different data, code or settings. Use a new --artifact-dir.")
    write_json(manifest, contract)
    fold_rows, assignments = [], []
    for fold, (fit, validation) in enumerate(folds, 1):
        validation_trips = np.unique(train.trip_codes[validation])
        assignments.extend(dict(fold=fold, trip_id=train.trips[i]) for i in validation_trips)
        fold_rows.append(dict(fold=fold, fit_trips=len(np.unique(train.trip_codes[fit])),
                              validation_trips=len(validation_trips), fit_rows=len(fit),
                              validation_rows=len(validation), fit_positive=int(train.y[fit].sum()),
                              validation_positive=int(train.y[validation].sum()), overlap=0))
    write_csv(args.report_dir / "cross_validation_folds.csv", fold_rows)
    write_csv(args.report_dir / "fold_assignments.csv", assignments)

    cv_metrics, comparisons, per_trip, runtimes, slices, baselines = [], [], [], [], [], []
    for name in MODELS:
        oof = np.full(len(train.y), np.nan)
        for fold, (fit, validation) in enumerate(folds, 1):
            scores, timing = _fold_fit(train, name, fold, fit, validation, args, cache)
            oof[validation] = scores
            metrics, _, _ = evaluate(train, scores, validation)
            cv_metrics.append(dict(model=name, fold=fold, **metrics))
            runtimes.append(dict(model=name, stage="cross_validation", fold=fold, **timing))
            print(f"{name} fold {fold}/5, top1={metrics['top1']:.4f}, "
                  f"fit={timing['fit_s']:.2f}s, predict={timing['predict_s']:.2f}s", flush=True)
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete out-of-fold predictions")
        metrics, by_trip, ranking = evaluate(train, oof)
        comparisons.append(dict(model=name, **metrics))
        per_trip.extend(dict(partition="train_oof", model=name, **row) for row in by_trip)
        slices.extend(dict(partition="train_oof", model=name, **row) for row in _slice_errors(train, ranking))
        write_csv(args.report_dir / "cross_validation_metrics.csv", cv_metrics)
        write_csv(args.report_dir / "model_comparison.csv", sorted(comparisons, key=selection_key))
        write_csv(args.report_dir / "training_runtime.csv", runtimes)

    comparisons.sort(key=selection_key)
    selected = comparisons[0]["model"]
    train_base, base_trips, base_ranks = evaluate(train, -train.X[:, FEATURES.index("distance_rank")],
                                                 probabilities=False)
    baselines.append(dict(partition="train_oof", **train_base))
    per_trip.extend(dict(partition="train_oof", model="nearest", **row) for row in base_trips)
    slices.extend(dict(partition="train_oof", model="nearest", **row) for row in _slice_errors(train, base_ranks))
    selection = dict(selected=selected, source="training_grouped_oof",
                     tie_break=["observation-weighted Top-1 descending", "trip-macro Top-1 descending",
                                "candidate log loss ascending", "model identifier ascending"],
                     ranked_models=[r["model"] for r in comparisons], policy_id=policy.policy_id)
    write_json(args.report_dir / "model_selection.json", selection)
    print(f"Training-only selection: {selected}. Fitting the five final pipelines.", flush=True)
    for name in MODELS:
        path = args.artifact_dir / f"{name}.joblib"
        timing_path = args.artifact_dir / f"{name}_fit.json"
        if not (path.is_file() and timing_path.is_file()):
            start = perf_counter()
            model = fit_pipeline(train, name, slice(None), args.workers)
            fit_seconds = perf_counter() - start
            probe = train.X[: min(len(train.y), 1024)]
            scores = positive_scores(model, probe, args.batch_size, args.workers)
            tmp = path.with_suffix(".tmp")
            joblib.dump(dict(model_id=name, features=FEATURES, pipeline=model, manifest=contract,
                             selection=selection), tmp, compress=3)
            del model
            reloaded = load_model(tmp)
            np.testing.assert_array_equal(scores, positive_scores(reloaded["pipeline"], probe,
                                                                  args.batch_size, args.workers))
            del reloaded
            tmp.replace(path)
            write_json(timing_path, dict(fit_s=fit_seconds, process_peak_rss_mib=peak_mib()))
        print(f"Final {name}: {path.stat().st_size:,} bytes", flush=True)

    # The held-out data are first opened after CV, the selection decision and final fitting.
    print("Building held-out offline evaluation groups with frozen alignment", flush=True)
    student = build_dataset(args.data_dir, "student_test", policy, args.alignment_dir)
    held_out, importances = [], []
    base, by_trip, ranking = evaluate(student, -student.X[:, FEATURES.index("distance_rank")],
                                      probabilities=False)
    baselines.append(dict(partition="student_test", **base))
    per_trip.extend(dict(partition="student_test", model="nearest", **row) for row in by_trip)
    slices.extend(dict(partition="student_test", model="nearest", **row) for row in _slice_errors(student, ranking))
    for name in MODELS:
        path = args.artifact_dir / f"{name}.joblib"
        bundle = load_model(path)
        if bundle["manifest"] != contract or bundle["selection"] != selection:
            raise ValueError(f"Incompatible model metadata in {path}")
        start = perf_counter()
        scores = positive_scores(bundle["pipeline"], student.X, args.batch_size, args.workers)
        end = perf_counter()
        metrics, by_trip, ranking = evaluate(student, scores)
        estimator = bundle["pipeline"].named_steps["classifier"]
        values = getattr(estimator, "feature_importances_", None)
        kind = "impurity_decrease"
        if name == "logistic_regression":
            values, kind = estimator.coef_[0], "signed_coefficient"
        if values is not None:
            importances.extend(dict(model=name, feature=feature, method=kind, value=float(value))
                               for feature, value in zip(FEATURES, values, strict=True))
        held_out.append(dict(model=name, **metrics))
        per_trip.extend(dict(partition="student_test", model=name, **row) for row in by_trip)
        slices.extend(dict(partition="student_test", model=name, **row) for row in _slice_errors(student, ranking))
        fit_time = json.loads((args.artifact_dir / f"{name}_fit.json").read_text())
        runtimes.append(dict(model=name, stage="final_student_test", fold=None, **fit_time,
                             predict_s=end-start, predict_rows_per_s=len(scores)/(end-start),
                             artifact_bytes=path.stat().st_size))
        print(f"Held out {name}, top1={metrics['top1']:.4f}, predict={end-start:.2f}s", flush=True)

    coverage = train.coverage + student.coverage
    coverage_summary = []
    for dataset in (train, student):
        row = dict(partition=dataset.partition, trips=len(dataset.trips), **{
            key: sum(item[key] for item in dataset.coverage) for key in
            ("original", "qualified", "eligible", "misses", "unqualified", "candidate_rows",
             "positives", "negatives")})
        coverage_summary.append(dict(**row, proxy_label_coverage=row["qualified"]/row["original"],
                                     candidate_recall=row["eligible"]/row["qualified"]))
    for name, rows in {
        "student_test_metrics": held_out, "per_trip_metrics": per_trip, "coverage_summary": coverage_summary,
        "coverage_by_trip": coverage, "exclusions_by_trip": train.exclusions + student.exclusions,
        "nearest_baseline": baselines, "training_runtime": runtimes, "error_slices": slices,
        "feature_importance": importances,
        "difficult_trip_metrics": [r for r in per_trip if r["partition"] == "train_oof" and r["trip_id"] in DIFFICULT],
    }.items():
        write_csv(args.report_dir / f"{name}.csv", rows)
    write_json(args.report_dir / "experiment_manifest.json", contract)
    render_reports(args.report_dir, comparisons, held_out, baselines, cv_metrics, per_trip,
                   runtimes, coverage_summary, selection, importances)
    print(f"Reports: {args.report_dir}. Process high-water RSS: {peak_mib():.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--alignment-dir", type=Path, default=Path("data/alignments"))
    parser.add_argument("--policy", type=Path, default=Path("reports/phase2b/train/policy.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/models/phase3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
