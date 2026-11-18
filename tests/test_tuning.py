from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.alignment.core import AlignmentPolicy
from src.models.dataset import Dataset
from src.models.evaluation import evaluate, ranked_targets
from src.models.inference import FEATURES, build_model, load_model, positive_scores
from src.tuning import __main__ as experiment
from src.tuning import search
from src.tuning.artifacts import bind_directory, fingerprint, read_json, read_record, save_record, score_checkpoint
from src.tuning.validation import fit_model, grouped_folds, pool_trial, trial_metrics


def dataset(partition="train"):
    trips = np.repeat(np.arange(6), 48)
    obs = np.tile(np.repeat(np.arange(24), 2), 6)
    targets = np.tile([1, 0], 144)
    x = np.zeros((len(targets), len(FEATURES)), dtype=np.float32)
    x[:, 0] = np.tile([0.5, 24], 144)
    x[:, 2] = np.tile([1, 2], 144)
    x[:, 3] = np.nan
    return Dataset(partition, tuple(f"{i:08d}" for i in range(6)), x, targets, trips,
                   obs, np.tile([11, 8], 144), [], [])


def baseline():
    classifier = build_model("gradient_boosting").named_steps["classifier"]
    return {name: classifier.get_params()[name] for name in search.SPACE}


@pytest.mark.parametrize("patch", [{"max_iter": True}, {"max_iter": 301}, {"max_depth": 100},
                                   {"learning_rate": float("nan")}, {"l2_regularization": 0},
                                   {"max_leaf_nodes": 20.5}, {"unrelated": 1}])
def test_invalid_parameters(patch):
    with pytest.raises(ValueError):
        search.validate_params({**baseline(), **patch})


def test_seeded_tpe_and_selection_order():
    history = []
    for i in range(10):
        params = search.propose("outer_01", i, history, baseline())
        assert params == search.propose("outer_01", i, history, baseline(), 2026)
        search.validate_params(params)
        history.append(dict(state="COMPLETE", params=params,
                            metrics=dict(correct=i % 5, trip_macro_top1=i / 20, log_loss=0.2)))
    assert history[0]["params"] == baseline()
    assert len({fingerprint(r["params"]) for r in history}) == len(history)
    assert search.propose("outer_01", 10, history, baseline(), 2) != search.propose(
        "outer_01", 10, history, baseline(), 2026)
    key = lambda correct, macro, loss: search.trial_key(
        baseline(), dict(correct=correct, trip_macro_top1=macro, log_loss=loss))
    assert key(12, .7, .2) < key(11, .9, .1)
    assert key(12, .7, .2) < key(12, .6, .1)
    assert key(12, .7, .2) < key(12, .7, .3)


def test_grouped_folds_cover_exact_scope_and_fit_local_imputer():
    data = dataset()
    outer = grouped_folds(data)
    for a, b in zip(outer, grouped_folds(data), strict=True):
        np.testing.assert_array_equal(a.fit, b.fit)
        np.testing.assert_array_equal(a.valid, b.valid)
        assert not set(data.trip_codes[a.fit]) & set(data.trip_codes[a.valid])
        inner = grouped_folds(data, a.fit, 3)
        np.testing.assert_array_equal(np.sort(np.concatenate([f.valid for f in inner])), a.fit)
        for f in inner:
            assert not set(data.trip_codes[f.fit]) & set(data.trip_codes[f.valid])
            assert not set(data.trip_codes[a.valid]) & set(data.trip_codes[f.valid])
    f = outer[0]
    data.X[f.valid, 0] = 900
    model = fit_model(data, baseline(), f.fit, 1)
    imputer = model.named_steps["features"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == np.median(data.X[f.fit, 0]) == 12.25
    assert not model.named_steps["classifier"].early_stopping
    assert model.named_steps["classifier"].random_state == 2026
    assert model.n_features_in_ == 15
    with pytest.raises(ValueError):
        fit_model(dataset("student_test"), baseline(), slice(None), 1)
    with pytest.raises(ValueError):
        grouped_folds(dataset("student_test"))


def test_pooled_objective_and_tie_ranking():
    data = dataset()
    scores = np.full(len(data.y), .5)
    assert np.all(ranked_targets(data, scores)[2] == 1)
    folds = grouped_folds(data)
    trial = pool_trial([trial_metrics(data, scores[f.valid], f.valid) for f in folds])
    result = evaluate(data, scores)[0]
    for key in ("correct", "eligible", "top1", "top3", "mrr", "trip_macro_top1", "log_loss"):
        assert trial[key] == pytest.approx(result[key])
    flipped = scores + np.tile([-.1, .1], len(scores) // 2)
    actual = evaluate(data, flipped)[0]
    assert actual["top1"] == 0 and actual["top3"] == 1
    bad = dataset()
    bad.y[0] = 0
    with pytest.raises(ValueError, match="one positive"):
        ranked_targets(bad, scores)


def test_checkpoint_integrity_and_manifest_rejection(tmp_path):
    for key in ("train_digest", "policy_id", "search_space"):
        directory = tmp_path / key
        bind_directory(directory, {key: "first"})
        with pytest.raises(ValueError, match="different"):
            bind_directory(directory, {key: "changed"})
    checkpoint = tmp_path / "metrics.json"
    save_record(checkpoint, "abc", {"count": 3})
    assert read_record(checkpoint, "abc") == {"count": 3}
    with pytest.raises(ValueError):
        read_record(checkpoint, "stale")
    checkpoint.write_text(checkpoint.read_text().replace('"count": 3', '"count": 9'))
    with pytest.raises(ValueError, match="corrupt"):
        read_record(checkpoint, "abc")
    path = tmp_path / "scores.npz"
    np.testing.assert_array_equal(score_checkpoint(path, "a", 2, lambda: (np.array([.2, .8]), {}))[0], [.2, .8])
    with pytest.raises(ValueError, match="Invalid"):
        score_checkpoint(path, "b", 2, lambda: pytest.fail("should load"))


def test_trial_resumes_completed_folds_after_interruption(tmp_path, monkeypatch):
    args = Namespace(seed=2026, workers=1, batch_size=256)
    data = dataset()
    folds = grouped_folds(data, n_splits=3)
    calls = []

    def score(_data, _params, fold, *_settings):
        calls.append(len(fold.valid))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return np.where(data.y[fold.valid], .9, .1), .02, .01

    monkeypatch.setattr(search, "score_fold", score)
    with pytest.raises(KeyboardInterrupt):
        search.search_study(data, "outer", folds, baseline(), 1, args, tmp_path, "test")
    history, best = search.search_study(data, "outer", folds, baseline(), 1, args, tmp_path, "test")
    assert len(calls) == 4
    assert best["metrics"]["top1"] == 1 and len(history) == 1
    search.search_study(data, "outer", folds, baseline(), 1, args, tmp_path, "test")
    assert len(calls) == 4


def test_failed_trial_preserves_error_in_journal(tmp_path, monkeypatch):
    args = Namespace(seed=2026, workers=1, batch_size=256)
    data = dataset()
    folds = grouped_folds(data, n_splits=3)

    def bad(*_):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(search, "score_fold", bad)
    with pytest.raises(RuntimeError, match="trial 1 failed"):
        search.search_study(data, "final", folds, baseline(), 1, args, tmp_path, "test")
    payload = read_json(tmp_path / "final/0000.json")["payload"]
    assert payload["state"] == "FAIL" and "synthetic failure" in payload["error"]
    with pytest.raises(ValueError, match="recorded failed trial"):
        search.search_study(data, "final", folds, baseline(), 1, args, tmp_path, "test")


def test_synthetic_end_to_end_freeze_before_student_and_preserve_phase3(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("train", "student_test", "evaluator_test"):
        (raw / name).mkdir()
    artifacts, reports = tmp_path / "models", tmp_path / "reports"
    args = Namespace(data_dir=raw, alignment_dir=tmp_path / "alignments",
                     policy=Path("reports/phase2b/train/policy.json"), artifact_dir=artifacts, report_dir=reports,
                     outer_folds=3, inner_folds=2, final_folds=3, outer_trials=2, final_trials=2,
                     workers=1, batch_size=128, seed=2026, training_only=False)
    original = experiment.records
    seen = []

    def guarded_load(view, partition, policy, *_):
        assert policy.policy_id == AlignmentPolicy().policy_id
        seen.append(partition)
        assert {p.name for p in view.iterdir()} == (
            {"train"} if partition == "train" else {"train", "student_test"})
        if partition == "student_test":
            assert (artifacts / "frozen_model.json").is_file()
            assert (artifacts / "studies/final/0001.json").is_file()
        return dataset(partition)

    def guarded_records(path):
        if path.name == "student_test_metrics.csv":
            student = dataset("student_test")
            metrics = evaluate(student, np.where(student.y, .9, .1))[0]
            return [dict(model="gradient_boosting", **{k: str(v) for k, v in metrics.items()})]
        return original(path)

    monkeypatch.setattr(experiment, "build_dataset", guarded_load)
    monkeypatch.setattr(experiment, "records", guarded_records)
    monkeypatch.setattr(experiment, "phase3_reference", lambda *_: "recomputed_same_outer_folds")
    protected = Path("reports/phase3/model_comparison.csv").read_bytes()
    result = experiment.run(args)
    assert result["fixed"]["eligible"] == result["tuned"]["eligible"] == 144
    assert result["student"]["metrics"]["top1"] == 1.0
    assert seen == ["train", "student_test"]
    old_results = {p.name: p.read_bytes() for p in reports.iterdir() if p.name != "training_runtime.csv"}
    experiment.run(args)
    assert seen == ["train", "student_test", "train"]
    assert old_results == {p.name: p.read_bytes() for p in reports.iterdir() if p.name != "training_runtime.csv"}
    assert len(list(reports.glob("*.png"))) == 6
    assert Path("reports/phase3/model_comparison.csv").read_bytes() == protected
    model = load_model(artifacts / "gradient_boosting.joblib")["pipeline"]
    np.testing.assert_array_equal(positive_scores(model, dataset().X[:0]), [])
    with pytest.raises(ValueError, match="separate"):
        experiment.check_outputs(Namespace(**{**vars(args), "report_dir": Path("reports/phase3")}))
