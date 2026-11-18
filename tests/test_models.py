from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.alignment.core import AlignmentPolicy, align_trip
from src.candidates.pipeline import CandidateConfig, InferenceTrip, candidate_batches
from src.models.__main__ import fit_pipeline
from src.models.dataset import Dataset, join_targets, _student_rows, _training_rows
from src.models.evaluation import binary_metrics, evaluate, make_folds, selection_key
from src.models.inference import FEATURES, MODELS, feature_matrix, load_model, positive_scores, predict_trip
from src.preprocessing.loaders import Trip, TripFiles


def sample(partition="train"):
    track = np.array([[0.0002, 0, 0], [0.0003, 0, 1], [4, 5, 2]])
    nodes = np.array([[0, 0], [0.001, 0], [0, 0.00012], [0.001, 0.00012]])
    arcs = np.array([[0, 1], [2, 3]])
    return InferenceTrip(partition, "00000007", track, nodes, arcs)


def test_complete_join_keys_strictly_validated_and_ordered():
    obs, candidates = next(candidate_batches(sample(), CandidateConfig()))
    prefix = ("train", "00000007", 0)
    targets = [(*prefix, 1, 0), (*prefix, 0, 1)]
    assert join_targets(obs, candidates, targets) == (1, 0)
    assert join_targets(obs, candidates, []) == ()
    for invalid in (targets + targets[:1], targets[:1], [(*prefix, 0, 0), targets[0]],
                    [(*prefix, 0, 2), targets[0]], [("student_test", *targets[0][1:]), targets[1]]):
        with pytest.raises(ValueError):
            join_targets(obs, candidates, invalid)
    with pytest.raises(ValueError, match="Duplicate feature"):
        join_targets(obs, (candidates[0], candidates[0]), [])


def test_training_join_exclusions_and_frozen_candidate_config():
    obs_groups = list(candidate_batches(sample(), CandidateConfig()))
    policy = AlignmentPolicy()
    audits = [dict(partition="train", trip_id="00000007", observation_id=i,
                   outcome=outcome, reason=reason, policy_id=policy.policy_id,
                   radius_m=50, top_k=8, candidate_count=len(group), target_rank_within_radius=1)
              for i, ((_, group), (outcome, reason)) in enumerate(zip(obs_groups, (
                  ("included", "target_present"), ("unqualified", "heading_disagreement"),
                  ("candidate_miss", "empty_candidate_group")), strict=True))]
    targets = [dict(partition="train", trip_id="00000007", observation_id=0, arc_id=arc,
                    target=int(arc == 0), method="qualified", provenance="ordered_reference_proxy_v1",
                    policy_id=policy.policy_id) for arc in (0, 1)]

    def joined():
        return list(_training_rows(iter(audits), iter([(("00000007", 0), iter(targets))]),
                                   policy, "train", "00000007", iter(obs_groups)))

    assert [row[2] for row in joined()] == [(1, 0), (), ()]
    audits[1]["radius_m"] = 25
    with pytest.raises(ValueError, match="Incompatible policy, candidate configuration"):
        joined()


def test_explicit_feature_allowlist_and_route_independence():
    trip = sample()
    raw = Trip(TripFiles("train", trip.trip_id, Path(".")), trip.track, trip.nodes, trip.arcs,
               np.array([0]), [])
    cfg = CandidateConfig()
    features = list(candidate_batches(trip, cfg))
    first_label = align_trip(raw, AlignmentPolicy())["qualified"][0]
    raw.route = np.array([1])
    second_label = align_trip(raw, AlignmentPolicy())["qualified"][0]
    assert first_label.arc_id != second_label.arc_id
    assert features == list(candidate_batches(trip, cfg))
    assert FEATURES == (
        "distance_m", "arc_length_m", "distance_rank", "delta_t_s", "speed_mps",
        "heading_difference_deg", "projection_fraction", "from_in_degree", "from_out_degree",
        "to_in_degree", "to_out_degree", "time_step_valid", "speed_valid",
        "heading_difference_valid", "projection_valid",
    )

    class WithReferences:
        reference_arc_id = 999
        alternative_margin = -10
        target = 0

        def __getattr__(self, name):
            return getattr(features[0][1][0], name)

    np.testing.assert_array_equal(feature_matrix([WithReferences()]), feature_matrix(features[0][1][:1]))


def dataset(partition="train"):
    # Each of six trips has two candidates for each of 20 GPS fixes. Lower
    # geometric rank is always the known target, with missing motion values.
    x = np.zeros((240, len(FEATURES)), dtype=np.float32)
    y = np.tile([1, 0], 120).astype(np.int8)
    x[:, 0] = np.tile([1, 12], 120)
    x[:, 2] = np.tile([1, 2], 120)
    x[:, 3:6] = np.nan
    trip_ids = tuple(f"{j:08}" for j in range(6))
    return Dataset(partition, trip_ids, x, y, np.repeat(np.arange(6), 40),
                   np.tile(np.repeat(np.arange(20), 2), 6), np.tile([12, 9], 120), [], [])


def test_same_five_trip_folds_and_fold_only_scaler_even_with_heldout_outlier():
    data = dataset()
    folds = make_folds(data)
    assert len(folds) == 5
    assert sorted(np.concatenate([valid for _, valid in folds])) == list(range(len(data.y)))
    for train, valid in folds:
        assert not set(data.trip_codes[train]) & set(data.trip_codes[valid])
    train, valid = folds[0]
    data.X[valid, 0] = 5000
    pipe = fit_pipeline(data, "logistic_regression", train, 1)
    scaler = pipe.named_steps["features"].named_transformers_["numeric"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(6.5)
    assert data.X[:, 0].mean() > 6.5
    with pytest.raises(ValueError, match="Only training"):
        fit_pipeline(dataset("student_test"), "logistic_regression", slice(None), 1)
    with pytest.raises(ValueError, match="only the training"):
        make_folds(dataset("student_test"))


@pytest.mark.parametrize("name", MODELS)
def test_five_model_synthetic_cv_with_identical_folds_and_reloads(name, tmp_path):
    data = dataset()
    oof = np.full(len(data.y), np.nan)
    for train, valid in make_folds(data):
        pipe = fit_pipeline(data, name, train, 1)
        oof[valid] = positive_scores(pipe, data.X[valid], 40)
    metrics, by_trip, _ = evaluate(data, oof)
    assert np.all((oof >= 0) & (oof <= 1))
    assert metrics["top1"] == 1 and metrics["eligible"] == 120 and len(by_trip) == 6
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(dict(model_id=name, features=FEATURES, pipeline=pipe), artifact)
    loaded = load_model(artifact)["pipeline"]
    first = positive_scores(loaded, data.X, 40)
    second = positive_scores(fit_pipeline(data, name, train, 1), data.X, 40)
    np.testing.assert_array_equal(first, second)


def test_ranking_ties_zero_candidates_and_binary_metrics():
    class Constant:
        classes_ = np.array([0, 1])

        def predict_proba(self, matrix):
            return np.full((len(matrix), 2), 0.5)

    outcomes = list(predict_trip(Constant(), "constant", candidate_batches(sample(), CandidateConfig()), 2))
    assert [row.observation_id for row, _ in outcomes] == [0, 1, 2]
    assert [row.arc_id for row in outcomes[0][1]] == [0, 1]
    assert [row.prediction_rank for row in outcomes[0][1]] == [1, 2]
    assert outcomes[2][1] == ()
    data = dataset()
    metrics, _, _ = evaluate(data, np.full(len(data.y), 0.5))
    assert metrics["top1"] == 1 and metrics["tp"] == 0
    figures = binary_metrics(np.array([1, 1, 0, 0, 0]), np.array([.9, .1, .8, .2, .5]))
    assert (figures["tp"], figures["fp"], figures["fn"], figures["tn"]) == (1, 1, 1, 2)
    assert figures["precision"] == figures["recall"] == figures["f1"] == 0.5


def test_student_evaluation_creates_targets_only_offline(tmp_path):
    trip = sample("student_test")
    files = TripFiles("student_test", trip.trip_id, tmp_path)
    for extension, values in (("track", trip.track), ("nodes", trip.nodes),
                              ("arcs", trip.arcs), ("route", [0])):
        np.savetxt(files.path(extension), values)
    # IDs require integer source tokens.
    np.savetxt(files.path("arcs"), trip.arcs, fmt="%d")
    np.savetxt(files.path("route"), [0], fmt="%d")
    result = list(_student_rows(files, candidate_batches(trip, CandidateConfig()), AlignmentPolicy()))
    assert [row[2] for row in result] == [(1, 0), (1, 0), ()]
    assert not (tmp_path / "candidate_targets.csv").exists()


def test_selection_key_uses_only_oof_metrics():
    first = dict(top1=.95, trip_macro_top1=.90, log_loss=.17, model="a", student_top1=.0)
    second = dict(top1=.94, trip_macro_top1=.98, log_loss=.10, model="b", student_top1=1.)
    assert sorted((second, first), key=selection_key)[0] == first


def test_synthetic_experiment_writes_complete_reports_before_held_out_evaluation(tmp_path, monkeypatch):
    import argparse
    import csv
    import json
    from dataclasses import asdict

    from src.models import __main__ as experiment

    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(asdict(AlignmentPolicy())))
    seen = []

    def fixture_data(_, partition, *_args):
        seen.append(partition)
        if partition == "student_test":
            assert (reports / "model_selection.json").is_file()
            assert len(list((artifacts / "cv_checkpoints").glob("*.npz"))) == 25
            assert len(list(artifacts.glob("*.joblib"))) == 5
        result = dataset(partition)
        result.coverage.append(dict(original=125, qualified=123, eligible=120, misses=3,
                                    unqualified=2, candidate_rows=240, positives=120, negatives=120))
        return result

    monkeypatch.setattr(experiment, "build_dataset", fixture_data)
    experiment.run(argparse.Namespace(data_dir=tmp_path / "raw", alignment_dir=tmp_path / "alignment",
                                      policy=policy, artifact_dir=artifacts, report_dir=reports,
                                      workers=1, batch_size=128))
    assert seen == ["train", "student_test"]
    with (reports / "coverage_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["partition"] for row in rows] == seen
    assert float(rows[0]["proxy_label_coverage"]) == 123 / 125
    assert float(rows[0]["candidate_recall"]) == 120 / 123
    with (reports / "student_test_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert {row["model"] for row in metrics} == set(MODELS)
    assert {float(row["top1"]) for row in metrics} == {1.0}
