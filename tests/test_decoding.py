"""Phase 5 synthetic graph, inference, schema and offline scoring regressions."""

from dataclasses import asdict
import json
from math import log
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from src.candidates.pipeline import InferenceTrip, load_inference_trip
from src.candidates.spatial import RoadIndex
from src.decoding import CANDIDATES, DEFAULT_MODEL, MODELS, decode_trip, model_bundle
from src.decoding.__main__ import _files, _output_directory, main
from src.decoding.evaluation import add_rates, observation_metrics, route_metrics
from src.decoding.graph import Path as RoadPath, RoadGraph
from src.decoding.hmm import DecoderConfig, State, Transition, emission, transition
from src.decoding.viterbi import Viterbi
from src.models.inference import FEATURES
from src.preprocessing.loaders import TripFiles


class FixedClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        scores = np.full(len(features), .75)
        return np.column_stack((1 - scores, scores))


@pytest.fixture
def bundle():
    return {"pipeline": FixedClassifier(), "model_id": "gradient_boosting",
            "features": FEATURES, "manifest": {"candidate_config": asdict(CANDIDATES)}}


def graph(nodes, arcs, **kwargs):
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs, dtype=int)
    return RoadGraph(nodes, RoadIndex(nodes, arcs), **kwargs)


def state(arc, fraction=.5, rank=1, score=.75):
    return State(arc, rank, score, fraction, emission(score))


def test_directed_graph_preserves_parallel_arcs_and_excludes_degenerate_geometry():
    road = graph([[0, 0], [.001, 0], [.002, 0], [np.nan, 0]],
                 [[0, 1], [0, 1], [1, 2], [2, 1], [1, 1], [1, 3], [-1, 0]])
    assert list(road.arcs) == [0, 1, 2, 3]
    assert road.outgoing[0] == [0, 1]
    assert road.shortest(0, 2, 300)[0].arcs == (0, 2)
    assert road.shortest(2, 1, 300)[0].arcs == (3,)
    assert road.shortest(2, 0, 300)[0] is None
    assert road.connect(state(1, 0), state(1, 1), 200)[0].distance_m == pytest.approx(111.2, abs=.1)
    assert road.connect(state(1, 1), state(1, 0), 300)[0] is None


def test_backward_on_same_arc_requires_a_directed_loop():
    road = graph([[0, 0], [.001, 0]], [[0, 1], [1, 0]])
    path, reason = road.connect(state(0, .8), state(0, .2), 200)
    assert reason is None
    assert path.arcs == (1,)
    assert path.distance_m == pytest.approx(1.4 * road.arcs[0].length_m)
    assert road.connect(state(0, .8), state(0, .2), 100)[0] is None
    assert road.connect(state(0), state(0), 0)[0] == RoadPath(0, ())


def test_search_cache_respects_radius_and_does_not_cache_an_incomplete_search():
    nodes = [[0, 0], [.001, 0], [.002, 0], [.003, 0], [.004, .004]]
    arcs = [[0, 1], [1, 2], [2, 3]]
    road = graph(nodes, arcs, cache_size=1)
    assert road.shortest(0, 3, 160)[0] is None
    assert road.shortest(0, 3, 100)[0] is None
    assert (road.searches, road.cache_hits) == (1, 1)
    assert road.shortest(0, 3, 500)[0].arcs == (0, 1, 2)
    assert road.shortest(0, 3, 150)[0] is None
    assert road.shortest(0, 4, 500)[0] is None
    assert list(road.cache) == [(0, 4)]
    limited = graph(nodes, arcs, search_budget=2)
    for _ in range(2):
        assert limited.shortest(0, 3, 500) == (None, "routing_budget_exceeded")
    assert limited.budget_exhaustions == 2 and not limited.cache


@pytest.mark.parametrize("probability", [None, np.nan, np.inf, -1, 1.1])
def test_emissions_reject_invalid_probabilities(probability):
    assert emission(probability) is None


def test_emissions_and_transition_scores_use_log_evidence_and_speed_limits():
    assert emission(0) < emission(.2) == log(.2) < emission(1) < 0
    road = graph([[0, 0], [.001, 0]], [[0, 1]])
    config = DecoderConfig(position_slack_m=0)
    first, last = state(0, 0), state(0, 1)
    length = road.arcs[0].length_m
    assert transition(road, first, last, length, 3, 55, config)[0].score == 0
    assert transition(road, first, last, 0, 3, 55, config)[0].score < 0
    assert transition(road, first, last, length, 1, 55, config) == (None, "movement_limit")


def test_viterbi_backtracks_the_optimal_path_and_breaks_ties_by_state_order():
    first = [state(4), state(5, score=.50), state(9)]
    second = [state(7), state(8)]
    viterbi = Viterbi()
    viterbi.start(0, first)
    penalty = Transition(-2, RoadPath(0, ()), False)
    best = Transition(0, RoadPath(0, ()), False)
    assert viterbi.advance(1, second, [(0, 0, penalty), (1, 0, best), (2, 0, penalty)])
    assert [s.arc_id for _, s in viterbi.active()] == [7]
    assert [s.arc_id for _, s, _, _ in viterbi.finish()] == [5, 7]
    viterbi.start(0, first)
    assert viterbi.advance(1, second, [(0, 0, best), (2, 0, best), (2, 1, best)])
    assert [s.arc_id for _, s, _, _ in viterbi.finish()] == [4, 7]
    assert viterbi.finish() == []


def test_failed_pair_may_win_with_a_penalized_explicit_restart():
    viterbi = Viterbi()
    viterbi.start(0, [state(4)])
    slow_detour = Transition(-8, RoadPath(400, (3,)), False)
    assert viterbi.advance(1, [state(7), state(8)], [(0, 0, slow_detour)],
                           [(0, 1, "no_feasible_transition")], restart_cost=4)
    assert [(candidate.arc_id, link, restart) for _, candidate, link, restart in
            viterbi.finish()] == [(4, None, None), (8, None, "no_feasible_transition")]


def test_reconstruction_loop_and_stationary_fix_preserve_occurrences(bundle):
    trip = InferenceTrip("train", "00000001",
                         np.array([[.0032, 0, 0], [.0032, 0, 3], [.0008, 0, 30]]),
                         np.array([[0, 0], [.004, 0], [.002, .004]]),
                         np.array([[0, 1], [1, 2], [2, 0]]))
    result = decode_trip(trip, bundle).result
    assert [m["arc_id"] for m in result["matches"]] == [0, 0, 0]
    segment, = result["route_segments"]
    assert segment["arc_ids"] == [0, 1, 2, 0]
    assert [o["role"] for o in segment["occurrences"]] == [
        "selected", "connector", "connector", "selected"]
    assert segment["occurrences"][0]["last_observation_id"] == 1
    assert result["route_geojson"]["features"][0]["geometry"]["coordinates"] == [
        [0, 0], [.004, 0], [.002, .004], [0, 0], [.004, 0]]


def test_empty_groups_bad_gps_and_unreliable_time_reset_segments(bundle):
    trip = InferenceTrip(
        "train", "00000001",
        np.array([[.0005, 1, 0], [.0005, 1, 31], [.0005, 1, np.nan], [.0005, 1, 32],
                  [.0005, 1, 32], [.0005, 1, 31], [np.nan, 1, 33], [.0015, 1, 34],
                  [1, 1, 35], [.0035, 1, 36]]),
        np.array([[0, 1], [.002, 1], [.003, 1], [.005, 1]]), np.array([[0, 1], [2, 3]]))
    run = decode_trip(trip, bundle, batch_size=1)
    matches = run.result["matches"]
    assert [m["observation_id"] for m in matches] == list(range(10))
    assert [m["break_reason"] for m in matches] == [
        None, "gps_gap", "invalid_timestamp", "invalid_timestamp_pair",
        "nonpositive_time_step", "nonpositive_time_step", "invalid_gps", "invalid_gps",
        "no_candidates", "no_candidates"]
    assert matches[6]["arc_id"] is matches[8]["arc_id"] is None
    assert run.result["status"] == "disconnected"
    assert all(len(s["arc_ids"]) == 1 for s in run.result["route_segments"])
    second = decode_trip(trip, bundle, batch_size=100).result
    assert {k: v for k, v in run.result.items() if k != "processing_time_s"} == {
        k: v for k, v in second.items() if k != "processing_time_s"}


def test_disconnected_candidates_never_fabricate_a_connector(bundle):
    trip = InferenceTrip("train", "01", np.array([[.0005, 0, 0], [.0025, 0, 8]]),
                         np.array([[0, 0], [.001, 0], [.002, 0], [.003, 0]]),
                         np.array([[0, 1], [2, 3]]))
    run = decode_trip(trip, bundle)
    assert [s["arc_ids"] for s in run.result["route_segments"]] == [[0], [1]]
    assert run.result["matches"][1]["break_reason"] == "no_feasible_transition"
    metrics = route_metrics(run.result, run.graph, [0, 1, -1])
    assert metrics["arc_precision"] == metrics["arc_recall"] == 1
    assert metrics["reference_disconnected_transitions"] == 1
    assert metrics["reference_uncheckable_transitions"] == 1
    assert metrics["predicted_internal_disconnections"] == 0


def test_fast_jump_and_only_zero_length_candidate(bundle):
    trip = InferenceTrip("train", "01",
                         np.array([[.0005, 0, 0], [.0015, 0, 1], [.002, .002, 10]]),
                         np.array([[0, 0], [.002, 0], [.002, .002]]),
                         np.array([[0, 1], [2, 2]]))
    result = decode_trip(trip, bundle).result
    assert result["matches"][1]["break_reason"] == "speed_above_limit"
    assert result["matches"][2]["break_reason"] == "no_usable_states"
    assert result["matches"][2]["arc_id"] is None


def test_proxy_comparison_uses_identical_eligible_denominator():
    statuses = [(4, 4, (4, 3), 3, "accepted"), (None, 2, (2,), 2, "accepted"),
                (2, 2, (2,), 8, "accepted"), (5, 5, (5,), 5, "ambiguous")]
    matches, candidates, labels = [], [], []
    for i, (hmm, classifier, group, target, status) in enumerate(statuses):
        key = dict(partition="train", trip_id="01", observation_id=i)
        matches.append(dict(**key, arc_id=hmm, classifier_arc_id=classifier))
        candidates.append(group)
        labels.append(SimpleNamespace(**key, arc_id=target, status=status, method="qualified",
                                      provenance="test", policy_id="test"))
    row = add_rates(observation_metrics({"matches": matches}, candidates, labels))
    assert (row["original"], row["qualified"], row["eligible"], row["candidate_misses"]) == (4, 3, 2, 1)
    assert (row["classifier_top1"], row["hmm_top1"], row["top1_delta_pp"]) == (.5, 0.0, -50)


def test_cli_inference_never_checks_or_opens_reference_file(tmp_path, bundle, monkeypatch):
    root = tmp_path / "raw"
    directory = root / "train" / "00000001"
    directory.mkdir(parents=True)
    for extension, content in {"nodes": "0 0\n.001 0\n", "arcs": "0 1\n",
                               "track": ".0005 0 1\n"}.items():
        (directory / f"00000001.{extension}").write_text(content)

    original = TripFiles.path

    def guarded(file, extension):
        assert extension != "route"
        return original(file, extension)

    monkeypatch.setattr(TripFiles, "path", guarded)
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)
    output = tmp_path / "output"
    main(["--data-root", str(root), "--trip-id", "00000001", "--model-path", str(model_path),
          "--output-dir", str(output)])
    result = json.loads((output / "train/00000001.pathsnap.json").read_text())
    assert result["status"] == "connected"
    assert result["model_id"] == "phase3/gradient_boosting"
    assert result["route_geojson"] == json.loads((output / "train/00000001.pathsnap.geojson").read_text())
    assert load_inference_trip(_files(root, "train", ["00000001"], False)[0]).track.shape == (1, 3)


def test_empty_result_and_model_compatibility(tmp_path, bundle):
    assert MODELS[DEFAULT_MODEL] == Path("artifacts/models/phase3/gradient_boosting.joblib")
    assert MODELS["phase4"] == Path("artifacts/models/phase4/gradient_boosting.joblib")
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = model_bundle(path=path)
    trip = InferenceTrip("train", "01", np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2), int))
    result = decode_trip(trip, loaded).result
    assert (result["status"], result["observation_count"], result["route_segments"]) == ("empty", 0, [])
    bundle["manifest"]["candidate_config"]["top_k"] = 2
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="candidate contract"):
        model_bundle(path=path)
    bundle["features"] = ()
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="feature contract"):
        model_bundle(path=path)
    with pytest.raises(ValueError, match="outside raw data"):
        _output_directory(tmp_path / "raw/output", tmp_path / "raw")
