import csv
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from src.alignment.__main__ import run
from src.alignment.core import AlignmentPolicy, Options, align_trip
from src.alignment.sequence import monotonic_path
from src.alignment.targets import label_candidates
from src.candidates.pipeline import CandidateConfig, InferenceTrip, candidate_batches
from src.preprocessing.geometry import EARTH_RADIUS_M
from src.preprocessing.loaders import Trip, TripFiles


METRES_PER_DEGREE = EARTH_RADIUS_M * np.pi / 180


def make_trip(points, nodes, arcs, route, times=None, partition="train"):
    gps = np.asarray(points, dtype=float).reshape(-1, 2) / METRES_PER_DEGREE
    times = np.arange(len(gps)) if times is None else times
    return Trip(TripFiles(partition, "00000001", Path(".")), np.column_stack((gps, times)),
                np.asarray(nodes, dtype=float).reshape(-1, 2) / METRES_PER_DEGREE,
                np.asarray(arcs, dtype=int).reshape(-1, 2), np.asarray(route, dtype=int), [])


def test_nearest_projection_keeps_reference_occurrence_and_raw_directed_id():
    trip = make_trip([(20, 3), (100, 50)], [(0, 0), (70, 0), (70, 50)],
                     [(1, 2), (0, 1)], [1, 0])
    first, second = align_trip(trip, AlignmentPolicy())["nearest"]
    assert (first.arc_id, first.occurrence_index) == (1, 0)
    assert first.distance_m == pytest.approx(3)
    assert first.projection_fraction == pytest.approx(20 / 70)
    assert (second.arc_id, second.occurrence_index) == (0, 1)
    assert second.distance_m == pytest.approx(30)
    assert second.projection_fraction == pytest.approx(1)
    assert align_trip(trip, AlignmentPolicy(proposal_radius_m=25))["nearest"][1].status == "unmatched"


def test_monotonic_progression_matches_multiple_gps_fixes_to_one_arc():
    trip = make_trip([(10, 1), (25, 1), (40, 1), (65, 1), (80, 1)],
                     [(0, 0), (50, 0), (100, 0)], [(0, 1), (1, 2)], [0, 1])
    labels = align_trip(trip, AlignmentPolicy())
    for method in ("monotonic", "qualified"):
        assert [row.occurrence_index for row in labels[method]] == [0, 0, 0, 1, 1]
        assert all(row.status == "accepted" for row in labels[method])


def test_revisit_of_same_arc_uses_second_occurrence_after_loop():
    trip = make_trip([(12, 1), (40, 1), (59, 18), (59, 46), (42, 41), (14, 15), (20, 1), (43, 1)],
                     [(0, 0), (60, 0), (60, 60)], [(0, 1), (1, 2), (2, 0)], [0, 1, 2, 0])
    labels = align_trip(trip, AlignmentPolicy())
    assert [r.occurrence_index for r in labels["monotonic"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert labels["nearest"][-1].status == "ambiguous"
    assert labels["nearest"][-1].distance_ties == 1
    assert labels["qualified"][-1].status == "accepted"
    assert (labels["qualified"][-1].arc_id, labels["qualified"][-1].occurrence_index) == (0, 3)


def test_reliable_heading_resolves_reverse_geometry():
    trip = make_trip([(80, 0), (60, 0), (40, 0)], [(0, 0), (100, 0)],
                     [(0, 1), (1, 0)], [0, 1])
    labels = align_trip(trip, AlignmentPolicy())
    assert labels["nearest"][1].arc_id == 0
    assert labels["nearest"][1].status == "ambiguous"
    assert labels["nearest"][1].distance_ties == 1
    assert labels["qualified"][1].arc_id == 1
    assert labels["qualified"][1].heading_difference_deg == pytest.approx(0)
    assert labels["qualified"][1].status == "accepted"


def test_sequence_cannot_use_its_own_tied_matches_as_ambiguity_anchors():
    trip = make_trip([(40, 0)] * 8, [(0, 0), (100, 0)],
                     [(0, 1), (1, 0)], [0, 1])
    labels = align_trip(trip, AlignmentPolicy())
    assert all(row.arc_id is not None for row in labels["monotonic"])
    assert all(row.alternative_margin == 0 for row in labels["qualified"])
    assert {row.status for row in labels["qualified"]} == {"ambiguous"}


def test_parallel_roads_with_close_score_cause_abstention():
    trip = make_trip([(40, 1)], [(0, 0), (100, 0), (100, 4), (0, 4)],
                     [(0, 1), (1, 2), (2, 3)], [0, 1, 2])
    labels = align_trip(trip, AlignmentPolicy())
    assert labels["nearest"][0].status == labels["monotonic"][0].status == "accepted"
    row = labels["qualified"][0]
    assert row.arc_id == 0 and row.alternative_margin == pytest.approx(2)
    assert row.status == "ambiguous" and row.reason == "small_sequence_feasible_margin"
    assert align_trip(trip, AlignmentPolicy(qualified_margin=1))["qualified"][0].status == "accepted"


def test_identical_directed_geometry_of_distinct_arc_ids_is_ambiguous():
    trip = make_trip([(40, 0)], [(0, 0), (100, 0), (0, 0), (100, 0)],
                     [(0, 1), (2, 3)], [0])
    labels = align_trip(trip, AlignmentPolicy())
    for rows in labels.values():
        assert rows[0].status == "ambiguous"
        assert rows[0].reason == "duplicate_directed_geometry" and rows[0].duplicate_directed_geometry


def test_disconnected_reference_localizes_abstention():
    trip = make_trip([(50, 0), (97, 0), (203, 0), (230, 0)], [(0, 0), (100, 0), (200, 0), (300, 0)],
                     [(0, 1), (2, 3)], [0, 1], times=[0, 2, 8, 10])
    labels = align_trip(trip, AlignmentPolicy())
    assert [r.section_index for r in labels["monotonic"]] == [0, 0, 1, 1]
    assert [r.status for r in labels["qualified"]] == [
        "accepted", "disconnected_reference", "disconnected_reference", "accepted",
    ]


def test_missing_reference_occurrences_keep_original_index():
    trip = make_trip([(30, 0), (260, 0), (280, 0)], [(0, 0), (100, 0), (200, 0), (300, 0)],
                     [(0, 1), (2, 3)], [0, 917, 1], times=[0, 20, 22])
    rows = align_trip(trip, AlignmentPolicy())["monotonic"]
    assert [r.occurrence_index for r in rows] == [0, 2, 2]
    assert rows[1].status == "disconnected_reference"


def test_invalid_observations_gaps_and_unmatched_preserve_all_ids():
    trip = make_trip([(10, 0), (20, 0), (70, 0), (75, 0), (np.nan, 0), (84, 0),
                      (90, 0), (95, 900)], [(0, 0), (100, 0)], [(0, 1)], [0],
                     times=[0, 2, 100, 101, 102, np.nan, 104, 105])
    rows = align_trip(trip, AlignmentPolicy())["qualified"]
    assert [r.observation_id for r in rows] == list(range(8))
    assert [r.status for r in rows] == [
        "accepted", "accepted", "ambiguous", "accepted", "invalid_observation",
        "invalid_observation", "ambiguous", "unmatched",
    ]
    assert rows[2].gap_before and rows[2].reason == "large_timestamp_gap"
    assert rows[6].reason == "unreliable_timestamp_step"
    assert rows[-1].reason == "outside_proposal_radius"


def test_missing_and_zero_length_references_abstain():
    trip = make_trip([(0, 0)], [(0, 0)], [(0, 0)], [])
    assert align_trip(trip, AlignmentPolicy())["nearest"][0].reason == "no_drawable_reference"
    trip.route = np.array([0])
    rows = align_trip(trip, AlignmentPolicy())["qualified"]
    assert rows[0].status == "ambiguous"
    assert rows[0].reason == "zero_length_reference_arc"


def test_sequence_skips_expensive_backward_match():
    trip = make_trip([(15, 0), (160, 0), (22, 0), (170, 0)],
                     [(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2)], [0, 1],
                     times=[0, 4, 8, 12])
    rows = align_trip(trip, AlignmentPolicy())["monotonic"]
    assert rows[0].occurrence_index == 0 and rows[-1].occurrence_index == 1
    assert not (rows[1].occurrence_index == 1 and rows[2].occurrence_index == 0)
    assert "sequence_skipped" in {r.reason for r in rows}


def test_sparse_sequence_score_matches_exhaustive_small_example():
    occs = [[0, 1], [0, 1, 3], [], [1, 2, 4], [3, 4]]
    costs = [[1, 2], [4, 1, 3], [], [1, 2, 2], [2, 1]]
    sections = np.array([0, 0, 0, 1, 1])
    gaps = np.array([False, False, True, False, False])
    options = [Options(np.array(occ), np.array(cost), np.array([]), np.array([]), np.array(cost))
               for occ, cost in zip(occs, costs)]
    path = monotonic_path(options, gaps, sections, match_reward=5, skip_cost=2, break_cost=1.5)

    def score(choices):
        reward = 0
        last = None
        for row, j in enumerate(choices):
            if j is None:
                continue
            occ = occs[row][j]
            if last is not None:
                if occ < last:
                    return -np.inf
                # The gap is immediately before row 3's first available options.
                reward -= (0 if row == 3 else 2) * max(0, occ - last - 1)
                reward -= 1.5 * (sections[occ] - sections[last])
            reward += 5 - costs[row][j]
            last = occ
        return reward

    exhaustive = product(*([None, *range(len(occ))] for occ in occs))
    assert score([path.get(i) for i in range(len(occs))]) == max(map(score, exhaustive))


def test_candidate_binary_joins_preserve_keys_and_never_create_all_negative_groups():
    trip = make_trip([(40, 5)], [(0, 0), (100, 0), (0, 5), (100, 5)],
                     [(0, 1), (2, 3)], [0])
    policy = AlignmentPolicy()
    data = InferenceTrip(trip.files.partition, trip.files.trip_id, trip.track, trip.nodes, trip.arcs)
    row = align_trip(trip, policy)["qualified"][0]
    assert row.status == "accepted"
    _, both = next(candidate_batches(data, CandidateConfig(radius_m=10, top_k=2)))
    outcome, targets = label_candidates(row, both)
    assert outcome == "included"
    assert [(c.arc_id, c.target) for c in targets] == [(1, 0), (0, 1)]
    assert all(c.observation_id == 0 and c.policy_id == policy.policy_id for c in targets)
    assert label_candidates(row, both[:1]) == ("candidate_miss", ())
    assert label_candidates(row, ()) == ("candidate_miss", ())
    assert label_candidates(replace(row, status="unmatched"), both) == ("unqualified", ())
    with pytest.raises(ValueError, match="does not match"):
        label_candidates(row, (replace(both[0], partition="student_test"),))
    with pytest.raises(ValueError, match="Duplicate candidate"):
        label_candidates(row, (*both, both[0]))


def test_alignment_is_deterministic_and_reference_changes_leave_inference_identical():
    trip = make_trip([(40, 1), (50, 1)], [(0, 0), (100, 0), (0, 4), (100, 4)],
                     [(0, 1), (2, 3)], [0])
    data = InferenceTrip(trip.files.partition, trip.files.trip_id, trip.track, trip.nodes, trip.arcs)
    cfg = CandidateConfig()
    policy = AlignmentPolicy()
    first_candidates = list(candidate_batches(data, cfg))
    first_labels = align_trip(trip, policy)
    assert first_labels == align_trip(trip, policy)
    trip.route = np.array([1])
    second_labels = align_trip(trip, policy)
    assert first_labels["qualified"][0].arc_id != second_labels["qualified"][0].arc_id
    assert first_candidates == list(candidate_batches(data, cfg))


def test_partition_reports_and_label_exports(tmp_path):
    raw = tmp_path / "raw"
    for part in ("train", "student_test", "evaluator_test"):
        directory = raw / part / "00000001"
        directory.mkdir(parents=True)
        for suffix, data in {
            "track": "0.0003 0 0\n0.0004 0 1\n",
            "nodes": "0 0\n0.001 0\n0 0.0001\n0.001 0.0001\n",
            "arcs": "0 1\n2 3\n", "route": "0\n",
        }.items():
            (directory / f"00000001.{suffix}").write_text(data)
    reports, exports = tmp_path / "reports", tmp_path / "exports"
    cfg = AlignmentPolicy()
    run(raw, "train", reports, exports, cfg)
    first_export = {p.name: p.read_bytes() for p in (exports / "train").iterdir()}
    run(raw, "train", reports, exports, cfg)
    assert first_export == {p.name: p.read_bytes() for p in (exports / "train").iterdir()}
    run(raw, "student_test", reports, exports, cfg)
    assert (reports / "train" / "policy.json").read_bytes() == (
        reports / "student_test" / "policy.json").read_bytes()
    with (exports / "train" / "candidate_targets.csv").open(newline="") as stream:
        targets = list(csv.DictReader(stream))
    assert len(targets) == 4 and sum(int(row["target"]) for row in targets) == 2
    assert {r["policy_id"] for r in targets} == {cfg.policy_id}
    assert set(p.name for p in (exports / "student_test").iterdir()) == {
        "labels.csv", "candidate_join_status.csv",
    }
    assert not (reports / "student_test" / "training_sensitivity.csv").exists()
    with pytest.raises(ValueError, match="Select one supported partition"):
        run(raw, "evaluator_test", reports, exports, cfg)
