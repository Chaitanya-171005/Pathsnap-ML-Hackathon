import csv
import subprocess
import sys

import numpy as np
import pytest

from src.candidates.pipeline import (
    CandidateConfig, InferenceTrip, candidate_batches, load_inference_trip,
)
from src.candidates.spatial import RoadIndex, project_onto_segments
from src.preprocessing.geometry import EARTH_RADIUS_M
from src.preprocessing.loaders import TripFiles


METRES_PER_DEGREE = EARTH_RADIUS_M * np.pi / 180


def trip(track, nodes, arcs):
    return InferenceTrip("train", "00000007", np.array(track, dtype=float).reshape(-1, 3),
                         np.array(nodes, dtype=float).reshape(-1, 2),
                         np.array(arcs, dtype=int).reshape(-1, 2))


def test_segment_clamps_and_directed_fractions():
    starts = np.array([[0, 0], [10, 0], [0, 0], [10, 0], [5, 4]])
    ends = np.array([[10, 0], [0, 0], [1, 0], [8, 0], [5, 4]])
    distance, fraction = project_onto_segments(np.array([2, 3]), starts, ends)
    np.testing.assert_allclose(distance, [3, 3, np.sqrt(10), np.sqrt(45), np.sqrt(10)])
    np.testing.assert_allclose(fraction, [0.2, 0.8, 1, 1, np.nan])


def test_parallel_duplicate_reverse_and_tie_order():
    data = trip([[0.00002, 0, 0], [0.00004, 0, 1]],
                [[0, 0], [0.0001, 0], [0, 0.0001], [0.0001, 0.0001]],
                [[99, 1], [0, 1], [0, 1], [1, 0], [2, 3]])
    config = CandidateConfig(radius_m=20, top_k=4)
    obs, rows = list(candidate_batches(data, config))[1]
    assert obs.candidates_within_radius == obs.candidate_count == 4
    assert [r.arc_id for r in rows] == [1, 2, 3, 4]
    assert [r.distance_rank for r in rows] == [1, 2, 3, 4]
    assert [r.distance_m for r in rows] == pytest.approx(
        [0, 0, 0, 0.0001 * METRES_PER_DEGREE])
    assert [r.arc_length_m for r in rows] == pytest.approx([0.0001 * METRES_PER_DEGREE] * 4)
    assert [r.heading_difference_deg for r in rows] == pytest.approx([0, 0, 180, 0])
    assert [r.projection_fraction for r in rows] == pytest.approx([0.4, 0.4, 0.6, 0.4])
    assert rows[0].trip_id == "00000007" and rows[0].observation_id == 1
    assert (rows[0].from_in_degree, rows[0].from_out_degree,
            rows[0].to_in_degree, rows[0].to_out_degree) == (1, 2, 2, 1)
    limited, candidates = list(candidate_batches(data, CandidateConfig(radius_m=20, top_k=2)))[1]
    assert limited.candidates_within_radius == 4
    assert [r.arc_id for r in candidates] == [1, 2]


def test_distinct_segments_tied_at_radius_are_ordered_by_arc_id():
    nodes = np.array([[0, 0], [0, -1 / METRES_PER_DEGREE], [0, 1 / METRES_PER_DEGREE]])
    arcs = np.array([[1, 1], [2, 2]])
    index = RoadIndex(nodes, arcs)
    result, distance, _ = next(index.nearby(index.projection.project(nodes[:1]), 1))
    np.testing.assert_array_equal(result, [0, 1])
    np.testing.assert_allclose(distance, [1, 1])
    below, *_ = next(index.nearby(index.projection.project(nodes[:1]), np.nextafter(1.0, 0)))
    assert not len(below)


def test_zero_length_kept_and_invalid_geometry_excluded():
    data = trip([[0, 0, 0], [0.00002, 0, 1]], [[0, 0], [0, 0], [0, 91]],
                [[0, 1], [0, 2], [-1, 1]])
    _, candidates = list(candidate_batches(data, CandidateConfig()))[1]
    [row] = candidates
    assert row.arc_id == 0 and row.arc_length_m == 0
    assert row.distance_m == pytest.approx(0.00002 * METRES_PER_DEGREE)
    assert row.projection_fraction is None and not row.projection_valid
    assert row.heading_difference_deg is None and not row.heading_difference_valid
    assert row.from_out_degree == 2  # Endpoint-valid arcs count even without geometry.


@pytest.mark.parametrize("have_network", [True, False])
def test_empty_groups_preserve_observation_ids(have_network):
    data = trip([[40, 40, 7], [np.nan, 0, 8], [0, 0, np.nan]],
                [[0, 0]] if have_network else [], [[0, 0]] if have_network else [])
    batches = list(candidate_batches(data, CandidateConfig()))
    assert [obs.observation_id for obs, _ in batches] == [0, 1, 2]
    assert [obs.candidate_status for obs, _ in batches] == [
        "no_candidates", "invalid_gps", "ok" if have_network else "no_candidates",
    ]
    assert [obs.timestamp_valid for obs, _ in batches] == [True, True, False]
    if have_network:
        assert batches[-1][1][0].delta_t_s is None
        assert not batches[-1][1][0].time_step_valid


def test_gaps_missing_and_unreliable_motion():
    data = trip([[0, 0, 0], [0, 0, 1], [0.00003, 0, 31], [0.00006, 0, 62],
                 [0.0001, 0, 62], [0.00012, 0, 61], [np.nan, 0, 63],
                 [0.00014, 0, 64], [0.001, 0, 65], [0.001, 0, np.inf],
                 [0.001, 0, 71]], [[0, 0], [0.01, 0]], [[0, 1]])
    batches = list(candidate_batches(data, CandidateConfig()))
    assert [obs.motion_status for obs, _ in batches] == [
        "first_observation", "short_step", "ok", "gps_gap", "nonpositive_time_step",
        "nonpositive_time_step", "invalid_gps_pair", "invalid_gps_pair",
        "speed_above_limit", "invalid_timestamp_pair", "invalid_timestamp_pair",
    ]
    stationary = batches[1][1][0]
    assert stationary.speed_mps == 0 and stationary.speed_valid
    assert stationary.heading_difference_deg is None and not stationary.heading_difference_valid
    moving = batches[2][1][0]
    assert moving.delta_t_s == 30 and moving.time_step_valid and moving.heading_difference_valid
    gap = batches[3][1][0]
    assert gap.delta_t_s == 31 and gap.speed_mps is None and not gap.time_step_valid
    after_missing = batches[7][1][0]
    assert after_missing.time_step_valid and not after_missing.speed_valid
    assert batches[5][1][0].delta_t_s == -1


@pytest.mark.parametrize("config", [
    {"radius_m": 0}, {"radius_m": np.nan}, {"radius_m": np.inf}, {"top_k": 0},
    {"top_k": 2.0}, {"top_k": True}, {"max_gap_s": -1}, {"min_heading_step_m": np.inf},
])
def test_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        CandidateConfig(**config)


def test_large_index_matches_brute_force_and_long_segments_near_endpoints():
    rng = np.random.default_rng(2026)
    xy = rng.uniform(-5000, 5000, (65_000, 2))
    xy[0], xy[1] = [0, 0], [4999, 0]
    nodes = xy / METRES_PER_DEGREE
    pairs = np.column_stack((np.arange(0, len(nodes), 2), np.arange(1, len(nodes), 2)))
    nodes[3::2] = nodes[2::2] + rng.uniform(-0.001, 0.001, nodes[3::2].shape)
    index = RoadIndex(nodes, pairs)
    points = index.projection.project(np.array([[0, 5 / METRES_PER_DEGREE], [0.01, 0.002]]))
    starts, ends = index.start, index.end
    delta = ends - starts
    assert len(index.arc_ids) == 32_500
    for point, (ids, distances, _) in zip(points, index.nearby(points, 15), strict=True):
        fraction = ((point - starts) * delta).sum(axis=1) / (delta**2).sum(axis=1)
        projected = starts + np.clip(fraction, 0, 1)[:, None] * delta
        brute_force = np.linalg.norm(point - projected, axis=1)
        expected = np.flatnonzero(brute_force <= 15)
        expected = expected[np.lexsort((expected, brute_force[expected]))]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(distances, brute_force[expected], atol=1e-9)
    result = list(index.nearby(points, 15))
    assert 0 in result[0][0]  # Its midpoint is about 2.5 km from the query.


def test_trip_local_projection_across_dateline():
    data = trip([[180, 0.0001, 0]], [[179.999, 0], [-179.999, 0]], [[0, 1]])
    _, candidates = next(candidate_batches(data, CandidateConfig(radius_m=12)))
    assert candidates[0].distance_m == pytest.approx(11.1195, abs=0.001)
    assert candidates[0].projection_fraction == pytest.approx(0.5)


def test_inference_and_export_are_independent_of_the_route(tmp_path):
    root = tmp_path / "raw"
    path = root / "train" / "00005"
    path.mkdir(parents=True)
    files = TripFiles("train", "00005", path)
    files.path("track").write_text("0 0 0\nNaN 0 1\n")
    files.path("nodes").write_text("0 0\n0.001 0\n")
    files.path("arcs").write_text("0 1\n")
    # A directory here would cause a file-level issue if the inference loader opened it.
    files.path("route").mkdir()
    data = load_inference_trip(files)
    assert data.issues == ()
    output = tmp_path / "out"
    command = [sys.executable, "-m", "src.candidates", "--data-dir", str(root), "--partition",
               "train", "--trip-id", "00005", "--output-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    results = output / "train" / "00005"
    original = {f.name: f.read_bytes() for f in results.iterdir()}
    files.path("route").rmdir()
    files.path("route").write_text("999\n-1\n")
    subprocess.run(command, check=True, capture_output=True)
    assert {f.name: f.read_bytes() for f in results.iterdir()} == original
    with (results / "candidates.csv").open(newline="") as stream:
        [candidate] = list(csv.DictReader(stream))
    assert candidate["trip_id"] == "00005" and candidate["arc_id"] == "0"
    assert candidate["heading_difference_deg"] == ""
    assert candidate["heading_difference_valid"] == "False"
    blocked = subprocess.run([*command[:-1], str(root)], capture_output=True, text=True)
    assert blocked.returncode == 2 and "outside the raw dataset" in blocked.stderr
