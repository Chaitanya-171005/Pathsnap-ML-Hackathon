import numpy as np
import pytest

from src.preprocessing.geometry import haversine_m, nearest_segment_m
from src.preprocessing.graph_analysis import topology
from src.preprocessing.loaders import (
    TripFiles, discover_trips, load_arcs, load_nodes, load_route, load_track, load_trip,
)
from src.preprocessing.statistics import gps_statistics, network_statistics, route_statistics
from src.preprocessing.validation import validate_trip, valid_coordinates


@pytest.mark.parametrize(("loader", "extension", "content", "expected"), [
    (load_track, "track", "-1.5\t 48.2  20\r\n-1.4 48.1\t19\n",
     [[-1.5, 48.2, 20], [-1.4, 48.1, 19]]),
    (load_nodes, "nodes", "-1.5 48.2\n -1.4\t48.1", [[-1.5, 48.2], [-1.4, 48.1]]),
    (load_arcs, "arcs", "2 1\n1\t2\n", [[2, 1], [1, 2]]),
    (load_route, "route", " 1\n\t0\n 1 ", [1, 0, 1]),
])
def test_file_parsing(tmp_path, loader, extension, content, expected):
    path = tmp_path / f"00000002.{extension}"
    path.write_text(content)
    rows, issues = loader(path)
    np.testing.assert_array_equal(rows, expected)
    assert issues == []
    assert rows.dtype == (np.int64 if extension in {"arcs", "route"} else np.float64)


@pytest.mark.parametrize(("loader", "shape"), [
    (load_track, (0, 3)), (load_nodes, (0, 2)), (load_arcs, (0, 2)), (load_route, (0,)),
])
@pytest.mark.parametrize("exists", [True, False])
def test_missing_or_empty_file(tmp_path, loader, shape, exists):
    path = tmp_path / "trip.data"
    if exists:
        path.touch()
    rows, issues = loader(path)
    assert rows.shape == shape
    assert [issue.code for issue in issues] == ["empty_file" if exists else "missing_file"]


def test_malformed_tracks_and_missing_values_keep_original_rows(tmp_path):
    path = tmp_path / "trip.track"
    path.write_bytes(b"1 2 100\n\nNaN 3\n1 2 bad\n1 2 9 extra\n3 4 5\n3 \xff 6")
    data, issues = load_track(path)
    assert data.shape == (7, 3)
    assert [(i.code, i.line, i.column) for i in issues] == [
        ("column_count", 2, None), ("column_count", 3, None),
        ("invalid_number", 4, 3), ("column_count", 5, None), ("invalid_number", 7, 2),
    ]
    assert np.isnan(data[1]).all() and np.isnan(data[4]).all()
    assert np.isnan(data[2, 0]) and np.isnan(data[2, 2])
    np.testing.assert_equal(data[5], [3, 4, 5])


def test_integer_format_and_overflow_preserve_arc_positions(tmp_path):
    path = tmp_path / "trip.arcs"
    path.write_text("0 1\n1 1.0\n3 9223372036854775808\n2 0")
    rows, issues = load_arcs(path)
    np.testing.assert_array_equal(rows, [[0, 1], [1, -1], [3, -1], [2, 0]])
    assert [i.line for i in issues] == [2, 3]


@pytest.fixture
def example_trip(tmp_path):
    files = TripFiles("train", "00000009", tmp_path)
    for ext, content in {
        "track": "0 0 1\n0 0 1\n0.01 0 9\n0.01 0 8\ninf 0 nan\n0.02 0 10\n0 91 11",
        "nodes": "0 0\n0.01 0\n0.02 0\n0 91",
        "arcs": "0 1\n1 2\n2 1\n2 4\n2 3\n-1 0",
        "route": "0\n1\n0\n9\n3\n4\n2",
    }.items():
        files.path(ext).write_text(content)
    return load_trip(files)


def test_invalid_ids_geometry_and_directed_continuity(example_trip):
    valid = validate_trip(example_trip)
    assert valid.arcs.tolist() == [True, True, True, False, True, False]
    assert valid.arc_geometry.tolist() == [True, True, True, False, False, False]
    assert valid.route_ids.tolist() == [True, True, True, False, True, True, True]
    assert valid.route_disconnections.tolist() == [False, True, False, False, False, True]
    assert valid.route_uncheckable.tolist() == [False, False, True, True, True, False]
    metrics = route_statistics(example_trip, valid)
    assert metrics["invalid_reference_ids"] == 1
    assert metrics["repeated_reference_arcs"] == 1
    assert metrics["arcs_without_geometry"] == 3
    assert metrics["route_arc_sum_m"] == pytest.approx(4447.8, abs=0.1)
    assert metrics["start_lon"] == 0
    assert metrics["end_lon"] == 0.01
    assert metrics["gps_points_compared"] == 5
    assert metrics["gps_to_route_m_max"] < 0.001
    graph = network_statistics(example_trip, valid)
    assert graph["invalid_node_coordinates"] == 1
    assert graph["invalid_node_refs"] == 2


def test_duplicate_and_non_monotonic_timestamps(example_trip):
    valid = validate_trip(example_trip)
    np.testing.assert_equal(valid.timestamp_deltas, [0, 8, -1, np.nan, np.nan, 1])
    metrics, intervals = gps_statistics(example_trip, valid, (5, 10), 55)
    np.testing.assert_array_equal(intervals, [8, 1])
    assert metrics["duplicate_consecutive_positions"] == 2
    assert metrics["duplicate_consecutive_observations"] == 1
    assert metrics["duplicate_timestamp_rows"] == 1
    assert metrics["timestamp_order"] == "decreasing"
    assert metrics["uncheckable_intervals"] == 2
    assert metrics["time_diff_s_min"] == -1
    assert metrics["valid_speed_pairs"] == metrics["implausible_speed_pairs"] == 1
    assert metrics["gap_gt_5_s"] == 1
    assert metrics["gap_gt_10_s"] == 0


def test_distance_longitude_latitude_and_dateline():
    assert haversine_m([0, 0], [1, 0]) == pytest.approx(111_195, abs=2)
    assert haversine_m([0, 60], [1, 60]) == pytest.approx(55_597, abs=2)
    assert haversine_m([0, 60], [0, 61]) == pytest.approx(111_195, abs=2)
    assert haversine_m([179.9, 0], [-179.9, 0]) == pytest.approx(22_239, abs=2)
    assert valid_coordinates(np.array([[122, 48], [48, 122], [0, np.inf]])).tolist() == [
        True, False, False,
    ]


def test_point_to_segment_interior_endpoint_and_degenerate_geometry():
    points = np.array([[0.01, 0.001], [0.03, 0], [1, 0]])
    segments = np.array([[[0, 0], [0.02, 0]], [[1, 0], [1, 0]]])
    np.testing.assert_allclose(nearest_segment_m(points, segments), [111.195, 1111.951, 0],
                               atol=0.002)
    np.testing.assert_allclose(nearest_segment_m(np.array([[-179.95, 0]]),
                                                 np.array([[[179.9, 0], [-179.9, 0]]])),
                               [0], atol=1e-6)


def test_sparse_direction_and_isolated_components():
    arcs = np.array([[0, 1], [0, 1], [1, 0], [1, 2], [3, 3]])
    actual = topology(5, arcs)
    assert actual["duplicate_directed_edges"] == actual["isolated_nodes"] == actual["self_loops"] == 1
    assert actual["weak_size_frequency"] == {1: 2, 3: 1}
    assert actual["strong_size_frequency"] == {1: 3, 2: 1}
    assert topology(0, np.empty((0, 2), dtype=int))["weak_components"] == 0


def test_deterministic_inventory_excludes_evaluator_and_preserves_zeroes(tmp_path):
    for path in ("train/00003", "train/00001", "student_test/00002",
                 "evaluator_test/00004", "train/not_a_trip"):
        (tmp_path / path).mkdir(parents=True)
    (tmp_path / "train" / "00005").touch()
    expected = [("train", "00001"), ("train", "00003"), ("student_test", "00002")]
    for _ in range(2):
        assert [(f.partition, f.trip_id) for f in discover_trips(tmp_path)] == expected
