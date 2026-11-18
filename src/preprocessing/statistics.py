"""Compute descriptive statistics on original adjacent rows and directed arcs."""

import numpy as np

from .geometry import haversine_m, nearest_segment_m
from .graph_analysis import topology
from .loaders import Trip
from .validation import TripValidity


def distribution(name: str, values: np.ndarray) -> dict:
    if not len(values):
        return {f"{name}_{stat}": None for stat in ("min", "median", "mean", "p95", "max")}
    median, p95 = np.percentile(values, [50, 95])
    return {
        f"{name}_min": float(values.min()),
        f"{name}_median": float(median),
        f"{name}_mean": float(values.mean()),
        f"{name}_p95": float(p95),
        f"{name}_max": float(values.max()),
    }


def extent(points: np.ndarray) -> dict:
    if not len(points):
        return dict.fromkeys(("lon_min", "lat_min", "lon_max", "lat_max"))
    lon_min, lat_min = points.min(axis=0)
    lon_max, lat_max = points.max(axis=0)
    return dict(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)


def gps_statistics(trip: Trip, valid: TripValidity, gaps: tuple[int, ...],
                   speed_limit: float) -> tuple[dict, np.ndarray]:
    coordinates = trip.track[:, :2]
    deltas = valid.timestamp_deltas
    finite_deltas = deltas[np.isfinite(deltas)]
    positive = finite_deltas[finite_deltas > 0]
    usable_pairs = valid.gps[:-1] & valid.gps[1:]
    distances = np.full(len(deltas), np.nan)
    distances[usable_pairs] = haversine_m(coordinates[:-1][usable_pairs],
                                         coordinates[1:][usable_pairs])
    speed_pairs = usable_pairs & (deltas > 0)
    with np.errstate(over="ignore"):
        speeds = distances[speed_pairs] / deltas[speed_pairs]
    finite_speeds = speeds[np.isfinite(speeds)]
    times = trip.track[valid.timestamps, 2]
    with np.errstate(over="ignore"):
        duration = times.max() - times.min() if len(times) else np.nan
    decreases = int((finite_deltas < 0).sum())
    status = "decreasing" if decreases else "nondecreasing"
    if not decreases and len(finite_deltas) != len(deltas):
        status = "uncheckable"
    elif not len(finite_deltas):
        status = "insufficient"
    same_position = usable_pairs & (coordinates[:-1] == coordinates[1:]).all(axis=1)
    return {
        "gps_observations": len(trip.track),
        "missing_coordinate_rows": int(np.isnan(coordinates).any(axis=1).sum()),
        "invalid_coordinate_rows": int((~valid.gps).sum()),
        "invalid_timestamp_rows": int((~valid.timestamps).sum()),
        "duplicate_consecutive_positions": int(same_position.sum()),
        "duplicate_consecutive_observations": int((same_position & (deltas == 0)).sum()),
        "duplicate_timestamp_rows": len(times) - len(np.unique(times)),
        "timestamp_decreases": decreases,
        "uncheckable_intervals": len(deltas) - len(finite_deltas),
        "timestamp_order": status,
        "positive_intervals": len(positive),
        "one_second_intervals": int((positive == 1).sum()),
        "duration_s": float(duration) if np.isfinite(duration) else None,
        "gps_polyline_m": float(np.nansum(distances)) if usable_pairs.any() else None,
        "valid_distance_pairs": int(usable_pairs.sum()),
        "valid_speed_pairs": len(finite_speeds),
        "nonfinite_speed_pairs": len(speeds) - len(finite_speeds),
        "implausible_speed_pairs": int((speeds > speed_limit).sum()),
        **{f"gap_gt_{gap}_s": int((positive > gap).sum()) for gap in gaps},
        **extent(coordinates[valid.gps]),
        **distribution("time_diff_s", finite_deltas),
        **distribution("interval_s", positive),
        **distribution("step_m", distances[usable_pairs]),
        **distribution("speed_mps", finite_speeds),
    }, positive


def network_statistics(trip: Trip, valid: TripValidity) -> dict:
    arc_segments = trip.nodes[trip.arcs[valid.arc_geometry]]
    lengths = haversine_m(arc_segments[:, 0], arc_segments[:, 1])
    return {
        "nodes": len(trip.nodes),
        "directed_arcs": len(trip.arcs),
        "invalid_node_coordinates": int((~valid.nodes).sum()),
        "invalid_node_refs": int((~valid.arcs).sum()),
        "arcs_without_geometry": int((~valid.arc_geometry).sum()),
        "zero_length_arcs": int((lengths == 0).sum()),
        **topology(len(trip.nodes), trip.arcs[valid.arcs]),
        **extent(trip.nodes[valid.nodes]),
        **distribution("arc_length_m", lengths),
    }


def route_statistics(trip: Trip, valid: TripValidity) -> dict:
    arcs = trip.route[valid.route_ids]
    segments = trip.nodes[trip.arcs[trip.route[valid.route_geometry]]]
    lengths = haversine_m(segments[:, 0], segments[:, 1])
    start = segments[0, 0] if len(segments) and valid.route_geometry[0] else (None, None)
    end = segments[-1, 1] if len(segments) and valid.route_geometry[-1] else (None, None)
    offsets = nearest_segment_m(trip.track[valid.gps, :2], segments) if len(segments) else np.empty(0)
    return {
        "reference_arcs": len(trip.route),
        "invalid_reference_ids": int((~valid.route_ids).sum()),
        "repeated_reference_arcs": len(arcs) - len(np.unique(arcs)),
        "arcs_without_geometry": int((~valid.route_geometry).sum()),
        "disconnected_pairs": int(valid.route_disconnections.sum()),
        "uncheckable_pairs": int(valid.route_uncheckable.sum()),
        "route_arc_sum_m": float(lengths.sum()) if len(lengths) else None,
        "reference_minus_gps_count": len(trip.route) - len(trip.track),
        "start_lon": start[0], "start_lat": start[1], "end_lon": end[0], "end_lat": end[1],
        "gps_points_compared": len(offsets),
        "gps_offset_gt_30_m": int((offsets > 30).sum()),
        "gps_offset_gt_100_m": int((offsets > 100).sum()),
        **distribution("gps_to_route_m", offsets),
    }
