"""Reference-route diagnostics for training trip 00000054, isolated from inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.preprocessing.geometry import haversine_m, nearest_segment_m
from src.preprocessing.loaders import discover_trips, load_trip
from src.preprocessing.validation import validate_trip

from .spatial import RoadIndex


def diagnose(data_dir: Path) -> dict:
    files = next((files for files in discover_trips(data_dir)
                  if (files.partition, files.trip_id) == ("train", "00000054")), None)
    if files is None:
        raise ValueError("Training trip 00000054 was not found")
    trip = load_trip(files)
    valid = validate_trip(trip)
    if not valid.gps.all() or not valid.timestamps.all() or not valid.route_geometry.all():
        raise ValueError("Diagnostic requires valid GPS coordinates, timestamps and route geometry")
    if len(trip.track) < 2 or not len(trip.route):
        raise ValueError("Diagnostic requires a nonempty reference and at least two GPS rows")

    gps = trip.track[:, :2]
    segments = trip.nodes[trip.arcs[trip.route]]
    reference_offsets = nearest_segment_m(gps, segments)
    endpoint_offsets = haversine_m(gps, segments[-1, 1])
    closest_end = int(np.argmin(endpoint_offsets))
    tail = closest_end + 1
    tail_points = gps[tail:]
    off_reference = reference_offsets > 100
    index = RoadIndex(trip.nodes, trip.arcs)
    nearest_roads = np.fromiter(
        (distances[0] if len(distances) else np.inf
         for _, distances, _ in index.nearby(index.projection.project(gps), 100.0)),
        dtype=float, count=len(gps),
    )
    tail_steps = haversine_m(tail_points[:-1], tail_points[1:])
    return {
        "trip_id": files.trip_id,
        "gps_rows": len(gps),
        "reference_arcs": len(segments),
        "reference_directed_breaks": int(valid.route_disconnections.sum()),
        "reference_arc_sum_m": float(haversine_m(segments[:, 0], segments[:, 1]).sum()),
        "gps_first_to_reference_start_m": float(haversine_m(gps[0], segments[0, 0])),
        "gps_last_to_reference_end_m": float(endpoint_offsets[-1]),
        "closest_to_ref_end_observation_id": closest_end,
        "closest_to_ref_end_distance_m": float(endpoint_offsets[closest_end]),
        "off_reference_gt_100_m": int(off_reference.sum()),
        **{f"off_reference_with_road_within_{radius}_m":
           int((off_reference & (nearest_roads <= radius)).sum()) for radius in (25, 50, 100)},
        "tail_start_observation_id": tail if len(tail_points) else None,
        "tail_gps_rows": len(tail_points),
        "tail_off_reference_gt_100_m": int(off_reference[tail:].sum()),
        "tail_duration_s": float(trip.track[-1, 2] - trip.track[tail, 2]) if len(tail_points) else None,
        "tail_max_radius_from_reference_end_m": float(endpoint_offsets[tail:].max())
        if len(tail_points) else None,
        "tail_path_m": float(tail_steps.sum()),
        "tail_step_median_m": float(np.median(tail_steps)) if len(tail_steps) else None,
        "tail_net_displacement_m": float(haversine_m(tail_points[0], tail_points[-1]))
        if len(tail_points) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(json.dumps(diagnose(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
