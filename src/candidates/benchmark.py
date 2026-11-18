"""Benchmark inference candidates on representative student-side training trips."""

import argparse
from array import array
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from src.preprocessing.loaders import discover_trips

from .pipeline import CandidateConfig, candidate_batches, load_inference_trip
from .spatial import RoadIndex


# Short, typical, GPS/reference anomaly, and the largest road network by arc count.
DEFAULT_TRIPS = ("00000026", "00000083", "00000054", "00000013")


def measure(data_dir: Path, trip_id: str, radius: float, k: int) -> dict:
    files = next((files for files in discover_trips(data_dir)
                  if files.partition == "train" and files.trip_id == trip_id), None)
    if files is None:
        raise ValueError(f"Training trip {trip_id} was not found")
    trip = load_inference_trip(files)
    config = CandidateConfig(radius_m=radius, top_k=k)
    start = perf_counter()
    index = RoadIndex(trip.nodes, trip.arcs)
    built = perf_counter()
    counts = []
    distances = array("d")
    nearest = array("d")
    valid_without_candidates = capped = 0
    for observation, candidates in candidate_batches(trip, config, index):
        counts.append(observation.candidate_count)
        valid_without_candidates += observation.candidate_status == "no_candidates"
        capped += observation.candidates_within_radius > k
        if candidates:
            nearest.append(candidates[0].distance_m)
            distances.extend(row.distance_m for row in candidates)
    elapsed = perf_counter() - built
    units_per_mib = 1024**2 if sys.platform == "darwin" else 1024
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / units_per_mib

    def percentile(values, q):
        return round(float(np.percentile(values, q)), 3) if len(values) else None

    return {
        "trip_id": trip_id, "radius_m": radius, "top_k": k,
        "gps_rows": len(trip.track), "road_arc_rows": len(trip.arcs),
        "candidate_rows": len(distances),
        "mean_candidates": round(float(np.mean(counts)), 3) if counts else None,
        "observations_without_candidates": counts.count(0),
        "valid_gps_without_candidates": valid_without_candidates,
        "observations_capped": capped,
        "nearest_p50_m": percentile(nearest, 50), "nearest_p95_m": percentile(nearest, 95),
        "candidate_p50_m": percentile(distances, 50), "candidate_p95_m": percentile(distances, 95),
        "candidate_max_m": round(max(distances), 3) if len(distances) else None,
        "build_s": round(built - start, 4), "query_features_s": round(elapsed, 4),
        "peak_rss_mib": round(peak_rss_mib, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--trips", nargs="+", default=DEFAULT_TRIPS)
    parser.add_argument("--radii", nargs="+", type=float, default=[25, 50, 100])
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 8, 16])
    parser.add_argument("--output", type=Path, default=Path("reports/phase2a/benchmark.csv"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(measure(args.data_dir, args.trips[0], args.radii[0], args.top_k[0])))
        return

    if any(args.output.resolve().is_relative_to(root)
           for root in (args.data_dir.resolve(), Path("data/raw").resolve())):
        parser.error("Benchmark output must be outside the raw dataset")
    rows = []
    for trip_id in args.trips:
        for radius in args.radii:
            for k in args.top_k:
                command = [
                    sys.executable, "-m", "src.candidates.benchmark", "--worker",
                    "--data-dir", str(args.data_dir), "--trips", trip_id,
                    "--radii", str(radius), "--top-k", str(k),
                ]
                rows.append(json.loads(subprocess.check_output(command, text=True)))
        print(f"Benchmarked training trip {trip_id}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
