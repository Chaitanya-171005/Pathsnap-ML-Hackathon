"""Export per-observation and candidate features for one development trip."""

import argparse
import csv
from dataclasses import astuple, fields
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from src.preprocessing.loaders import PARTITIONS, discover_trips

from .pipeline import (
    CandidateConfig, CandidateRecord, ObservationRecord, candidate_batches, load_inference_trip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/candidates"))
    defaults = CandidateConfig()
    parser.add_argument("--radius-m", type=float, default=defaults.radius_m)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--max-gap-s", type=float, default=defaults.max_gap_s)
    parser.add_argument("--max-speed-mps", type=float, default=defaults.max_speed_mps)
    parser.add_argument("--min-heading-step-m", type=float, default=defaults.min_heading_step_m)
    args = parser.parse_args()

    files = next((files for files in discover_trips(args.data_dir)
                  if (files.partition, files.trip_id) == (args.partition, args.trip_id)), None)
    if files is None:
        parser.error("Trip was not found in the selected development partition")
    try:
        config = CandidateConfig(args.radius_m, args.top_k, args.max_gap_s,
                                 args.max_speed_mps, args.min_heading_step_m)
    except ValueError as error:
        parser.error(str(error))

    output = args.output_dir / args.partition / args.trip_id
    candidate_path, observation_path = output / "candidates.csv", output / "observations.csv"
    raw_roots = (args.data_dir.resolve(), Path("data/raw").resolve())
    if any(path.resolve().is_relative_to(root) for path in (candidate_path, observation_path)
           for root in raw_roots):
        parser.error("Output paths must be outside the raw dataset")

    trip = load_inference_trip(files)
    for issue in trip.issues:
        print(f"{issue.file} {issue.code}, line={issue.line}, column={issue.column}",
              file=sys.stderr)
    output.mkdir(parents=True, exist_ok=True)
    observations = candidates = 0
    with TemporaryDirectory(dir=output) as staging:
        staged_candidates = Path(staging) / candidate_path.name
        staged_observations = Path(staging) / observation_path.name
        with (staged_candidates.open("w", newline="", encoding="utf-8") as candidate_file,
              staged_observations.open("w", newline="", encoding="utf-8") as observation_file):
            candidate_writer = csv.writer(candidate_file, lineterminator="\n")
            observation_writer = csv.writer(observation_file, lineterminator="\n")
            candidate_writer.writerow(field.name for field in fields(CandidateRecord))
            observation_writer.writerow(field.name for field in fields(ObservationRecord))
            for observation, rows in candidate_batches(trip, config):
                observation_writer.writerow(astuple(observation))
                candidate_writer.writerows(map(astuple, rows))
                observations += 1
                candidates += len(rows)
        staged_candidates.replace(candidate_path)
        staged_observations.replace(observation_path)
    print(f"{args.partition}/{args.trip_id}: {observations:,} observations, "
          f"{candidates:,} candidates at {config.radius_m:g} m / top {config.top_k}; {output}")


if __name__ == "__main__":
    main()
