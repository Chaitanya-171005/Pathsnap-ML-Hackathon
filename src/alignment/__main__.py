"""Run offline proxy alignment and candidate inclusion reports for one partition."""

import argparse
from array import array
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
import csv
import json
from operator import attrgetter
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from src.candidates.spatial import RoadIndex
from src.preprocessing.loaders import PARTITIONS, discover_trips, load_trip
from src.preprocessing.validation import validate_trip

from .core import AlignmentPolicy, LabelRecord, align_trip, qualification
from .report import (CAPS, DIFFICULT_IDS, RADII, aggregate_recall, comparison_rows, coverage,
                     difficult_rows, plot_difficult, recall_rows, write_summary)
from .targets import CandidateTarget, targets_for_arc_ids


def _write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _record_writer(stream, record_type):
    names = [f.name for f in fields(record_type)]
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(names)
    return writer, attrgetter(*names)


def _measure_candidate_ranks(trip, labels, join_writer, target_writer, target_values):
    """One label-independent 100 m road query serves all radius and cap combinations."""
    index = RoadIndex(trip.nodes, trip.arcs)
    valid = validate_trip(trip)
    valid_ids = np.flatnonzero(valid.gps)
    points = index.projection.project(trip.track[valid_ids, :2])
    n = len(trip.track)
    per_method = {method: np.array([r.arc_id if r.status == "accepted" else -1 for r in records])
                  for method, records in labels.items()}
    ranks = {(method, radius): np.zeros(n, dtype=int) for method in labels for radius in RADII}
    empty_at = {radius: np.ones(n, dtype=bool) for radius in RADII}
    hits = iter(index.nearby(points, max(RADII)))
    qualified = labels["qualified"]
    for row in qualified:
        obs = row.observation_id
        road_ids = ()
        if valid.gps[obs]:
            nearby, distances, _ = next(hits)
            ids = index.arc_ids[nearby]
            counts = np.searchsorted(distances, RADII, side="right")
            road_ids = ids[:min(8, counts[1])]
            for radius, count in zip(RADII, counts, strict=True):
                empty_at[radius][obs] = count == 0
            for method, expected in per_method.items():
                if expected[obs] < 0:
                    continue
                at = np.flatnonzero(ids == expected[obs])
                if len(at):
                    rank = int(at[0]) + 1
                    for radius, count in zip(RADII, counts, strict=True):
                        if rank <= count:
                            ranks[method, radius][obs] = rank
        outcome, targets = targets_for_arc_ids(row, map(int, road_ids))
        rank = ranks["qualified", 50][obs]
        reason = {"included": "target_present", "unqualified": row.reason}.get(outcome)
        if outcome == "candidate_miss":
            reason = ("empty_candidate_group" if empty_at[50][obs] else
                      "top_k_truncation" if rank > 8 else "target_outside_radius")
        join_writer.writerow((row.partition, row.trip_id, obs, outcome, reason,
                              len(road_ids), int(rank) if rank else None, 50, 8, row.policy_id))
        if target_writer is not None:
            target_writer.writerows(map(target_values, targets))
    return list(recall_rows(trip.files.trip_id, labels, valid_ids, empty_at, ranks))


def run(data_dir: Path, partition: str, report_dir: Path, export_dir: Path,
        policy: AlignmentPolicy, trip_ids: list[str] | None = None, plots: bool = False) -> None:
    if partition not in PARTITIONS:
        raise ValueError("Select one supported partition")
    started = perf_counter()
    files = [f for f in discover_trips(data_dir) if f.partition == partition]
    if trip_ids is not None:
        if set(trip_ids) - {f.trip_id for f in files}:
            raise ValueError("One or more trip IDs were absent from the selected partition")
        files = [f for f in files if f.trip_id in trip_ids]
    if not files:
        raise ValueError("No trips were found in the selected partition")
    report_dir, export_dir = report_dir / partition, export_dir / partition
    if report_dir.resolve() == export_dir.resolve():
        raise ValueError("Report and export directories must differ")
    raw_roots = (data_dir.resolve(), Path("data/raw").resolve())
    if any(path.resolve().is_relative_to(root) for path in (report_dir, export_dir)
           for root in raw_roots):
        raise ValueError("Outputs must be outside raw data directories")
    report_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    trip_rows, all_recalls, all_pairs, difficult, times = [], [], [], [], []
    accepted_distances = {}
    reasons = Counter()
    sensitivity = Counter()
    variants = {}
    if partition == "train":
        for d, margin in ((15, 3), (20, 1), (20, 3), (20, 5), (25, 3)):
            if d <= policy.proposal_radius_m:
                variants[d, margin] = replace(policy, qualified_distance_m=d, qualified_margin=margin)

    # Stage each output set until all selected trips finish.
    with TemporaryDirectory(dir=report_dir) as temp_report, TemporaryDirectory(dir=export_dir) as temp_export:
        reports, exports = Path(temp_report), Path(temp_export)
        with ((exports / "labels.csv").open("w", newline="", encoding="utf-8") as label_file,
              (exports / "candidate_join_status.csv").open("w", newline="", encoding="utf-8") as join_file,
              (exports / "candidate_targets.csv").open("w", newline="", encoding="utf-8")
              if partition == "train" else nullcontext() as target_file):
            label_writer, label_values = _record_writer(label_file, LabelRecord)
            join_writer = csv.writer(join_file, lineterminator="\n")
            join_writer.writerow(("partition", "trip_id", "observation_id", "outcome", "reason",
                                  "candidate_count", "target_rank_within_radius", "radius_m",
                                  "top_k", "policy_id"))
            target_writer, target_values = (_record_writer(target_file, CandidateTarget)
                                            if partition == "train" else (None, None))
            for number, file in enumerate(files, 1):
                trip_started = perf_counter()
                trip = load_trip(file)
                loaded = perf_counter()
                timing = {}
                labels = align_trip(trip, policy, timing)
                aligned = perf_counter()
                valid = validate_trip(trip)
                route_breaks = int(valid.route_disconnections.sum())
                repeated = len(trip.route[valid.route_ids]) - len(np.unique(trip.route[valid.route_ids]))
                for method, rows in labels.items():
                    trip_rows.append(dict(trip_id=file.trip_id, method=method, **coverage(rows),
                                          reference_breaks=route_breaks, repeated_reference_arcs=repeated,
                                          proposals_with_distance_ties=sum(r.distance_ties > 0 for r in rows),
                                          boundary_risk=sum(r.boundary_risk for r in rows),
                                          gap_rows=sum(r.gap_before for r in rows)))
                    accepted_distances.setdefault(method, array("d")).extend(
                        r.distance_m for r in rows if r.status == "accepted")
                    reasons.update((method, row.status, row.reason) for row in rows)
                    label_writer.writerows(map(label_values, rows))
                all_pairs.extend(dict(trip_id=file.trip_id, **r) for r in comparison_rows(labels))
                if file.trip_id in DIFFICULT_IDS and partition == "train":
                    difficult.extend(difficult_rows(trip, labels, route_breaks))
                    if plots and len(trip.track):
                        plot_difficult(labels, exports / f"alignment_{file.trip_id}.png")
                for (distance, margin), variant in variants.items():
                    sensitivity[distance, margin] += sum(
                        qualification(row, variant)[0] == "accepted" for row in labels["monotonic"])
                measured = perf_counter()
                all_recalls.extend(_measure_candidate_ranks(trip, labels, join_writer,
                                                            target_writer, target_values))
                ended = perf_counter()
                times.append(dict(trip_id=file.trip_id, gps_rows=len(trip.track), load_s=loaded-trip_started,
                                  alignment_s=aligned-loaded, **timing,
                                  summaries_and_label_export_s=measured-aligned,
                                  candidate_diagnostics_and_target_export_s=ended-measured))
                if number % 10 == 0 or number == len(files):
                    print(f"{partition}: {number}/{len(files)} trips", flush=True)

        recalls = list(aggregate_recall(all_recalls))
        pair_totals = []
        for first, second in dict.fromkeys((r["first_method"], r["second_method"]) for r in all_pairs):
            selected = [r for r in all_pairs if (r["first_method"], r["second_method"]) == (first, second)]
            pair_totals.append(dict(first_method=first, second_method=second, **{
                key: sum(r[key] for r in selected) for key in selected[0]
                if key not in ("trip_id", "first_method", "second_method")}))
        observations = sum(r["observations"] for r in trip_rows if r["method"] == "qualified")
        sensitivity_rows = [
            dict(qualified_distance_m=d, qualified_margin=m, accepted=sensitivity[d, m],
                 observations=observations, coverage=sensitivity[d, m] / observations if observations else None)
            for d, m in variants
        ]
        tables = {
            "trip_coverage": trip_rows, "method_comparison_by_trip": all_pairs,
            "method_comparison": pair_totals, "candidate_recall_by_trip": all_recalls,
            "candidate_recall": recalls, "difficult_cases": difficult,
            "training_sensitivity": sensitivity_rows,
        }
        for name, rows in tables.items():
            if rows:
                _write_csv(reports / f"{name}.csv", (dict(partition=partition, **row) for row in rows))
        policy_values = asdict(policy)
        (reports / "policy.json").write_text(json.dumps(policy_values, indent=2) + "\n", encoding="utf-8")
        write_summary(reports / "summary.md", partition, trip_rows, accepted_distances, recalls,
                      pair_totals, reasons, policy_values)
        _write_csv(reports / "runtime_by_trip.csv", times)
        rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
        runtime = dict(partition=partition, trips=len(files), wall_s=perf_counter()-started,
                       peak_process_rss_mib=peak_rss, platform=sys.platform, python=sys.version.split()[0])
        for key in times[0]:
            if key.endswith("_s"):
                runtime[key] = sum(item[key] for item in times)
        (reports / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
        for target, source in ((report_dir, reports), (export_dir, exports)):
            for path in sorted(source.iterdir()):
                path.replace(target / path.name)
    print(f"Reports: {report_dir}\nLocal exports: {export_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase2b"))
    parser.add_argument("--export-dir", type=Path, default=Path("data/alignments"))
    parser.add_argument("--trips", nargs="+")
    parser.add_argument("--policy", type=Path, help="Load a frozen policy JSON, such as a training run's policy.json")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    try:
        policy = AlignmentPolicy(**json.loads(args.policy.read_text(encoding="utf-8"))) if args.policy else AlignmentPolicy()
        run(args.data_dir, args.partition, args.report_dir, args.export_dir, policy, args.trips, args.plots)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
