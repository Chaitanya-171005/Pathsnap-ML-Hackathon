"""Run the complete, deterministic Phase 1 analysis on the two student partitions."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from src.preprocessing.loaders import EXTENSIONS, PARTITIONS, discover_trips, load_trip
from src.preprocessing.statistics import gps_statistics, network_statistics, route_statistics
from src.preprocessing.validation import validate_trip
from src.visualization.folium_maps import write_map
from src.visualization.plots import write_plots


ISSUE_FIELDS = ("partition", "trip_id", "file", "issue", "count", "line", "column")
QUALITY_METRICS = {
    ".track": ("missing_coordinate_rows", "invalid_coordinate_rows", "invalid_timestamp_rows",
               "duplicate_consecutive_positions", "duplicate_consecutive_observations",
               "duplicate_timestamp_rows", "timestamp_decreases", "uncheckable_intervals",
               "nonfinite_speed_pairs", "implausible_speed_pairs"),
    ".nodes": ("invalid_node_coordinates", "isolated_nodes"),
    ".arcs": ("invalid_node_refs", "arcs_without_geometry", "duplicate_directed_edges",
              "self_loops", "zero_length_arcs"),
    ".route": ("invalid_reference_ids", "repeated_reference_arcs", "arcs_without_geometry",
                "disconnected_pairs", "uncheckable_pairs", "gps_offset_gt_100_m"),
}


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, dict) else v
                             for k, v in row.items()})


def inventory_row(files, trip) -> dict:
    result = {}
    for extension in EXTENSIONS:
        path = files.path(extension)
        try:
            size = path.stat().st_size if path.is_file() else None
        except OSError:
            size = None
        result[f"{extension}_present"] = size is not None
        result[f"{extension}_bytes"] = size
        result[f"{extension}_rows"] = len(getattr(trip, extension))
    return result


def representative_trips(training: list, gaps: tuple[int, ...]) -> dict:
    if not training:
        return {}
    median = np.median([gps["gps_observations"] for _, gps, _ in training])
    short = min(training, key=lambda item: (item[1]["gps_observations"], item[0].trip_id))
    typical = min((item for item in training if item != short),
                  key=lambda item: (abs(item[1]["gps_observations"] - median), item[0].trip_id),
                  default=short)
    short, typical = short[0], typical[0]
    chosen = {"short": short}
    if short != typical:
        chosen = {"typical": typical, **chosen}

    def anomaly_priority(item):
        _, gps, route = item
        return (route["disconnected_pairs"], gps[f"gap_gt_{gaps[-1]}_s"],
                gps["implausible_speed_pairs"], route["gps_offset_gt_100_m"])

    other = [item for item in training if item[0] not in chosen.values()]
    if other:
        anomaly = max(other, key=anomaly_priority)
        if any(anomaly_priority(anomaly)):
            chosen["anomaly"] = anomaly[0]
    return chosen


def write_summary(path, inventory, gps, networks, routes, issues, gaps, speed_limit, maps) -> None:
    lines = [
        "# PathSnap Phase 1 EDA", "",
        f"GPS gap thresholds are {', '.join(map(str, gaps))} s (strictly greater). "
        f"The speed screening threshold is {speed_limit:g} m/s.", "",
    ]
    for partition, expected in PARTITIONS.items():
        inv, track, graph, route = [
            [row for row in rows if row["partition"] == partition]
            for rows in (inventory, gps, networks, routes)
        ]
        partition_issues = [row for row in issues if row["partition"] == partition]
        lines.extend([f"## {partition}", ""])
        if not inv:
            lines.extend([f"No trips found (expected {expected}).", ""])
            continue

        def total(rows, field):
            return sum(row[field] for row in rows)

        seconds = total(track, "one_second_intervals")
        positives = total(track, "positive_intervals")
        interval_share = f"{seconds / positives:.2%}" if positives else "n/a"
        offsets = total(route, "gps_points_compared")
        over_30 = total(route, "gps_offset_gt_30_m")
        over_100 = total(route, "gps_offset_gt_100_m")
        offset_share = f"{over_100 / offsets:.2%}" if offsets else "n/a"
        equal = sum(row["reference_minus_gps_count"] == 0 for row in route)
        connected = sum(row["reference_arcs"] > 0 and row["invalid_reference_ids"] == 0
                        and row["arcs_without_geometry"] == 0 and row["disconnected_pairs"] == 0
                        and row["uncheckable_pairs"] == 0 for row in route)
        largest_gap = max((row["interval_s_max"] for row in track
                           if row["interval_s_max"] is not None), default=None)
        gap_text = f"{largest_gap:g} s" if largest_gap is not None else "n/a"
        lines.extend([
            f"- **{len(inv)} trips** (expected {expected}), with "
            f"**{sum(total(inv, f'{ext}_present') for ext in EXTENSIONS)} files present**.",
            f"- {total(track, 'gps_observations'):,} GPS rows, {total(graph, 'nodes'):,} nodes, "
            f"{total(graph, 'directed_arcs'):,} directed arcs and {total(route, 'reference_arcs'):,} reference rows.",
            f"- Exactly 1 s for {seconds:,} of {positives:,} positive adjacent intervals "
            f"({interval_share}). Largest positive interval is {gap_text}.",
            f"- {total(track, 'duplicate_timestamp_rows'):,} duplicate timestamp rows "
            f"and {total(track, 'timestamp_decreases'):,} adjacent timestamp decreases.",
            f"- {equal}/{len(route)} trips have equal GPS and reference counts. "
            f"{connected}/{len(route)} have a nonempty, fully checkable reference path with geometry.",
            f"- {total(route, 'disconnected_pairs'):,} directed reference breaks "
            f"and {total(route, 'uncheckable_pairs'):,} uncheckable reference pairs.",
            f"- Across {offsets:,} GPS points compared against the closest reference segment, "
            f"{over_30:,} exceed 30 m and {over_100:,} exceed 100 m ({offset_share}).",
            f"- {sum(row['weak_components'] > 1 for row in graph)} road networks have multiple "
            f"weak components and {sum(row['strong_components'] > 1 for row in graph)} have multiple strong components.",
            "",
            "| File | Finding | Count |",
            "|---|---|---:|",
        ])
        counts = Counter()
        for issue in partition_issues:
            counts[issue["file"], issue["issue"]] += issue["count"]
        lines.extend(f"| {file} | {code} | {count:,} |"
                     for (file, code), count in sorted(counts.items()))
        if not counts:
            lines.append("| — | No findings | 0 |")
        lines.append("")

    lines.extend(["## Training map selection", ""])
    lines.extend(f"- {category.title()} — `{trip_id}` (`{name}`)"
                 for category, trip_id, name in maps)
    lines.extend([
        "", "## Interpretation", "",
        "Reference offsets are unsigned distances to any valid reference segment. They measure "
        "spatial agreement without asserting direction, traversal time or an observation label. "
        "Route length sums endpoint-to-endpoint arc lengths, counting repeats and adding no connectors. "
        "Parser errors and semantic quality findings can refer to the same rows.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_dir: Path, report_dir: Path, visual_dir: Path, gaps: tuple[int, ...],
        speed_limit: float, road_limit: int) -> None:
    if any(path.resolve().is_relative_to(data_dir.resolve()) for path in (report_dir, visual_dir)):
        raise ValueError("Output directories must be outside the input dataset directory.")
    files = discover_trips(data_dir)
    if not files:
        raise ValueError(f"No trips found in {data_dir}/train or {data_dir}/student_test")
    report_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    inventory, gps, networks, routes, issues, training = [], [], [], [], [], []
    intervals = {partition: [] for partition in PARTITIONS}

    for file_set in files:
        trip = load_trip(file_set)
        valid = validate_trip(trip)
        key = dict(partition=file_set.partition, trip_id=file_set.trip_id)
        inventory.append(dict(key, **inventory_row(file_set, trip)))
        track, time_steps = gps_statistics(trip, valid, gaps, speed_limit)
        graph = network_statistics(trip, valid)
        route = route_statistics(trip, valid)
        gps.append(dict(key, **track))
        networks.append(dict(key, **graph))
        routes.append(dict(key, **route))
        intervals[file_set.partition].append(time_steps)
        issues.extend(dict(key, file=i.file, issue=i.code, count=1, line=i.line, column=i.column)
                      for i in trip.issues)
        for file, stats in zip(QUALITY_METRICS, (track, graph, graph, route), strict=True):
            fields = QUALITY_METRICS[file] + (tuple(f"gap_gt_{gap}_s" for gap in gaps)
                                              if file == ".track" else ())
            issues.extend(dict(key, file=file, issue=field, count=stats[field])
                          for field in fields if stats[field])
        if file_set.partition == "train" and valid.gps.any():
            training.append((file_set, track, route))

    for partition, expected in PARTITIONS.items():
        actual = sum(f.partition == partition for f in files)
        if actual != expected:
            issues.append(dict(partition=partition, file="partition",
                               issue=f"expected_{expected}_trips_found_{actual}", count=1))
        print(f"{partition}: {actual} trips (expected {expected})", flush=True)
    train_ids = {f.trip_id for f in files if f.partition == "train"}
    issues.extend(dict(partition="student_test", trip_id=f.trip_id, file="partition",
                       issue="trip_id_also_in_train", count=1)
                  for f in files if f.partition == "student_test" and f.trip_id in train_ids)

    for name, rows in (("dataset_inventory", inventory), ("gps_statistics", gps),
                       ("network_statistics", networks), ("reference_routes", routes)):
        write_csv(report_dir / f"{name}.csv", rows)
    write_csv(report_dir / "data_quality_issues.csv", issues, ISSUE_FIELDS)

    selections = representative_trips(training, gaps)
    maps = []
    for category in ("typical", "short", "anomaly"):
        path = visual_dir / f"training_{category}.html"
        if category not in selections:
            path.unlink(missing_ok=True)
            continue
        trip = load_trip(selections[category])
        write_map(trip, validate_trip(trip), category, path, road_limit)
        maps.append((category, trip.files.trip_id, path.name))
    if not selections:
        (visual_dir / "07_training_trajectories.png").unlink(missing_ok=True)
    interval_arrays = {part: np.concatenate(batches) if batches else np.empty(0)
                       for part, batches in intervals.items()}
    write_plots(visual_dir, gps, networks, routes, interval_arrays, gaps, selections)
    write_summary(report_dir / "summary.md", inventory, gps, networks, routes,
                  issues, gaps, speed_limit, maps)
    print(f"Reports: {report_dir}\nVisualizations: {visual_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/eda"))
    parser.add_argument("--visual-dir", type=Path, default=Path("visualizations/eda"))
    parser.add_argument("--gap-seconds", nargs="+", type=int, default=[5, 10, 30, 60])
    parser.add_argument("--speed-mps", type=float, default=55.0)
    parser.add_argument("--map-road-limit", type=int, default=12_000)
    args = parser.parse_args()
    if (min(args.gap_seconds) <= 0 or not np.isfinite(args.speed_mps)
            or args.speed_mps <= 0 or args.map_road_limit <= 0):
        parser.error("thresholds and road limit must be positive and finite")
    try:
        run(args.data_dir, args.report_dir, args.visual_dir,
            tuple(sorted(set(args.gap_seconds))), args.speed_mps, args.map_road_limit)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
