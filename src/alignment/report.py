"""Partition-local coverage, disagreement and proxy recall reports."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .core import METHODS, STATUSES


RADII = (25, 50, 100)
CAPS = (3, 8, 16)
DIFFICULT_IDS = {"00000019", "00000054"}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def quantiles(values) -> dict[str, float | None]:
    if not len(values):
        return {"p10": None, "p50": None, "p90": None, "p95": None}
    return {f"p{q}": float(value)
            for q, value in zip((10, 50, 90, 95), np.percentile(values, (10, 50, 90, 95)), strict=True)}


def coverage(records) -> dict:
    counts = Counter(row.status for row in records)
    distances = [row.distance_m for row in records if row.status == "accepted"]
    return {
        "observations": len(records), **{status: counts[status] for status in STATUSES},
        "coverage": fraction(counts["accepted"], len(records)),
        "abstention_rate": fraction(len(records) - counts["accepted"], len(records)),
        **{f"accepted_distance_{key}_m": value for key, value in quantiles(distances).items()},
    }


def compare(first, second) -> dict:
    metrics = Counter(dict(both_proposed=0, same_proposed_arc=0, same_proposed_occurrence=0,
                           both_accepted=0, same_accepted_arc=0, same_accepted_occurrence=0,
                           first_only_accepted=0, second_only_accepted=0))
    for a, b in zip(first, second, strict=True):
        if a.arc_id is not None and b.arc_id is not None:
            metrics["both_proposed"] += 1
            metrics["same_proposed_arc"] += a.arc_id == b.arc_id
            metrics["same_proposed_occurrence"] += a.occurrence_index == b.occurrence_index
        if a.status == "accepted" and b.status == "accepted":
            metrics["both_accepted"] += 1
            metrics["same_accepted_arc"] += a.arc_id == b.arc_id
            metrics["same_accepted_occurrence"] += a.occurrence_index == b.occurrence_index
        else:
            metrics["first_only_accepted"] += a.status == "accepted"
            metrics["second_only_accepted"] += b.status == "accepted"
    metrics["different_proposed_arc"] = metrics["both_proposed"] - metrics["same_proposed_arc"]
    metrics["different_accepted_arc"] = metrics["both_accepted"] - metrics["same_accepted_arc"]
    return dict(metrics)


def comparison_rows(labels):
    for first, second in combinations(METHODS, 2):
        yield {"first_method": first, "second_method": second,
               **compare(labels[first], labels[second])}


def recall_rows(trip_id, labels, valid_ids, empty_at, ranks):
    for method, records in labels.items():
        accepted_ids = np.array([row.observation_id for row in records if row.status == "accepted"],
                                dtype=int)
        for radius in RADII:
            target_ranks = ranks[method, radius][accepted_ids]
            empty = empty_at[radius][accepted_ids]
            for cap in CAPS:
                included = int(((target_ranks > 0) & (target_ranks <= cap)).sum())
                accepted = len(accepted_ids)
                yield dict(trip_id=trip_id, method=method, radius_m=radius, top_k=cap,
                           accepted=accepted, included=included, misses=accepted-included,
                           accepted_with_no_candidates=int(empty.sum()),
                           target_outside_radius_with_candidates=int(((target_ranks == 0) & ~empty).sum()),
                           target_truncated_by_top_k=int((target_ranks > cap).sum()),
                           gps_rows_without_candidates=int(empty_at[radius][valid_ids].sum()),
                           candidate_recall=fraction(included, accepted),
                           candidate_miss_rate=fraction(accepted-included, accepted))


def aggregate_recall(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["radius_m"], row["top_k"])].append(row)
    for (method, radius, cap), items in groups.items():
        fields = ("accepted", "included", "misses", "accepted_with_no_candidates",
                  "target_outside_radius_with_candidates", "target_truncated_by_top_k",
                  "gps_rows_without_candidates")
        totals = {key: sum(item[key] for item in items) for key in fields}
        trip_rates = [item["candidate_recall"] for item in items if item["accepted"]]
        yield dict(method=method, radius_m=radius, top_k=cap, **totals,
                   candidate_recall=fraction(totals["included"], totals["accepted"]),
                   candidate_miss_rate=fraction(totals["misses"], totals["accepted"]),
                   trips_with_accepted=len(trip_rates),
                   **{f"per_trip_recall_{q}": v for q, v in quantiles(trip_rates).items()})


def difficult_rows(trip, labels, breaks):
    """Aggregate targeted slices without exporting locations or individual labels."""
    n = len(trip.track)
    slices = {"whole_trip": np.ones(n, dtype=bool)}
    if trip.files.trip_id == "00000054":
        # The endpoint approach was independently located by the Phase 2A diagnostic.
        slices["through_endpoint_approach"] = np.arange(n) <= 2702
        slices["after_endpoint_approach"] = np.arange(n) > 2702
    for method, records in labels.items():
        for name, mask in slices.items():
            selected = [row for row in records if mask[row.observation_id]]
            yield dict(trip_id=trip.files.trip_id, method=method, subset=name,
                       route_breaks=breaks, **coverage(selected),
                       boundary_flags=sum(row.boundary_risk for row in selected),
                       gap_rows=sum(row.gap_before for row in selected))


def plot_difficult(labels, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(accepted="#087957", ambiguous="#ae6600", unmatched="#969eaa",
                  invalid_observation="#852c95", disconnected_reference="#bf2438")
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.2}):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6), layout="constrained")
        for ax, (method, records) in zip(axes, labels.items(), strict=True):
            for status, color in colors.items():
                rows = [r for r in records if r.status == status]
                if rows:
                    ax.scatter([r.observation_id for r in rows],
                               [r.distance_m if r.distance_m is not None else -3 for r in rows],
                               s=6, alpha=0.65, color=color, label=status.replace("_", " "))
            ax.set_title(method.capitalize(), loc="left")
            ax.set_ylabel("Offset (m)")
            ax.grid()
            ax.legend(loc="upper right", ncol=3, fontsize=7)
        axes[-1].set_xlabel("Original GPS observation ID · points at −3 indicate no proposal")
        fig.suptitle(f"Reference alignment diagnostics · {records[0].trip_id}")
        fig.savefig(path, dpi=140)
        plt.close(fig)


def write_summary(path, partition, trip_rows, accepted_distances, recalls, pairs, reasons, policy):
    lines = [f"# Phase 2B proxy-label alignment · {partition}", "",
             "These measurements describe supplied-reference proxy labels and candidate inclusion.", "",
             f"Trips analysed: {len({row['trip_id'] for row in trip_rows})}.", "",
             "| Method | Accepted / observations | Coverage | Ambiguous | Unmatched | Invalid | Disconnected | Abstention | Per-trip coverage p10 / p50 / p90 | Accepted offset p50 / p95 (m) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(value):
        return f"{value:.2%}" if value is not None else "—"

    for method in METHODS:
        rows = [row for row in trip_rows if row["method"] == method]
        counts = {key: sum(row[key] for row in rows) for key in ("observations", *STATUSES)}
        q = quantiles([r["coverage"] for r in rows if r["coverage"] is not None])
        d = quantiles(accepted_distances[method])
        offsets = f"{d['p50']:.2f} / {d['p95']:.2f}" if d["p50"] is not None else "—"
        lines.append(
            f"| {method} | {counts['accepted']:,} / {counts['observations']:,} | "
            f"{pct(fraction(counts['accepted'], counts['observations']))} | "
            f"{counts['ambiguous']:,} | {counts['unmatched']:,} | {counts['invalid_observation']:,} | "
            f"{counts['disconnected_reference']:,} | "
            f"{pct(fraction(counts['observations'] - counts['accepted'], counts['observations']))} | "
            f"{pct(q['p10'])} / {pct(q['p50'])} / {pct(q['p90'])} | {offsets} |")
    lines.extend(["", "## Agreement across jointly accepted observations", "",
                  "| Methods | Jointly accepted | Same directed arc | Different directed arc | Same route occurrence |",
                  "|---|---:|---:|---:|---:|"])
    for row in pairs:
        lines.append(f"| {row['first_method']} / {row['second_method']} | {row['both_accepted']:,} | "
                     f"{row['same_accepted_arc']:,} | {row['different_accepted_arc']:,} | "
                     f"{row['same_accepted_occurrence']:,} |")
    lines.extend(["", "## Qualified proxy recall", "",
                  "| Radius (m) | Top-K | Included / qualified | Recall | Miss rate | Per-trip recall p10 / p50 / p90 |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for r in recalls:
        if r["method"] == "qualified":
            lines.append(f"| {r['radius_m']} | {r['top_k']} | {r['included']:,} / {r['accepted']:,} | "
                         f"{pct(r['candidate_recall'])} | {pct(r['candidate_miss_rate'])} | "
                         f"{pct(r['per_trip_recall_p10'])} / {pct(r['per_trip_recall_p50'])} / "
                         f"{pct(r['per_trip_recall_p90'])} |")
    lines.extend(["", "## Qualified abstention reasons", ""])
    lines.extend(f"- `{reason}` — {n:,}" for (method, status, reason), n in sorted(reasons.items())
                 if method == "qualified" and status != "accepted")
    lines.extend(["", "## Policy", "", "```json", json.dumps(policy, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
