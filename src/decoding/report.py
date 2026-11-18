"""Aggregate Phase 5 reports. Raw predictions are written separately."""

from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean

from .evaluation import add_rates


COUNTS = ("original", "matched", "with_candidates", "qualified", "eligible", "candidate_misses",
          "classifier_correct", "hmm_correct")
DIFFICULT = {
    "00000019": "reference discontinuities",
    "00000054": "anomalous GPS tail",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def figures(directory, pooled, point_rows, route_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=150)
        plt.close(fig)

    labels = [row["partition"].replace("student_test", "Student testing") for row in pooled]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for offset, field, name, color in (
        (-0.18, "classifier_top1", "Classifier only", "#395a82"),
        (0.18, "hmm_top1", "HMM", "#177565"),
    ):
        values = [100 * row[field] if row[field] is not None else 0 for row in pooled]
        bars = ax.bar([i + offset for i in x], values, width=.34, label=name, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}%" if row[field] is not None else "n/a"
                                  for v, row in zip(values, pooled, strict=True)], padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Qualified Top-1 agreement (%)", ylim=(0, 112))
    ax.legend(loc="lower right")
    save(fig, "classifier_vs_hmm.png")

    labels = [row["trip_id"] for row in point_rows]
    fig, ax = plt.subplots(figsize=(8.8, max(3.2, .35 * len(labels))))
    y = list(range(len(labels)))
    deltas = [row["top1_delta_pp"] or 0 for row in point_rows]
    ax.barh(y, deltas, color=["#177565" if value >= 0 else "#aa5435" for value in deltas])
    ax.set(yticks=y, yticklabels=labels, xlabel="HMM minus classifier Top-1 (percentage points)")
    ax.axvline(0, color="#425466", linewidth=.8)
    ax.invert_yaxis()
    save(fig, "per_trip_improvement.png")

    fig, (coverage, segments) = plt.subplots(1, 2, sharey=True, figsize=(9, .35 * len(labels) + 1.4))
    coverage.barh(y, [100 * (row["match_coverage"] or 0) for row in point_rows], color="#177565")
    coverage.set(xlabel="Original fixes matched (%)", xlim=(0, 105), yticks=y, yticklabels=labels)
    segments.barh(y, [row["segments"] for row in route_rows], color="#395a82")
    segments.set(xlabel="Reconstructed segments")
    coverage.invert_yaxis()
    save(fig, "route_connectivity.png")


def write_report(directory: Path, points, routes, runtime, model_id, artifact_hash, config,
                 difficult_details):
    directory.mkdir(parents=True, exist_ok=True)
    partitions = defaultdict(list)
    for row in points:
        partitions[row["partition"]].append(row)
    pooled = [add_rates(dict(partition=partition, trips=len(rows), **{
        key: sum(row[key] for row in rows) for key in COUNTS
    })) for partition, rows in partitions.items()]
    difficult = [
        {**point, **route, "case": DIFFICULT[point["trip_id"]],
         **difficult_details[point["trip_id"]]}
        for point, route in zip(points, routes, strict=True)
        if point["partition"] == "train" and point["trip_id"] in DIFFICULT
    ]
    for filename, rows in (
        ("classifier_vs_hmm.csv", pooled), ("per_trip_metrics.csv", points),
        ("route_metrics.csv", routes), ("runtime.csv", runtime),
        ("difficult_trip_metrics.csv", difficult),
    ):
        write_csv(directory / filename, rows)
    figures(directory, pooled, points, routes)
    summary = [
        "# PathSnap Phase 5 measured results", "",
        f"Model `{model_id}` with SHA-256 `{artifact_hash}`.", "",
        f"Frozen decoder settings `{config}`.", "",
        "Training rows are smoke checks using the fitted classifier. Student testing is a reused "
        "development holdout. The paired Top-1 denominator includes qualified labels present in "
        "Phase 2A Top-8. Decoder abstentions on these rows count as incorrect.", "",
        "| Partition | Trips | Original | Qualified | Candidate misses | Eligible | Classifier | HMM | Change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled:
        scores = [f"{row[k]:.3%}" if row[k] is not None else "n/a"
                  for k in ("classifier_top1", "hmm_top1")]
        summary.append(
            f"| {row['partition']} | {row['trips']} | {row['original']:,} | {row['qualified']:,} | "
            f"{row['candidate_misses']:,} | {row['eligible']:,} | {scores[0]} | {scores[1]} | "
            f"{row['correct_delta']:+,} fixes |")
    peak = max((r["peak_rss_mib"] for r in runtime if r["peak_rss_mib"] is not None), default=None)
    summary.extend(["", "## Route and runtime", "",
                    "| Partition | Segments | Internal disconnections | Reference disconnections | No feasible transitions | Inference seconds |",
                    "|---|---:|---:|---:|---:|---:|"])
    for partition in partitions:
        subset = [row for row in routes if row["partition"] == partition]
        totals = [sum(row[k] for row in subset) for k in ("segments", "predicted_internal_disconnections",
                                                         "reference_disconnected_transitions", "no_feasible_transitions")]
        seconds = sum(r["inference_s"] for r in runtime if r["partition"] == partition)
        summary.append(f"| {partition} | {totals[0]} | {totals[1]} | {totals[2]} | {totals[3]} | {seconds:.2f} |")
    summary.extend([
        "",
        "| Partition | Classifier arc precision | HMM arc precision | Classifier arc recall | HMM arc recall | Classifier bigram F1 | HMM bigram F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for partition in partitions:
        subset = [row for row in routes if row["partition"] == partition]
        metrics = []
        for key in ("classifier_arc_precision", "arc_precision", "classifier_arc_recall",
                    "arc_recall", "classifier_directed_bigram_f1", "directed_bigram_f1"):
            values = [row[key] for row in subset if row[key] is not None]
            metrics.append(f"{mean(values):.2%}" if values else "n/a")
        summary.append(f"| {partition} | {' | '.join(metrics)} |")
    summary.extend([
        "",
        (f"The decoding and evaluation process reached a {peak:.1f} MiB RSS high-water mark "
         "at the inference checkpoints." if peak is not None else
         "Peak process RSS is unavailable on this platform."),
        "Per-trip inference stages, load and offline evaluation time are in [runtime.csv](runtime.csv).",
        "The route agreement table averages per-trip rates. Arc precision and recall use unique "
        "directed IDs. Bigram F1 uses connected adjacent occurrence counts. The classifier route "
        "diagnostic uses its raw Top-1 sequence with consecutive duplicates collapsed. HMM "
        "lengths sum full arc occurrences or snapped transitions, depending on the column.",
        "",
        "See [per-trip proxy metrics](per_trip_metrics.csv), [route metrics](route_metrics.csv) and "
        "[difficult trip diagnostics](difficult_trip_metrics.csv).",
    ])
    (directory / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
