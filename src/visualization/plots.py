"""Headless, partition-labelled figures for the Phase 1 report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.loaders import load_trip
from src.preprocessing.validation import validate_trip


COLORS = {"train": "#1761a0", "student_test": "#ad561b"}
LABELS = {"train": "Training", "student_test": "Student testing"}


def _save(fig, output: Path, filename: str) -> None:
    fig.savefig(output / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _partitions(ax, rows, x, y, *, indexed=False) -> None:
    for partition, color in COLORS.items():
        selected = [row for row in rows if row["partition"] == partition
                    and row[y] is not None and (indexed or row[x] is not None)]
        if not selected:
            continue
        positions = [row["trip_id"] for row in selected] if indexed else [row[x] for row in selected]
        ax.scatter(positions, [row[y] for row in selected], s=17, alpha=0.75,
                   c=color, label=LABELS[partition])
    if ax.collections:
        ax.legend()


def _distribution(ax, values, xlabel, title) -> None:
    positive = [data[data > 0] for data in values.values() if len(data)]
    positive = [data for data in positive if len(data)]
    if positive:
        low = min(data.min() for data in positive)
        high = max(max(data.max() for data in positive), low * 1.05)
        bins = np.geomspace(low, high * 1.001, 47)
        for partition, data in values.items():
            if (data > 0).any():
                ax.hist(data[data > 0], bins=bins, histtype="step", linewidth=1.8,
                        label=LABELS[partition], color=COLORS[partition])
        ax.set(xscale="log", yscale="log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No positive values", ha="center", transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel="Count", title=title)


@plt.rc_context({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "axes.axisbelow": True,
    "figure.constrained_layout.use": True,
})
def write_plots(output, gps, networks, routes, intervals, gaps, selections) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    _partitions(ax, gps, "trip_id", "gps_observations", indexed=True)
    ax.set(xticks=[], xlabel="Trips in ID order within each partition",
           ylabel="GPS observations", title="GPS observation count by trip")
    _save(fig, output, "01_observation_counts.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _distribution(ax, intervals, "Positive interval (s)", "GPS sampling intervals")
    _save(fig, output, "02_timestamp_intervals.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for partition, color in COLORS.items():
        data = intervals[partition]
        axes[0].plot(gaps, [(data > threshold).sum() for threshold in gaps], "o-",
                     color=color, label=LABELS[partition])
    axes[0].set(xlabel="Gap threshold (s)", ylabel="Intervals above threshold",
                title="GPS gap frequency")
    axes[0].legend()
    _distribution(axes[1], {p: data[data > gaps[0]] for p, data in intervals.items()},
                  "Interval (s)", f"GPS gap magnitude above {gaps[0]} s")
    _save(fig, output, "03_gps_gaps.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    lengths = {p: np.asarray([r["gps_polyline_m"] / 1000 for r in gps
                             if r["partition"] == p and r["gps_polyline_m"] is not None])
               for p in COLORS}
    _distribution(ax, lengths, "GPS polyline length (km)", "GPS trajectory length by trip")
    _save(fig, output, "04_trajectory_lengths.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _partitions(ax, networks, "nodes", "directed_arcs")
    ax.set(xlabel="Nodes", ylabel="Directed arcs", xscale="symlog", yscale="symlog",
           title="Road-network size by trip")
    _save(fig, output, "05_road_network_size.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    _partitions(ax, networks, "weak_components", "strong_components")
    ax.set(xlabel="Weakly connected components", ylabel="Strongly connected components",
           xscale="symlog", yscale="symlog", title="Road-network connectivity by trip")
    _save(fig, output, "06_connectivity.png")

    if selections:
        fig, axes = plt.subplots(1, len(selections), figsize=(4.5 * len(selections), 4),
                                 squeeze=False)
        for (category, files), ax in zip(selections.items(), axes[0]):
            trip = load_trip(files)
            valid = validate_trip(trip)
            gps_line = trip.track[:, :2].copy()
            gps_line[~valid.gps] = np.nan
            ax.plot(*gps_line.T, color=COLORS["train"], alpha=0.7, linewidth=0.8)
            points = gps_line[valid.gps]
            if len(points):
                ax.scatter(*points[[0, -1]].T, c=["#087748", "#b93c30"], s=20, zorder=3)
                ax.set_aspect(1 / max(0.01, np.cos(np.radians(np.median(points[:, 1])))))
            ax.ticklabel_format(useOffset=False)
            ax.tick_params(axis="x", labelrotation=20)
            ax.set(title=f"{category.title()} · {files.trip_id}",
                   xlabel="Longitude (°)", ylabel="Latitude (°)")
        fig.suptitle("Selected training GPS trajectories · first point green, last red")
        _save(fig, output, "07_training_trajectories.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    route_rows = [
        dict(row, gps_observations=track["gps_observations"],
             route_km=row["route_arc_sum_m"] / 1000 if row["route_arc_sum_m"] is not None else None)
        for row, track in zip(routes, gps, strict=True)
    ]
    _partitions(ax, route_rows, "gps_observations", "route_km")
    ax.set(xlabel="GPS observations (count)", ylabel="Reference arc lengths summed (km)",
           title="Reference route length versus GPS observations")
    fig.text(0.5, -0.025, "Each point represents one trip. Counts do not imply row-level alignment.",
             ha="center")
    _save(fig, output, "08_reference_length_vs_gps_count.png")
