"""Render aggregate Phase 4 results and diagnostics."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .artifacts import atomic_path, canonical, fingerprint, read_json, write_json
from .search import SPACE


METRICS = ("eligible", "candidate_rows", "correct", "top1", "trip_macro_top1", "top3", "mrr",
           "precision", "recall", "f1", "log_loss")
RATES = {"top1", "trip_macro_top1", "top3", "mrr", "precision", "recall", "f1"}


def csv_report(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = sorted({key for row in rows for key in row})
    with atomic_path(path) as temporary, temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(output, name, draw):
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                         "axes.grid": True, "grid.alpha": 0.23, "figure.facecolor": "white",
                         "axes.titlesize": 13, "axes.titleweight": "bold"}):
        fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
        draw(ax)
        with atomic_path(output / f"{name}.png") as temporary:
            fig.savefig(temporary, format="png", dpi=150)
        plt.close(fig)


def _importance(trials, seed):
    features = []
    for row in trials:
        params = row["params"]
        features.append([SPACE[name].index(params[name]) - 1 if name == "max_depth" else
                         np.log(params[name]) if SPACE[name][0] == "log" else params[name]
                         for name in SPACE])
    model = ExtraTreesRegressor(n_estimators=192, min_samples_leaf=2, random_state=seed, n_jobs=1)
    if len(trials) < 6:
        return dict.fromkeys(SPACE, 0.0)
    model.fit(features, [row["metrics"]["top1"] for row in trials])
    return dict(zip(SPACE, model.feature_importances_, strict=True))


def figures(result, output):
    trials = [row for row in result["all_trials"] if row["study"] == "final" and row["state"] == "COMPLETE"]
    xs = [r["number"] + 1 for r in trials]
    ys = [r["metrics"]["top1"] * 100 for r in trials]
    teal, navy = "#137e72", "#354f79"

    def history(ax):
        ax.scatter(xs, ys, s=22, color=navy, alpha=0.6, label="Completed trial")
        ax.step(xs, np.maximum.accumulate(ys), color=teal, where="post", label="Best so far")
        ax.set(title="Full-training search history", xlabel="Trial", ylabel="Grouped validation Top-1 (%)")
        ax.legend()

    def importance(ax):
        weights = sorted(_importance(trials, result["manifest"]["seed"]).items(), key=lambda item: item[1])
        ax.barh([n for n, _ in weights], [v * 100 for _, v in weights], color=teal)
        ax.set(title="Hyperparameter importance in the full-training search",
               xlabel="Exploratory surrogate impurity importance (%)")
        if len(trials) < 6:
            ax.text(0.5, 0.1, "Insufficient trials to fit the surrogate", transform=ax.transAxes, ha="center")

    def comparison(ax):
        names = ["top1", "trip_macro_top1", "top3", "mrr", "f1"]
        xs2 = np.arange(len(names))
        for offset, key, label, color in ((-.19, "fixed", "Fixed Phase 3", navy), (.19, "tuned", "Nested tuning", teal)):
            values = [result[key][name] * 100 for name in names]
            bars = ax.bar(xs2 + offset, values, width=0.37, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set(title="Fixed baseline and nested tuning on identical outer groups", ylabel="Metric (%)",
               xticks=xs2, xticklabels=["Top-1", "Trip macro Top-1", "Top-3", "MRR", "Binary F1"],
               ylim=(0, 108))
        ax.legend(loc="lower right")

    def folds(ax):
        for model, label, color, mark in (("fixed", "Fixed Phase 3", navy, "o"), ("tuned", "Nested tuning", teal, "s")):
            rows = [r for r in result["outer_metrics"] if r["model"] == model]
            ax.plot([r["fold"] for r in rows], [r["top1"] * 100 for r in rows], color=color, marker=mark, label=label)
        ax.set(title="Paired Top-1 agreement by outer fold", xlabel="Outer validation fold",
               ylabel="Top-1 proxy agreement (%)", xticks=range(1, result["manifest"]["outer_folds"] + 1))
        ax.legend()

    def trips(ax):
        values = sorted(r["top1_difference"] * 100 for r in result["per_trip"])
        ax.bar(np.arange(1, len(values) + 1), values, color=[teal if v >= 0 else "#b65643" for v in values])
        ax.axhline(0, color=navy, linewidth=1)
        ax.set(title="Paired Top-1 difference across training trips", xlabel="Trip rank by observed difference",
               ylabel="Tuned minus fixed (percentage points)")

    def tradeoff(ax):
        points = ax.scatter([r["metrics"]["log_loss"] for r in trials], ys, c=xs, cmap="viridis", s=38)
        ax.figure.colorbar(points, ax=ax, label="Trial")
        ax.set(title="Trial ranking agreement and candidate log loss", xlabel="Grouped validation candidate log loss",
               ylabel="Grouped validation Top-1 (%)")

    for name, draw in (("optimization_history", history), ("hyperparameter_importance", importance),
                       ("baseline_vs_tuned", comparison), ("outer_fold_comparison", folds),
                       ("per_trip_improvement", trips), ("top1_vs_logloss", tradeoff)):
        _plot(output, name, draw)


def render_reports(result, output):
    manifest = result["manifest"]
    selection, frozen, student = result["selection"], result["frozen"], result["student"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "optimization_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Use a separate report directory for a different experiment")
    write_json(manifest_path, manifest)
    histories = result["all_trials"]
    inner = [dict(study=t["study"], trial=t["number"] + 1, **r) for t in histories for r in t["folds"]]
    csv_report(output / "optimization_trials.csv",
               (dict(study=t["study"], trial=t["number"] + 1, state=t["state"], **t["params"],
                     **t.get("metrics", {}), error=t.get("error")) for t in histories))
    timing_fields = {"fit_s", "predict_s", "rss_mib"}
    csv_report(output / "inner_fold_metrics.csv",
               ({k: v for k, v in row.items() if k not in timing_fields} for row in inner))
    csv_report(output / "outer_fold_metrics.csv", result["outer_metrics"])
    csv_report(output / "fold_assignments.csv", result["assignments"])
    csv_report(output / "per_trip_comparison.csv", result["per_trip"])
    if result["difficult"]:
        csv_report(output / "difficult_trip_metrics.csv", result["difficult"])
    slices = [dict(partition="training_outer", **r) for r in result["slices"]]
    if student:
        slices += [dict(partition="student_test", **r, tuned_top1=r["top1"]) for r in student["slices"]]
    csv_report(output / "error_slices.csv", slices)
    comparison = []
    partitions = [("training_outer", result["fixed"], result["tuned"], result["nearest"])]
    if student:
        partitions.append(("student_test", student["baseline"], student["metrics"], {}))
        csv_report(output / "student_test_per_trip.csv", student["per_trip"])
    for partition, fixed, tuned, nearest in partitions:
        for metric in METRICS:
            comparison.append(dict(partition=partition, metric=metric, fixed=fixed[metric], tuned=tuned[metric],
                                   difference=tuned[metric] - fixed[metric], nearest=nearest.get(metric)))
    csv_report(output / "baseline_vs_tuned.csv", comparison)
    with Path("reports/phase3/training_runtime.csv").open(newline="", encoding="utf-8") as stream:
        phase3 = next(row for row in csv.DictReader(stream)
                      if row["model"] == "gradient_boosting" and row["stage"] == "final_student_test")
    runtime = [dict(stage="inner_fold", **{k: v for k, v in r.items()
                                           if k in timing_fields | {"study", "trial", "fold"}}) for r in inner]
    runtime += [dict(stage="trial", study=t["study"], trial=t["number"] + 1, wall_s=t["wall_s"])
                for t in histories]
    runtime += result["runtimes"]
    runtime += [dict(stage="final_fit", model="tuned", fit_s=frozen["fit_s"], model_bytes=frozen["model_bytes"]),
                dict(stage="archived_phase3_final", model="fixed", fit_s=float(phase3["fit_s"]),
                     predict_s=float(phase3["predict_s"]), model_bytes=int(phase3["artifact_bytes"]))]
    if student:
        runtime.append(dict(stage="student_test", model="tuned", predict_s=student["predict_s"],
                            rss_mib=student["rss_mib"]))
    runtime.append(dict(stage="session", wall_s=result["session_wall_s"], rss_mib=result["session_rss_mib"]))
    csv_report(output / "training_runtime.csv", runtime)
    write_json(output / "best_hyperparameters.json", selection["final_params"])
    write_json(output / "model_selection.json", selection)
    write_json(output / "reproducibility.json", dict(
        manifest_sha256=fingerprint(manifest), frozen_model_sha256=frozen["sha256"],
        reload_probe_sha256=frozen["reload_probe_sha256"], reload_probe_rows=frozen["reload_rows"],
        student_evaluation_saved=student is not None,
        student_data_digest=None if student is None else student["student_digest"],
        trial_record_count=len(histories), inner_fit_count=len(inner),
        paired_outer_groups=result["fixed"]["eligible"],
        trial_sequence="Sequential deterministic proposals from seed, study ID, trial number and completed history",
        fold_overlap="All group checks passed", baseline_source=manifest["baseline_source"]))
    figures(result, output)

    counts = {s: sum(r["state"] == s for r in histories) for s in ("COMPLETE", "FAIL", "PRUNED")}
    interval = selection["paired_trip_interval"]
    improved = sum(r["top1_difference"] > 0 for r in result["per_trip"])
    declined = sum(r["top1_difference"] < 0 for r in result["per_trip"])
    lines = ["# PathSnap Phase 4 Bayesian optimisation", "",
             "These results measure agreement with qualified route-proxy labels.", "",
             f"- Fixed Phase 3 training baseline source: `{manifest['baseline_source']}`.",
             f"- Outer / inner / final CV folds: {manifest['outer_folds']} / {manifest['inner_folds']} / {manifest['final_folds']}.",
             f"- Trial budgets: {manifest['outer_trials']} per outer study and {manifest['final_trials']} for full training. "
             f"Completed {counts['COMPLETE']}, failed {counts['FAIL']}, pruned {counts['PRUNED']}.",
             f"- Seed: {manifest['seed']}. Final parameters: `{canonical(selection['final_params'])}`.",
             f"- Paired outer Top-1 change: **{selection['nested_top1_difference'] * 100:+.3f} percentage points**. "
             f"95% trip-bootstrap interval [{interval['lower']*100:+.3f}, {interval['upper']*100:+.3f}] percentage points.",
             f"- Per trip: {improved} improved, {declined} declined and "
             f"{len(result['per_trip']) - improved - declined} tied.",
             f"- Training-only recommendation: **{selection['recommendation']}**.", "",
             "| Metric | Fixed outer CV | Nested tuned outer CV | Held-out Phase 3 | Held-out Phase 4 |",
             "|---|---:|---:|---:|---:|"]
    for metric in METRICS:
        def format_value(value):
            if metric in RATES:
                return f"{value:.4%}"
            if metric == "log_loss":
                return f"{value:.6f}"
            return f"{int(value):,}"
        values = [format_value(result["fixed"][metric]), format_value(result["tuned"][metric])]
        values += [format_value(student[k][metric]) for k in ("baseline", "metrics")] if student else ["—", "—"]
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    lines.extend(["", "The model family was selected in Phase 3 using training CV. The Phase 4 outer "
                  "scores assess inner-fold tuning conditional on that earlier model-family choice. "
                  "The final configuration comes from its own full-training study.", "",
                  "Student testing was previously used for Phase 3 reporting. Its Phase 4 evaluation "
                  "used the frozen model and serves as a reused development holdout.", "",
                  "Timing and memory are in `training_runtime.csv`. The full trial table is in "
                  "`optimization_trials.csv`, with validation counts and metrics in `inner_fold_metrics.csv`."])
    with atomic_path(output / "summary.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
