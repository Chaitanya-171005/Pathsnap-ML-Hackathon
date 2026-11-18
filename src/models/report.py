"""Render aggregate Phase 3 results from measured trial records."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_reports(output, comparison, held_out, baselines, folds, per_trip, timings, coverage, selection, importances):
    ids = [r["model"] for r in comparison]
    names = [name.replace("_", " ").title() for name in ids]
    test = {r["model"]: r for r in held_out}
    final = {r["model"]: r for r in timings if r["stage"] == "final_student_test"}
    lines = [
        "# Phase 3 classifier comparison", "",
        "All pointwise metrics measure agreement with qualified supplied-route proxy labels.", "",
        f"Training-only selection: **{selection['selected'].replace('_', ' ')}**.", "",
        "| Model | OOF Top-1 | OOF trip macro | OOF Top-3 | OOF F1 | OOF log loss | Student Top-1 | Student trip macro | Student F1 | Final fit (s) | Student predict (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        name = row["model"]
        s = test[name]
        lines.append(
            f"| {name} | {row['top1']:.3%} | {row['trip_macro_top1']:.3%} | "
            f"{row['top3']:.3%} | {row['f1']:.4f} | {row['log_loss']:.4f} | "
            f"{s['top1']:.3%} | {s['trip_macro_top1']:.3%} | {s['f1']:.4f} | "
            f"{final[name]['fit_s']:.2f} | {final[name]['predict_s']:.2f} |")
    lines.extend(["", "## Nearest geometric candidate", ""])
    lines.extend(f"- {row['partition']} — {row['correct']:,}/{row['eligible']:,} ({row['top1']:.3%})."
                 for row in baselines)
    lines.extend(["", "## Denominators", ""])
    lines.extend(
        f"- {r['partition']} — {r['original']:,} original, {r['qualified']:,} qualified "
        f"({r['proxy_label_coverage']:.3%}), {r['eligible']:,} eligible "
        f"({r['candidate_recall']:.3%} of qualified), {r['misses']:,} candidate misses."
        for r in coverage)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.2, "axes.axisbelow": True})
    x = np.arange(len(ids))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    for i, (title, data) in enumerate((("Training OOF", comparison), ("Student testing", [test[m] for m in ids]))):
        ax.bar(x + (i - 0.5) * 0.36, [r["top1"] for r in data], width=0.36, label=title)
    ax.axhline(baselines[0]["top1"], linestyle="--", color="#445265", label="Training nearest")
    ax.set(title="Observation Top-1 proxy agreement", ylim=(0, 1))
    ax.legend()
    ax = axes[0, 1]
    ax.bar(x, [row["f1"] for row in comparison], color="#257e73")
    ax.set(title="OOF candidate binary F1", ylim=(0, 1))
    ax = axes[1, 0]
    for idx, model in enumerate(ids):
        model_folds = [r for r in folds if r["model"] == model]
        ax.scatter(np.full(len(model_folds), idx), [r["top1"] for r in model_folds], alpha=0.8)
    ax.set(title="Five grouped validation folds", ylabel="Top-1 agreement", ylim=(0, 1))
    ax = axes[1, 1]
    for i, (label, key) in enumerate((("Final fit", "fit_s"), ("Student prediction", "predict_s"))):
        ax.bar(x + (i - 0.5) * 0.36, [final[m][key] for m in ids], width=0.36, label=label)
    ax.set(title="Runtime in seconds (log scale)", yscale="log")
    ax.legend()
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=20, ha="right")
    fig.savefig(output / "model_comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for i, model in enumerate(ids):
        values = [r["top1"] for r in per_trip if r["model"] == model and r["partition"] == "train_oof"]
        ax.boxplot(values, positions=[i], manage_ticks=False)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Top-1 proxy agreement",
           title="Per-trip training OOF distribution")
    fig.savefig(output / "per_trip_top1.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ids), figsize=(14, 3.6), layout="constrained")
    for ax, row, name in zip(axes, comparison, names, strict=True):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}\n{rates[i, j]:.1%}", ha="center", va="center",
                    color="white" if rates[i, j] > 0.55 else "#142334", fontsize=9)
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class")
        ax.grid(False)
    axes[0].set_ylabel("Proxy class")
    fig.suptitle("OOF candidate confusion matrices · counts and actual-class percentages")
    fig.savefig(output / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    for ax, model in zip(axes, ("logistic_regression", "decision_tree", "random_forest"), strict=True):
        rows = sorted((r for r in importances if r["model"] == model),
                      key=lambda r: abs(r["value"]))[-8:]
        ax.barh([r["feature"] for r in rows], [r["value"] for r in rows], color="#257e73")
        ax.set(title=model.replace("_", " ").title(),
               xlabel="Signed coefficient" if model == "logistic_regression" else "Impurity decrease")
    fig.suptitle("Eight largest feature weights in fitted training pipelines")
    fig.savefig(output / "feature_importance.png", dpi=140)
    plt.close(fig)
