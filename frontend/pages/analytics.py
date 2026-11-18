import pandas as pd
import plotly.express as px
import streamlit as st
from frontend.components.charts import COLORS, model_comparison, show_chart
from frontend.components.metric_cards import header
from frontend.services.report_loader import LABELS

METRICS = {"top1": "Top-1 proxy agreement", "trip_macro_top1": "Trip-macro Top-1", "top3": "Top-3 proxy agreement",
    "f1": "Binary candidate F1", "log_loss": "Candidate log loss", "mrr": "Mean reciprocal rank"}


def render(ctx):
    reports = ctx["reports"]
    header("Evidence workspace / analytics", "Model Analytics", "Five classifiers. One fixed default. Every figure backed by repository reports.")
    st.info("All classifier scores describe agreement with qualified reference-route proxy labels. They are not independently verified physical map-matching accuracy or route-level evaluation.")
    phase3, phase4, definitions = st.tabs(["Phase 3 · Model comparison", "Phase 4 · Optimisation", "Metric guide"])
    with phase3:
        metric = st.selectbox("Comparison metric", list(METRICS), format_func=METRICS.get)
        data = reports.comparison_data(metric)
        if data.empty:
            st.info("Phase 3 comparison reports unavailable in this checkout.")
        else:
            show_chart(model_comparison(data, metric), percent=metric != "log_loss")
            st.caption("★ marks the selected fixed Histogram Gradient Boosting model. Training OOF and held-out student testing use separate denominators. Lower log loss is better.")
        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("Nearest-road baseline")
            nearest = reports.table("nearest")
            if nearest.empty:
                st.caption("Not available")
            else:
                show_chart(px.bar(nearest, x="partition", y="top1", color="partition", color_discrete_sequence=["#7891ad", "#a4b3c6"],
                    labels={"top1": "Top-1 proxy agreement", "partition": "Evaluation partition"}), percent=True)
        with right:
            st.subheader("Training model ranking")
            ranking = reports.table("comparison")
            if not ranking.empty:
                ranking = ranking.sort_values(["top1", "trip_macro_top1", "log_loss", "model"], ascending=[False, False, True, True])
                st.dataframe(ranking[["model", "top1", "trip_macro_top1", "top3", "f1", "log_loss"]], hide_index=True, width="stretch")
                st.caption(f"Selected from training grouped OOF only: {LABELS.get(reports.default_model, reports.default_model)}")
        st.subheader("Per-trip performance")
        trips = reports.table("per_trip")
        if trips.empty:
            st.caption("Not available")
        else:
            partition = st.radio("Per-trip evaluation partition", list(trips.partition.unique()), horizontal=True)
            models = st.multiselect("Classifiers", list(LABELS), default=[reports.default_model, "decision_tree"], format_func=LABELS.get)
            plotted = trips[(trips.partition == partition) & trips.model.isin(models)].copy()
            plotted["classifier"] = plotted.model.map(LABELS)
            if plotted.empty:
                st.info("Choose at least one classifier.")
            else:
                fig = px.line(plotted, x="trip_id", y="top1", color="classifier", markers=True, color_discrete_sequence=COLORS,
                    hover_data=["eligible", "correct"], labels={"trip_id": "Trip ID", "top1": "Top-1 proxy agreement"})
                fig.update_xaxes(type="category")
                show_chart(fig, percent=True)
        with st.expander("Coverage and denominators"):
            st.dataframe(reports.table("coverage"), hide_index=True, width="stretch")
    with phase4:
        render_phase4(reports)
    with definitions:
        st.markdown("""
**Top-1 / Top-3:** qualified proxy target ranked first / within the first three candidates, divided by eligible groups.

**Trip-macro Top-1:** average of trip-level Top-1 values, weighting each represented trip equally.

**MRR:** average reciprocal rank of the qualified target within its candidate group.

**Binary F1:** candidate classification F1 at the recorded positive-class threshold (> 0.5). This differs from observation-level ranking.

**Candidate log loss:** binary candidate probability loss. Lower is better; probabilities are not normalised within each observation group.

**Qualified coverage:** accepted proxy observations / original GPS observations. **Candidate recall:** qualified targets present / qualified observations. Neither is Top-1 agreement.

**Route continuity and route-level evaluation:** require backend-supplied route results and permitted server-side evaluation. They cannot be inferred from these classifier reports.
""")


def render_phase4(reports):
    if not reports.phase4_available:
        st.info("Phase 4 results unavailable in this checkout")
        return
    selection = reports.selection(4)
    delta = selection.get("nested_top1_difference")
    interval = selection.get("paired_trip_interval", {})
    st.subheader("Fixed Phase 3 vs tuned Phase 4")
    st.success("Fixed Phase 3 Histogram Gradient Boosting retained as default. Tuned Phase 4 is experimental.")
    if delta is not None and {"lower", "upper"} <= interval.keys():
        st.write(f"Nested Top-1 change: **{delta * 100:+.3f} percentage points** · 95% paired trip-bootstrap interval **[{interval['lower'] * 100:+.3f}, {interval['upper'] * 100:+.3f}]** percentage points.")
        st.caption("The interval includes zero. Candidate log loss increased; recorded student-scoring time approximately doubled. These single-run timings depend on the machine.")
    rows = reports.table("phase4")
    partition = st.radio("Phase 4 evaluation partition", list(rows.partition.unique()), horizontal=True)
    metric = st.selectbox("Phase 4 metric", list(METRICS), format_func=METRICS.get)
    selected = rows[(rows.partition == partition) & (rows.metric == metric)]
    if selected.empty:
        st.caption("Not available")
    else:
        plot = selected.melt(id_vars=["metric"], value_vars=["fixed", "tuned"], var_name="Model", value_name="Value")
        show_chart(px.bar(plot, x="Model", y="Value", color="Model", color_discrete_sequence=COLORS), percent=metric != "log_loss")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Recorded runtime & model size")
    runtime = reports.table("phase4_runtime")
    if not runtime.empty:
        recorded = runtime[runtime.stage.isin(["final_fit", "archived_phase3_final", "student_test"])].copy()
        st.dataframe(recorded[["model", "stage", "fit_s", "predict_s", "model_bytes"]], hide_index=True, width="stretch")
        panels = st.columns(3)
        for col, field, label in zip(panels, ["fit_s", "predict_s", "model_bytes"], ["Final fit (s)", "Student scoring (s)", "Model size (bytes)"]):
            with col:
                present = recorded.dropna(subset=[field])
                if not present.empty:
                    show_chart(px.bar(present, x="model", y=field, color="model", color_discrete_sequence=COLORS, labels={field: label}))
    st.subheader("Optimisation history")
    trials = reports.table("trials")
    if not trials.empty:
        complete = trials[trials.state == "COMPLETE"].sort_values(["study", "trial"])
        study = st.selectbox("Optimisation study", list(complete.study.unique()))
        show_chart(px.line(complete[complete.study == study], x="trial", y="top1", markers=True,
            labels={"trial": "Trial number", "top1": "Grouped validation Top-1"}, color_discrete_sequence=COLORS), percent=True)
        st.caption("Inner-search scores describe that study's validation data; they are distinct from nested outer evaluation.")
