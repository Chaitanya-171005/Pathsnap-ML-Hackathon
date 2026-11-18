import streamlit as st
from frontend.components.metric_cards import header, metric_cards
from frontend.services.report_loader import LABELS


def render(ctx):
    reports = ctx["reports"]
    header("Geospatial intelligence / overview", "PathSnap", "Intelligent GPS Map Matching and Route Reconstruction")
    st.markdown('<div class="hero"><span class="pill">FROM NOISY SIGNALS TO PLAUSIBLE JOURNEYS</span><h2>Every journey has a path.<br>Find the one that fits.</h2><p>PathSnap combines machine learning and sequential route modelling to reconstruct plausible vehicle journeys from noisy GPS observations and directed road networks.</p></div>', unsafe_allow_html=True)
    st.caption("EXPERIMENT SNAPSHOT · Actual repository reports · Qualified proxy-label agreement")
    coverage, train, test = reports.table("coverage"), reports.table("comparison"), reports.table("student")
    def value(table, column, field, match):
        if table.empty or column not in table:
            return None
        rows = table[table[field] == match]
        return rows.iloc[0][column] if not rows.empty else None
    default = reports.default_model
    metric_cards([
        ("Training trips", value(coverage, "trips", "partition", "train"), "number"),
        ("Student-testing trips", value(coverage, "trips", "partition", "student_test"), "number"),
        ("Models trained", train.model.nunique() if not train.empty else None, "number"),
        ("Training GPS observations", value(coverage, "original", "partition", "train"), "number"),
        ("Eligible training groups", value(coverage, "eligible", "partition", "train"), "number"),
        ("Training OOF Top-1", value(train, "top1", "model", default), "percent"),
        ("Student-testing Top-1", value(test, "top1", "model", default), "percent"),
    ])
    st.caption("Top-1 measures agreement with qualified reference-route proxy labels; it does not establish independently verified physical map-matching accuracy.")
    st.subheader("The reconstruction pipeline")
    st.markdown('<div class="pipeline">GPS input <b>→</b> Preprocessing <b>→</b> Candidates <b>→</b> Features <b>→</b> Classifier <b>→</b> <span class="pending">HMM / Viterbi · implemented, awaiting API connection</span> <b>→</b> Route visualisation</div>', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Project readiness")
        st.dataframe([
            {"Phase": "01 · Data exploration", "State": "Complete"},
            {"Phase": "02 · Candidates & proxy alignment", "State": "Complete"},
            {"Phase": "03 · Classifier training", "State": "Complete · fixed default"},
            {"Phase": "04 · Bayesian optimisation", "State": "Merged in starting Main · reports present" if reports.phase4_available else "Phase 4 results unavailable in this checkout"},
            {"Phase": "05 · HMM / Viterbi", "State": "Complete · schema 1.0 UI adapter ready · awaiting API connection"},
        ], hide_index=True, width="stretch")
    with right:
        st.subheader("Default classifier")
        with st.container(border=True):
            st.markdown(f"**{LABELS.get(default, default)}**")
            st.write("Fixed Phase 3 model")
            st.caption("Phase 4 remains experimental. Its paired improvement interval includes zero, and the fixed model remains the recommended default.")
    st.subheader("Explore the workspace")
    for col, name, text in zip(st.columns(3), ["Trip Explorer", "Model Analytics", "Route Prediction"],
        ["Inspect GPS observations and directed roads.", "Compare classifiers and tuning evidence.", "Run synthetic predictions or connect a backend."]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(text)
                st.button(f"Open {name}", key=f"go-{name}", on_click=lambda page=name: st.session_state.update(nav=page))
