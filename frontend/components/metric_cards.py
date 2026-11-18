import math
import streamlit as st


def format_metric(value, kind="number"):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "Not available"
    if kind == "percent":
        return f"{value * 100:.3f}%"
    if kind == "seconds":
        return f"{value:.3f} s"
    if kind == "decimal":
        return f"{value:.6f}"
    return f"{value:,.0f}"


def metric_cards(items):
    for start in range(0, len(items), 4):
        batch = items[start:start + 4]
        for col, (label, value, kind) in zip(st.columns(len(batch)), batch):
            with col:
                st.metric(label, format_metric(value, kind), border=True)


def header(eyebrow, title, description):
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)
