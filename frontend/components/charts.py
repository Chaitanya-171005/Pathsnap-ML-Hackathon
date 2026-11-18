import plotly.express as px
import streamlit as st

COLORS = ["#27d9cb", "#818cf8", "#f8b86a", "#f472b6", "#5caaff"]


def show_chart(fig, percent=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#dce8f5"),
        margin=dict(l=12, r=16, t=35, b=16), legend_title_text="", hovermode="closest")
    fig.update_yaxes(gridcolor="#20334c", tickformat=".1%" if percent else None)
    fig.update_xaxes(gridcolor="#20334c")
    st.plotly_chart(fig, width="stretch")


def model_comparison(data, metric):
    data = data.copy()
    data["label"] = data["label"] + data["selected"].map({True: " ★ default", False: ""})
    fig = px.bar(data, x="label", y="value", color="partition", barmode="group",
        color_discrete_sequence=COLORS, labels={"label": "Classifier", "value": metric.replace("_", " ").title()},
        hover_data={"model": True, "selected": True})
    fig.update_traces(hovertemplate="%{x}<br>%{y:.6f}<extra>%{fullData.name}</extra>")
    return fig
