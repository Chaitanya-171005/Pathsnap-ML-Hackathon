import pandas as pd
import streamlit as st
from frontend.components.metric_cards import header, metric_cards
from frontend.components.navigation import fetch_cached, trip_selector
from frontend.components.map_view import render_map


def observation_frame(geometry):
    return pd.DataFrame([{"Observation ID": o.observation_id, "Longitude": o.coordinate[0],
        "Latitude": o.coordinate[1], "Relative time (s)": o.relative_timestamp_s} for o in geometry.observations])


def render(ctx):
    header("Spatial workspace / explorer", "Trip Explorer", "Trace observations. Inspect directed roads. Understand the journey.")
    if not ctx["caps"]:
        st.info("Connect a backend or select Demo mode to explore trips.")
        return
    selected = trip_selector(ctx["client"], ctx["caps"], ctx["namespace"], "explorer")
    if not selected:
        return
    with st.spinner("Loading trip geometry…"):
        trip = fetch_cached(ctx["client"], ctx["namespace"], "trip", *selected)
        geometry = fetch_cached(ctx["client"], ctx["namespace"], "geometry", *selected)
    st.caption(f"TRIP {trip.trip_id} · {trip.description}")
    metric_cards([("GPS observations", trip.observation_count, "number"), ("Directed arcs", len(geometry.arcs), "number")])
    if not geometry.observations:
        st.info("This trip contains no GPS observations.")
    result = st.session_state.get("prediction_result")
    if result and (result.partition, result.trip_id) != selected:
        result = None
    render_map(geometry, result, key=f"explorer-{selected[0]}-{selected[1]}")
    with st.expander("Inspect GPS observations", expanded=False):
        st.dataframe(observation_frame(geometry), hide_index=True, width="stretch")
