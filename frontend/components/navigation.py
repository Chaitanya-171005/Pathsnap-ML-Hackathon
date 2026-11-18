"""Selection and fetch helpers shared by explorer and prediction pages."""
import streamlit as st
from frontend.services.api_client import ClientError


def fetch_cached(client, namespace, kind, *args):
    # Session-local cache prevents live trip geometry leaking into Demo mode or another URL.
    import time
    cache = st.session_state.setdefault("service_cache", {})
    key = (namespace, kind, *args)
    entry = cache.get(key)
    if entry is None or time.monotonic() - entry[0] > 60:
        value = getattr(client, kind)(*args)
        if kind in {"trip", "geometry"} and (value.partition, value.trip_id) != args:
            raise ClientError("Backend returned a different trip or partition than requested.")
        if kind == "trips" and any(t.partition != args[0] for t in value.trips):
            raise ClientError("Backend returned trips from a different partition.")
        if len(cache) >= 12:
            cache.pop(next(iter(cache)))
        cache[key] = (time.monotonic(), value)
    return cache[key][1]


def trip_selector(client, caps, namespace, key):
    if not caps.partitions:
        st.info("No trip partitions are exposed by the backend.")
        return None
    left, right = st.columns(2)
    partition = left.selectbox("Dataset partition", caps.partitions, key=f"{key}-partition")
    with st.spinner("Loading available trips…"):
        trips = fetch_cached(client, namespace, "trips", partition).trips
    if not trips:
        st.info("No trips available in this partition.")
        return None
    trip_id = right.selectbox("Trip ID", [t.trip_id for t in trips], key=f"{key}-trip")
    return partition, trip_id


def available_modes(caps):
    return [mode for mode in caps.modes if mode.available]


def mode_selector(caps):
    modes = available_modes(caps)
    if not modes:
        st.info("No prediction modes are currently available.")
        return None
    ids = [m.id for m in modes]
    selected = st.selectbox("Prediction mode", ids, index=ids.index(caps.default_mode) if caps.default_mode in ids else 0,
        format_func=lambda value: next(m.label for m in modes if m.id == value))
    for mode in caps.modes:
        if not mode.available:
            st.caption(f"Unavailable · {mode.label}: {mode.reason or 'Not enabled by backend'}")
    return selected
