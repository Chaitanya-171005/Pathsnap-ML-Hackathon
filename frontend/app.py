"""Launch from the repository root: python -m streamlit run frontend/app.py."""
import sys
import time
from pathlib import Path

# Streamlit executes app.py as a script; resolve imports independently of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from frontend import __version__
from frontend.config import Config
from frontend.services.api_client import ClientError, HttpPathSnapClient
from frontend.services.mock_client import MockPathSnapClient
from frontend.services.report_loader import ReportLoader
from frontend.pages import analytics, overview, predictions, system_status, trip_explorer

PAGES = {"Overview": overview, "Trip Explorer": trip_explorer, "Route Prediction": predictions,
         "Model Analytics": analytics, "System Status": system_status}


def main():
    st.set_page_config(page_title="PathSnap · Geospatial intelligence", page_icon="📍", layout="wide",
        initial_sidebar_state="expanded")
    st.markdown(f"<style>{(Path(__file__).parent / 'assets/styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    try:
        config = Config.from_env()
    except ValueError as exc:
        st.error(f"Frontend configuration error: {exc}")
        return
    with st.sidebar:
        st.markdown('<div class="brand">Path<span>Snap</span> <small>↗</small></div>', unsafe_allow_html=True)
        st.caption("GEOSPATIAL INTELLIGENCE")
        st.divider()
        page = st.radio("Workspace", list(PAGES), key="nav")
        st.divider()
        mode = st.segmented_control("Data source", ["Demo", "Live"], default="Demo", key="data_source")
        mode = mode or "Demo"
    namespace = (mode, config.api_url, config.timeout)
    if st.session_state.get("client_namespace") != namespace:
        st.session_state.client_namespace = namespace
        for key in ("prediction_result", "prediction_request", "pending_job", "connection_snapshot", "health_latency_ms", "service_cache"):
            st.session_state.pop(key, None)
        st.session_state.client = MockPathSnapClient() if mode == "Demo" else (HttpPathSnapClient(config.api_url, config.timeout) if config.api_url else None)
    client = st.session_state.client
    snapshot = st.session_state.get("connection_snapshot")
    if snapshot is None or time.monotonic() - snapshot[0] > 30:
        health, caps, error = None, None, None
        if client:
            try:
                health = client.health()
                st.session_state.health_latency_ms = client.last_request.latency_ms
                caps = client.capabilities()
                if mode == "Live" and caps.synthetic:
                    raise ClientError("Backend advertises synthetic data. Select Demo or connect a real inference service.")
            except ClientError as exc:
                error = str(exc)
                caps = None
        else:
            error = "PATHSNAP_API_URL is not configured. Set the environment variable and restart Streamlit."
        snapshot = (time.monotonic(), health, caps, error)
        st.session_state.connection_snapshot = snapshot
    _, health, caps, error = snapshot
    with st.sidebar:
        if error:
            st.error("Backend unavailable")
        elif mode == "Demo":
            st.success("Demo service ready · synthetic")
        elif health and health.status == "ok":
            st.success("Backend connected")
        else:
            st.warning("Backend degraded")
        st.caption("Available prediction modes")
        if caps:
            for prediction_mode in caps.modes:
                if prediction_mode.available:
                    st.caption(f"● {prediction_mode.label}")
        else:
            st.caption("Not available")
        st.divider()
        st.caption(f"FRONTEND v{__version__} · API v1 proposed")
        st.caption("Default: fixed Phase 3 classifier")
        st.caption("Streamlit / Folium / Plotly")
    if error:
        st.warning(error)
    if mode == "Demo":
        st.caption("DEMO MODE · All trip geometry and prediction examples are synthetic. Analytics use actual repository reports.")
    ctx = dict(config=config, mode=mode, client=client, namespace=namespace, caps=caps, health=health,
        reports=ReportLoader(), connection_error=error)
    def render_page():
        try:
            PAGES[page].render(ctx)
        except ClientError as exc:
            st.error(str(exc))
            st.info("Check System Status, refresh the connection, or switch to Demo mode.")
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Data could not be displayed: {exc}")
    # Public navigation API disables Streamlit's automatic pages/ discovery.
    st.navigation([st.Page(render_page, title="PathSnap")], position="hidden").run()


if __name__ == "__main__":
    main()
