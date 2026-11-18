from dataclasses import asdict
import streamlit as st
from frontend import __version__
from frontend.components.metric_cards import header


def render(ctx):
    header("Operations / health & evaluation", "System Status", "Connection details, backend capabilities and permitted evaluation results.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.write(f"**Frontend version:** {__version__}")
            st.write(f"**Mode:** {ctx['mode']}")
            st.write(f"**API URL:** {ctx['config'].api_url or 'Not configured'}")
            health = ctx.get("health")
            st.write(f"**Backend:** {health.status if health else 'Not connected'}")
            st.write(f"**Backend version:** {health.version if health else 'Not available'}")
            latency = st.session_state.get("health_latency_ms")
            st.write(f"**Health response time:** {latency:.1f} ms" if latency is not None else "**Health response time:** Not available")
            if ctx["mode"] == "Demo":
                st.caption("Health and capabilities refer to the local synthetic service; no real backend is connected.")
            st.button("Refresh connection & capabilities", on_click=lambda: st.session_state.pop("connection_snapshot", None))
    with right:
        with st.container(border=True):
            st.write("**Last service request**")
            if ctx["client"]:
                st.json(asdict(ctx["client"].last_request))
            else:
                st.caption("No requests yet")
            if ctx.get("connection_error"):
                st.error(ctx["connection_error"])
    st.subheader("Available models & prediction modes")
    if ctx["caps"]:
        st.dataframe([m.model_dump() for m in ctx["caps"].modes], hide_index=True, width="stretch")
        st.caption(f"API contract: {ctx['caps'].api_version} · Backend default: {ctx['caps'].default_mode}")
    else:
        st.info("Not available · connect a backend to discover capabilities.")
    st.subheader("Evaluation returned by the service")
    result = st.session_state.get("prediction_result")
    if result:
        st.write(f"Result {result.result_id} · Trip {result.trip_id}")
        if result.synthetic:
            st.info("Synthetic demo · no reference-based evaluation is performed.")
        st.write("**Proxy-label / permitted evaluation metrics**")
        st.json(result.evaluation_metrics) if result.evaluation_metrics else st.caption("Not available")
        st.write("**Route continuity metrics**")
        st.json(result.continuity_metrics) if result.continuity_metrics else st.caption("Not available")
        st.write("**HMM route reconstruction**")
        if result.phase5:
            st.write(f"Phase 5 schema {result.phase5.schema_version} · {result.phase5.status} · {len(result.phase5.route_segments)} route segments")
            st.caption(f"Inference runtime: {result.phase5.processing_time_s:.3f} s")
        else:
            st.caption(result.decoded_route.provenance if result.decoded_route else "Not available")
    else:
        st.info("Not available · run a prediction to inspect backend-returned metrics.")
    st.markdown("""
| Evidence | Meaning |
|---|---|
| Classifier candidate ranking | Independent directed road selection per GPS observation |
| Proxy-label agreement | Agreement on qualified reference-derived groups; report denominators separately |
| HMM route reconstruction | A sequential route supplied by the decoder, with explicit provenance |
| Route-level evaluation | Server-computed metrics on a decoded route, when permitted |
""")
    st.caption("The frontend never computes reference-based evaluation or reads evaluator datasets, .route files or model binaries.")
