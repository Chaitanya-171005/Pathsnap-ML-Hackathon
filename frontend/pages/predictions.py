import streamlit as st
from frontend.models.schemas import FINAL_PARTITION, PredictionRequest
from frontend.services.api_client import ClientError
from frontend.components.metric_cards import header, metric_cards
from frontend.components.navigation import fetch_cached, trip_selector, mode_selector
from frontend.components.map_view import render_map


def validate_result(result, request):
    if (result.partition, result.trip_id, result.mode) != (request.partition, request.trip_id, request.mode):
        raise ClientError("Backend returned a result for a different trip or prediction mode.")
    return result


def render(ctx):
    header("Inference workspace / predictions", "Route Prediction", "Select a trip and an available mode, then inspect the returned selections.")
    if not ctx["caps"]:
        st.info("Connect a backend or select Demo mode to run predictions.")
        return
    selected = trip_selector(ctx["client"], ctx["caps"], ctx["namespace"], "prediction",
                             explicit_final_selection=True)
    if not selected:
        st.session_state.pop("final_auto_request", None)
        return
    final = selected[0] == FINAL_PARTITION
    mode = "hmm_viterbi" if final else mode_selector(ctx["caps"])
    if not mode:
        return
    request = PredictionRequest(partition=selected[0], trip_id=selected[1], mode=mode)
    pending_job = bool(st.session_state.get("pending_job"))
    if final:
        st.caption("Final evaluation · frozen HMM / Viterbi")
        auto_run = not pending_job and st.session_state.get("final_auto_request") != request
        run_clicked = st.button("Run Again", disabled=pending_job)
        if auto_run or run_clicked:
            st.session_state.final_auto_request = request
    else:
        st.session_state.pop("final_auto_request", None)
        auto_run = False
        run_clicked = st.button("Run Prediction", type="primary", disabled=pending_job)
    if auto_run or run_clicked:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_request", None)
        with st.status("Submitting prediction…", expanded=True) as status:
            st.write("The service receives trip identifiers and the selected mode; inference stays on the backend.")
            job = ctx["client"].predict(request)
            if job.status == "failed":
                status.update(label="Prediction failed", state="error")
                st.error(job.error or "Backend inference failed.")
            elif job.status == "completed":
                st.session_state.prediction_result = validate_result(job.result, request)
                st.session_state.prediction_request = request
                status.update(label="Prediction completed", state="complete")
            else:
                st.session_state.pending_job = (job.result_id, request)
                status.update(label="Prediction accepted · check progress below", state="complete")
    pending = st.session_state.get("pending_job")
    if pending:
        pending_notice = st.empty()
        pending_notice.info(f"Job {pending[0]} is queued or running. Check progress to retrieve its result.")
        left, right = st.columns(2)
        if left.button("Check prediction progress"):
            job = ctx["client"].result(pending[0])
            if job.result_id != pending[0]:
                raise ClientError("Backend returned a different job ID.")
            if job.status == "completed":
                st.session_state.prediction_result = validate_result(job.result, pending[1])
                st.session_state.prediction_request = pending[1]
                st.session_state.pop("pending_job", None)
                pending_notice.empty()
            elif job.status == "failed":
                st.error(job.error or "Backend inference failed.")
                st.session_state.pop("pending_job", None)
            else:
                st.caption(f"Current job status: {job.status}")
        if right.button("Dismiss local job tracking"):
            st.session_state.pop("pending_job", None)
            st.rerun()
    result = st.session_state.get("prediction_result")
    if result is None:
        st.info("No prediction result yet. Choose an available mode and run a prediction.")
        return
    if (result.partition, result.trip_id, result.mode) != (request.partition, request.trip_id, request.mode):
        st.info("The saved result is for a different selection. Run a new prediction or restore its trip and mode.")
        return
    st.success("Synthetic demonstration completed" if result.synthetic else "Backend prediction completed")
    st.write(f"Trip **{result.trip_id}** · Model **{result.model_id}**")
    matched_count = sum(m.status == "matched" for m in result.phase5.matches) if result.phase5 else len(result.selections)
    metric_cards([("GPS observations", result.observation_count, "number"), ("Matched observations" if result.phase5 else "Selected candidates", matched_count, "number"),
        ("Empty candidate groups", len(result.empty_observation_ids), "number"), ("Backend runtime", result.runtime_s, "seconds")])
    if result.phase5:
        render_decoder_details(result.phase5)
    st.caption("Probabilities are raw positive-class candidate scores, not group-normalised probabilities. Synthetic examples have no measured inference runtime or evaluation metrics.")
    with st.spinner("Loading result map…"):
        geometry = fetch_cached(ctx["client"], ctx["namespace"], "geometry", result.partition, result.trip_id)
    if geometry.synthetic != result.synthetic:
        raise ClientError("Result and geometry disagree about synthetic provenance.")
    render_map(geometry, result, key=f"result-{result.result_id}")
    with st.expander("GPS observation-to-road matches" if result.phase5 else "Predicted directed arcs & candidate probabilities", expanded=True):
        if result.phase5:
            st.dataframe([m.model_dump() for m in result.phase5.matches], hide_index=True, width="stretch")
            st.caption("arc_id / model_score describe the decoded match; classifier_arc_id is the independent raw classifier Top-1. Unmatched observations retain null selected values.")
        elif result.selections:
            st.dataframe([s.model_dump() for s in result.selections], hide_index=True, width="stretch")
        else:
            st.info("No candidates were selected. Empty groups are preserved in the response.")
        if result.empty_observation_ids:
            st.write("Observations with empty candidate groups", result.empty_observation_ids)
    st.subheader("Backend metrics")
    for label, values in (("Route continuity", result.continuity_metrics), ("Permitted evaluation", result.evaluation_metrics)):
        st.write(f"**{label}**")
        if values:
            st.json(values)
        elif final and label == "Permitted evaluation":
            st.caption("Score pending organiser confirmation or unavailable for this reference.")
        else:
            st.caption("Not available")
    st.download_button("Download validated result JSON", result.model_dump_json(indent=2), file_name=f"pathsnap-{result.trip_id}.json", mime="application/json")
    if result.phase5:
        st.download_button("Download Phase 5 JSON", result.phase5.model_dump_json(indent=2, exclude_unset=True),
            file_name=f"{result.trip_id}.pathsnap.json", mime="application/json")
        st.download_button("Download route GeoJSON", result.phase5.route_geojson.model_dump_json(indent=2),
            file_name=f"{result.trip_id}.pathsnap.geojson", mime="application/geo+json")


def render_decoder_details(decoded):
    """Present supplied decoder state and diagnostics; no route inference."""
    st.write(f"**Route status:** {decoded.status}")
    metric_cards([("Route segments", len(decoded.route_segments), "number"),
        ("Unmatched observations", sum(m.status == "unmatched" for m in decoded.matches), "number")])
    if decoded.status == "empty":
        st.info("No decoded route segments. Observation records and break reasons remain available below.")
    elif decoded.status == "partial":
        st.warning("Partial route · a single decoded segment with unmatched GPS observations.")
    with st.expander("Directed route segments & break diagnostics"):
        if decoded.route_segments:
            st.dataframe([s.model_dump() for s in decoded.route_segments], hide_index=True, width="stretch")
        else:
            st.caption("No route segments")
        st.write("**Break reasons (annotated observation records)**")
        st.json(decoded.break_counts)
        st.caption("A break can annotate an unmatched row and a subsequent restart. Break counts are not route-segment counts.")
        st.write("**Backend routing diagnostics**")
        st.json(decoded.routing.model_dump())
        st.write("**Decoder configuration used by backend**")
        st.json(decoded.decoder_config)
