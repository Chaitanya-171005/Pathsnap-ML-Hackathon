"""Translate backend result formats into presentation data, without inference."""
from uuid import uuid4
from frontend.models.phase5 import Phase5Result
from frontend.models.schemas import CandidateSelection, PredictionJob, PredictionResult


def adapt_phase5(payload, result_id=None, *, synthetic=False):
    decoded = Phase5Result.model_validate(payload)
    return PredictionResult(
        result_id=result_id or f"inline-{uuid4().hex}", partition=decoded.partition,
        trip_id=decoded.trip_id, mode="hmm_viterbi", model_id=decoded.model_id,
        observation_count=decoded.observation_count, phase5=decoded,
        runtime_s=decoded.processing_time_s, synthetic=synthetic,
        empty_observation_ids=[m.observation_id for m in decoded.matches if m.candidate_count == 0],
    )


def parse_prediction_response(payload, result_id=None):
    """Accept direct decode_trip JSON or a proposed API job wrapping that JSON.

    Job lifecycle status stays separate from Phase 5 route connectivity status.
    Existing classifier and synthetic-demo envelopes remain compatible.
    """
    if isinstance(payload, dict) and "schema_version" in payload:
        result = adapt_phase5(payload, result_id)
        return PredictionJob(result_id=result.result_id, status="completed", result=result)
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict) and "schema_version" in payload["result"]:
        payload = dict(payload)
        payload["result"] = adapt_phase5(payload["result"], payload.get("result_id")).model_dump()
    return PredictionJob.model_validate(payload)


def classifier_selections(result):
    """Keep raw Top-1 classifier arcs distinct from decoder-selected matches.

    Phase 5 does not supply the raw Top-1 score/rank when it differs from the
    decoded arc, so those fields stay unavailable instead of borrowing HMM data.
    """
    if not result.phase5:
        return result.selections
    return [CandidateSelection(observation_id=m.observation_id, arc_id=m.classifier_arc_id,
        candidate_rank=m.candidate_rank if m.arc_id == m.classifier_arc_id else None,
        probability=m.model_score if m.arc_id == m.classifier_arc_id else None)
        for m in result.phase5.matches if m.classifier_arc_id is not None]


def has_route(result):
    return bool(result and (result.decoded_route or (result.phase5 and result.phase5.route_geojson.features)))
