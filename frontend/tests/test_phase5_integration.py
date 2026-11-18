"""Contract-shaped synthetic fixtures; never run decoder/model/evaluator code."""
from collections import Counter
from pathlib import Path
from unittest.mock import Mock
import folium
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest
from frontend.models.phase5 import Phase5Result
from frontend.models.schemas import Geometry, PredictionRequest
from frontend.services.api_client import ClientError, HttpPathSnapClient
from frontend.services.mock_client import MockPathSnapClient
from frontend.services.prediction_adapter import adapt_phase5, classifier_selections, parse_prediction_response
from frontend.components.map_view import build_map, route_lines
from frontend.pages.predictions import validate_result


def phase5_payload(status="disconnected"):
    """Invented schema 1.0 example with independent directed line features."""
    identity = dict(partition="synthetic_city", trip_id="00000083")
    model_id = "phase3/gradient_boosting"
    matches = [dict(**identity, observation_id=i, arc_id=17+i, candidate_rank=2,
        model_score=.7, segment_id=0, classifier_arc_id=40+i, candidate_count=2,
        status="matched", break_reason=None) for i in range(3)]
    groups = [[0, 1, 2]]
    if status == "partial":
        groups = [[0, 1]]
    elif status == "disconnected":
        groups = [[0], [2]]
        matches[2].update(segment_id=1, break_reason="no_usable_states")
    elif status == "empty":
        groups = []
    selected_ids = {i for group in groups for i in group}
    for i, match in enumerate(matches):
        if i not in selected_ids:
            match.update(arc_id=None, candidate_rank=None, model_score=None, segment_id=None,
                status="unmatched", break_reason="no_usable_states")
            if status == "empty" and i != 1:
                match.update(candidate_count=0, classifier_arc_id=None, break_reason="no_candidates")
    segments, features = [], []
    for segment_id, ids in enumerate(groups):
        arc_ids = [matches[i]["arc_id"] for i in ids]
        reason = matches[ids[0]]["break_reason"]
        segments.append(dict(segment_id=segment_id, break_reason=reason,
            first_observation_id=ids[0], last_observation_id=ids[-1], arc_ids=arc_ids,
            occurrences=[dict(arc_id=matches[i]["arc_id"], role="selected", first_observation_id=i,
                last_observation_id=i) for i in ids], matched_distance_m=0 if len(ids)==1 else 100,
            full_arc_length_m=150 * len(ids)))
        # Deliberately distant lines reveal any accidental bridge between segments.
        coords = [[77+segment_id, 12+segment_id], [77.001+segment_id, 12.001+segment_id]]
        features.append(dict(type="Feature", id=segment_id,
            properties=dict(**identity, model_id=model_id, segment_id=segment_id,
                break_reason=reason, arc_count=len(arc_ids)),
            geometry=dict(type="LineString", coordinates=coords)))
    return dict(schema_version="1.0", **identity, model_id=model_id,
        observation_count=3, status=status, decoder_config={"restart_cost": 12}, matches=matches,
        route_segments=segments, route_geojson=dict(type="FeatureCollection", features=features),
        break_counts=dict(Counter(m["break_reason"] for m in matches if m["break_reason"])),
        routing=dict(searches=2, cache_hits=1, budget_exhaustions=0), processing_time_s=.125)


@pytest.mark.parametrize("status", ["connected", "partial", "disconnected", "empty"])
def test_phase5_status_and_roundtrip(status):
    payload = phase5_payload(status)
    parsed = Phase5Result.model_validate(payload)
    assert Phase5Result.model_validate_json(parsed.model_dump_json()) == parsed
    result = adapt_phase5(payload, "job-1")
    assert result.mode == "hmm_viterbi" and result.model_id == "phase3/gradient_boosting"
    assert result.runtime_s == payload["processing_time_s"]
    assert result.trip_id == "00000083" and result.phase5.status == status
    assert result.decoded_route is None and result.selections == []


def test_unmatched_is_not_always_empty_candidate_group():
    result = adapt_phase5(phase5_payload("empty"))
    assert len(result.phase5.matches) == 3
    assert result.empty_observation_ids == [0, 2]
    assert result.phase5.matches[1].candidate_count == 2
    assert result.phase5.matches[1].model_score is None
    assert result.phase5.route_geojson.features == []


def test_zero_observation_output():
    payload = phase5_payload("empty")
    payload.update(observation_count=0, matches=[], break_counts={})
    result = adapt_phase5(payload)
    assert result.observation_count == 0 and result.empty_observation_ids == []
    assert route_lines(result) == []


def test_do_not_borrow_decoder_probability_for_raw_classifier():
    payload = phase5_payload()
    payload["matches"][0]["classifier_arc_id"] = payload["matches"][0]["arc_id"]
    result = adapt_phase5(payload)
    raw = classifier_selections(result)
    assert raw[0].probability == .7 and raw[0].candidate_rank == 2
    assert raw[1].probability is None and raw[1].candidate_rank is None
    assert raw[1].arc_id == payload["matches"][1]["classifier_arc_id"]


def test_map_draws_disconnected_lines_independently():
    payload = phase5_payload()
    result = adapt_phase5(payload)
    before = result.phase5.model_dump_json()
    lines = route_lines(result)
    assert len(lines) == 2 and all(len(coordinates) == 2 for _, coordinates in lines)
    geometry = Geometry(partition=result.partition, trip_id=result.trip_id, observations=[], arcs=[])
    view = build_map(geometry, result, layers=["Decoded route"])
    group = next(c for c in view._children.values() if isinstance(c, folium.FeatureGroup) and "HMM" in c.layer_name)
    polylines = [c for c in group._children.values() if isinstance(c, folium.PolyLine)]
    assert len(polylines) == 2
    assert polylines[0].locations == [[12, 77], [12.001, 77.001]]
    assert polylines[1].locations == [[13, 78], [13.001, 78.001]]
    assert result.phase5.model_dump_json() == before
    assert "disconnected" in view.get_root().render()


def test_direct_payload_and_job_lifecycle_status_are_separate():
    payload = phase5_payload("empty")
    direct = parse_prediction_response(payload)
    assert direct.status == "completed" and direct.result.phase5.status == "empty"
    envelope = parse_prediction_response(dict(result_id="backend-job", status="completed", result=payload))
    assert envelope.result_id == envelope.result.result_id == "backend-job"
    poll = parse_prediction_response(payload, "polled-job")
    assert poll.result_id == "polled-job"
    assert parse_prediction_response(dict(result_id="pending", status="running")).status == "running"


@pytest.mark.parametrize("invalid", ["version", "status", "trip_id", "coordinates", "linestring", "match_order",
    "match_count", "match_identity", "null_match", "rank", "segment_id", "feature_identity", "feature_count", "arc_order", "break_counts", "runtime"])
def test_invalid_phase5_responses_rejected(invalid):
    payload = phase5_payload()
    if invalid == "version": payload["schema_version"] = "2.0"
    elif invalid == "status": payload["status"] = "connected"
    elif invalid == "trip_id": payload["trip_id"] = 83
    elif invalid == "coordinates": payload["route_geojson"]["features"][0]["geometry"]["coordinates"][0] = [12, 181]
    elif invalid == "linestring": payload["route_geojson"]["features"][0]["geometry"]["type"] = "MultiLineString"
    elif invalid == "match_order": payload["matches"].reverse()
    elif invalid == "match_count": payload["observation_count"] = 4
    elif invalid == "match_identity": payload["matches"][0]["partition"] = "wrong"
    elif invalid == "null_match": payload["matches"][1]["arc_id"] = 17
    elif invalid == "rank": payload["matches"][0]["candidate_rank"] = 3
    elif invalid == "segment_id": payload["matches"][0]["segment_id"] = 8
    elif invalid == "feature_identity": payload["route_geojson"]["features"][0]["properties"]["trip_id"] = "00000001"
    elif invalid == "feature_count": payload["route_geojson"]["features"].pop()
    elif invalid == "arc_order": payload["route_segments"][0]["arc_ids"] = [999]
    elif invalid == "break_counts": payload["break_counts"] = {}
    elif invalid == "runtime": payload["processing_time_s"] = float("nan")
    with pytest.raises(ValidationError):
        adapt_phase5(payload)


def test_repeated_arc_occurrences_and_connectors_preserved():
    payload = phase5_payload("connected")
    segment = payload["route_segments"][0]
    segment["arc_ids"] = [17, 99, 18, 17]
    segment["occurrences"] = [segment["occurrences"][0], dict(arc_id=99, role="connector"),
        segment["occurrences"][1], dict(arc_id=17, role="selected", first_observation_id=2, last_observation_id=2)]
    payload["matches"][2]["arc_id"] = 17
    payload["route_geojson"]["features"][0]["properties"]["arc_count"] = 4
    result = adapt_phase5(payload)
    assert result.phase5.route_segments[0].arc_ids == [17, 99, 18, 17]
    assert result.phase5.route_segments[0].occurrences[1].role == "connector"


def test_http_phase5_post_and_result_endpoint():
    demo = MockPathSnapClient()
    caps = demo.capabilities()
    caps.synthetic = False
    for mode in caps.modes:
        mode.available = mode.id == "hmm_viterbi"
    response = Mock(status_code=200)
    response.json.side_effect = [caps.model_dump(), phase5_payload(), phase5_payload()]
    response.raise_for_status.return_value = None
    session = Mock()
    session.request.return_value = response
    client = HttpPathSnapClient("http://backend:8000", session=session)
    request = PredictionRequest(partition="synthetic_city", trip_id="00000083", mode="hmm_viterbi")
    job = client.predict(request)
    assert job.result.phase5.status == "disconnected"
    assert session.request.call_args.kwargs["json"] == request.model_dump()
    assert client.result("backend-job").result_id == "backend-job"
    validate_result(job.result, request)
    with pytest.raises(ClientError):
        validate_result(job.result, request.model_copy(update={"mode": "phase3_fixed"}))


@pytest.mark.parametrize("status", ["disconnected", "empty", "partial"])
def test_phase5_results_page_preserves_mock_mode(monkeypatch, status):
    original = MockPathSnapClient.capabilities
    def capabilities(client):
        caps = original(client)
        for mode in caps.modes:
            if mode.id == "hmm_viterbi": mode.available = True
        return caps
    # Enable a contract-shaped synthetic response only inside the test harness.
    monkeypatch.setattr(MockPathSnapClient, "capabilities", capabilities)
    def predict(client, request):
        result = adapt_phase5(phase5_payload(status), "phase5-test", synthetic=True)
        from frontend.models.schemas import PredictionJob
        return PredictionJob(result_id=result.result_id, status="completed", result=result)
    monkeypatch.setattr(MockPathSnapClient, "predict", predict)
    root = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(root / "frontend/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Route Prediction").run()
    assert app.selectbox[2].value == "phase3_fixed"
    app.selectbox[2].set_value("hmm_viterbi").run()
    next(b for b in app.button if b.label == "Run Prediction").click().run()
    assert not app.exception and not app.error
    assert app.session_state["prediction_result"].phase5.status == status
    assert any("0.125 s" == metric.value for metric in app.metric)
    if status == "disconnected":
        assert any("Disconnected route" in message.value for message in app.warning)
    app.sidebar.radio[0].set_value("System Status").run()
    assert not app.exception and not app.error
