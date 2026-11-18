"""Focused UI/service tests. No raw datasets, model loading or training."""
import importlib
import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest
import requests
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest
from frontend.config import Config
from frontend.models.schemas import (
    Capabilities, Geometry, PredictionJob, PredictionRequest, PredictionResult,
)
from frontend.services.api_client import ClientError, HttpPathSnapClient
from frontend.services.mock_client import MockPathSnapClient
from frontend.services.report_loader import ReportLoader
from frontend.components.metric_cards import format_metric
from frontend.components.map_view import build_map, display_indices, latlon
from frontend.components.charts import model_comparison
from frontend.components.navigation import available_modes
from frontend.pages.predictions import validate_result

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo():
    return MockPathSnapClient()


@pytest.fixture
def request_data():
    return PredictionRequest(partition="synthetic_city", trip_id="00000083")


@pytest.mark.parametrize("module", ["app", "config", "pages.overview", "pages.analytics", "pages.predictions", "pages.trip_explorer", "pages.system_status"])
def test_module_imports(module):
    assert importlib.import_module(f"frontend.{module}")


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("PATHSNAP_API_URL", raising=False)
    monkeypatch.delenv("PATHSNAP_REQUEST_TIMEOUT", raising=False)
    assert Config.from_env() == Config()


def test_config_lan_url(monkeypatch):
    monkeypatch.setenv("PATHSNAP_API_URL", "http://192.168.1.15:8000/")
    monkeypatch.setenv("PATHSNAP_REQUEST_TIMEOUT", "8")
    assert Config.from_env() == Config("http://192.168.1.15:8000", 8)


@pytest.mark.parametrize("url", ["file:///tmp/backend", "localhost", "http://user:password@host", "http://host?token=x"])
def test_config_invalid_url(monkeypatch, url):
    monkeypatch.setenv("PATHSNAP_API_URL", url)
    with pytest.raises(ValueError):
        Config.from_env()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "121", "invalid"])
def test_invalid_timeout(monkeypatch, timeout):
    monkeypatch.setenv("PATHSNAP_REQUEST_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        Config.from_env()


@pytest.mark.parametrize("value,kind,expected", [(None, "number", "Not available"), (float("nan"), "percent", "Not available"),
    (.9504933413, "percent", "95.049%"), (208417, "number", "208,417"), (.2, "seconds", "0.200 s")])
def test_format(value, kind, expected):
    assert format_metric(value, kind) == expected


def test_report_metrics_and_default():
    reports = ReportLoader()
    assert reports.default_model == "gradient_boosting"
    assert reports.table("coverage").iloc[0].original == 208417
    assert len(reports.table("comparison")) == 5
    assert reports.table("per_trip").iloc[0].trip_id == "00000001"


def test_optional_phase4():
    reports = ReportLoader(ROOT / "frontend/tests/absent_reports")
    assert not reports.phase4_available
    assert reports.table("phase4").empty
    assert reports.default_model == "gradient_boosting"
    assert reports.comparison_data("top1").empty


def test_phase4_actual_reports():
    reports = ReportLoader()
    assert reports.phase4_available
    assert reports.selection(4)["recommendation"] == "phase3"
    assert reports.selection(4)["paired_trip_interval"]["lower"] < 0
    row = reports.table("phase4").query("partition == 'training_outer' and metric == 'top1'").iloc[0]
    assert row.fixed == pytest.approx(.9504933413297624)


@pytest.mark.parametrize("metric", ["top1", "top3", "f1", "log_loss", "trip_macro_top1", "mrr"])
def test_chart_data(metric):
    reports = ReportLoader()
    data = reports.comparison_data(metric)
    assert len(data) == 10
    assert data.selected.sum() == 2
    assert len(model_comparison(data, metric).data) == 2
    expected = reports.table("comparison").set_index("model").loc["gradient_boosting", metric]
    assert data.query("model == 'gradient_boosting' and partition == 'Training OOF'").iloc[0].value == expected


@pytest.mark.parametrize("trip_id", MockPathSnapClient.IDS)
def test_demo_data(demo, trip_id):
    trip = demo.trip("synthetic_city", trip_id)
    geometry = demo.geometry("synthetic_city", trip_id)
    assert trip.trip_id == trip_id
    assert len(geometry.observations) == trip.observation_count
    assert geometry.synthetic
    assert geometry == MockPathSnapClient().geometry("synthetic_city", trip_id)
    assert all(a.coordinates[0] != a.coordinates[-1] for a in geometry.arcs)
    assert Geometry.model_validate_json(geometry.model_dump_json()) == geometry


def test_zero_padding_and_trip_selection(demo, request_data):
    assert demo.trips("synthetic_city").trips[0].trip_id == "00000083"
    assert request_data.model_dump()["trip_id"] == "00000083"
    with pytest.raises(ValidationError):
        PredictionRequest(partition="train", trip_id=83)
    with pytest.raises(ClientError):
        demo.trip("synthetic_city", "83")


def test_prediction_empty_groups_and_missing_hmm(demo, request_data):
    job = demo.predict(request_data)
    result = job.result
    assert result.synthetic and result.decoded_route is None
    assert result.empty_observation_ids
    assert len(result.selections) + len(result.empty_observation_ids) == result.observation_count
    assert not result.evaluation_metrics and result.runtime_s is None
    assert demo.result(job.result_id) == job
    assert PredictionJob.model_validate_json(job.model_dump_json()) == job


def test_unavailable_modes(demo):
    assert [m.id for m in available_modes(demo.capabilities())] == ["phase3_fixed", "demo_route"]
    with pytest.raises(ClientError):
        demo.predict(PredictionRequest(partition="synthetic_city", trip_id="00000083", mode="hmm_viterbi"))


def test_synthetic_route_continuity(demo):
    result = demo.predict(PredictionRequest(partition="synthetic_city", trip_id="00000083", mode="demo_route")).result
    route = result.decoded_route
    assert route.provenance == "synthetic_demo"
    arcs = {a.arc_id: a for a in demo.geometry("synthetic_city", "00000083").arcs}
    assert all(arcs[a].to_node == arcs[b].from_node for a, b in zip(route.arc_ids, route.arc_ids[1:]))


def test_coordinate_conversion_and_display_indices():
    assert latlon((77.5, 12.9)) == (12.9, 77.5)
    indices = display_indices(20000, 150)
    assert len(indices) == 150 and indices[0] == 0 and indices[-1] == 19999


def test_map_layers_do_not_fabricate_route(demo, request_data):
    geometry = demo.geometry(request_data.partition, request_data.trip_id)
    result = demo.predict(request_data).result
    original = geometry.model_dump_json()
    html = build_map(geometry, result).get_root().render()
    assert "Raw GPS" in html and "Directed road network" in html
    assert "independent arcs" in html
    assert "Backend HMM / Viterbi decoded route" not in html
    assert geometry.model_dump_json() == original
    assert "Raw GPS" not in build_map(geometry, layers=[]).get_root().render()


def test_map_synthetic_route_layer(demo):
    result = demo.predict(PredictionRequest(partition="synthetic_city", trip_id="00000083", mode="demo_route")).result
    html = build_map(demo.geometry(result.partition, result.trip_id), result).get_root().render()
    assert "SYNTHETIC route example" in html


def test_all_empty_prediction_groups_and_empty_map(demo, request_data):
    result = demo.predict(request_data).result.model_dump()
    result.update(selections=[], empty_observation_ids=list(range(result["observation_count"])))
    assert PredictionResult.model_validate(result).selections == []
    empty = Geometry(partition="train", trip_id="00000083", observations=[], arcs=[])
    assert build_map(empty)


def http_client(payload=None, error=None, status=200):
    response = Mock(status_code=status)
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session = Mock()
    session.request.side_effect = error
    if error is None:
        session.request.return_value = response
    return HttpPathSnapClient("http://backend:8000", timeout=3, session=session), response


def test_http_parsing_and_query(demo):
    client, _ = http_client(demo.trips("synthetic_city").model_dump())
    assert client.trips("synthetic_city").trips[0].trip_id == "00000083"
    client.session.request.assert_called_once_with("GET", "http://backend:8000/api/v1/trips", timeout=3, params={"partition": "synthetic_city"})
    assert client.last_request.error is None


@pytest.mark.parametrize("error,text", [(requests.Timeout(), "timed out"), (requests.ConnectionError(), "Cannot connect"),
    (requests.RequestException(), "request failed")])
def test_http_transport_failures(error, text):
    client, _ = http_client(error=error)
    with pytest.raises(ClientError, match=text):
        client.health()
    assert client.last_request.status == "Failed"
    assert client.last_request.latency_ms is not None


def test_http_invalid_json():
    client, response = http_client()
    response.json.side_effect = ValueError("invalid JSON")
    with pytest.raises(ClientError, match="invalid JSON"):
        client.health()


def test_http_invalid_schema():
    client, _ = http_client({"status": "ok", "version": "x", "unexpected": True})
    with pytest.raises(ClientError, match="contract"):
        client.health()


def test_http_status_error():
    client, response = http_client(status=503)
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    with pytest.raises(ClientError, match="503"):
        client.health()


def test_http_post_request_and_unavailable_mode(demo, request_data):
    caps = demo.capabilities().model_dump()
    job = demo.predict(request_data).model_dump()
    client, response = http_client()
    response.json.side_effect = [caps, job]
    assert client.predict(request_data).status == "completed"
    assert client.session.request.call_args.kwargs["json"] == request_data.model_dump()
    client, response = http_client(caps)
    with pytest.raises(ClientError, match="unavailable"):
        client.predict(request_data.model_copy(update={"mode": "hmm_viterbi"}))
    assert client.session.request.call_count == 1


def test_job_and_result_identity(demo, request_data):
    result = demo.predict(request_data).result
    with pytest.raises(ClientError, match="different trip"):
        validate_result(result.model_copy(update={"trip_id": "00000026"}), request_data)
    with pytest.raises(ValidationError):
        PredictionJob(result_id="wrong", status="completed", result=result)
    data = result.model_dump()
    data["empty_observation_ids"].append(data["selections"][0]["observation_id"])
    with pytest.raises(ValidationError):
        PredictionResult.model_validate(data)


def test_invalid_geometry_and_capabilities(demo):
    geometry = demo.geometry("synthetic_city", "00000083").model_dump()
    geometry["observations"][0]["coordinate"] = (77, 140)
    with pytest.raises(ValidationError):
        Geometry.model_validate(geometry)
    caps = demo.capabilities().model_dump()
    caps["synthetic"] = False
    with pytest.raises(ValidationError):
        Capabilities.model_validate(caps)


def test_apptest_all_pages_and_prediction(monkeypatch):
    monkeypatch.delenv("PATHSNAP_API_URL", raising=False)
    app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "PathSnap"
    assert "95.049%" in [m.value for m in app.metric]
    for page in ["Trip Explorer", "Model Analytics", "System Status", "Route Prediction"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
    next(b for b in app.button if b.label == "Run Prediction").click().run()
    assert not app.exception and not app.error
    assert app.session_state["prediction_result"].trip_id == "00000083"
    assert app.session_state["prediction_result"].decoded_route is None
    app.selectbox[2].set_value("demo_route").run()
    next(b for b in app.button if b.label == "Run Prediction").click().run()
    assert not app.exception and not app.error
    assert app.session_state["prediction_result"].decoded_route.provenance == "synthetic_demo"


def test_apptest_live_missing_url(monkeypatch):
    monkeypatch.delenv("PATHSNAP_API_URL", raising=False)
    app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
    app.segmented_control[0].set_value("Live").run()
    assert not app.exception
    assert any("PATHSNAP_API_URL" in w.value for w in app.warning)
    app.sidebar.radio[0].set_value("Route Prediction").run()
    assert not app.exception
    assert any("Connect a backend" in info.value for info in app.info)


def test_apptest_async_job_and_source_reset(monkeypatch):
    original = MockPathSnapClient.predict
    def queued(client, request):
        completed = original(client, request)
        return PredictionJob(result_id=completed.result_id, status="queued")
    monkeypatch.setattr(MockPathSnapClient, "predict", queued)
    app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Route Prediction").run()
    next(b for b in app.button if b.label == "Run Prediction").click().run()
    assert app.session_state["pending_job"]
    next(b for b in app.button if b.label == "Check prediction progress").click().run()
    assert not app.exception and not app.error
    assert app.session_state["prediction_result"].trip_id == "00000083"
    app.segmented_control[0].set_value("Live").run()
    assert "prediction_result" not in app.session_state
    assert "pending_job" not in app.session_state
