"""Exercise final mode using invented catalog entries and references only."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from backend import final_evaluation, main, service
from frontend.models.schemas import FINAL_PARTITION, PredictionJob
from frontend.services.api_client import HttpPathSnapClient
from src.alignment.core import AlignmentPolicy, align_trip
from src.candidates.pipeline import load_inference_trip
from src.decoding import DecoderConfig, decode_trip
from src.preprocessing.loaders import Trip, TripFiles


TRIPS = tuple(f"{number:08d}" for number in range(901, 911))
TOKEN = "syntheticTestTokenOnlyDoNotUseOnLan"
BASE = "http://192.168.50.11:8011"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FILTER = {"partition": FINAL_PARTITION}
ROOT = Path(__file__).resolve().parents[2]


class FixedClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.tile([.05, .95], (len(features), 1))


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    root = tmp_path / "invented-trips"
    for number, trip_id in enumerate(TRIPS):
        folder = root / trip_id
        folder.mkdir(parents=True)
        files = TripFiles(FINAL_PARTITION, trip_id, folder)
        files.path("track").write_text(f"0.0001 1 1\n0.0002 1 2\n0.0012 1 {15 + number}\n")
        files.path("nodes").write_text("0 1\n0.001 1\n0.002 1\n")
        files.path("arcs").write_text("0 1\n1 2\n")
        files.path("route").write_text("0\n1\n")
    monkeypatch.setenv("PATHSNAP_FINAL_EVALUATION", "1")
    monkeypatch.setenv("PATHSNAP_FINAL_ROOT", str(root))
    monkeypatch.setenv("PATHSNAP_FINAL_TOKEN", TOKEN)
    monkeypatch.delenv("PATHSNAP_FINAL_SCORING", raising=False)
    return root


@pytest.fixture
def application(tmp_path, catalog, monkeypatch):
    dev = tmp_path / "dev"
    (dev / "train").mkdir(parents=True)
    models = {phase: tmp_path / f"{phase}.joblib" for phase in ("phase3", "phase4")}
    models["phase3"].touch()
    policy = AlignmentPolicy()
    monkeypatch.setattr(main, "verify_freeze", lambda *_: (DecoderConfig(), policy))
    monkeypatch.setattr(service, "model_bundle", lambda *_: {
        "pipeline": FixedClassifier(), "model_id": "gradient_boosting",
        "manifest": {"policy_id": policy.policy_id},
    })
    return main.create_app(data_root=dev, model_paths=models)


def client_for(application, *, peer="192.168.50.12", base=BASE):
    return TestClient(application, base_url=base, client=(peer, 50000))


def predict(client, trip_id=TRIPS[0], mode="hmm_viterbi", headers=AUTH):
    return client.post("/api/v1/predictions", headers=headers,
                       json={**FILTER, "trip_id": trip_id, "mode": mode})


def assert_hidden(client, headers):
    assert FINAL_PARTITION not in client.get("/api/v1/capabilities", headers=headers).json()["partitions"]
    for suffix in ("", f"/{TRIPS[0]}", f"/{TRIPS[0]}/geometry"):
        response = client.get("/api/v1/trips" + suffix, params=FILTER, headers=headers)
        assert (response.status_code, response.json()["code"]) == (422, "invalid_partition")
    response = predict(client, headers=headers)
    assert (response.status_code, response.json()["code"]) == (422, "invalid_partition")


def test_disabled_does_not_open_catalog(application, monkeypatch):
    monkeypatch.delenv("PATHSNAP_FINAL_EVALUATION")
    monkeypatch.setenv("PATHSNAP_FINAL_ROOT", "/nonexistent/unused-in-disabled-mode")
    must_not_call = Mock(side_effect=AssertionError("Final evaluation setup ran"))
    monkeypatch.setattr(main, "verify_freeze", must_not_call)
    with client_for(application) as client:
        assert_hidden(client, AUTH)
        assert client.get("/api/v1/trips", params={"partition": "train"}).json() == {"trips": []}
    must_not_call.assert_not_called()


def test_private_authenticated_access_only(application):
    with client_for(application) as client:
        assert_hidden(client, {})
        assert_hidden(client, {"Authorization": "Bearer wrong"})
        assert_hidden(client, {**AUTH, "X-Forwarded-Host": "public.example"})
        assert FINAL_PARTITION in client.get("/api/v1/capabilities", headers=AUTH).json()["partitions"]
    with client_for(application, base="https://public.example") as client:
        assert_hidden(client, AUTH)
    with client_for(application, peer="8.8.8.8") as client:
        assert_hidden(client, AUTH)


def test_all_ten_choices_and_real_decoder_without_reading_references(application, monkeypatch, caplog):
    original = TripFiles.path

    def only_inference(files, extension):
        assert extension in ("track", "nodes", "arcs")
        return original(files, extension)

    monkeypatch.setattr(TripFiles, "path", only_inference)
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with client_for(application) as client:
        listing = client.get("/api/v1/trips", params=FILTER, headers=AUTH)
        assert listing.headers["cache-control"] == "no-store"
        assert [row["trip_id"] for row in listing.json()["trips"]] == list(TRIPS)
        for trip_id in TRIPS:
            response = predict(client, trip_id)
            assert response.status_code == 200, response.text
            assert response.headers["cache-control"] == "no-store"
            job = PredictionJob.model_validate(response.json())
            result = job.result
            assert response.headers["x-request-id"] == job.result_id
            assert result.trip_id == trip_id and result.phase5.route_segments[0].arc_ids == [0, 1]
            assert result.runtime_s > 0 and result.continuity_metrics["predicted_internal_disconnections"] == 0
            assert result.evaluation_metrics == {}
            assert f"trip_id={trip_id} request_id={job.result_id}" in caplog.text
            assert client.get(f"/api/v1/results/{job.result_id}").status_code == 404
    assert TOKEN not in caplog.text


def test_reference_only_after_decode_and_only_aggregates_returned(application, catalog, monkeypatch):
    monkeypatch.setenv("PATHSNAP_FINAL_SCORING", final_evaluation.DOCUMENTED_SCORING)
    events = []
    decode = final_evaluation._decode_trip
    original_path = TripFiles.path

    def decode_then_record(trip, *args, **kwargs):
        assert not hasattr(trip, "route")
        events.append("inference_started")
        result = decode(trip, *args, **kwargs)
        events.append("inference_completed")
        return result

    def route_after_inference(files, extension):
        if extension == "route":
            assert events[-1] == "inference_completed"
            events.append("reference_accessed")
        return original_path(files, extension)

    monkeypatch.setattr(final_evaluation, "_decode_trip", decode_then_record)
    monkeypatch.setattr(TripFiles, "path", route_after_inference)
    with client_for(application) as client:
        geometry = client.get(f"/api/v1/trips/{TRIPS[0]}/geometry", params=FILTER, headers=AUTH)
        assert geometry.status_code == 200
        assert set(geometry.json()) == {"partition", "trip_id", "synthetic", "observations", "arcs"}
        response = predict(client)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert events == ["inference_started", "inference_completed", "reference_accessed"]
        assert result["evaluation_metrics"] == {
            "qualified_proxy_observations_count": 3, "eligible_proxy_observations_count": 3,
            "qualified_candidate_misses_count": 0, "classifier_proxy_top1_fraction": 1.0,
            "hmm_proxy_top1_fraction": 1.0, "hmm_directed_arc_precision_fraction": 1.0,
            "hmm_directed_arc_recall_fraction": 1.0, "hmm_directed_arc_pair_f1_fraction": 1.0,
        }
        assert "reference" not in response.text.lower()
        assert {m["arc_id"] for m in result["phase5"]["matches"]} == {0, 1}

        files = TripFiles(FINAL_PARTITION, TRIPS[0], catalog / TRIPS[0])
        with pytest.raises(ValueError, match="Only train and student_test"):
            decode_trip(load_inference_trip(files), None)
        with pytest.raises(ValueError, match="only training and student-testing"):
            inference = load_inference_trip(files)
            align_trip(Trip(files, inference.track, inference.nodes, inference.arcs, np.array([0, 1]), []),
                       AlignmentPolicy())


def test_changing_only_reference_changes_metrics_not_inference(application, catalog, monkeypatch):
    monkeypatch.setenv("PATHSNAP_FINAL_SCORING", final_evaluation.DOCUMENTED_SCORING)
    path = catalog / TRIPS[0] / f"{TRIPS[0]}.route"
    with client_for(application) as client:
        baseline = predict(client).json()["result"]
        path.write_text("1\n")
        changed = predict(client).json()["result"]
    assert changed["phase5"]["matches"] == baseline["phase5"]["matches"]
    assert changed["phase5"]["route_geojson"] == baseline["phase5"]["route_geojson"]
    assert changed["continuity_metrics"] == baseline["continuity_metrics"]
    assert changed["evaluation_metrics"]["hmm_directed_arc_precision_fraction"] == .5
    assert changed["evaluation_metrics"]["hmm_directed_arc_recall_fraction"] == 1
    assert changed["evaluation_metrics"]["hmm_directed_arc_pair_f1_fraction"] == 0


@pytest.mark.parametrize("reference", ["missing", "invalid", "symlink"])
def test_missing_or_unusable_reference_keeps_prediction(application, catalog, monkeypatch, reference):
    monkeypatch.setenv("PATHSNAP_FINAL_SCORING", final_evaluation.DOCUMENTED_SCORING)
    path = catalog / TRIPS[0] / f"{TRIPS[0]}.route"
    path.unlink()
    if reference == "invalid":
        path.write_text("9999\n")
    if reference == "symlink":
        path.symlink_to(catalog / TRIPS[1] / f"{TRIPS[1]}.route")
    with client_for(application) as client:
        response = predict(client)
        assert response.status_code == 200
        assert response.json()["result"]["evaluation_metrics"] == {}


def test_bad_requests_and_missing_inference_files(application, catalog):
    with client_for(application) as client:
        assert predict(client, "../outside").status_code == 422
        assert predict(client, "00009999").status_code == 404
        assert predict(client, mode="phase3_fixed").status_code == 409
        (catalog / TRIPS[0] / f"{TRIPS[0]}.nodes").unlink()
        assert predict(client).status_code == 404


def test_incomplete_catalog_and_wrong_model_fail_closed(application, catalog, tmp_path):
    (catalog / TRIPS[0] / f"{TRIPS[0]}.track").unlink()
    with pytest.raises(ValueError, match="incomplete or linked"):
        with client_for(application):
            pass
    model = tmp_path / "incorrect.joblib"
    model.write_bytes(b"not the frozen model")
    with pytest.raises(ValueError, match="exact frozen"):
        final_evaluation.verify_freeze(ROOT, model)


def test_streamlit_selects_and_renders_real_api_result(application, monkeypatch):
    monkeypatch.setenv("PATHSNAP_FINAL_SCORING", final_evaluation.DOCUMENTED_SCORING)
    monkeypatch.setenv("PATHSNAP_API_URL", BASE)
    posts = []

    class RequestsTransport:
        def request(self, method, url, *, timeout, allow_redirects, **kwargs):
            assert timeout > 0 and not allow_redirects
            return transport.request(method, url, follow_redirects=allow_redirects, **kwargs)

    class AsgiClient(HttpPathSnapClient):
        def __init__(self, url, timeout, *, final_token):
            super().__init__(url, timeout, session=RequestsTransport(), final_token=final_token)

        def _request(self, method, path, schema, **kwargs):
            if method == "POST":
                posts.append(kwargs["json"]["trip_id"])
            return super()._request(method, path, schema, **kwargs)

    monkeypatch.setattr("frontend.services.api_client.HttpPathSnapClient", AsgiClient)
    with client_for(application) as transport:
        app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Route Prediction").run()
        assert FINAL_PARTITION not in app.selectbox[0].options
        app.segmented_control[0].set_value("Live").run()
        app.selectbox(key="prediction-partition").set_value(FINAL_PARTITION).run()
        selector = app.selectbox(key="prediction-final-trip")
        assert selector.options == list(TRIPS) and selector.value is None and posts == []
        selector.set_value(TRIPS[-1]).run()
        assert not app.exception and not app.error
        result = app.session_state["prediction_result"]
        assert result.trip_id == TRIPS[-1] and result.phase5.route_geojson.features
        assert result.evaluation_metrics["hmm_proxy_top1_fraction"] == 1
        assert any("Permitted evaluation" in item.value for item in app.markdown)
        assert any("hmm_proxy_top1_fraction" in item.value for item in app.json)
        assert any("Route status" in item.value for item in app.markdown)
        app.run()
        assert posts == [TRIPS[-1]]
        app.selectbox(key="prediction-final-trip").set_value(TRIPS[0]).run()
        assert posts == [TRIPS[-1], TRIPS[0]]


def test_frozen_classifier_runs_through_api(catalog, tmp_path):
    path = os.getenv("PATHSNAP_TEST_PHASE3_MODEL")
    if not path:
        pytest.skip("Set PATHSNAP_TEST_PHASE3_MODEL to the trusted frozen model for this check")
    app = main.create_app(data_root=tmp_path / "dev", model_paths={
        "phase3": Path(path), "phase4": tmp_path / "unused.joblib",
    })
    with client_for(app) as client:
        assert client.get("/health", headers=AUTH).json()["status"] == "ok"
        for trip_id in TRIPS:
            response = predict(client, trip_id)
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            assert result["trip_id"] == trip_id
            assert result["model_id"] == "phase3/gradient_boosting"
            assert result["phase5"]["route_geojson"]["features"]
            assert result["phase5"]["decoder_config"] == json.loads(
                (ROOT / "reports/phase5/decoder_freeze.json").read_text()
            )["decoder_config"]
