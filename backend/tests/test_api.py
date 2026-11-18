from threading import Event

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import service
from backend.main import create_app
from frontend.models.phase5 import Phase5Result
from frontend.models.schemas import Capabilities, Geometry, PredictionJob
from src.preprocessing.loaders import TripFiles


TRIP = "00000026"
PREDICTIONS = "/api/v1/predictions"
GEOMETRY = f"/api/v1/trips/{TRIP}/geometry"


class FixedClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.tile([.1, .9], (len(features), 1))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    trip = root / "train" / TRIP
    trip.mkdir(parents=True)
    (root / "student_test").mkdir()
    contents = {"nodes": "0 1\n.001 1\n.003 1\n.004 1\n",
                "arcs": "0 1\n1 0\n2 3\n99 1\n",
                "track": ".0005 1 100\n.0035 1 140\nnan 1 141\n"}
    for extension, text in contents.items():
        (trip / f"{TRIP}.{extension}").write_text(text)
    model_paths = {phase: tmp_path / f"{phase}.joblib" for phase in ("phase3", "phase4")}
    model_paths["phase3"].touch()
    loaded = []

    def load(phase, path):
        loaded.append((phase, path))
        return {"pipeline": FixedClassifier(), "model_id": "gradient_boosting"}

    original_path = TripFiles.path

    def inference_only(files, extension):
        assert extension != "route"
        return original_path(files, extension)

    monkeypatch.setattr(service, "model_bundle", load)
    monkeypatch.setattr(TripFiles, "path", inference_only)
    return root, model_paths, loaded


def payload(mode="hmm_viterbi"):
    return {"partition": "train", "trip_id": TRIP, "mode": mode}


def test_all_endpoints_and_cached_model_and_trip(paths, monkeypatch):
    root, models, loaded = paths
    reads = []
    original = service.load_inference_trip

    def read(files):
        reads.append(files.trip_id)
        return original(files)

    monkeypatch.setattr(service, "load_inference_trip", read)
    with TestClient(create_app(data_root=root, model_paths=models)) as client:
        assert client.get("/health").json()["status"] == "ok"
        modes = Capabilities.model_validate(client.get("/api/v1/capabilities").json())
        assert not modes.synthetic and modes.default_mode == "phase3_fixed"
        assert {mode.id for mode in modes.modes if mode.available} == {"phase3_fixed", "hmm_viterbi"}
        listing = client.get("/api/v1/trips", params={"partition": "train"}).json()
        summary = client.get(f"/api/v1/trips/{TRIP}", params={"partition": "train"}).json()
        assert listing == {"trips": [summary]}
        assert (summary["trip_id"], summary["observation_count"]) == (TRIP, 3)
        geom = Geometry.model_validate(client.get(GEOMETRY, params={"partition": "train"}).json())
        assert [obs.observation_id for obs in geom.observations] == [0, 1]
        assert geom.observations[1].relative_timestamp_s == 40
        assert geom.arcs[0].coordinates == list(reversed(geom.arcs[1].coordinates))
        assert [arc.arc_id for arc in geom.arcs] == [0, 1, 2]
        body = client.post(PREDICTIONS, json=payload()).json()
        hmm = Phase5Result.model_validate(body)
        assert "result_id" not in body
        assert hmm.status == "disconnected"
        assert [segment.arc_ids for segment in hmm.route_segments] == [[0], [2]]
        assert hmm.matches[-1].arc_id is None
        response = client.post(PREDICTIONS, json=payload("phase3_fixed"))
        job = PredictionJob.model_validate(response.json())
        assert job.result.empty_observation_ids == [2]
        assert [match.arc_id for match in job.result.selections] == [0, 2]
        assert client.get(f"/api/v1/results/{job.result_id}").json() == response.json()
        assert client.get("/api/v1/results/unknown").status_code == 404
    assert loaded == [("phase3", models["phase3"])]
    assert reads == [TRIP]


def test_model_availability_and_missing_data(paths):
    root, models, _ = paths
    models["phase3"].unlink()
    models["phase4"].touch()
    with TestClient(create_app(data_root=root, model_paths=models)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        modes = client.get("/api/v1/capabilities").json()["modes"]
        assert {mode["id"] for mode in modes if mode["available"]} == {"phase4_tuned"}
        assert client.post(PREDICTIONS, json=payload()).json()["code"] == "model_unavailable"
        assert client.post(PREDICTIONS, json=payload("phase4_tuned")).json()["result"]["model_id"] == (
            "gradient_boosting_tuned")
        (root / "student_test").rmdir()
        assert client.get("/api/v1/trips", params={"partition": "student_test"}).status_code == 503
        (root / "train" / TRIP / f"{TRIP}.nodes").unlink()
        assert client.get(GEOMETRY, params={"partition": "train"}).status_code == 404


def test_evaluator_invalid_trip_ids_and_symlinks_are_rejected(paths):
    root, models, _ = paths
    (root / "train" / "symlink").symlink_to(root / "train" / TRIP, target_is_directory=True)
    with TestClient(create_app(data_root=root, model_paths=models)) as client:
        for update, status in (
            ({"trip_id": 26}, 422), ({"trip_id": "../00000026"}, 422),
            ({"partition": "evaluator_test"}, 422), ({"partition": "../train"}, 422),
            ({"trip_id": "symlink"}, 404), ({"trip_id": "00009999"}, 404),
            ({"mode": "demo_route"}, 409), ({"other": "field"}, 422),
        ):
            response = client.post(PREDICTIONS, json={**payload(), **update})
            assert response.status_code == status
            assert set(response.json()) == {"code", "message"}
        assert client.get(GEOMETRY, params={"partition": "evaluator_test"}).status_code == 422
        track = root / "train" / TRIP / f"{TRIP}.track"
        track.rename(track.with_suffix(".backup"))
        track.symlink_to(track.with_suffix(".backup"))
        assert client.get(GEOMETRY, params={"partition": "train"}).status_code == 404


def test_empty_hmm_is_a_success(paths):
    root, models, _ = paths
    (root / "train" / TRIP / f"{TRIP}.track").write_text("nan 1 0\n")
    with TestClient(create_app(data_root=root, model_paths=models)) as client:
        response = client.post(PREDICTIONS, json=payload())
        assert response.status_code == 200
        result = Phase5Result.model_validate(response.json())
        assert result.status == "empty" and len(result.matches) == 1
        assert result.route_geojson.features == []


def test_prediction_timeout_leaves_slot_busy_until_worker_finishes(paths, monkeypatch):
    root, models, _ = paths
    entered, release = Event(), Event()

    def slow(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return {}

    monkeypatch.setattr(service, "decode_trip", slow)
    with TestClient(create_app(data_root=root, model_paths=models, timeout_s=.3)) as client:
        try:
            response = client.post(PREDICTIONS, json=payload())
            assert entered.is_set()
            assert (response.status_code, response.json()["code"]) == (504, "prediction_timeout")
            assert client.post(PREDICTIONS, json=payload()).json()["code"] == "prediction_busy"
            assert client.get("/health").status_code == 200
        finally:
            release.set()


def test_predictions_return_safe_server_errors(paths, monkeypatch):
    root, models, _ = paths

    def crash(*args, **kwargs):
        raise RuntimeError("/sensitive/path")

    monkeypatch.setattr(service, "decode_trip", crash)
    with TestClient(create_app(data_root=root, model_paths=models),
                    raise_server_exceptions=False) as client:
        response = client.post(PREDICTIONS, json=payload())
        assert response.status_code == 500
        assert response.json()["code"] == "server_error"
        assert "/sensitive/path" not in response.text
