"""Opt-in smoke against a running Uvicorn instance and actual training data."""

import os

import folium
import pytest

from frontend.models.schemas import PredictionRequest
from frontend.services.api_client import HttpPathSnapClient


def test_frontend_live_http():
    url = os.getenv("PATHSNAP_SMOKE_URL")
    uds = os.getenv("PATHSNAP_SMOKE_UDS")
    if not (url or uds):
        pytest.skip("Set PATHSNAP_SMOKE_URL or PATHSNAP_SMOKE_UDS to run the live smoke")

    session = None
    if uds:
        import httpx

        session = httpx.Client(transport=httpx.HTTPTransport(uds=uds))

    client = HttpPathSnapClient(url or "http://localhost", session=session)
    try:
        assert_live_frontend(client)
    finally:
        client.session.close()


def assert_live_frontend(client):
    from frontend.components.map_view import build_map

    assert client.health().status == "ok"
    modes = client.capabilities()
    assert not modes.synthetic and "train" in modes.partitions
    assert any(mode.id == "hmm_viterbi" and mode.available for mode in modes.modes)
    trip_id = "00000026"
    summary = client.trip("train", trip_id)
    assert summary in client.trips("train").trips
    geometry = client.geometry("train", trip_id)
    assert geometry.observations and geometry.arcs and not geometry.synthetic
    completed = client.predict(PredictionRequest(partition="train", trip_id=trip_id, mode="hmm_viterbi"))
    assert completed.status == "completed" and completed.result.phase5
    result = completed.result.phase5
    assert result.model_id == "phase3/gradient_boosting"
    assert result.observation_count == summary.observation_count
    assert len(result.route_segments) > 1
    view = build_map(geometry, completed.result, layers=["Decoded route"])
    route = next(group for group in view._children.values()
                 if isinstance(group, folium.FeatureGroup) and group.layer_name.startswith("Backend HMM"))
    lines = [layer for layer in route._children.values() if isinstance(layer, folium.PolyLine)]
    assert len(lines) == len(result.route_segments)
    for line, feature in zip(lines, result.route_geojson.features, strict=True):
        assert line.locations == [[lat, lon] for lon, lat in feature.geometry.coordinates]
    assert "Segment 0" in view.get_root().render()
    print(f"{trip_id}: {result.observation_count} observations, {len(geometry.arcs)} directed roads, "
          f"{len(lines)} Folium route lines, {result.processing_time_s:.3f}s backend inference")
