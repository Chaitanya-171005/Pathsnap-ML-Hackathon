"""Equivalent synchronous transport interfaces for mock and HTTP backends."""
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
from urllib.parse import quote, urlsplit
import requests
from pydantic import ValidationError
from frontend.config import private_evaluation_host
from frontend.models.schemas import (
    Capabilities, Geometry, Health, PredictionJob, PredictionRequest, TripList, TripSummary,
)
from frontend.services.prediction_adapter import parse_prediction_response


class ClientError(RuntimeError):
    pass


@dataclass
class RequestStatus:
    endpoint: str = "No requests yet"
    status: str = "Idle"
    latency_ms: float | None = None
    error: str | None = None


class PathSnapClient(Protocol):
    last_request: RequestStatus
    def health(self) -> Health: ...
    def capabilities(self) -> Capabilities: ...
    def trips(self, partition: str) -> TripList: ...
    def trip(self, partition: str, trip_id: str) -> TripSummary: ...
    def geometry(self, partition: str, trip_id: str) -> Geometry: ...
    def predict(self, request: PredictionRequest) -> PredictionJob: ...
    def result(self, result_id: str) -> PredictionJob: ...


class HttpPathSnapClient:
    def __init__(self, base_url: str, timeout: float = 15, session=None, *, final_token=""):
        if not base_url:
            raise ClientError("Set PATHSNAP_API_URL before selecting Live mode.")
        if final_token and not private_evaluation_host(urlsplit(base_url).hostname):
            raise ClientError("Final evaluation requires a private IPv4 API URL.")
        self.final_token = final_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        if final_token and session is None:
            self.session.trust_env = False
        self.last_request = RequestStatus()

    def _request(self, method, path, schema, *, parser=None, **kwargs):
        start = perf_counter()
        self.last_request = RequestStatus(endpoint=f"{method} {path}", status="Running")
        if self.final_token:
            kwargs.update(headers={"Authorization": f"Bearer {self.final_token}"}, allow_redirects=False)
        try:
            response = self.session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            parsed = parser(response.json()) if parser else schema.model_validate(response.json())
            self.last_request.status = f"HTTP {response.status_code} · Validated"
            return parsed
        except requests.Timeout as exc:
            message = f"Backend timed out after {self.timeout:g}s. Check System Status and retry."
            raise ClientError(message) from exc
        except requests.ConnectionError as exc:
            message = "Cannot connect to backend. Check the API URL, LAN connection and backend process."
            raise ClientError(message) from exc
        except requests.HTTPError as exc:
            message = f"Backend returned HTTP {exc.response.status_code}."
            raise ClientError(message) from exc
        except (ValueError, ValidationError) as exc:
            message = "Backend returned invalid JSON or a response incompatible with the proposed v1 contract."
            raise ClientError(message) from exc
        except requests.RequestException as exc:
            message = "Backend request failed. Check the configured API URL and transport."
            raise ClientError(message) from exc
        finally:
            self.last_request.latency_ms = (perf_counter() - start) * 1000
            if self.last_request.status == "Running":
                self.last_request.status = "Failed"
                self.last_request.error = locals().get("message", "Request failed")

    def health(self):
        return self._request("GET", "/health", Health)

    def capabilities(self):
        return self._request("GET", "/api/v1/capabilities", Capabilities)

    def trips(self, partition):
        return self._request("GET", "/api/v1/trips", TripList, params={"partition": partition})

    def trip(self, partition, trip_id):
        return self._request("GET", f"/api/v1/trips/{quote(trip_id, safe='')}", TripSummary, params={"partition": partition})

    def geometry(self, partition, trip_id):
        return self._request("GET", f"/api/v1/trips/{quote(trip_id, safe='')}/geometry", Geometry, params={"partition": partition})

    def predict(self, request):
        supported = self.capabilities()
        if request.partition not in supported.partitions or not any(m.id == request.mode and m.available for m in supported.modes):
            raise ClientError("Prediction partition or mode is unavailable on this backend.")
        return self._request("POST", "/api/v1/predictions", PredictionJob,
            parser=parse_prediction_response, json=request.model_dump())

    def result(self, result_id):
        return self._request("GET", f"/api/v1/results/{quote(result_id, safe='')}", PredictionJob,
            parser=lambda payload: parse_prediction_response(payload, result_id))
