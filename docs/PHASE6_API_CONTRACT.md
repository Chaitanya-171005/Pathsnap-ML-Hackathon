# PathSnap proposed frontend API · v1

## Phase 5 schema 1.0 integration update

The implemented decoder's authoritative body is now documented in [PHASE5_OUTPUT_CONTRACT.md](PHASE5_OUTPUT_CONTRACT.md), retrieved from remote Main at `fcb64d6`. Its response fields supersede the earlier proposed single `decoded_route` for Phase 5. No backend code was merged into the UI branch. `frontend/models/phase5.py` validates the actual decoder contract, and `frontend/services/prediction_adapter.py` adapts it for display without importing or running any decoder logic.

The existing proposed endpoint paths below remain unchanged until FastAPI is connected. POST `/api/v1/predictions` may return **the complete native Phase 5 JSON directly** (the body of `decode_trip(...).result`). No `result_id` or UI-specific fields need to be added to that body. The UI assigns an `inline-...` identifier for this completed synchronous response. Alternatively, the proposed asynchronous envelope may embed the native Phase 5 JSON in its `result` field:

```json
{"result_id":"backend-job-id","status":"completed","result":{"schema_version":"1.0","partition":"train","trip_id":"00000083","model_id":"phase3/gradient_boosting","observation_count":0,"status":"empty","decoder_config":{},"matches":[],"route_segments":[],"route_geojson":{"type":"FeatureCollection","features":[]},"break_counts":{},"routing":{"searches":0,"cache_hits":0,"budget_exhaustions":0},"processing_time_s":0.125}}
```

This is an illustrative schema example, not a measured result. `GET /api/v1/results/{result_id}` accepts either the native completed payload or the job envelope. Queued/running/failed responses still use the job envelope. Envelope `status` describes the **job lifecycle**; native result `status` describes **route connectivity** (`connected`, `partial`, `disconnected`, `empty`). A completed request does not imply a connected route. Result partition/trip/mode must agree with the submitted HMM request. An HMM response to a classifier-only request is rejected.

Native fields retained by the adapter:

- `matches`: every source observation in order, including unmatched rows with null selected values. These are decoded GPS-to-road matches. `classifier_arc_id` remains distinct from the decoded `arc_id`. `model_score` is the raw positive-class candidate probability of the decoded choice. When the raw classifier choice differs, its probability/rank are unavailable; the UI never borrows the decoded score for that choice.
- `route_segments`: ordered directed arc occurrences, including connectors and repeated arcs. No sorting/deduplication, path finding or stitching is performed in the frontend. Segment count comes from this list, not `break_counts`.
- `route_geojson`: one Feature/LineString per segment, in longitude/latitude order. The map draws each feature as a separate Folium polyline, retaining disconnected gaps even when nearby endpoints could be joined. Empty feature collections are valid.
- `status`: shown explicitly alongside matched/unmatched counts and segment count. The schema validates its documented relationship to segment and match counts; it does not calculate route continuity or evaluate accuracy.
- `processing_time_s`: mapped directly to the displayed backend runtime; never replaced with HTTP latency.
- `decoder_config`, `break_counts`, `routing`: retained and displayed as supplied diagnostics. Break counts count annotated records and can include both an unmatched row and its restart. They are not evaluation metrics.

Validation also checks schema version, identity, observation order/count, selected/null match fields, segment/feature association, directed occurrence order, finite coordinates and runtime. GeoJSON must use LineString per feature; unknown geometry types are rejected instead of flattened. No route-level evaluation is inferred from route status.

Internally the presentation envelope stores the validated decoder body under `PredictionResult.phase5`. This is a UI representation; the backend need not generate it. The existing `selections` / `decoded_route` fields remain compatible with classifier responses and the synthetic Demo mode. Downloads offer the original decoder field structure as `.pathsnap.json` and its map-ready `.pathsnap.geojson`. HMM mode is still enabled only when capabilities advertise it. `phase3/gradient_boosting` remains the decoder's default classifier; Phase 4 selection is explicit and experimental. Existing demo examples stay synthetic and require no backend.

The older illustrative decoded-route format below remains supported for prior clients and the synthetic example; new Phase 5 integration should send schema 1.0 outputs above.

This document proposes the FastAPI integration contract. **These endpoints are not an assertion about an existing backend.** No backend implementation is included. Authoritative typed schemas: `frontend/models/schemas.py`. Clients: `frontend/services/api_client.py` and `mock_client.py`.

The Streamlit Python process sends HTTP requests from Laptop 2 to Laptop 1. Model inference, candidate generation, decoder logic, evaluation permissions and reference data stay on Laptop 1. The browser never receives reference routes or labels. Frontend analytics only read allowlisted aggregate reports.

## Transport, identity and errors

- UTF-8 `application/json`; v1 resources live under `/api/v1`. `/health` is unversioned.
- Partition names are strings advertised by capabilities. Trip IDs **must be strings**, preserving zero padding, matching `[A-Za-z0-9_-]{1,80}`. Numeric trip IDs are rejected.
- All coordinate pairs are `[longitude, latitude]` in WGS84 geographic degrees. GeoJSON is not required in v1; if added, it must retain this order. Relative timestamps are seconds from a trip-local origin, not Unix dates.
- Arc IDs identify directed arcs, local to a trip. Geometry vertices run from source to target. Reverse arcs have distinct IDs. Observation arrays retain backend observation order; IDs need not be consecutive.
- Finite numeric values only. Unknown fields are rejected in v1. New optional fields require coordinated schema/client updates; incompatible changes require a new major API path.
- Default client timeout is 15 seconds, configurable up to 120. Requests are synchronous HTTP; long inference uses a job resource. The UI never retries POST automatically.
- Error responses: `{"code":"trip_not_found","message":"Trip unavailable","request_id":"optional-trace-id"}` (`ErrorResponse` schema). Do not include reference data or secrets in messages. The UI reports HTTP status, transport failure, or schema failure, and records the last endpoint, status, error and latency.
- Status codes: 200 successful reads / synchronous completed prediction; 202 accepted job; 400 invalid combination; 404 absent trip/result; 409 unavailable mode or job conflict; 422 invalid request schema; 500 inference failure; 503 unavailable service. The client handles all non-2xx as failures. Failed inference may alternatively be represented as a `failed` job from the result endpoint.

## Endpoints

| Method | Endpoint | Request | Successful response |
|---|---|---|---|
| GET | `/health` | None | `Health` |
| GET | `/api/v1/capabilities` | None | `Capabilities` |
| GET | `/api/v1/trips` | `?partition=...` | `TripList` |
| GET | `/api/v1/trips/{trip_id}` | `?partition=...` | `TripSummary` |
| GET | `/api/v1/trips/{trip_id}/geometry` | `?partition=...` | `Geometry` |
| POST | `/api/v1/predictions` | `PredictionRequest` JSON | `PredictionJob` (200 or 202) |
| GET | `/api/v1/results/{result_id}` | None | `PredictionJob` |

Health returns `{"status":"ok","version":"backend-version"}`; status can also be `degraded`.

Capabilities example:

```json
{
  "api_version": "v1",
  "partitions": ["train", "student_test"],
  "default_mode": "phase3_fixed",
  "synthetic": false,
  "modes": [
    {"id":"phase3_fixed","label":"Phase 3 Fixed Classifier","model_id":"gradient_boosting","available":true,"reason":null},
    {"id":"phase4_tuned","label":"Phase 4 Tuned Classifier (experimental)","model_id":"gradient_boosting_tuned","available":false,"reason":"Not loaded"},
    {"id":"hmm_viterbi","label":"HMM / Viterbi Route Reconstruction","model_id":"hmm_viterbi","available":false,"reason":"Phase 5 integration pending"}
  ]
}
```

Mode IDs are `phase3_fixed`, `phase4_tuned`, `hmm_viterbi`, and synthetic-only `demo_route`. IDs must be unique. The advertised default must be listed. Unavailable modes are explained and excluded from selection; if a backend default is unavailable, the UI selects the first available mode. A live backend must advertise `synthetic:false`; demo-only capabilities are rejected in Live mode. Recommended default remains the fixed Phase 3 Histogram Gradient Boosting model. The existence of Phase 4 reports does not enable a prediction mode.

`TripList` is `{"trips":[TripSummary,...]}`. Empty lists are valid. `TripSummary`:

```json
{"partition":"student_test","trip_id":"00000083","observation_count":2,"description":"Backend permitted trip"}
```

`Geometry` contains only inference-visible geometry:

```json
{
  "partition":"student_test",
  "trip_id":"00000083",
  "synthetic":false,
  "observations":[
    {"observation_id":0,"coordinate":[77.5901,12.9611],"relative_timestamp_s":0},
    {"observation_id":1,"coordinate":[77.5912,12.9610],"relative_timestamp_s":2}
  ],
  "arcs":[
    {"arc_id":17,"coordinates":[[77.59,12.961],[77.592,12.961]],"from_node":3,"to_node":4}
  ]
}
```

Coordinates above are illustrative, not organiser data. Observation IDs and arc IDs must each be unique within geometry. Observation and arc arrays may be empty. Arc lines need at least two vertices. Node IDs and timestamps may be null/omitted. Selected arcs should be included in geometry so they can be drawn. The UI warns about missing selected-arc geometry. Geometry and metadata must match the requested partition/trip; the UI rejects mismatches.

Prediction request:

```json
{"partition":"student_test","trip_id":"00000083","mode":"phase3_fixed"}
```

Queued/running job: `{"result_id":"job-123","status":"queued","error":null,"result":null}`. Allowed statuses: `queued`, `running`, `completed`, `failed`. Failed jobs carry an optional safe error message. Completed jobs must embed a result with the same result ID; non-completed jobs must not embed results. The UI retains the submitted request while polling manually, independently of current selector changes. Dismissing tracking is local only.

Completed job example (illustrative values, no measured evaluation):

```json
{
  "result_id":"job-123",
  "status":"completed",
  "error":null,
  "result":{
    "result_id":"job-123",
    "partition":"student_test",
    "trip_id":"00000083",
    "mode":"phase3_fixed",
    "model_id":"gradient_boosting",
    "observation_count":2,
    "selections":[{"observation_id":1,"arc_id":17,"candidate_rank":2,"probability":0.91}],
    "empty_observation_ids":[0],
    "decoded_route":null,
    "runtime_s":null,
    "continuity_metrics":{},
    "evaluation_metrics":{},
    "synthetic":false
  }
}
```

Selections contain one chosen candidate per nonempty observation group. `candidate_rank` is the original one-based distance rank; the returned candidate is the model's Top-1 selection. `probability` is a nullable raw positive-class binary probability in [0,1], **not a group-normalised probability**. Selected and empty observation IDs must be unique/disjoint and their counts must sum to `observation_count`. The UI does not drop empty groups or estimate missing selections. Results must match the submitted trip, partition and mode.

`runtime_s` is nullable nonnegative measured backend processing time. `continuity_metrics` and `evaluation_metrics` are maps of metric names to finite numeric values or null. They are rendered verbatim, never computed from references in the UI. Backend metric names should include units/denominators where needed (for example `route_discontinuities_count`, `proxy_top1_fraction`). Omitted metrics display **Not available**. The backend is responsible for permission checks and deciding which aggregate evaluation metrics can be exposed.

## Phase 5 integration

Load the Phase 3 fixed model and decoder on Laptop 1; advertise `hmm_viterbi.available:true` only when ready. Submit inference through the same prediction endpoint. For a genuine decoded route return:

```json
{
  "provenance":"hmm_viterbi",
  "arc_ids":[17,21],
  "coordinates":[[77.59,12.961],[77.592,12.961],[77.593,12.962]]
}
```

Put this object in `PredictionResult.decoded_route`, with `mode:"hmm_viterbi"`. Coordinates represent actual decoder output, including real backend-resolved connecting geometry. A route requires at least two vertices. If decoding fails or no route is available, return null and appropriate job/error/metric details. Never connect independent classifier selections to invent a decoded route. Synthetic examples use `provenance:"synthetic_demo"` and `synthetic:true`; they are never labelled genuine HMM output.

## Two-laptop network setup

On Laptop 1, the backend owner binds FastAPI to the LAN interface (commonly `0.0.0.0:8000`) and permits the chosen port on a trusted private network. On Laptop 2 set `PATHSNAP_API_URL=http://<Laptop-1-private-IP>:8000`. Both laptops must be mutually reachable; guest Wi-Fi isolation, firewall rules and VPN routing can prevent this. Verify `/health` from Laptop 2 before selecting Live mode. No network configuration is changed by the frontend. Do not expose an unauthenticated inference server publicly; use your explicitly configured private network or an authenticated deployment. Browser CORS is not required for the Streamlit-to-FastAPI server request path. Streamlit itself may stay bound to Laptop 2's loopback when only its local browser is used.

Authentication, trip pagination, viewport geometry, cancellation and streaming progress are not included in this proposed v1 contract. Coordinate/model payload size policies and evaluation access remain backend responsibilities. Contract tests currently use validated synthetic fixtures and mocked HTTP; end-to-end live compatibility awaits Laptop 1.
