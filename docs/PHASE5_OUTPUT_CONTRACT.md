# PathSnap Phase 5 output contract

The inference CLI writes `<output-dir>/<partition>/<trip-id>.pathsnap.json` and `<trip-id>.pathsnap.geojson`. The latter is the `route_geojson` object extracted from the first file. Paths with these suffixes and the default `data/decoding/` directory are Git-ignored. The JSON values are native numbers, strings, arrays, objects and nulls. Road IDs and observation IDs are zero-based source row indices and are scoped to the trip. Segment IDs are zero-based in route order.

## Result fields

| Field | Meaning |
|---|---|
| `schema_version` | `"1.0"` |
| `partition`, `trip_id` | Development partition and original trip ID, with leading zeroes retained |
| `model_id` | `phase3/gradient_boosting` by default, or `phase4/gradient_boosting` with explicit selection |
| `observation_count` | Number of original track rows |
| `status` | `empty` for zero segments, `disconnected` for two or more, `partial` for one segment with unmatched rows, and `connected` for one segment covering every row |
| `decoder_config` | Actual decoder parameters used |
| `matches` | One observation record per source row, in original order |
| `route_segments` | Ordered segment records and directed arc occurrences |
| `route_geojson` | A GeoJSON FeatureCollection with one LineString per segment |
| `break_counts` | Frequencies of non-null reasons in the observation records |
| `routing` | Directed searches, cache hits and exhausted search budgets |
| `processing_time_s` | Inference wall time from spatial index creation through reconstruction, in seconds |

Each match carries `partition`, `trip_id`, `observation_id`, `arc_id`, `candidate_rank` (the distance rank), `model_score` (the original positive-class probability), `segment_id`, `classifier_arc_id` (raw classifier Top-1), `candidate_count`, `status` (`matched` or `unmatched`) and `break_reason`. Unmatched rows have null selected arc, rank, score and segment ID. An observation with an unusable timestamp can have an isolated match. Empty outputs retain the match records and contain an empty FeatureCollection.

A match at a segment start may carry the reason the preceding chain ended. An unmatched row can carry its own reason, so `break_counts` counts annotated records, including both the empty row and a subsequent restart where applicable. Use `route_segments` to count route segments. Reasons include `no_candidates`, `no_usable_states`, `invalid_gps`, `invalid_timestamp`, `invalid_timestamp_pair`, `nonpositive_time_step`, `gps_gap`, `invalid_gps_pair`, `speed_above_limit`, `movement_limit`, `no_feasible_transition`, `routing_budget_exceeded` and `invalid_transition_score`.

Each segment carries `segment_id`, `break_reason`, `first_observation_id`, `last_observation_id`, the ordered `arc_ids`, `occurrences`, `matched_distance_m` and `full_arc_length_m`. Each occurrence has an `arc_id` and a role of `selected` or `connector`. Selected occurrences include the first and last observation IDs assigned to that occurrence. Consecutive forward fixes on one directed arc extend its existing occurrence. Returning to an arc through a loop creates a new occurrence. Full arc lengths include the portions of the first and last roads outside the snapped observations. The matched distance sums only projected, connected movement and is zero for a one-fix segment.

GeoJSON feature properties carry `segment_id`, `break_reason`, `partition`, `trip_id`, `model_id` and `arc_count`. A LineString contains directed road endpoints in longitude, latitude order with shared junction coordinates emitted once. Render each feature independently in Folium. The geometry uses full arc endpoints and carries no original GPS coordinates. Avoid concatenating segment coordinates into a single line.

## Backend use

Load a trusted model once and reuse it across trip requests. Pass an inference trip loaded through the Phase 2A loader. Serialize `run.result` as the response body. For map-only requests, serve `run.result["route_geojson"]`.

```python
from src.candidates.pipeline import load_inference_trip
from src.decoding import decode_trip, model_bundle
from src.preprocessing.loaders import TripFiles

bundle = model_bundle()

def decode(files: TripFiles) -> dict:
    trip = load_inference_trip(files)
    return decode_trip(trip, bundle).result
```

Treat exact trip predictions and route geometry as local outputs. The offline reports provide aggregate metrics for sharing.
