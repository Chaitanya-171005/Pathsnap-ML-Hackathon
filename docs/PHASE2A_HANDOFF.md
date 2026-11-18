# PathSnap Phase 2A handoff

Phase 2A adds directed road candidates and inference-time features on top of Phase 1 commit `28d3ffb`. Both student partitions use the same candidate pipeline. The inference loader opens `.track`, `.nodes` and `.arcs`. The investigation of trip `00000054` reads the reference route through a separate diagnostic entry point.

## Run the pipeline

From the repository root, use Python 3.13 and the existing requirements. Place the original trip directories under `data/raw/train/` and `data/raw/student_test/`. See the [Phase 1 handoff](PHASE1_HANDOFF.md) for installation and layout.

```bash
source .venv/bin/activate
python -m src.candidates --data-dir data/raw --partition train --trip-id 00000083 --radius-m 50 --top-k 8
python -m src.candidates --data-dir data/raw --partition student_test --trip-id 00000052
python -m src.candidates.benchmark --data-dir data/raw --radii 25 50 100 --top-k 3 8 16
python -m src.candidates.anomaly --data-dir data/raw > reports/phase2a/trip_00000054.json
python -m pytest -q
```

The export command writes `observations.csv` and `candidates.csv` under `data/candidates/<partition>/<trip_id>/`, which is Git-ignored. Select different export roots using `--output-dir` when comparing settings for the same trip. To change motion screening, set `--max-gap-s`, `--max-speed-mps` or `--min-heading-step-m`. Defaults are 30 s, 55 m/s and 2 m. The 50 m radius and Top-8 are operational defaults for reproducibility.

## Algorithms

The spatial index projects valid nodes and GPS points into the same trip-local equirectangular plane in metres using Phase 1's spherical Earth radius. It uses the first valid node longitude for local wrapping and the mean valid node latitude to scale east-west distances. It indexes segment midpoints with SciPy `cKDTree`, grouped by powers-of-two half-length ranges. Each tree is queried out to the requested radius plus its group's maximum half-length. By the triangle inequality, this includes any segment close enough to the observation, including long arcs whose midpoints are far away. Exact clamped point-to-segment distance in the projected plane then filters the shortlist to the inclusive radius. The nearest Top-K within the radius are retained by ascending distance and then ascending original arc ID on equal computed distances.

Each duplicate or reverse arc retains its own ID. A shared endpoint order in the distance calculation gives duplicate and reverse geometry the same distance. The fraction and heading preserve the original arc direction. A zero-length arc is indexed as a point. It has a valid distance, zero length and unavailable fraction and heading difference. Invalid endpoint IDs or coordinates exclude an arc from the spatial index. Topology counts the incident endpoint-valid directed rows, including duplicate edges, loops and edges with missing geometry.

Motion uses the immediately preceding source row. Missing coordinates in either row make motion unavailable. Speed uses Phase 1's haversine distance divided by the time difference. GPS heading uses the initial spherical bearing and is screened when movement is short. Arc heading uses the projected endpoint vector. Heading differences wrap around north and are reported in `[0, 180]` degrees. A positive timestamp delta larger than the gap threshold is retained as a value and flagged as unreliable for motion. Speeds above the screening limit are unavailable, with the reason retained on the observation record.

## ML interface

Use `load_inference_trip` and `candidate_batches` from `src.candidates.pipeline`. Each batch contains an `ObservationRecord` and a tuple of zero or more `CandidateRecord` values. A `RoadIndex` for the same trip can be supplied to reuse the index across settings. Select predictor columns explicitly. Identifiers support joins and grouping, and carry no target meaning.

| Candidate fields | Python type | Meaning |
|---|---|---|
| `partition`, `trip_id` | `str` | Partition and zero-padded trip directory name |
| `observation_id`, `arc_id` | `int` | Original zero-based track row and trip-local directed arc row |
| `distance_m`, `arc_length_m` | `float` | Clamped GPS-to-segment distance and endpoint length in metres |
| `distance_rank` | `int` | One-based distance rank within this observation |
| `delta_t_s`, `speed_mps` | `float \| None` | Adjacent timestamp difference in seconds and screened GPS speed in m/s |
| `heading_difference_deg` | `float \| None` | Absolute circular difference in degrees between movement bearing and arc heading |
| `projection_fraction` | `float \| None` | Clamped position from the arc's start (0) to its end (1) |
| `from_in_degree`, `from_out_degree`, `to_in_degree`, `to_out_degree` | `int` | Endpoint in-degrees and out-degrees by directed arc rows |
| `time_step_valid`, `speed_valid`, `heading_difference_valid`, `projection_valid` | `bool` | Time-step validity means finite delta in `(0, max_gap_s]`. Other flags indicate a usable feature value |

| Observation fields | Python type | Meaning |
|---|---|---|
| `partition`, `trip_id`, `observation_id` | `str`, `str`, `int` | A unique key for every original GPS row |
| `gps_valid`, `timestamp_valid` | `bool` | Valid coordinate pair and finite timestamp on the current row |
| `candidate_status` | `str` | `ok`, `invalid_gps` or `no_candidates` |
| `candidates_within_radius`, `candidate_count` | `int` | Count before Top-K and count retained |
| `delta_t_s`, `speed_mps`, `heading_deg` | `float \| None` | Time delta, screened speed and GPS bearing clockwise from north |
| `motion_status` | `str` | `ok`, `first_observation`, `invalid_timestamp_pair`, `nonpositive_time_step`, `gps_gap`, `invalid_gps_pair`, `speed_above_limit` or `short_step` |

CSV uses blanks for Python `None`, decimal numbers for floats and integers, and `True` / `False` for booleans. Read `trip_id` as text to preserve leading zeroes. Observations are emitted by ascending source row ID. Candidates are grouped in that same order and emitted by distance rank. Empty groups appear in the observation table and have no fabricated candidate row. Future labels should live in a separate table with provenance and join on the observation and candidate keys.

## Measured benchmark

The [36-run CSV](../reports/phase2a/benchmark.csv) covers 25, 50 and 100 m crossed with Top-3, Top-8 and Top-16. Each row represents one fresh process on Darwin arm64, Python 3.13.9, NumPy 2.5.5 and SciPy 1.18.1. Build time excludes disk parsing. Query time includes search, feature records and metric collection, excluding CSV export and percentile calculation. Peak RSS is the whole worker process high-water mark at the end of the query, including imports, raw parsing, index construction and metric buffers. Timing and RSS are single-run measurements.

| Training trip at 50 m / Top-8 | GPS rows | Directed arcs | Mean candidates / GPS row | Empty GPS groups | Candidate distance p50 / p95 (m) | Build / query (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `00000026` — short | 361 | 5,043 | 7.546 | 0 | 12.499 / 40.150 | 0.003 / 0.056 | 65.81 |
| `00000083` — typical | 1,507 | 25,393 | 7.324 | 0 | 17.930 / 43.998 | 0.013 / 0.237 | 73.80 |
| `00000054` — spatial anomaly | 19,313 | 116,916 | 1.008 | 16,519 | 18.900 / 42.313 | 0.065 / 1.852 | 96.31 |
| `00000013` — largest network | 2,580 | 307,392 | 5.993 | 0 | 18.388 / 45.006 | 0.171 / 0.376 | 135.55 |

- At 25, 50 and 100 m, the typical trip had 3, 0 and 0 empty groups. Trip `00000054` had 16,604, 16,519 and 1,034. At 100 m the anomaly's nearest-candidate median was 82.68 m across observations with candidates, reflecting the inclusion of much more distant roads.
- Increasing Top-K cannot fill an empty radius search. On the largest network at 50 m, Top-3, Top-8 and Top-16 yielded 2.879, 5.993 and 8.056 candidates per observation. Query time was 0.317, 0.376 and 0.408 s. At Top-8 the cap truncated 956 observations.
- A separate smoke run at 50 m / Top-8 processed all 90 trips. The 80 training trips produced 1,085,198 candidates across 208,417 GPS rows, with 18,229 empty groups. The 10 student-testing trips produced 113,691 candidates across 19,160 GPS rows, with zero empty groups. Every candidate was within radius, in rank order and linked to an existing directed arc row.
- These results measure geometric coverage and resource use. The data do not establish per-observation true-arc recall. Choose a configuration after alignment validation and assess performance by trip, including anomalous cases.

## Training trip `00000054`

The [diagnostic JSON](../reports/phase2a/trip_00000054.json) shows 422 connected directed reference arcs with 31.36 km of total arc length. The first and final GPS rows are 21.0 m and 22.1 m from their corresponding reference endpoints. The closest GPS row to the reference end is row 2,702 at 7.4 m.

After that row, 16,610 GPS fixes span 16,656 s (4 h 37 min 36 s), all within 240.8 m of the reference endpoint. The tail accumulates 126.48 km of consecutive-point distances, with a 7.06 m median step and just 7.45 m net displacement from its first to last point. All 16,156 fixes more than 100 m from the reference geometry occur in this tail. Of those fixes, 0 have any supplied road arc within 25 m, 65 have one within 50 m and 15,122 have one within 100 m.

The offsets are concentrated in a prolonged local tail, consistent with GPS drift, local movements beyond the supplied reference trip, or both. Proximity and endpoint timing cannot establish which explanation is correct or the moment when referenced travel ends. The raw trip and all of its observation IDs are retained. A later labelling policy should allow abstention and report this case separately. Forcing a label onto every tail observation could let a single trip dominate supervised training and distort pointwise metrics.

## Limitations and Phase 2B decisions

- Distances and directions use endpoint segments and a local spherical approximation. The results depend on the scale of the trip and the supplied network. Geographic comparisons near the search threshold can differ from higher-fidelity geodesic or curved-segment calculations.
- A batch can still produce a large shortlist for a very broad radius or a dense set of long segments. The Top-K cap bounds returned rows. Smaller radii reduce shortlist work but may omit the correct road; identical-distance duplicates can consume several slots.
- Decide what qualifies as observed, ambiguous and unmatched alignment, including opposite directions, repeated arcs, discontinuities and extended gaps. Record alignment provenance and group train/validation splits by trip before fitting transformations or models.
- Separate coverage of accepted alignment labels, candidate recall for those labels, and empty candidate rates. Decide whether repeated copies of the same directed geometry count as one equivalence class for evaluation. Confirm the evaluation protocol and whether an observation-level reference mapping exists.
- Tests cover geometric projection and ties, direction, invalid input, gaps, a 32,500-arc brute-force check and an inference-export test with altered references. The complete suite passed 40 tests, including all 22 Phase 1 tests. The training export and anomaly JSON reproduced byte for byte, as did the tracked Phase 1 reports and plots in a separate EDA run. SHA-256 hashes matched for all 360 raw files.

## Files

| Path | Purpose |
|---|---|
| `src/candidates/__init__.py`, `src/candidates/__main__.py` | Package and single-trip CSV export |
| `src/candidates/spatial.py`, `src/candidates/pipeline.py` | Spatial search, directed geometry, inference loading and feature records |
| `src/candidates/benchmark.py`, `reports/phase2a/benchmark.csv` | Reproducible benchmark and measured results |
| `src/candidates/anomaly.py`, `reports/phase2a/trip_00000054.json` | Separate diagnostic and measured results |
| `tests/test_candidates.py` | Phase 2A tests |
| `.gitignore`, `README.md`, `docs/PHASE2A_HANDOFF.md` | Local export exclusions, commands and implementation handoff |
