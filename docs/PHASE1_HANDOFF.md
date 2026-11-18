# Phase 1 handoff

## Setup and reproduction

Use Python 3.13 (verified on 3.13.9). From the repository root, place the organiser-supplied trip folders under `data/raw/train/` and `data/raw/student_test/`. Keep their filenames and leading zeros intact. The data and local organiser PDFs are excluded from Git. The [dataset schema](DATASET_SCHEMA.md) describes the four file types and array shapes.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.eda.run_eda --data-dir data/raw
python -m pytest -q
```

Recognised options are `--report-dir`, `--visual-dir`, `--gap-seconds`, `--speed-mps` and `--map-road-limit`. For example, use `--gap-seconds 5 10 30 60 --speed-mps 55 --map-road-limit 12000` to spell out the defaults. Read trip IDs as text when importing CSVs so leading zeros survive, particularly in spreadsheet tools. The default map HTML uses Leaflet assets and OpenStreetMap tiles, which require network access when opening the map. PNG plots are available offline.

## Files and outputs

| Path | Purpose |
|---|---|
| `.gitignore`, `requirements.txt`, `README.md` | Data exclusions, dependencies and how to run Phase 1 |
| `src/__init__.py`, `src/preprocessing/__init__.py`, `src/eda/__init__.py`, `src/visualization/__init__.py` | Python packages |
| `src/preprocessing/loaders.py`, `src/preprocessing/validation.py` | Row-preserving parsing, inventory discovery and validity masks |
| `src/preprocessing/geometry.py`, `src/preprocessing/graph_analysis.py`, `src/preprocessing/statistics.py` | Metre-based distances, sparse topology and per-trip measurements |
| `src/eda/run_eda.py` | EDA entry point, CSV export, map selection and Markdown summary |
| `src/visualization/plots.py`, `src/visualization/folium_maps.py` | Static figures and interactive training maps |
| `tests/test_preprocessing.py`, `tests/test_eda.py` | Synthetic loader, validation, geometry, graph and end-to-end tests |
| `docs/DATASET_SCHEMA.md`, `docs/EDA_FINDINGS.md`, `docs/DATA_CONTRACT.md`, `docs/PHASE1_HANDOFF.md` | Schemas, measured findings, future interface and this handoff |
| `reports/eda/dataset_inventory.csv` | Partition, trip ID, presence, byte size and loaded row counts |
| `reports/eda/gps_statistics.csv` | GPS bounds, quality, time, length and speed metrics |
| `reports/eda/network_statistics.csv` | Road bounds, topology, connected component sizes and arc lengths |
| `reports/eda/reference_routes.csv` | Reference integrity, directed connectivity, endpoints and GPS offsets |
| `reports/eda/data_quality_issues.csv`, `reports/eda/summary.md` | Individual parse issues when applicable, per-trip diagnostic counts and aggregate report |
| `visualizations/eda/01_observation_counts.png`, `visualizations/eda/02_timestamp_intervals.png`, `visualizations/eda/03_gps_gaps.png`, `visualizations/eda/04_trajectory_lengths.png` | GPS observation counts, sampling, gaps and trip lengths |
| `visualizations/eda/05_road_network_size.png`, `visualizations/eda/06_connectivity.png`, `visualizations/eda/07_training_trajectories.png`, `visualizations/eda/08_reference_length_vs_gps_count.png` | Graph size and connectivity, selected trajectories and route length versus observation count |
| `visualizations/eda/training_typical.html`, `visualizations/eda/training_short.html`, `visualizations/eda/training_anomaly.html` | Three training maps with selectable GPS, road and reference layers |

Location-bearing exports are Git-ignored by default. Use the checked-in generation code and locally obtained data to recreate them. The EDA pipeline writes only to output directories. Parse errors retain source row positions. Geometry and speed measurements exclude the inputs that fail their validity checks, and the issue CSV records diagnostics.

## Verification

All 22 pytest cases passed. A second full run reproduced the five CSVs, one Markdown summary, eight PNGs and three HTML maps byte for byte on the same environment. SHA-256 hashes matched for all 360 raw files before and after. The three HTML maps were checked by executing their inline scripts against a test implementation of the Leaflet interface and comparing the registered GPS points, ordered route arcs and sampled road geometries directly with the raw files. Live browser rendering and basemap tiles remain unverified.

## Carry into candidate generation

- The observed partitions contain 80 and 10 trips, with 360 files and no invalid coordinate, endpoint or reference IDs. The main problems are gaps, speed flags, reference discontinuities and GPS/reference spatial discrepancies. See [EDA findings](EDA_FINDINGS.md).
- Use arc direction and preserve arc IDs even when their endpoints are identical to another arc's endpoints. Decide how to handle zero-length arcs and duplicated geometry in candidate ranking.
- Define a physical distance-based search and a candidate-miss policy. Evaluate it by trip and report how many observations can be assigned a qualified reference label. Keep the reference route out of inference features.
- Decide how downstream components should handle long gaps, near ties and disconnected candidate sets. Verify any observation-level reference mapping or proxy with the organisers. See the [data contract](DATA_CONTRACT.md).
- GPS distance and speed are endpoint-based estimates. Road length uses node endpoints, and the reference projection ignores timing and traversal order. Avoid drawing gap connectors or silently repairing incomplete route sequences.
