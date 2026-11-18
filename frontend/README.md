# PathSnap frontend

Phase 5 decoder integration is ready: see [the actual schema 1.0 contract](../docs/PHASE5_OUTPUT_CONTRACT.md). The HTTP client accepts native decoder JSON directly or inside a completed prediction-job envelope. Matches, route status, processing runtime, directed segment occurrences and break diagnostics are displayed. GeoJSON features are drawn independently; disconnected segments are never joined. Native JSON and route GeoJSON can be downloaded. This is a presentation adapter only; no model or HMM logic runs in the frontend. Demo mode and the fixed Phase 3 default are preserved. Real HMM inference remains capability-gated until FastAPI is connected.

Streamlit + Folium + Plotly dashboard, independent of the ML environment. Python 3.12+ is supported for the frontend; the existing pipeline has its own Python/dependency requirements.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

If the environment is activated, the launch command is:

```powershell
python -m streamlit run frontend/app.py
```

On this development computer Python was not on PATH. The environment was created with the bundled runtime. The already-created `.venv\Scripts\python.exe` can be used directly with the commands above. Do not install the ML requirements just to run this UI.

Demo starts by default. All trip coordinates, directed roads and candidate selections are deterministic invented examples, even though their string IDs resemble project IDs. No trained classifiers or raw data are loaded. The fixed-classifier demo simulates the layout only; the optional **Synthetic route example** is explicitly not HMM output. Phase 3 and Phase 4 analytics read actual aggregate reports from `reports/`. Missing reports produce empty states. The report loader is restricted to named aggregate files.

## Pages

- **Overview:** report-derived counts, selected model, pipeline status, shortcuts.
- **Trip Explorer:** backend-exposed partitions, zero-padded trip IDs, metadata, GPS table, interactive layer controls, directed arc inspection, start/end markers and fit bounds.
- **Route Prediction:** capability-gated modes, submit/status/result flow, independent selected arcs, empty groups, backend metrics and JSON download.
- **Model Analytics:** grouped train/test charts for six metrics, geometric baseline, ranking, per-trip chart, coverage, actual Phase 4 comparison, confidence interval, runtime, model size and optimisation studies.
- **System Status:** connection, health latency, last service request, capabilities and backend-only evaluation metrics.

## Live backend

```powershell
$env:PATHSNAP_API_URL = "http://BACKEND_IP:8000"
$env:PATHSNAP_REQUEST_TIMEOUT = "15"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Select **Live** in the sidebar. No localhost backend is assumed. `.env.example` is a reference; environment files are not loaded automatically. Restart Streamlit after changing shell configuration. Missing URLs and network errors leave Overview/Analytics usable. System Status can refresh health/capabilities. Ordinary health/capability refresh is cached for 30 seconds; trip data is cached per browser session and source URL for 60 seconds with a 12-entry bound. Failed requests are not cached. Switching source clears pending jobs, geometry and prediction results.

`HttpPathSnapClient` and `MockPathSnapClient` implement the same `PathSnapClient` interface. The HTTP endpoints are **proposed**, not implemented in this branch. See [API contract](../docs/PHASE6_API_CONTRACT.md). Requests use a bounded timeout. Prediction POSTs are never automatically retried. Asynchronous jobs are supported with **Check prediction progress**; dismissing local tracking does not cancel backend work.

## Maps and performance

Wire/GeoJSON coordinates use `(longitude, latitude)`; Folium receives `(latitude, longitude)`. Directed arc vertices retain source-to-target order. Classifier selections are separate arc geometries and are never joined into a route. Route geometry is only displayed with explicit backend/synthetic provenance.

The full GPS polyline is retained. Only inspection markers are sampled (default 150); directed road and selected-arc rendering has a visible cap (default 1,500), configurable in the display expander. Leaflet line smoothing is display-only. Prediction requests contain trip identifiers and mode, so display settings cannot alter inference. For very large backend payloads, map generation still scales with the GPS count; viewport/paginated geometry is a future contract extension. Every observation remains inspectable in the GPS table. Selections whose arc geometry is missing are flagged and remain listed in results.

Synthetic generation requires no network. Folium's JavaScript/CSS assets and OpenStreetMap tiles load from external providers; a fresh browser therefore needs internet to render the map component. Use **Vectors only** to hide the basemap after assets have loaded. The dashboard and data services remain usable without map tiles. OpenStreetMap requires attribution (provided by Folium) and its normal public-tile usage policy applies.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest frontend/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

Tests use Streamlit AppTest, synthetic responses and mocked HTTP transports. They never open raw datasets, model binaries or reference-route files. See [handoff](../docs/PHASE6_UI_HANDOFF.md) for measured verification and limitations.

The root `.streamlit/config.toml` defines the dashboard theme; launch from the repository root to apply it. CSS uses public widget test identifiers and own markup, without browser scripts or DOM mutation. Sidebar navigation collapses on narrow viewports, and metric columns stack.
