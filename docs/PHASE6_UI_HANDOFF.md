# Phase 6 UI handoff

## Phase 5 output integration follow-up

Phase 5 implementation is complete upstream. The UI now follows `docs/PHASE5_OUTPUT_CONTRACT.md`, read from refreshed remote Main at `fcb64d6` and copied as documentation only into this UI checkout. No backend code was merged or changed. The remainder of this document records the original Phase 6 delivery; statements there about waiting for decoder implementation are superseded by this update. FastAPI connectivity is still pending.

Added `frontend/models/base.py` for common wire types, `frontend/models/phase5.py` for strict schema 1.0 response validation, `frontend/services/prediction_adapter.py` for presentation conversion, and `frontend/tests/test_phase5_integration.py` for contract-shaped synthetic fixtures. Updated response schemas/HTTP client, map component, prediction page, overview, system status, demo capability explanation, frontend README and API contract.

The HTTP client accepts native completed decoder bodies or completed job envelopes containing them. Polling can also return native bodies; source result IDs are retained when present, with a local inline ID for synchronous bodies lacking one. Job lifecycle and route connectivity statuses remain separate. The UI retains every match, null unmatched values, candidate counts, independent classifier arc IDs, ordered route segments/occurrences, supplied GeoJSON, diagnostics and processing time. Only actual empty candidate groups are counted as empty; unmatched observations with nonempty candidates remain unmatched. Raw classifier probabilities/ranks remain unavailable when the decoder chose a different arc.

Each GeoJSON LineString is drawn independently, with explicit disconnected/partial/empty states. Repeated directed arcs and connector roles are preserved. Native JSON and GeoJSON downloads are available. No routes are constructed, stitched, decoded or evaluated in the frontend. The fixed Phase 3 Gradient Boosting classifier remains the default; real HMM mode waits for backend capabilities. Existing synthetic Demo mode continues to work.

Verification: **86 frontend tests passed**, comprising the original 56 plus focused Phase 5 cases for connected/partial/disconnected/empty and zero-observation outputs; native/enveloped HTTP parsing and polling; coordinate/identity/order/schema rejection; unknown geometry rejection; independent map lines; raw classifier/decoded-score separation; repeated arcs and connectors; and AppTest rendering of partial/disconnected/empty results. `pip check` passed. No decoder execution, trained model loading or raw/evaluator data access occurs. A real FastAPI round trip remains unverified until the backend is connected.

Implemented on `feature/phase6-ui`. The existing clean UI branch originally pointed at Phase 3 (`82485cc`); it was fast-forwarded to the freshly fetched `origin/Main` at `341d6b34806b3c85c1a955d5e15d9fb8d5a66bc7` (merged PR #5). Thus Phase 4 reports are present and used directly. No Phase 4 code was copied manually. No commit, push, merge into another branch, or pull request was performed; only the UI branch was fast-forwarded to its requested starting point.

## Delivered behavior

Five pages are complete: Overview, Trip Explorer, Route Prediction, Model Analytics, System Status. They share a navy/cyan/teal theme, responsive cards and sidebar, accessible labelled controls, bounded session caching, loading/empty/error states and consistent metric formatting.

Overview derives the 80 training trips, 10 student-testing trips, five classifiers, 208,417 training observations, 121,721 eligible groups, 95.049% training OOF Top-1 and 95.144% student-testing Top-1 from the actual Phase 3 aggregate files. Qualified proxy-label agreement is distinguished from physical map-matching accuracy. The fixed Histogram Gradient Boosting model remains the default. HMM is shown as awaiting integration.

Trip Explorer uses service-exposed partitions/IDs, preserves zero padding, loads metadata/geometry on demand, and renders observations in their original order with inspection popups/table, start/end markers, fit bounds and directed road tooltips. Folium receives latitude/longitude; wire data retain longitude/latitude. A complete Folium document is embedded in a fixed-height responsive Streamlit iframe. Leaflet handles map interactions without Python round trips. Road geometry order and source/target tooltip IDs retain direction semantics. Independent classifier selections are drawn as individual arcs. Only a supplied decoded route or an explicitly labelled synthetic example creates a continuous route layer.

Route Prediction implements capability-gated modes, request submission, queued/running/completed/failed job states, manual progress retrieval, result identity validation, selected arcs/probabilities, empty groups, measured runtime if returned, backend metrics, result maps and JSON download. It never calculates reference-based metrics or normalises raw binary candidate probabilities. Changing source clears geometry, pending jobs and saved results. Changing the selected trip/mode hides mismatched saved results.

Analytics compares all five classifiers on training OOF/student testing for Top-1, Top-3, trip-macro Top-1, MRR, binary F1 and log loss, with a separate nearest-geometry baseline, per-trip charts, ranking, coverage and definitions. Actual Phase 4 CSV/JSON reports supply fixed-vs-tuned metrics, paired uncertainty, fit/scoring times, model sizes and optimisation studies. The +0.031 percentage-point nested improvement is presented with its interval including zero, increased log loss and approximately doubled recorded scoring time. Missing Phase 4 reports produce “Phase 4 results unavailable in this checkout”. No reports are regenerated.

System Status shows the source, configured URL, health state/version/latency, last request/error, model capabilities and backend-returned evaluation. Synthetic provenance is explicit. Unavailable metrics show “Not available”.

## File inventory

Created:

```text
.streamlit/config.toml
frontend/__init__.py
frontend/app.py
frontend/config.py
frontend/requirements.txt
frontend/.env.example
frontend/README.md
frontend/assets/styles.css
frontend/models/__init__.py
frontend/models/schemas.py
frontend/services/__init__.py
frontend/services/api_client.py
frontend/services/mock_client.py
frontend/services/report_loader.py
frontend/components/__init__.py
frontend/components/map_view.py
frontend/components/metric_cards.py
frontend/components/charts.py
frontend/components/navigation.py
frontend/pages/__init__.py
frontend/pages/overview.py
frontend/pages/trip_explorer.py
frontend/pages/predictions.py
frontend/pages/analytics.py
frontend/pages/system_status.py
frontend/tests/test_frontend.py
docs/PHASE6_UI_HANDOFF.md
docs/PHASE6_API_CONTRACT.md
```

Modified: `README.md` (frontend setup links, launch instructions, roadmap/status).

Local ignored artifacts: `.venv/`, Python bytecode and pytest cache. Browser verification preview is outside the repository in the chat's visualization workspace. No changes relative to starting `origin/Main` in `src/`, `reports/`, model artifacts or raw data. Frontend code imports none of the ML pipeline. No raw dataset or `.route` file was opened during implementation or tests.

Runtime dependencies: Streamlit, Folium, Plotly, Pandas, Pydantic and Requests. Pytest is included for verification. The frontend was tested with Python 3.12, Streamlit 1.67.0, Folium 0.20.0, Plotly 6.9.0, Pandas 2.3.3, Pydantic 2.14.3, Requests 2.34.2 and Pytest 9.2.0. The isolated frontend requirements do not install ML training libraries. `streamlit-folium` is not required: full-document embedding avoids component-script reinjection and canvas/SVG plugin incompatibilities discovered during browser testing.

## Verification performed

- `python -m pytest frontend/tests -q -p no:cacheprovider`: **56 passed**. Includes module imports/config, exact metric formatting, actual/missing reports, default selection, all comparison chart datasets, synthetic determinism, trip selection/zero padding, coordinate conversion, layer generation, no fabricated route, empty groups, explicit synthetic route continuity, capability gating, JSON/schema validation, HTTP query/POST payloads, identity rejection, invalid responses, HTTP status failures, timeouts and connection errors. Streamlit AppTest exercises all five pages, both demo prediction modes, missing Live configuration, asynchronous result retrieval and source-state reset.
- `python -m pip check`: **No broken requirements found**.
- Streamlit launch: **successful**, listening at `http://127.0.0.1:8502` during verification. Binding the listening socket required execution outside the sandbox. The default launch command still uses Streamlit's ordinary port selection; 8502 was supplied explicitly for this session.
- Actual browser smoke test: **passed** for Overview metrics/theme, Trip Explorer map rendering and vector-only layer, fixed-classifier synthetic prediction, synthetic route result/provenance/layer, Phase 3 charts, Phase 4 optimisation evidence, System Status and missing-URL Live state. Switching back to Demo restored the local service. Responsive layout was visually checked at 1440×1000 and 390×844; viewport override was reset afterward. The map failure found in the first component-based implementation was fixed and its final iframe rendering verified in the browser.
- Contract validation: strict Pydantic schemas shared by Demo and HTTP, round-trip demo geometry/job JSON and mocked HTTP transport validation passed. **No real FastAPI inference server was available**, so a real two-laptop API round trip has not been tested.
- Existing ML regression tests were **not run**; no training, optimisation, dataset exploration, evaluator access or report regeneration was performed. Existing results remain unchanged relative to starting Main.

An initial pytest run hit an inaccessible host temporary directory. The missing-report test was changed to use an absent read-only report path, removing unnecessary temporary writes. Final tests pass without temporary-directory access or pytest cache writes. A later focused test caught a redundant rerun after asynchronous retrieval that could discard the next input; it was removed and asynchronous/source reset coverage now passes.

## Run from repository root

PowerShell, with Python installed/on PATH for environment creation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
.\.venv\Scripts\python.exe -m pytest frontend/tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The local `.venv` has already been created and dependencies installed on this computer, so the last command runs immediately. With the environment activated: `python -m streamlit run frontend/app.py`.

Configure Live mode before launch:

```powershell
$env:PATHSNAP_API_URL = "http://BACKEND_IP:8000"
$env:PATHSNAP_REQUEST_TIMEOUT = "15"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Select Live in the sidebar. `.env.example` documents configuration; `.env` is not loaded automatically. The frontend never assumes a localhost backend URL.

## Integration readiness and limits

The proposed API requires `/health`, `/api/v1/capabilities`, `/api/v1/trips`, `/api/v1/trips/{trip_id}`, `/api/v1/trips/{trip_id}/geometry`, `/api/v1/predictions`, and `/api/v1/results/{result_id}` with the schemas in `frontend/models/schemas.py`. `MockPathSnapClient` and `HttpPathSnapClient` implement equivalent interfaces. Full payloads, errors/status codes, job lifecycle, versioning and two-laptop network setup are in [the contract](PHASE6_API_CONTRACT.md).

Phase 5 connects by advertising an available `hmm_viterbi` mode and returning real decoded geometry with `provenance:"hmm_viterbi"` from the same prediction/results endpoints. Phase 4 inference stays disabled until backend capabilities enable it, independent of local report availability. Genuine predictions, model probabilities, measured inference runtime, route continuity and permitted route-level evaluation await backend data. Demo uses deterministic invented examples and needs no model binary or raw organiser files.

Basemap tiles and Folium JavaScript/CSS assets need internet on a fresh browser load. Synthetic data itself is offline. Vector-only rendering works once assets have loaded; a fully bundled offline basemap/asset package is not included. Large trips keep the full GPS line; marker/road display caps and visual smoothing never change inference inputs. Map construction still scales with GPS line length. Async progress is user-triggered rather than automatic polling; cancellation, authentication, pagination and viewport geometry require future coordinated backend extensions.

**PASS — Phase 6 frontend complete and ready for integration.**
