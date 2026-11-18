# PathSnap FastAPI backend

This API serves the existing Phase 6 Streamlit client. It loads trusted artifacts once per process and uses the Phase 3 classifier by default. The HMM returns the [native Phase 5 output](../docs/PHASE5_OUTPUT_CONTRACT.md) with separate directed GeoJSON segments. Classifier modes return the [v1 completed prediction envelope](../docs/PHASE6_API_CONTRACT.md).

## Local startup

Run from the repository root with the existing ML virtual environment activated:

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The default inputs are `data/raw/train/`, `data/raw/student_test/`, and `artifacts/models/phase3/gradient_boosting.joblib`. Each trip needs its `.track`, `.nodes`, and `.arcs` files. `artifacts/models/phase4/gradient_boosting.joblib` enables the optional tuned classifier when loadable. Use `PATHSNAP_DATA_ROOT`, `PATHSNAP_PHASE3_MODEL`, or `PATHSNAP_PHASE4_MODEL` to point at local inputs elsewhere. Supply only trusted joblib artifacts.

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/v1/capabilities
python -m pip install -r frontend/requirements.txt
PATHSNAP_API_URL=http://127.0.0.1:8000 python -m streamlit run frontend/app.py
```

Select **Live**, then **train**, **00000026**, and **HMM / Viterbi Route Reconstruction**. For the automated live check, leave the backend running and use an environment with the frontend and backend test dependencies:

```bash
PATHSNAP_SMOKE_URL=http://127.0.0.1:8000 python -m pytest -qs backend/tests/test_frontend_http.py
python -m pytest -q
```

## Two laptops

1. On Laptop 1, start Uvicorn with `--host 0.0.0.0 --port 8000` and allow TCP 8000 on a trusted private LAN. Keep the artifacts and development data on that laptop.
2. On Laptop 2, verify `curl -fsS http://<Laptop-1-private-IP>:8000/health`. Install `frontend/requirements.txt`, set `PATHSNAP_API_URL=http://<Laptop-1-private-IP>:8000`, and run `python -m streamlit run frontend/app.py`. In PowerShell use `$env:PATHSNAP_API_URL="http://<Laptop-1-private-IP>:8000"` before launching.
3. Use Live mode in Streamlit. This server is intended for a trusted local or private network. A public deployment needs an authenticated proxy and transport security.

## Response and operating limits

- `/health` returns `degraded` when the fixed classifier or development partitions are unavailable. Query capabilities for available model modes and partitions.
- Geometry includes valid GPS coordinates with their original source row IDs and valid directed arcs with their original arc IDs. Invalid coordinates are omitted from geometry. Prediction responses preserve the original observation count and explicitly represent unmatched rows. Timestamps are relative to the first finite source timestamp.
- The API uses only inference files. Reference routes, reference labels and evaluator data are outside this API.
- One synchronous inference can be active per server process. The server returns HTTP 504 after `PATHSNAP_PREDICTION_TIMEOUT_S` (default 12 seconds), and 503 to new predictions while that worker completes. A client timeout does not kill a running model. Set `PATHSNAP_REQUEST_TIMEOUT` on the frontend above the backend deadline if you change it.
- HMM results are inline and have no server result ID. The results endpoint serves the eight most recent completed classifier envelopes up to 4 MiB each, until eviction or process restart. Larger classifier responses are returned inline without retention. The system has no job queue.
