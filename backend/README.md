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

## Authorised final evaluation

Final evaluation is disabled unless Laptop 1 is started with
`PATHSNAP_FINAL_EVALUATION=1`. Use a dedicated server on a trusted private
IPv4 LAN for the assessment. Share a unique assessment token privately with
the Laptop 2 operator. Keep the public ngrok tunnel attached to the existing
development service. Final requests require the token and private IPv4 source
and destination addresses. Requests with forwarding headers are rejected.

### Laptop 1 (macOS)

Run from the isolated PathSnap checkout after the organiser authorises the session.
Set `PATHSNAP_FINAL_ROOT` to the absolute path of the authorised catalog containing
ten trip directories directly. Each trip must supply its own `.track`, `.nodes`
and `.arcs`. The current development data configuration can be supplied
independently through `PATHSNAP_DATA_ROOT`.

```bash
cd /tmp/pathsnap-final-evaluation
repo=/Users/apple/Documents/Pathsnap-ML-Hackathon
lan_ip=192.168.1.25  # replace with Laptop 1's private IPv4 address
printf 'Assessment token: '
read -rs PATHSNAP_FINAL_TOKEN
printf '\n'
export PATHSNAP_FINAL_TOKEN
export PATHSNAP_FINAL_EVALUATION=1
export PATHSNAP_FINAL_ROOT=/absolute/path/to/authorised-trip-catalog
export PATHSNAP_FINAL_SCORING=pending
export PATHSNAP_DATA_ROOT="$repo/data/raw"
export PATHSNAP_PHASE3_MODEL="$repo/artifacts/models/phase3/gradient_boosting.joblib"
"$repo/.venv/bin/python" -m uvicorn backend.main:app \
  --host "$lan_ip" --port 8011 --no-proxy-headers --access-log
```

The token must contain at least 32 ASCII letters or digits. Startup validates the
ten-trip catalog and the model hash, candidate configuration, alignment policy
and HMM defaults against the committed Phase 5 freeze. The assessor may select
a trip while scoring is pending. A successful request logs the trip ID, matching
request ID, measured inference time and scoring status. The access log includes
the prediction POST and HTTP status.

If organisers explicitly approve the existing Phase 5 qualified-proxy scoring
protocol, set `PATHSNAP_FINAL_SCORING=confirmed_phase5_proxy` and restart the
assessment server. The response then includes the qualified and eligible proxy
counts, candidate misses, two proxy Top-1 fractions, and HMM directed arc
precision, recall and adjacent-arc-pair F1. The denominator for Top-1 is
qualified proxies with the reference arc in the candidate set. Approved scoring
loads only the selected trip's reference after inference. Reference contents
and per-observation labels stay server-side. Unavailable references leave the
aggregate evaluation fields empty. If the organiser selects a different
protocol, keep scoring pending until that protocol is implemented and checked.

### Laptop 2 (macOS)

Use a copy of the same implementation and keep Streamlit bound to Laptop 2
loopback for the on-site assessor. Enter the same token as Laptop 1.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r frontend/requirements.txt
printf 'Assessment token: '
read -rs PATHSNAP_FINAL_TOKEN
printf '\n'
export PATHSNAP_FINAL_TOKEN
export PATHSNAP_API_URL=http://192.168.1.25:8011
.venv/bin/python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

### Laptop 2 (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
$secureToken = Read-Host -Prompt "Assessment token" -AsSecureString
$env:PATHSNAP_FINAL_TOKEN = ([System.Net.NetworkCredential]::new("", $secureToken)).Password
$env:PATHSNAP_API_URL = "http://192.168.1.25:8011"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.address 127.0.0.1
```

Replace the example IP on Laptop 2 with Laptop 1's actual private IP. Open the
local Streamlit URL and choose **Live → Route Prediction → evaluator_test**.
The UI presents ten options. Choosing one initiates the real inference request
and displays the segmented route, runtime, continuity and any approved score.
Check the paired POST access log and final inference log on Laptop 1. When the
assessment ends, stop the dedicated server and remove the token from both
terminal sessions.

### Readiness checklist

- [ ] Organiser has authorised activation and the supplied catalog.
- [ ] Organiser has confirmed the score definition, denominators and missing-reference handling.
- [ ] Both laptops have the same implementation and dependencies.
- [ ] Laptop 1 uses the verified Phase 3 artifact and a dedicated private listener.
- [ ] Laptop 2 reaches that listener from the actual assessment network.
- [ ] Ten choices, on-selection request, route, metrics and logs have been rehearsed with synthetic inputs.
- [ ] Authorised demo is scheduled with the relevant score configuration.

For the synthetic integration checks, run `python -m pytest -q
backend/tests/test_final_evaluation.py`. Optional
`PATHSNAP_TEST_PHASE3_MODEL=/absolute/path/to/gradient_boosting.joblib` adds
real frozen-model inference on the ten synthetic trips. The test suite creates
its own catalog and never reads the configured final dataset.
