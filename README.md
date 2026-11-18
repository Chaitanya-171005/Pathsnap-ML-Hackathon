# PathSnap — ML Hackathon

**GPS map matching and route reconstruction with machine learning and Viterbi decoding**

PathSnap reconstructs vehicle routes from noisy GPS observations by matching them to directed road segments. The project was developed for the **UE24CS352A Machine Learning Hackathon**, Department of Computer Science and Engineering, PES University.

Phases 1–6, the FastAPI integration and the gated final evaluator workflow are implemented. The final evaluator workflow landed in [PR #9](https://github.com/Chaitanya-171005/Pathsnap-ML-Hackathon/pull/9), and the frozen Phase 3 serving model landed in [PR #10](https://github.com/Chaitanya-171005/Pathsnap-ML-Hackathon/pull/10).

## Deliverable 1 — Public GitHub repository

### Deliverables and outputs directory

| Requirement | Implementation location | Outputs/evidence location | Status |
| --- | --- | --- | --- |
| Phase 1 · Dataset loading, preprocessing and EDA | [Preprocessing](src/preprocessing/), [EDA](src/eda/), [Phase 1 handoff](docs/PHASE1_HANDOFF.md) | [EDA reports](reports/eda/), [dataset inventory](reports/eda/dataset_inventory.csv), [data-quality issues](reports/eda/data_quality_issues.csv), [EDA summary](reports/eda/summary.md), [EDA figures](visualizations/eda/) | Implemented, aggregate outputs tracked |
| Phase 2A · Directed candidate generation and feature engineering | [Candidate pipeline](src/candidates/), [Phase 2A handoff](docs/PHASE2A_HANDOFF.md) | [Phase 2A results](reports/phase2a/), [benchmark](reports/phase2a/benchmark.csv), [trip diagnostic](reports/phase2a/trip_00000054.json) | Implemented |
| Phase 2B · Reference alignment and qualified proxy labels | [Alignment](src/alignment/), [Phase 2B handoff](docs/PHASE2B_HANDOFF.md) | [Phase 2B results](reports/phase2b/), [training policy](reports/phase2b/train/policy.json), [training summary](reports/phase2b/train/summary.md), [student summary](reports/phase2b/student_test/summary.md), [student candidate recall](reports/phase2b/student_test/candidate_recall.csv) | Implemented |
| Phase 3 · Five ML classifiers and model selection | [Model implementation](src/models/), [Phase 3 handoff](docs/PHASE3_HANDOFF.md) | [Phase 3 results](reports/phase3/), [experiment manifest](reports/phase3/experiment_manifest.json), [selection decision](reports/phase3/model_selection.json), [summary](reports/phase3/summary.md) | Five classifiers compared |
| Model comparison and diagnostics | [Phase 3 report generator](src/models/report.py) | [Comparison CSV](reports/phase3/model_comparison.csv), [comparison figure](reports/phase3/model_comparison.png), [confusion matrices](reports/phase3/confusion_matrices.png), [feature-importance figure](reports/phase3/feature_importance.png), [per-trip Top-1 figure](reports/phase3/per_trip_top1.png), [error slices](reports/phase3/error_slices.csv) | Tracked |
| Grouped cross-validation, student testing and runtime | [Evaluation code](src/models/evaluation.py) | [CV metrics](reports/phase3/cross_validation_metrics.csv), [fold assignments](reports/phase3/fold_assignments.csv), [student metrics](reports/phase3/student_test_metrics.csv), [coverage and exclusions](reports/phase3/coverage_summary.csv), [training and runtime records](reports/phase3/training_runtime.csv) | Tracked development evidence |
| Phase 4 · Bayesian hyperparameter optimisation | [Tuning implementation](src/tuning/), [Phase 4 handoff](docs/PHASE4_HANDOFF.md) | [Phase 4 results](reports/phase4/), [trial history](reports/phase4/optimization_trials.csv), [best hyperparameters](reports/phase4/best_hyperparameters.json), [search manifest](reports/phase4/optimization_manifest.json), [reproducibility record](reports/phase4/reproducibility.json) | 52 completed trials |
| Fixed-versus-tuned comparison and figures | [Phase 4 report generator](src/tuning/report.py) | [Paired comparison CSV](reports/phase4/baseline_vs_tuned.csv), [comparison plot](reports/phase4/baseline_vs_tuned.png), [optimisation history](reports/phase4/optimization_history.png), [outer-fold plot](reports/phase4/outer_fold_comparison.png), [hyperparameter importance](reports/phase4/hyperparameter_importance.png), [per-trip improvement](reports/phase4/per_trip_improvement.png), [Top-1 vs Log Loss](reports/phase4/top1_vs_logloss.png), [optimisation runtime](reports/phase4/training_runtime.csv), [summary](reports/phase4/summary.md) | Tracked |
| Phase 5 · HMM and Viterbi route reconstruction | [Decoder](src/decoding/), [Phase 5 handoff](docs/PHASE5_HANDOFF.md), [output schema](docs/PHASE5_OUTPUT_CONTRACT.md) | [Phase 5 results](reports/phase5/), [summary](reports/phase5/summary.md), [classifier vs HMM CSV](reports/phase5/classifier_vs_hmm.csv), [comparison figure](reports/phase5/classifier_vs_hmm.png), [per-trip results](reports/phase5/per_trip_metrics.csv), [route metrics](reports/phase5/route_metrics.csv), [connectivity figure](reports/phase5/route_connectivity.png), [per-trip improvement](reports/phase5/per_trip_improvement.png), [runtime](reports/phase5/runtime.csv) | Implemented, development results tracked |
| Frozen Phase 3 weights and Phase 5 decoder configuration | [Model loader](src/models/inference.py), [decoder defaults](src/decoding/hmm.py) | [Frozen classifier](artifacts/models/phase3/gradient_boosting.joblib), [decoder freeze](reports/phase5/decoder_freeze.json) | Serving model and configuration tracked |
| FastAPI inference | [API entry point](backend/main.py), [inference service](backend/service.py) | [API contract](docs/PHASE6_API_CONTRACT.md), [API tests](backend/tests/test_api.py), [live HTTP smoke test](backend/tests/test_frontend_http.py) | Implemented |
| Gated final evaluator API | [Final evaluator service](backend/final_evaluation.py), [API entry point](backend/main.py) | [Final evaluator tests](backend/tests/test_final_evaluation.py), [authorised launch instructions](backend/README.md#authorised-final-evaluation) | Implemented, synthetic coverage verified |
| Phase 6 · Streamlit and Folium | [App](frontend/app.py), [UI pages](frontend/pages/), [UI map renderer](frontend/components/map_view.py), [EDA map generator](src/visualization/folium_maps.py) | [UI handoff](docs/PHASE6_UI_HANDOFF.md), [frontend tests](frontend/tests/), [tracked EDA plots](visualizations/eda/) | Implemented, coordinate-bearing HTML generated locally |
| Automated tests | [ML and preprocessing tests](tests/), [backend tests](backend/tests/), [frontend tests](frontend/tests/) | [Test invocation](#installation-and-tests), [reported verification](#verification-and-limits) | 205 passed, live HTTP smoke skipped in last reported run |
| Dependencies | [ML requirements](requirements.txt), [backend requirements](backend/requirements.txt), [frontend requirements](frontend/requirements.txt) | [Frontend environment template](frontend/.env.example) | Tracked |
| Deployment and operation | [FastAPI launcher](backend/main.py), [Streamlit launcher](frontend/app.py) | [Backend runbook](backend/README.md), [frontend setup](frontend/README.md), [two-laptop instructions](#running-the-system) | Instructions tracked, physical two-laptop verified |

The Phase 3, 4 and 5 result directories contain the full sets of tracked CSVs and figures, including additional per-trip, error-slice and runtime records. The frozen serving artifact is the tracked exception to the model-directory ignore rule. Other model checkpoints and the optional tuned classifier artifact are local experiment outputs.

## Deliverable 2 — Consolidated technical PDF

The final consolidated PDF is a separate submission artifact. A completed or submitted final PDF has no verified location in this repository. Include the following evidence in the PDF:

1. **EDA and data quality** — Dataset structure, preprocessing, GPS and road-network figures, anomalies and partition boundaries. Start with the [EDA summary](reports/eda/summary.md), [data-quality CSV](reports/eda/data_quality_issues.csv) and [EDA figures](visualizations/eda/).
2. **Five-model comparison** — Algorithm rationale, key settings, grouped cross-validation, Macro F1, observation Top-1, applicable Log Loss, timing and generalisation. Include measured continuity where available and explain any N/A. Use the [comparison CSV](reports/phase3/model_comparison.csv), [student results](reports/phase3/student_test_metrics.csv) and [runtime CSV](reports/phase3/training_runtime.csv).
3. **Bayesian optimisation** — Search space, objective, trial count, selected configuration, fixed-versus-tuned results and why the fixed model was retained. See the [Phase 4 report](reports/phase4/summary.md).
4. **HMM and Viterbi** — States, emissions, transitions, route breaks, classifier-only versus decoded results, runtime and limitations. See the [Phase 5 handoff](docs/PHASE5_HANDOFF.md) and [measured results](reports/phase5/summary.md).
5. **Four-layer Folium map** — Use a permitted development trip to show GPS observations, road network, supplied reference route and predicted route. The [EDA map generator](src/visualization/folium_maps.py) provides GPS, road and reference layers. The [UI map renderer](frontend/components/map_view.py) shows predicted segments. Include the combined map and its screenshot as submission evidence.
6. **Actual UI and API evidence** — Capture real screenshots of the UI and a successful FastAPI request and response. Identify the environment used and record whether the evidence came from one or two devices.
7. **Reproducibility and technical explanation** — Record settings, commands, artifact checksum, metric denominators, limitations and any disclosure required by the organiser.

## Dataset and pipeline

| File | Contents |
| --- | --- |
| `.track` | GPS observations containing longitude, latitude and timestamp |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed connections between road-network nodes |
| `.route` | Ordered reference road-arc IDs |

Place permitted organiser-supplied development data in local zero-padded trip folders under `data/raw/train/` (80 trips) and `data/raw/student_test/` (10 trips). The 10 reserved evaluator trips are for authorised assessment only. Keep them out of training, tuning, feature and model selection, decoder changes and development tests. Raw data, local guideline PDFs, per-observation exports and coordinate-bearing maps are Git-ignored. See the [dataset schema](docs/DATASET_SCHEMA.md), [data contract](docs/DATA_CONTRACT.md) and [EDA findings](docs/EDA_FINDINGS.md).

PathSnap preprocesses GPS and road-network data, generates directed candidates within 50 m, retains up to eight candidates per observation and computes 15 features. The classifier scores those candidates. The HMM and Viterbi decoder use directed road connectivity to reconstruct separate route segments. The inference path consumes `.track`, `.nodes` and `.arcs`. Development alignment and approved post-inference scoring use reference routes separately. FastAPI runs inference on Laptop 1, and Streamlit on Laptop 2 displays the returned routes and available metrics.

## Measured development results

| Measure | Result | Committed evidence |
| --- | --- | --- |
| Classifiers trained and compared | 5 · Logistic Regression, KNN, Decision Tree, Random Forest and Histogram Gradient Boosting | [Model comparison](reports/phase3/model_comparison.csv) |
| Selected classifier, pooled training out-of-fold Top-1 | 95.049% · 115,695 / 121,721 | [Model comparison](reports/phase3/model_comparison.csv) |
| Selected classifier, student-testing Top-1 | 95.144% · 12,678 / 13,325 | [Student metrics](reports/phase3/student_test_metrics.csv) |
| HMM, student-testing Top-1 | 97.088% · 12,937 / 13,325 | [Classifier vs HMM](reports/phase5/classifier_vs_hmm.csv) |
| HMM, student-testing directed adjacent-arc-pair F1 | 70.39% · mean across trips, classifier baseline 56.06% | [Route metrics](reports/phase5/route_metrics.csv), [Phase 5 summary](reports/phase5/summary.md) |
| Bayesian optimisation | 52 completed trials | [Trial CSV](reports/phase4/optimization_trials.csv), [Phase 4 summary](reports/phase4/summary.md) |

Top-1 measures agreement with **qualified reference-derived proxy labels** whose arcs were present among the candidates. Independent ground-truth accuracy was not measured. Directed arc-pair F1 measures route agreement against supplied reference routes. The student-testing split is a reused development holdout. All figures above are from development partitions.

The [frozen Phase 3 Histogram Gradient Boosting model](artifacts/models/phase3/gradient_boosting.joblib) matches the SHA-256 recorded in the [decoder freeze](reports/phase5/decoder_freeze.json):

```text
6d52207d55f61468fe0a11c8e4c77fccbf1744f18adc309d9b4870c4158f69d4
```

## Installation and tests

Use Python 3.13 for the backend and model environment. From the repository root on macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
PATHSNAP_TEST_PHASE3_MODEL="$PWD/artifacts/models/phase3/gradient_boosting.joblib" python -m pytest -q -rs
```

The backend manifest includes the root ML requirements. Laptop 2 can install only the frontend requirements. See the [frontend setup](frontend/README.md) for a frontend-only and Windows PowerShell setup. Load joblib files only from trusted sources.

## Running the system

### Development mode on a trusted private LAN

**Laptop 1** — From the repository root with the backend environment active and local development data installed, start FastAPI as described in the [backend runbook](backend/README.md#two-laptops).

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

For same-device testing, bind to `127.0.0.1` and use that address for the API URL.

**Laptop 2** — From the repository root with the frontend environment active, replace the example address with Laptop 1's private IP.

```bash
export PATHSNAP_API_URL=http://192.168.1.25:8000
curl -fsS "$PATHSNAP_API_URL/health"
python -m streamlit run frontend/app.py
```

In PowerShell, set `$env:PATHSNAP_API_URL = "http://192.168.1.25:8000"` before launching Streamlit. Select **Live → Route Prediction**, select a permitted development trip and choose HMM/Viterbi. **Demo** mode uses synthetic data and can run independently of the backend.

### Authorised final assessment

Evaluator mode is disabled by default. After organiser authorisation, follow the [dedicated assessment runbook](backend/README.md#authorised-final-evaluation) from the current repository checkout. It contains the Laptop 1 and Laptop 2 launch commands for macOS and Windows. Replace its temporary-checkout example with the path to your current clone.

1. On Laptop 1, set `PATHSNAP_FINAL_EVALUATION=1`, point `PATHSNAP_FINAL_ROOT` at the authorised catalog of ten trip folders and configure an assessment token of at least 32 ASCII letters or digits. Bind the dedicated listener to Laptop 1's private IPv4 address on port `8011` with `--no-proxy-headers --access-log`. Keep this listener off public tunnels.
2. On Laptop 2, configure the same token and set `PATHSNAP_API_URL` to Laptop 1's private address on port `8011`. Launch Streamlit and choose **Live → Route Prediction → evaluator_test**. Selecting a trip sends a real inference request. Laptop 1 logs the trip ID and request ID.
3. Leave `PATHSNAP_FINAL_SCORING=pending` until the organiser confirms the scoring definition, denominator and missing-reference rules. Use `confirmed_phase5_proxy` only if the organiser approves that exact protocol. Approved scoring reads the selected reference after inference and returns permitted aggregates. Reference routes and per-observation labels stay on the server.

## Reproducing development evidence

The phase handoffs linked in the directory document the original commands and outputs. These diagnostic commands use permitted development data:

```bash
python -m src.eda.run_eda --data-dir data/raw
python -m src.candidates --partition train --trip-id 00000083 --radius-m 50 --top-k 8
python -m src.candidates.benchmark
python -m src.alignment --partition train --plots
python -m src.alignment --partition student_test --policy reports/phase2b/train/policy.json
```

EDA writes aggregate reports and plots to the directories listed above. GPS-specific outputs and Folium HTML stay local. Use the [Phase 1 handoff](docs/PHASE1_HANDOFF.md) for alternate output directories, gap thresholds and road-network display limits. Folium's browser assets and basemap tiles require network access. Use the [Phase 3 training](docs/PHASE3_HANDOFF.md), [Phase 4 tuning](docs/PHASE4_HANDOFF.md) and [Phase 5 evaluation](docs/PHASE5_HANDOFF.md) instructions for experiment reproduction in a separate work area.

## Verification and limits

- Most recent reported local suite **205 passed, 1 skipped, 0 failed**. The skipped case is the [live HTTP smoke test](backend/tests/test_frontend_http.py), which requires a running listener.
- The [final evaluator tests](backend/tests/test_final_evaluation.py) exercised ten synthetic evaluator trips. The frozen model was included in that run. Actual two-laptop connectivity is still unverified.
- Organiser confirmation of the official scoring protocol is pending. Reserved evaluator data is excluded from development and from the results reported here.

**Academic integrity.** Use measured results, real screenshots and accurate citations. Keep private credentials and raw data out of the public repository. Follow the organiser's disclosure rules.

**PES University | Department of Computer Science and Engineering | Machine Learning Hackathon**
