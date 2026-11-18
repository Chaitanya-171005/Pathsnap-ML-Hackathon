# PathSnap Phase 3 handoff

Phase 3 compares five classifiers for choosing one road candidate per GPS observation. The target is the qualified observation-to-arc proxy from the supplied reference route. Independent physical map-matching accuracy would require independently verified associations.

## Reproduce the experiment

Use the organiser dataset under `data/raw/train/` and `data/raw/student_test/`, preserving zero-padded directory names. The first alignment command is required when the ignored Phase 2B training exports are missing.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.alignment --partition train --policy reports/phase2b/train/policy.json
python -m pytest -q
python -m src.models --workers 4 --batch-size 8192
```

The same model command resumes matching completed folds and final fits using `artifacts/models/phase3/`. Checkpoints bind to the complete training-matrix digest, Phase 2B policy, Phase 2A candidate settings, estimator parameters, source hash, versions, worker count and batch size. Use new `--artifact-dir` and `--report-dir` paths for an independent rerun or changed settings. On a restricted Mac, `LOKY_MAX_CPU_COUNT=4` avoids a warning from physical-core detection. `MPLCONFIGDIR=/tmp/pathsnap-matplotlib` gives Matplotlib a writable cache if needed. KNN prediction can take substantially longer than fitting its index. Phase 3 never samples rows or drops a fold to meet a runtime target. Timing and process RSS vary across runs and machines. Single-process RSS is a cumulative high-water mark that includes earlier work, data buffers and model activity.

## Training-data contract

The Phase 2A inference pipeline generates candidates at 50 m / Top-8. Phase 2B training audit and targets are streamed in trip and observation order. The join verifies `(partition, trip_id, observation_id, arc_id)` for every eligible candidate, requires exactly one positive, and compares the positive rank to the frozen audit. Every existing competing candidate is retained in an eligible group. All other observations are counted by reason in `exclusions_by_trip.csv`, including explicit candidate misses. The Phase 2B frozen policy hash is checked before loading data.

Student-testing data are first opened after the training-only selection and final fits. The offline evaluator applies the frozen Phase 2B policy to the 10 held-out trips and joins the qualified target against the inference candidates in memory. No student-testing training export is produced. The inference module directly depends on Phase 2A candidates and contains no alignment calls.

| Predictor columns in matrix order | Type | Units or meaning |
|---|---|---|
| `distance_m`, `arc_length_m` | float | Metres |
| `distance_rank` | int converted to float | One-based nearest-candidate rank |
| `delta_t_s`, `speed_mps` | nullable float | Seconds, metres/second |
| `heading_difference_deg`, `projection_fraction` | nullable float | Degrees, fraction along the directed arc |
| `from_in_degree`, `from_out_degree`, `to_in_degree`, `to_out_degree` | int converted to float | Counts of incident directed road arcs |
| `time_step_valid`, `speed_valid`, `heading_difference_valid`, `projection_valid` | boolean converted to 0/1 | Phase 2A validity flags |

The matrix uses float32 with NaN for unavailable values. The predictor allowlist omits trip, partition, observation, and arc identifiers. It also omits the target and every reference-derived field, including occurrence, section, costs, scores, qualification and policy fields. A time step can retain its observed delta and carry a false validity flag, as in Phase 2A.

## Baselines and grouped validation

All classifiers use a scikit-learn pipeline. A column transformer median-imputes numeric columns and passes explicit validity flags through. Logistic Regression and KNN also standardise numeric columns. Each transformation fits inside its fold. A fully missing training-fold column is kept with its zero imputation. The final fits use all eligible training data. Seed 2026 is supplied to stochastic estimators.

| Model | Fixed baseline |
|---|---|
| Logistic Regression | L-BFGS, L2, C=1, iteration cap 300 |
| KNN | 15 neighbours, Euclidean KD-tree, leaf size 60, inverse-distance weights |
| Decision Tree | Gini, maximum depth 14, minimum 30 rows per leaf |
| Random Forest | 48 trees, Gini, maximum depth 14, minimum 20 rows per leaf, sqrt feature sampling, parallel fit and ordered single-worker scoring |
| Histogram Gradient Boosting | 150 stages, learning rate 0.1, at most 31 leaves per tree, minimum 30 rows per leaf, L2 regularisation 1, no early stopping |

All five models use the same `GroupKFold(n_splits=5)` assignment. Complete trips are held out and every candidate receives exactly one out-of-fold probability. `fold_assignments.csv` lists each validation trip and `cross_validation_folds.csv` gives row counts, positive counts and overlap checks. KNN uses bounded query batches against a searchable training index. The Random Forest fits its trees in parallel and scores them in a fixed single-threaded order to keep probability ties stable across artifact reloads.

The training selection order is pooled out-of-fold Top-1 descending, trip-macro Top-1 descending, candidate log loss ascending, then alphabetical model ID. Nearest geometric candidate is a separate comparator evaluated on identical groups. Binary precision, recall, F1, macro F1 and confusion counts use the positive-class probability threshold above 0.5. Top-1, Top-3 and reciprocal rank use model probabilities ordered by descending score, Phase 2A distance rank, then arc ID.

Report three denominators independently:

- Qualified coverage = accepted proxy observations / original GPS observations.
- Candidate recall = qualified targets present / accepted proxy observations.
- Top-1 agreement = correctly ranked groups / eligible evaluated groups. Trip macro gives each represented trip equal weight.

## Artifacts and the future decoder

Each trusted local `artifacts/models/phase3/<model_id>.joblib` bundle includes the fitted pipeline, ordered predictors, full parameters and runtime versions, training rows/trips and policy metadata, plus the selection record. Load joblib files only from trusted sources. This serialization format can execute code when opened.

```python
from pathlib import Path
from src.candidates.pipeline import CandidateConfig, candidate_batches, load_inference_trip
from src.models.inference import load_model, predict_trip

bundle = load_model(Path("artifacts/models/phase3/gradient_boosting.joblib"))
config = CandidateConfig(**bundle["manifest"]["candidate_config"])
# files is a TripFiles obtained from the Phase 2A inventory.
batches = candidate_batches(load_inference_trip(files), config)
for observation, predictions in predict_trip(bundle["pipeline"], bundle["model_id"], batches):
    # Empty groups have no predictions and preserve their observation record.
    print(observation.observation_id, predictions)
```

A prediction contains partition, trip ID, observation ID, candidate arc ID, the original candidate rank, model ID, raw positive-class probability and model prediction rank. Raw binary probabilities can sum to a value other than one within an observation. Phase 5 should define any calibration, normalisation, transition evidence and empty-group handling separately. For Phase 4, keep trip-grouped tuning on the training partition, retain the fixed Phase 3 baseline and avoid further model choices on the student-testing split. A final independent evaluation would need a separate untouched split or verified pointwise labels.

## Results and diagnostics

All 25 folds completed across 80 trips. The five validation folds had 15, 16, 16, 16 and 17 trips with 129,781, 130,002, 129,918, 130,066 and 131,211 candidate rows. Each trip appeared in one validation fold and overlap was zero. The held-out run used all 10 student-testing trips after selecting the model.

| Partition | Original fixes | Qualified proxies | Eligible groups | Misses | Unqualified | Positive / negative candidate rows |
|---|---:|---:|---:|---:|---:|---:|
| Training | 208,417 | 121,855 (58.467%) | 121,721 (99.890% of qualified) | 134 | 86,562 | 121,721 / 529,257 |
| Student testing | 19,160 | 13,335 (69.598%) | 13,325 (99.925% of qualified) | 10 | 5,825 | 13,325 / 58,996 |

| Classifier | OOF Top-1 | OOF trip macro | OOF binary F1 | OOF log loss | Held-out Top-1 | Final fit / held-out scoring (s) | Bundle bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 95.049% | 94.593% | 0.9436 | 0.0547 | 95.144% | 6.98 / 0.20 | 241,640 |
| Decision Tree | 94.787% | 94.269% | 0.9386 | 0.1098 | 94.934% | 3.89 / 0.02 | 48,186 |
| Random Forest | 94.671% | 93.989% | 0.9408 | 0.0662 | 94.627% | 13.45 / 0.30 | 2,838,590 |
| KNN | 94.288% | 93.441% | 0.9324 | 0.1964 | 94.169% | 1.30 / 23.44 | 25,220,866 |
| Logistic Regression | 94.264% | 93.505% | 0.8965 | 0.1054 | 94.334% | 1.33 / 0.02 | 3,410 |
| Nearest geometry | 60.085% | 62.172% | — | — | 58.919% | — | — |

Selected Histogram Gradient Boosting based on training-only pooled Top-1. It led the Decision Tree by 0.262 percentage points and also had the highest trip-macro Top-1 and lowest candidate log loss. The Decision Tree offered substantially lower scoring latency and a smaller artifact. KNN had low fit time but required 35–46 s to score each validation fold on this Mac and retained the largest bundle. The full corrected run peaked at about 987 MiB process RSS. Held-out prediction timings cover the 72,321 eligible candidate rows after feature computation, with pipeline transformation and probability prediction included.

### Difficult trips and slices

| Training trip | Eligible groups | Gradient Boosting OOF Top-1 | Nearest Top-1 |
|---|---:|---:|---:|
| `00000026` (short) | 212 | 92.92% | 45.75% |
| `00000083` (typical) | 831 | 93.50% | 75.33% |
| `00000013` (largest network) | 2,129 | 96.01% | 86.38% |
| `00000019` (directed reference breaks) | 1,691 | 90.95% | 51.09% |
| `00000054` (GPS tail anomaly) | 1,377 | 90.12% | 71.68% |
| `00000069` (repeated arcs) | 1,188 | 95.45% | 48.40% |

Trip `00000054` had 1,378 qualified labels out of 19,313 fixes (7.14%), of which 1,377 survived Top-8. The 16,610-fix tail contributed zero supervised groups. Its 90.12% agreement applies only to the eligible subset.

The selected model agreed with the proxy on 95.33% of 82,848 training groups whose nearest two candidates tied within 5 cm, and 97.29% of 79,562 groups with a reverse-direction feature signature. It dropped to 68.40% on 2,361 near-parallel signature groups, 56.65% on 4,491 groups with unavailable nearest-candidate heading and 73.72% on 940 groups without usable speed. The feature signatures use separation, length, projection position and heading among the nearest two candidates. They do not establish road topology or physical correctness. There were zero eligible groups with a gap over 30 s or with the duplicate-geometry signature in this particular slice. Phase 2B abstention removed many such labels before scoring. These slices overlap and the use of heading in proxy qualification can favour predictors using the same signal.

### Verification

69 tests passed, including 56 existing tests and 13 Phase 3 tests. Phase 1 tracked outputs reproduced byte for byte on the complete dataset. All 360 raw dataset file hashes matched before and after. The first run completed 25 folds but an exact serialization check caught Random Forest rounding differences at machine epsilon. The corrected run used ordered single-worker forest scoring and completed all 25 folds, all five final fits and all five held-out evaluations. Each bundle reproduced its training probe probabilities exactly after reload. An independent load reproduced all five held-out metric sets. The two full fits produced 20 bit-identical fold score arrays and five Random Forest arrays within 1e-14 after the scoring-order correction. The installed Python dependency set passed `pip check`.

## Files and reports

- Source: `src/models/__init__.py`, `__main__.py`, `dataset.py`, `evaluation.py`, `inference.py`, `report.py`.
- Regression tests: `tests/test_models.py`. Configuration and docs: `.gitignore`, `requirements.txt`, `README.md`, and this handoff.
- Aggregate model results: `reports/phase3/model_comparison.csv`, `cross_validation_metrics.csv`, `student_test_metrics.csv`, `per_trip_metrics.csv`, `nearest_baseline.csv`, `training_runtime.csv`, and `summary.md`.
- Grouping, data and selection: `cross_validation_folds.csv`, `fold_assignments.csv`, `coverage_summary.csv`, `coverage_by_trip.csv`, `exclusions_by_trip.csv`, `model_selection.json`, and `experiment_manifest.json`.
- Diagnostics and figures: `difficult_trip_metrics.csv`, `error_slices.csv`, `model_comparison.png`, `per_trip_top1.png`, `confusion_matrices.png`, `feature_importance.csv`, and `feature_importance.png`.
