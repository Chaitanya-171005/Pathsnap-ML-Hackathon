# PathSnap Phase 4 handoff

Phase 4 tunes the Phase 3 Histogram Gradient Boosting pipeline for observation-level Top-1 agreement with qualified supplied-route proxy labels. The 15 predictors, Phase 2A 50 m / Top-8 candidate policy, Phase 2B qualification, and frozen Phase 3 baseline are reused. Preprocessing fits separately inside every training fold. Unqualified fixes and qualified candidate misses retain their Phase 3 treatment.

## Protocol

- Five outer folds match the Phase 3 `GroupKFold` trip assignments. Each outer fitting partition has an independent three-fold grouped search. The selected parameters are refitted on the outer fitting trips and scored against its held-out trips.
- The final search uses its own five-fold run on the 80 training trips. It chooses parameters by pooled Top-1, trip-macro Top-1, candidate log loss, then canonical parameter order. The final model fits all eligible training rows.
- Student testing is opened after a hash-verified model freeze. A completed evaluation is written atomically and reused on resume. If a process stops during evaluation before its result is saved, the incomplete attempt can be replayed with the same frozen model. This development holdout was already reported in Phase 3.
- The fixed baseline uses the existing Phase 3 pipeline. Phase 3 fold scores are reused when the local manifest, runtime versions, prediction settings, exact fold assignments and frozen metrics agree. Otherwise the pipeline is fitted on the same outer folds. The nearest geometric candidate provides a second comparison on the identical groups.
- Probability scores are raw positive-class probabilities. Ties use Phase 2A distance rank, then arc ID. The outer comparison is conditional on the model family selected during Phase 3 training CV.

## Search and resume

The local [Tree-structured Parzen Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization) uses independent parameter densities on the best quarter of completed trials and the remaining trials. It proposes 48 joint samples from the better density and ranks them by its likelihood ratio against the other density. Truncated Gaussian kernels cover numeric parameters in normalised coordinates, with a uniform prior. Smoothed frequencies cover categorical depth. The first five configurations per study are the fixed baseline followed by seeded random proposals. The remaining configurations use TPE. The implementation uses the existing NumPy and SciPy dependencies. Optuna could not be installed in the execution environment.

| Parameter | Search space |
|---|---|
| Learning rate | 0.02–0.20, log scale |
| Boosting iterations | 80–300 |
| Leaf nodes | 15–63 |
| Maximum depth | Unlimited, 4, 6, 8 or 12 |
| Minimum samples per leaf | 10–80 |
| L2 regularisation | 0.001–10, log scale |

Early stopping is disabled. The experiment runs sequentially with bounded native threads. Seed 2026 controls estimator randomness, the trial sequence and the paired trip bootstrap. The sampler derives its random state from the study name, seed and trial number, then reconstructs its history from verified records. A run interrupted after a fold reuses that fold's statistics and can recreate the same proposals from the completed history. Trained frozen models and score checkpoints carry integrity hashes. A failed trial records its error and stops the run. The manifest binds code, folds, input matrix, policy, feature schema, search, budget, prediction settings and runtime versions. Run a different contract with fresh output directories.

From the repository root, use the following command for the measured initial budget. Regenerate the Phase 2B training exports with `python -m src.alignment --partition train --policy reports/phase2b/train/policy.json` if they are missing.

```bash
source .venv/bin/activate
python -m pip check
python -m pytest -q
python -m src.tuning --outer-trials 8 --final-trials 12 --workers 4 --batch-size 8192
```

Run the same command to resume. For the longer 20/40 search, supply fresh `--artifact-dir` and `--report-dir` locations and omit the trial-count flags. `--training-only` allows a smoke run with separate outputs without opening student-testing data.

## Results and limits

See the [measured summary](../reports/phase4/summary.md) and [complete trial table](../reports/phase4/optimization_trials.csv). [Outer fold pairs](../reports/phase4/outer_fold_metrics.csv), [per-trip results](../reports/phase4/per_trip_comparison.csv), [difficult trips](../reports/phase4/difficult_trip_metrics.csv) and [diagnostic subsets](../reports/phase4/error_slices.csv) show where tuning helped or hurt. The timing CSV gives fold and final-fit timings, the archived Phase 3 measurements and process RSS high-water marks. Session time reflects the current invocation. The hyperparameter importance plot is an exploratory tree-surrogate summary of the bounded search. The trip-cluster percentile interval measures resampling uncertainty in paired proxy agreement. Shared training folds and reference qualification can still affect these results.

The measured run completed 52 trials, with zero failed or pruned trials, 180 grouped search fits, five selected outer refits, and one full-training final fit. The fixed outer scores came from the verified Phase 3 checkpoints. The run took 25 minutes through student scoring and reached 860.6 MiB process RSS. The final trial selected 241 boosting iterations, learning rate 0.0946700463072859, 43 leaves, depth 8, 19 samples per leaf, and L2 regularisation 0.1833462779672531.

| Metric | Fixed Phase 3 | Phase 4 |
|---|---:|---:|
| Training pooled Top-1 | 95.0493% | 95.0806% |
| Training trip-macro Top-1 | 94.5933% | 94.6536% |
| Training Top-3 | 99.8234% | 99.8086% |
| Training MRR | 0.974109 | 0.974250 |
| Training candidate F1 | 0.943641 | 0.942911 |
| Training candidate log loss | 0.054739 | 0.055803 |
| Student-testing Top-1 | 95.1445% | 95.2495% |
| Final model size | 241,640 bytes | 495,871 bytes |
| Final fitting time | 6.98 s | 15.85 s |
| Student scoring time | 0.20 s | 0.40 s |

Nested tuning found 38 additional Top-1 matches among 121,721 eligible training fixes. Its paired change was +0.031 percentage points, with a 95% trip bootstrap interval from −0.033 to +0.123 points. Two outer folds improved, two declined, and one tied. Twenty-five trips improved, 26 declined, and 29 tied. The near-parallel signature slice gained 1.86 points on 2,361 groups, and the missing-heading slice lost 1.05 points on 4,491 groups. The anomalous trip `00000054` gained one correct match, the short trip `00000026` lost one, and the largest road-network trip `00000013` lost two. The other three named diagnostic trips tied. These slice results overlap and carry their own denominators.

Phase 3 remains the recommended default. The paired interval includes zero, outer log loss rose, and the tuned bundle took roughly twice the scoring time in these single-run measurements. The student-testing gain of 14 correct Top-1 choices among 13,325 eligible observation groups is reported as a reused-holdout result.

The training-only recommendation prefers the tuned model when the paired Top-1 interval is above zero and trip-macro Top-1 is preserved. Both models remain available. The test suite checks deterministic proposal replay, partial-trial recovery, incompatible checkpoint rejection, fold-local preprocessing, trip grouping, report regeneration, and delayed student access.

The final regression suite passed 83 tests. A checkpoint replay reproduced 20 of 21 report files byte for byte. The invocation timing row was the only report difference. The replay reused all completed trials and the frozen student result. All 360 development input files and 57 Phase 3 baseline files matched their original hashes.

## Load the frozen classifier

The tuned bundle uses the existing Phase 3 loader and candidate-ranking interface.

```python
from pathlib import Path
from src.candidates.pipeline import CandidateConfig
from src.models.inference import load_model, predict_trip

bundle = load_model(Path("artifacts/models/phase4/gradient_boosting.joblib"))
config = CandidateConfig(**bundle["manifest"]["candidate_config"])
# Build Phase 2A candidate batches with config, then pass them to predict_trip.
# predict_trip(bundle["pipeline"], bundle["model_id"], batches)
```

The selection record is `artifacts/models/phase4/model_selection.json`. The six resumable study journals are under `artifacts/models/phase4/studies/` and outer score checkpoints are under `artifacts/models/phase4/outer_checkpoints/`. `frozen_model.json` records the chosen bundle hash and reload probe result. Phase 3 fallback uses `artifacts/models/phase3/gradient_boosting.joblib`.
