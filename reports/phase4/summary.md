# PathSnap Phase 4 Bayesian optimisation

These results measure agreement with qualified route-proxy labels.

- Fixed Phase 3 training baseline source: `phase3_verified_checkpoints`.
- Outer / inner / final CV folds: 5 / 3 / 5.
- Trial budgets: 8 per outer study and 12 for full training. Completed 52, failed 0, pruned 0.
- Seed: 2026. Final parameters: `{"l2_regularization":0.1833462779672531,"learning_rate":0.0946700463072859,"max_depth":8,"max_iter":241,"max_leaf_nodes":43,"min_samples_leaf":19}`.
- Paired outer Top-1 change: **+0.031 percentage points**. 95% trip-bootstrap interval [-0.033, +0.123] percentage points.
- Per trip: 25 improved, 26 declined and 29 tied.
- Training-only recommendation: **phase3**.

| Metric | Fixed outer CV | Nested tuned outer CV | Held-out Phase 3 | Held-out Phase 4 |
|---|---:|---:|---:|---:|
| eligible | 121,721 | 121,721 | 13,325 | 13,325 |
| candidate_rows | 650,978 | 650,978 | 72,321 | 72,321 |
| correct | 115,695 | 115,733 | 12,678 | 12,692 |
| top1 | 95.0493% | 95.0806% | 95.1445% | 95.2495% |
| trip_macro_top1 | 94.5933% | 94.6536% | 95.1032% | 95.1721% |
| top3 | 99.8234% | 99.8086% | 99.8424% | 99.8499% |
| mrr | 97.4109% | 97.4250% | 97.4346% | 97.4883% |
| precision | 93.6401% | 93.7134% | 93.0820% | 93.0444% |
| recall | 95.0994% | 94.8760% | 95.4221% | 95.3696% |
| f1 | 94.3641% | 94.2911% | 94.2375% | 94.1926% |
| log_loss | 0.054739 | 0.055803 | 0.055203 | 0.055297 |

The model family was selected in Phase 3 using training CV. The Phase 4 outer scores assess inner-fold tuning conditional on that earlier model-family choice. The final configuration comes from its own full-training study.

Student testing was previously used for Phase 3 reporting. Its Phase 4 evaluation used the frozen model and serves as a reused development holdout.

Timing and memory are in `training_runtime.csv`. The full trial table is in `optimization_trials.csv`, with validation counts and metrics in `inner_fold_metrics.csv`.
