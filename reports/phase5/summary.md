# PathSnap Phase 5 measured results

Model `phase3/gradient_boosting` with SHA-256 `6d52207d55f61468fe0a11c8e4c77fccbf1744f18adc309d9b4870c4158f69d4`.

Frozen decoder settings `DecoderConfig(distance_scale_m=25.0, transition_weight=0.5, position_slack_m=15.0, search_budget=4096, cache_size=4096, restart_cost=4.0)`.

Training rows are smoke checks using the fitted classifier. Student testing is a reused development holdout. The paired Top-1 denominator includes qualified labels present in Phase 2A Top-8. Decoder abstentions on these rows count as incorrect.

| Partition | Trips | Original | Qualified | Candidate misses | Eligible | Classifier | HMM | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4 | 23,535 | 4,116 | 5 | 4,111 | 91.948% | 94.673% | +112 fixes |
| student_test | 10 | 19,160 | 13,335 | 10 | 13,325 | 95.144% | 97.088% | +259 fixes |

## Route and runtime

| Partition | Segments | Internal disconnections | Reference disconnections | No feasible transitions | Inference seconds |
|---|---:|---:|---:|---:|---:|
| train | 824 | 0 | 3 | 794 | 5.12 |
| student_test | 1517 | 0 | 1 | 1501 | 7.94 |

| Partition | Classifier arc precision | HMM arc precision | Classifier arc recall | HMM arc recall | Classifier bigram F1 | HMM bigram F1 |
|---|---:|---:|---:|---:|---:|---:|
| train | 66.51% | 71.59% | 84.68% | 97.59% | 52.36% | 60.88% |
| student_test | 73.87% | 78.51% | 87.04% | 97.38% | 56.06% | 70.39% |

The decoding and evaluation process reached a 315.5 MiB RSS high-water mark at the inference checkpoints.
Per-trip inference stages, load and offline evaluation time are in [runtime.csv](runtime.csv).
The route agreement table averages per-trip rates. Arc precision and recall use unique directed IDs. Bigram F1 uses connected adjacent occurrence counts. The classifier route diagnostic uses its raw Top-1 sequence with consecutive duplicates collapsed. HMM lengths sum full arc occurrences or snapped transitions, depending on the column.

See [per-trip proxy metrics](per_trip_metrics.csv), [route metrics](route_metrics.csv) and [difficult trip diagnostics](difficult_trip_metrics.csv).
