# Phase 3 classifier comparison

All pointwise metrics measure agreement with qualified supplied-route proxy labels.

Training-only selection: **gradient boosting**.

| Model | OOF Top-1 | OOF trip macro | OOF Top-3 | OOF F1 | OOF log loss | Student Top-1 | Student trip macro | Student F1 | Final fit (s) | Student predict (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 95.049% | 94.593% | 99.823% | 0.9436 | 0.0547 | 95.144% | 95.103% | 0.9424 | 6.98 | 0.20 |
| decision_tree | 94.787% | 94.269% | 99.513% | 0.9386 | 0.1098 | 94.934% | 94.884% | 0.9391 | 3.89 | 0.02 |
| random_forest | 94.671% | 93.989% | 99.700% | 0.9408 | 0.0662 | 94.627% | 94.458% | 0.9398 | 13.45 | 0.30 |
| knn | 94.288% | 93.441% | 99.215% | 0.9324 | 0.1964 | 94.169% | 93.976% | 0.9324 | 1.30 | 23.44 |
| logistic_regression | 94.264% | 93.505% | 99.269% | 0.8965 | 0.1054 | 94.334% | 94.054% | 0.9017 | 1.33 | 0.02 |

## Nearest geometric candidate

- train_oof — 73,136/121,721 (60.085%).
- student_test — 7,851/13,325 (58.919%).

## Denominators

- train — 208,417 original, 121,855 qualified (58.467%), 121,721 eligible (99.890% of qualified), 134 candidate misses.
- student_test — 19,160 original, 13,335 qualified (69.598%), 13,325 eligible (99.925% of qualified), 10 candidate misses.
