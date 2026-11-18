# PathSnap Phase 1 EDA

GPS gap thresholds are 5, 10, 30, 60 s (strictly greater). The speed screening threshold is 55 m/s.

## train

- **80 trips** (expected 80), with **320 files present**.
- 208,417 GPS rows, 1,933,485 nodes, 3,876,904 directed arcs and 31,336 reference rows.
- Exactly 1 s for 201,048 of 208,337 positive adjacent intervals (96.50%). Largest positive interval is 153064 s.
- 0 duplicate timestamp rows and 0 adjacent timestamp decreases.
- 0/80 trips have equal GPS and reference counts. 70/80 have a nonempty, fully checkable reference path with geometry.
- 13 directed reference breaks and 0 uncheckable reference pairs.
- Across 208,417 GPS points compared against the closest reference segment, 25,381 exceed 30 m and 19,992 exceed 100 m (9.59%).
- 79 road networks have multiple weak components and 80 have multiple strong components.

| File | Finding | Count |
|---|---|---:|
| .arcs | duplicate_directed_edges | 1,775 |
| .arcs | self_loops | 22 |
| .arcs | zero_length_arcs | 100 |
| .route | disconnected_pairs | 13 |
| .route | gps_offset_gt_100_m | 19,992 |
| .route | repeated_reference_arcs | 67 |
| .track | duplicate_consecutive_positions | 6 |
| .track | gap_gt_10_s | 590 |
| .track | gap_gt_30_s | 133 |
| .track | gap_gt_5_s | 1,216 |
| .track | gap_gt_60_s | 69 |
| .track | implausible_speed_pairs | 981 |

## student_test

- **10 trips** (expected 10), with **40 files present**.
- 19,160 GPS rows, 90,689 nodes, 185,041 directed arcs and 3,477 reference rows.
- Exactly 1 s for 18,886 of 19,150 positive adjacent intervals (98.62%). Largest positive interval is 210 s.
- 0 duplicate timestamp rows and 0 adjacent timestamp decreases.
- 0/10 trips have equal GPS and reference counts. 9/10 have a nonempty, fully checkable reference path with geometry.
- 1 directed reference breaks and 0 uncheckable reference pairs.
- Across 19,160 GPS points compared against the closest reference segment, 142 exceed 30 m and 3 exceed 100 m (0.02%).
- 10 road networks have multiple weak components and 10 have multiple strong components.

| File | Finding | Count |
|---|---|---:|
| .arcs | duplicate_directed_edges | 128 |
| .arcs | self_loops | 2 |
| .arcs | zero_length_arcs | 6 |
| .route | disconnected_pairs | 1 |
| .route | gps_offset_gt_100_m | 3 |
| .route | repeated_reference_arcs | 255 |
| .track | gap_gt_10_s | 27 |
| .track | gap_gt_30_s | 2 |
| .track | gap_gt_5_s | 44 |
| .track | gap_gt_60_s | 1 |
| .track | implausible_speed_pairs | 4 |

## Training map selection

- Typical — `00000083` (`training_typical.html`)
- Short — `00000026` (`training_short.html`)
- Anomaly — `00000019` (`training_anomaly.html`)

## Interpretation

Reference offsets are unsigned distances to any valid reference segment. They measure spatial agreement without asserting direction, traversal time or an observation label. Route length sums endpoint-to-endpoint arc lengths, counting repeats and adding no connectors. Parser errors and semantic quality findings can refer to the same rows.
