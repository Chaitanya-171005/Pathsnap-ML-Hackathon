# PathSnap Phase 2B handoff

Phase 2B derives observation-level proxy labels from supplied ordered reference routes. The references establish route order. The pointwise association comes from this pipeline's geometry, movement and sequence assumptions. Candidate features still come from the Phase 2A inference loader, which takes only the GPS track, nodes and arcs.

## Reproduce the runs

From the repository root with the existing Python 3.13 environment and original dataset layout:

```bash
source .venv/bin/activate
python -m src.alignment --partition train --trips 00000019 00000026 00000054 00000083 00000013 \
  --report-dir /tmp/pathsnap-representative/reports --export-dir /tmp/pathsnap-representative/exports --plots
python -m src.alignment --partition train --plots
python -m src.alignment --partition student_test --policy reports/phase2b/train/policy.json
python -m pytest -q
```

The partition flag selects one split. Each run writes its summary to `reports/phase2b/<partition>/` and local derived rows to `data/alignments/<partition>/`. Use separate output roots for subsets and experimental policies. The local export directory, including the observation-level plots, is Git-ignored. Student testing produces offline evaluation proxies and join diagnostics. Binary candidate training targets are exported for training only.

## Alignment policies

| Method | Procedure | Interpretation |
|---|---|---|
| `nearest` (A) | Search only drawable reference occurrences up to the 45 m proposal radius in the same local projected metre plane as Phase 2A. Choose the exact clamped point-to-segment minimum and then the smallest occurrence index for a deterministic tie proposal. | Distance baseline. A distinct occurrence can tie even when the arc ID is the same. The 0.05 m tie tolerance, duplicate geometry and section boundary checks can cause abstention. |
| `monotonic` (B) | Find a maximum-scoring chain of observation/occurrence pairs with nondecreasing occurrence index. Compare local scores among alternatives between independent evidence anchors. | Multiple observations can share an occurrence, a later revisit has a separate index, and the chain can skip observations. |
| `qualified` (C) | Apply independent acceptance gates to B's proposal and locally plausible alternatives. Retain B's proposed IDs and diagnostics even if the gates abstain. | Conservative proxy training policy, with diagnostic scores and reasons. |

The reference-only spatial index uses each route occurrence as a separate row. A missing reference arc or unusable endpoint splits the route, as does a mismatch between a directed arc's destination and the next arc's origin. The nearest and monotonic methods flag proposals within 12 projected metres along an arc of a broken boundary. B also flags the first selected observation after crossing to a new reference section. A section crossing does not imply the observed travel traversed any connecting road.

### Sparse sequence objective

A match contributes `32 - local_cost`, where `local_cost = projected_distance_m + 18 × heading_difference_deg / 180` when Phase 2A motion screening supplies a heading. An unavailable heading contributes no angular term. Moving forward charges 0.6 per skipped reference occurrence and 12 per crossed section. The 0.6 charge is waived at the first row with options after a timestamp gap over 30 seconds. Section costs still apply. Unassigned observations contribute zero and can occur anywhere. The chain may start and end anywhere on the supplied reference. Position fractions may fluctuate within an arc due to GPS jitter.

The recurrence maintains best scores ending at the same occurrence and prefix maxima over lower occurrences. Two Fenwick trees implement the prefix maxima with and without the per-skip penalty. A row publishes its updates only after all its options have been evaluated, allowing at most one match per observation. Backpointers recover the selected chain. For `M` spatially feasible observation/occurrence pairs and `R` reference rows, the sequence pass costs `O(M log(R + 1))` time and `O(M + R)` memory after proximity search. Equal cumulative scores prefer the earlier stored match in stable observation, distance and occurrence order.

### Evidence gates and failure modes

C requires a distance at most 20 m, a local alternative margin greater than 3 when a competitor is present, and a heading difference at most 75° where reliable. It abstains just after a large gap, on unreliable timestamp steps, and at flagged disconnected boundaries. Exact directed geometry shared by different road IDs is ambiguous, including duplicates outside the supplied reference. A zero-length arc can propose a location but has no direction or meaningful projection fraction.

To compute B and C alternatives, an anchor must be a selected match whose local cost beats **all** other nearby occurrences by more than 3, has positive standalone match reward, and is outside the immediate gap row. Only the nearest earlier and later such anchors bound which route occurrences are plausible. A run that keeps choosing one of two tied directions cannot supply its own corroborating anchors. Reliable movement can resolve opposite directions. A distinctive section in a loop can locate which traversal of a repeated arc is plausible.

The local margin is the cheapest remaining plausible alternative's local score minus the selected score. A negative margin means the sequence chose a locally more expensive match. A blank margin means the competitor set was empty within the search gate and bounds. Interpret the margin as a local, uncalibrated score conditional on the chosen sequence and corroborating anchors. Heading comes from adjacent fixes and can disagree with the arc at turns or on noisy segments. Geometry is projected straight between supplied endpoints. These choices can bias labels toward easier observations and discard difficult but valid matches.

## ML data contract and provenance

CSV blanks represent missing optional values, booleans are `True` and `False`, and `trip_id` must be read as text to retain leading zeroes. The observation key is `(partition, trip_id, observation_id)` and the Phase 2A candidate key adds `arc_id`. Observation IDs, arc IDs, route occurrence indices and section indices are zero-based. Partition exports are ordered by trip ID. Labels are grouped by method within a trip and then event order. Joins are grouped by observation and ordered by Phase 2A candidate rank.

### `labels.csv`

| Fields | Type | Meaning |
|---|---|---|
| `partition`, `trip_id`, `method` | string | Split, zero-padded trip ID and one of `nearest`, `monotonic`, `qualified` |
| `observation_id` | integer | Source GPS row |
| `arc_id`, `occurrence_index`, `section_index` | nullable integer | Proposed directed ID, specific occurrence and contiguous reference section |
| `distance_m`, `projection_fraction`, `heading_difference_deg` | nullable float | Projected metres, clamped directed fraction in [0,1], and circular heading difference in degrees |
| `local_cost`, `alternative_margin` | nullable float | Distance score for A, distance plus heading for B/C, and next-local-score margin |
| `plausible_alternatives`, `distance_ties` | integer | Count of viable competing occurrences and count within the projected-distance tie tolerance |
| `duplicate_directed_geometry`, `boundary_risk`, `gap_before` | boolean | Structural and temporal diagnostic flags |
| `motion_status`, `status`, `reason` | string | Phase 2A motion status, final qualification and its primary explanation |
| `provenance`, `policy_id` | string | `ordered_reference_proxy_v1` and SHA-256 of canonical sorted compact policy JSON |

Every method emits a row for every original GPS observation. The statuses are `accepted`, `ambiguous`, `unmatched`, `invalid_observation` and `disconnected_reference`. Proposed IDs can be present on an abstention. Invalid coordinates or non-finite current timestamps produce `invalid_observation`. A trace with no drawable nearby reference produces an unmatched observation before the qualification stage.

### Candidate labels and join audit

| File | Columns and types |
|---|---|
| `candidate_targets.csv` | `partition:str, trip_id:str, observation_id:int, arc_id:int, target:int, method:str, provenance:str, policy_id:str` |
| `candidate_join_status.csv` | `partition:str, trip_id:str, observation_id:int, outcome:str, reason:str, candidate_count:int, target_rank_within_radius:nullable int, radius_m:int, top_k:int, policy_id:str` |

The binary targets use C and the Phase 2A 50 m / Top-8 configuration. A group with exactly one present qualified directed target produces one positive and the existing competing candidates as negatives. An unqualified observation or a candidate miss produces zero binary rows and remains visible in the join audit. The audit separates empty groups, targets outside the radius and Top-K truncation. The rank is based on the uncapped radius search and so can exceed Top-K.

For model input, use the [Phase 2A feature schema](PHASE2A_HANDOFF.md), select only inference features, and join binary labels by the full candidate key with the same radius and cap. The candidate arc ID identifies the candidate. The target arc identity from alignment, reference occurrence, costs, status, reasons and provenance belong outside predictor columns. A different candidate configuration requires a fresh offline join via `label_candidates()` in `src/alignment/targets.py`.

## Measured results

The final representative run used training trips `00000026` (short), `00000083` (typical), `00000013` (largest road network), `00000019` (three reference breaks) and `00000054` (prolonged endpoint anomaly). The full runs covered 80 training trips and 10 student-testing trips. The sensitivity sweep changed only C's gates, with B's proposals and anchor bounds held fixed. At a 20 m gate, margins of 1, 3 and 5 accepted 64.26%, 58.47% and 54.64% of training observations. At margin 3, 15 m and 25 m accepted 56.68% and 59.12%. I retained 20 m / margin 3 as a conservative working point. Loosening the score margin recovered many more labels than widening the distance gate, with no independent pointwise labels available to verify those additional matches. The student-testing run loaded the byte-identical training policy JSON.

| Strategy | Training accepted / 208,417 | Training ambiguous | Student testing accepted / 19,160 | Student testing ambiguous |
|---|---:|---:|---:|---:|
| Nearest | 168,407 (80.80%) | 16,481 | 15,370 (80.22%) | 3,757 |
| Monotonic | 177,109 (84.98%) | 3,822 | 18,426 (96.17%) | 391 |
| Qualified | 121,855 (58.47%) | 56,230 | 13,335 (69.60%) | 5,213 |

Qualified abstention was 41.53% in training and 30.40% in student testing. Training had 30,082 unmatched and 250 disconnected-reference observations, while student testing had 607 and 5. The median per-trip qualified coverage was 66.04% for training and 68.32% for student testing. The observation-weighted median / 95th-percentile qualified distances were 4.43 / 13.55 m and 4.80 / 14.23 m. Nearest and monotonic differed on 4,566 of 162,686 jointly accepted training observations and 569 of 14,918 jointly accepted student observations. See the [training summary](../reports/phase2b/train/summary.md) and [student-testing summary](../reports/phase2b/student_test/summary.md) for per-trip distributions, distance quantiles and all agreement counts.

### Candidate inclusion

The qualified distance gate is 20 m in the same projection as candidate generation, and the smallest tested radius is 25 m. Thus, inclusion for accepted C labels was the same at 25, 50 and 100 m. All C misses at those radii were Top-K truncations. The report CSVs also include A and B, whose wider proposal gates produce radius-dependent results.

| Top-K | Training included / 121,855 | Training misses | Student testing included / 13,335 | Student testing misses |
|---|---:|---:|---:|---:|
| 3 | 119,374 (97.96%) | 2,481 | 12,997 (97.47%) | 338 |
| 8 | 121,721 (99.89%) | 134 | 13,325 (99.93%) | 10 |
| 16 | 121,822 (99.97%) | 33 | 13,335 (100%) | 0 |

Including every GPS row, empty candidate groups at 25, 50 and 100 m were 21,395, 18,229 and 1,346 for training and 223, 0 and 0 for student testing. This is a different denominator from proxy recall and makes the effect of the wider radius visible even when qualified pointwise recall is unchanged.

### Difficult examples

- `00000054` retained all 19,313 observation IDs. Through the endpoint approach, C accepted 1,378 of 2,703 fixes. In the remaining 16,610 fixes, A accepted 18, B accepted 4 and C accepted zero. The endpoint split at GPS row 2,702 comes from the Phase 2A diagnostic and is used only for reporting.
- `00000019` has three directed route breaks. The nearest method marked 20 observations as disconnected-reference. The monotonic and qualified methods marked 13 each. C accepted 1,693 of 2,354 GPS fixes away from its excluded or ambiguous groups.
- `00000069` contains 29 reference arc repeats. Nearest had 341 observations with distance ties, monotonic had 217 within its anchor bounds, and C accepted 1,194 of 2,067 fixes. Exact same-direction duplicate road rows are abstentions irrespective of reference membership. Opposite directions can be resolved with distinctive heading or order evidence. Tied stationary fixes and close parallel roads are covered by synthetic regression tests.
- Training had 133 timestamp gaps above 30 s. The explicit gap reason was assigned to 116 C abstentions. The other gap rows could already have a stronger reason, such as being too far from the route or having no proposal.

### Runtime and verification

On Darwin arm64 with Python 3.13.9, the training run took 82.35 s wall time and peaked at 317.14 MiB process RSS. Alignment took 37.04 s, including 23.33 s for shared proposal generation, 2.78 s for sparse sequence search, 8.67 s for baseline/sequence records and 1.99 s for qualification. The student-testing run took 6.93 s and peaked at 101.03 MiB, with 3.51 s spent on alignment. Timing and RSS are single-run observations, including Python and imported libraries in process memory.

All 56 tests passed, including all 40 existing tests and 16 Phase 2B tests. Coverage includes brute-force agreement with the sparse sequence objective, known synthetic assignments and qualified-only candidate joins. A full repeat reproduced both partitions' deterministic reports and exports byte for byte. SHA-256 hashes matched for all 360 raw files. Phase 1 and Phase 2A implementations, tests and reports were checked against base commit `515c9c0`.

The training export contains 650,978 candidate rows across 121,721 groups. Inspection verified exactly one matching positive per group. All 134 candidate misses and 86,562 unqualified training observations appeared in the join audit without binary target rows. Candidate IDs and ordering on trips `00000019`, `00000054` and `00000083` matched the existing Phase 2A inference entry point.

## Phase 3 readiness

The qualified training groups can support five proxy-supervised classifiers with trip-grouped validation. Report the fraction of all original observations eligible for pointwise evaluation, the candidate recall against qualified references, and classifier agreement within those groups. Evaluate held-out trips using the frozen alignment policy and report per-trip metrics alongside observation-weighted totals. Keep route-level evaluation against the ordered reference separate from the pointwise proxy comparison. Coverage and ambiguity warrant explicit reporting, particularly on trips with sparse qualified supervision. Ask the organisers to confirm the expected protocol for pointwise metrics if independent GPS-to-arc labels are available or required.

## Files

- `src/alignment/__init__.py`, `src/alignment/__main__.py`, `src/alignment/core.py`, `src/alignment/sequence.py`, `src/alignment/targets.py`, `src/alignment/report.py`
- `tests/test_alignment.py`, `docs/PHASE2B_HANDOFF.md`, `README.md`, `.gitignore`
- Each partition report directory contains `summary.md`, `policy.json`, `trip_coverage.csv`, `method_comparison.csv`, `method_comparison_by_trip.csv`, `candidate_recall.csv`, `candidate_recall_by_trip.csv`, `runtime.json`, and `runtime_by_trip.csv`.
- Training also contains `training_sensitivity.csv` and `difficult_cases.csv`. Local training plots are `alignment_00000019.png` and `alignment_00000054.png`.
