# Phase 2B proxy-label alignment · student_test

These measurements describe supplied-reference proxy labels and candidate inclusion.

Trips analysed: 10.

| Method | Accepted / observations | Coverage | Ambiguous | Unmatched | Invalid | Disconnected | Abstention | Per-trip coverage p10 / p50 / p90 | Accepted offset p50 / p95 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 15,370 / 19,160 | 80.22% | 3,757 | 28 | 0 | 5 | 19.78% | 66.48% / 94.72% / 98.88% | 5.12 / 17.05 |
| monotonic | 18,426 / 19,160 | 96.17% | 391 | 338 | 0 | 5 | 3.83% | 93.43% / 96.20% / 98.90% | 5.13 / 15.48 |
| qualified | 13,335 / 19,160 | 69.60% | 5,213 | 607 | 0 | 5 | 30.40% | 55.85% / 68.32% / 82.60% | 4.80 / 14.23 |

## Agreement across jointly accepted observations

| Methods | Jointly accepted | Same directed arc | Different directed arc | Same route occurrence |
|---|---:|---:|---:|---:|
| nearest / monotonic | 14,918 | 14,349 | 569 | 14,349 |
| nearest / qualified | 11,086 | 10,892 | 194 | 10,892 |
| monotonic / qualified | 13,335 | 13,335 | 0 | 13,335 |

## Qualified proxy recall

| Radius (m) | Top-K | Included / qualified | Recall | Miss rate | Per-trip recall p10 / p50 / p90 |
|---:|---:|---:|---:|---:|---:|
| 25 | 3 | 12,997 / 13,335 | 97.47% | 2.53% | 94.88% / 97.09% / 98.59% |
| 25 | 8 | 13,325 / 13,335 | 99.93% | 0.07% | 99.75% / 100.00% / 100.00% |
| 25 | 16 | 13,335 / 13,335 | 100.00% | 0.00% | 100.00% / 100.00% / 100.00% |
| 50 | 3 | 12,997 / 13,335 | 97.47% | 2.53% | 94.88% / 97.09% / 98.59% |
| 50 | 8 | 13,325 / 13,335 | 99.93% | 0.07% | 99.75% / 100.00% / 100.00% |
| 50 | 16 | 13,335 / 13,335 | 100.00% | 0.00% | 100.00% / 100.00% / 100.00% |
| 100 | 3 | 12,997 / 13,335 | 97.47% | 2.53% | 94.88% / 97.09% / 98.59% |
| 100 | 8 | 13,325 / 13,335 | 99.93% | 0.07% | 99.75% / 100.00% / 100.00% |
| 100 | 16 | 13,335 / 13,335 | 100.00% | 0.00% | 100.00% / 100.00% / 100.00% |

## Qualified abstention reasons

- `duplicate_directed_geometry` — 1
- `heading_disagreement` — 2,347
- `large_timestamp_gap` — 2
- `small_sequence_feasible_margin` — 2,863
- `reference_section_boundary` — 5
- `distance_above_qualification_limit` — 269
- `outside_proposal_radius` — 28
- `sequence_skipped` — 310

## Policy

```json
{
  "proposal_radius_m": 45.0,
  "qualified_distance_m": 20.0,
  "tie_margin_m": 0.05,
  "anchor_margin": 3.0,
  "qualified_margin": 3.0,
  "max_heading_difference_deg": 75.0,
  "heading_penalty": 18.0,
  "match_reward": 32.0,
  "skip_cost": 0.6,
  "break_cost": 12.0,
  "break_buffer_m": 12.0,
  "max_gap_s": 30.0,
  "max_speed_mps": 55.0,
  "min_heading_step_m": 2.0
}
```
