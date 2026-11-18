# Phase 2B proxy-label alignment · train

These measurements describe supplied-reference proxy labels and candidate inclusion.

Trips analysed: 80.

| Method | Accepted / observations | Coverage | Ambiguous | Unmatched | Invalid | Disconnected | Abstention | Per-trip coverage p10 / p50 / p90 | Accepted offset p50 / p95 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 168,407 / 208,417 | 80.80% | 16,481 | 23,140 | 0 | 389 | 19.20% | 65.48% / 94.31% / 98.46% | 4.79 / 16.70 |
| monotonic | 177,109 / 208,417 | 84.98% | 3,822 | 27,236 | 0 | 250 | 15.02% | 89.62% / 96.43% / 98.77% | 4.76 / 14.93 |
| qualified | 121,855 / 208,417 | 58.47% | 56,230 | 30,082 | 0 | 250 | 41.53% | 51.35% / 66.04% / 83.00% | 4.43 / 13.55 |

## Agreement across jointly accepted observations

| Methods | Jointly accepted | Same directed arc | Different directed arc | Same route occurrence |
|---|---:|---:|---:|---:|
| nearest / monotonic | 162,686 | 158,120 | 4,566 | 158,120 |
| nearest / qualified | 113,844 | 112,646 | 1,198 | 112,646 |
| monotonic / qualified | 121,855 | 121,855 | 0 | 121,855 |

## Qualified proxy recall

| Radius (m) | Top-K | Included / qualified | Recall | Miss rate | Per-trip recall p10 / p50 / p90 |
|---:|---:|---:|---:|---:|---:|
| 25 | 3 | 119,374 / 121,855 | 97.96% | 2.04% | 94.55% / 97.84% / 99.59% |
| 25 | 8 | 121,721 / 121,855 | 99.89% | 0.11% | 99.59% / 99.99% / 100.00% |
| 25 | 16 | 121,822 / 121,855 | 99.97% | 0.03% | 100.00% / 100.00% / 100.00% |
| 50 | 3 | 119,374 / 121,855 | 97.96% | 2.04% | 94.55% / 97.84% / 99.59% |
| 50 | 8 | 121,721 / 121,855 | 99.89% | 0.11% | 99.59% / 99.99% / 100.00% |
| 50 | 16 | 121,822 / 121,855 | 99.97% | 0.03% | 100.00% / 100.00% / 100.00% |
| 100 | 3 | 119,374 / 121,855 | 97.96% | 2.04% | 94.55% / 97.84% / 99.59% |
| 100 | 8 | 121,721 / 121,855 | 99.89% | 0.11% | 99.59% / 99.99% / 100.00% |
| 100 | 16 | 121,822 / 121,855 | 99.97% | 0.03% | 100.00% / 100.00% / 100.00% |

## Qualified abstention reasons

- `duplicate_directed_geometry` — 22
- `heading_disagreement` — 31,538
- `large_timestamp_gap` — 116
- `small_sequence_feasible_margin` — 24,554
- `reference_section_boundary` — 250
- `distance_above_qualification_limit` — 2,846
- `outside_proposal_radius` — 23,140
- `sequence_skipped` — 4,096

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
