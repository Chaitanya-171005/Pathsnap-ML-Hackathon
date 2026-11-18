# Phase 1 EDA findings

## Scope and measured results

The Phase 1 run analysed 360 files across 90 trips. It read the two student-side partitions separately. See the [generated summary](../reports/eda/summary.md) for the complete counts and threshold settings.

| Measurement | Training | Student testing |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Road nodes | 1,933,485 | 90,689 |
| Directed arcs | 3,876,904 | 185,041 |
| Reference arc rows | 31,336 | 3,477 |
| Timestamp decreases / duplicate timestamp rows | 0 / 0 | 0 / 0 |
| Exactly 1 s among positive intervals | 201,048 / 208,337 (96.50%) | 18,886 / 19,150 (98.62%) |
| Gaps over 60 s | 69 | 1 |
| Largest gap | 153,064 s, trip `00000090` | 210 s, trip `00000065` |
| Speed flags over 55 m/s | 981 | 4 |
| Duplicate directed edge rows beyond first occurrence | 1,775 | 128 |
| Self-loops / zero-length arc geometries | 22 / 100 | 2 / 6 |
| Isolated nodes | 0 | 0 |
| Directed reference breaks | 13 | 1 |
| GPS-to-reference offset over 100 m | 19,992 | 3 |

All four files were present for every trip. The run found no missing or invalid GPS coordinates, invalid timestamp values, invalid node coordinates, invalid endpoint IDs or invalid reference arc IDs. Six consecutive training observation pairs have identical positions and different timestamps. Across both partitions every reference sequence is shorter than its GPS track. See the [data contract](DATA_CONTRACT.md) for the implications for labels.

The 153,064-second step is approximately 42.5 hours. GPS duration is measured as the span from minimum to maximum observed timestamp and includes gaps. Approximate GPS polyline length connects consecutive valid source rows, including points separated by a long gap. The 55 m/s threshold is a diagnostic, and individual flags can reflect noise, observation gaps or actual fast travel. Training trip `00000032` contributes 642 speed flags. Reporting trip-level as well as pooled diagnostics helps distinguish a concentrated issue from a partition-wide pattern.

Road graphs include every node row and all endpoint-valid directed arcs. Sparse adjacency is used for connectivity; the component-size CSV columns contain JSON maps from component size to component count. A large strongly connected component count can reflect legal one-way structure as well as disconnected regions. Duplicate directed edge rows retain their source arc IDs.

Three training maps show typical, short and anomaly examples. The anomaly selection prioritises directed reference breaks, then long GPS gaps, speed flags and large offsets. This selects `00000019`, which has the most directed breaks. Road layers draw a deterministic sample when the arc count exceeds the rendering cap. The GPS and reference layers use all valid supplied geometry for the selected trip. Direct arc lengths and their displayed geometry are endpoint-to-endpoint approximations because the supplied files contain no intermediate shape vertices.

## Provenance

The five local organiser PDFs informed dataset semantics, evaluation separation and reproducibility expectations. The figures and statistics were produced by running the repository pipeline on the supplied student-side data. Codex assisted with implementation and documentation. Team members should reproduce the outputs, review the interpretation and disclose assistance under the organiser's assessment and deliverables policies.
