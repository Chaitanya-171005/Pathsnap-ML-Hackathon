# Data contract for candidate generation and route reconstruction

## Inputs and identifiers

Use the `Trip` structure from `src.preprocessing.loaders` plus the masks from `validate_trip`. Keep `(partition, trip_id, observation_id)` on observation rows and `(partition, trip_id, arc_id)` on road arcs. Arc endpoints are ordered. Parallel or duplicate rows retain distinct arc IDs. Use `arc_geometry` to identify drawable and spatially searchable arcs. Use `arcs` for endpoint-valid topology. Keep the input row ID when filtering any derived array. Treat a rejected GPS row as an explicit missing observation when planning downstream interfaces.

For future inference, the input consists of GPS observations, nodes and arcs. Reference routes belong in training-label derivation and evaluation. Fit preprocessing and cross-validation on training trips only, with groups defined by trip. Use student testing for development evaluation. Keep evaluator data outside development and evaluator references on the evaluation side.

## What the reference route establishes

The `.route` file gives an ordered traversal of trip-local directed arc IDs. It gives neither a timestamp for entering or leaving an arc nor an observation-to-arc association. All 90 trips have more GPS rows than reference rows. At trip scale, several GPS points can fall along one arc, and a sampling gap can span many arcs. Repeated arcs can represent revisits, and a nearby segment can be parallel to the travelled segment or coincide with its reverse direction.

| Reference diagnostic | Training | Student testing |
|---|---:|---:|
| Trips with equal GPS and reference row counts | 0/80 | 0/10 |
| Fully checkable directed paths with geometry | 70/80 | 9/10 |
| Directed breaks between adjacent reference rows | 13 across 10 trips | 1 across 1 trip |
| Reference arc repeats beyond first occurrence | 67 | 255 |
| GPS points over 30 m from every reference segment | 25,381 / 208,417 | 142 / 19,160 |
| GPS points over 100 m from every reference segment | 19,992 / 208,417 | 3 / 19,160 |

Every supplied reference ID resolves to an arc. Each reference arc is individually drawable. A break means the first arc's destination node ID differs from the next arc's origin node ID. This flags a discontinuity in the supplied sequence. It does not establish why the break exists or whether there is a plausible intermediate path. Training trip `00000019` has three such breaks. In training trip `00000054`, 16,156 of 19,313 GPS observations are over 100 m from every reference segment. That concentration is a reason to inspect the trip's alignment before treating the reference route as dense supervision.

The spatial diagnostic takes the shortest unsigned distance to any supplied valid reference segment. It ignores sequence position, heading and time, and therefore provides a measure of spatial agreement only. Arc lengths use a haversine sphere with mean radius 6,371,008.8 m and endpoint-to-endpoint chords on the road network. GPS-to-route offsets use straight segments in a local equirectangular approximation centred on each GPS point. These are suitable diagnostics at local road scales. Curved roads, global distances, polar regions and longitude-wrap bounds require additional handling for more precise use.

## Phase 2 alignment options to evaluate

| Strategy | What it uses | Where it is ambiguous |
|---|---|---|
| Closest point on reference geometry with a distance gate | Per-point spatial proximity | Crossings, parallel roads, reversed arcs and repeated segments can tie or be wrong |
| Monotonic sequence alignment over reference positions | GPS order, ordered arc positions, distance and optionally heading | Loops, reference discontinuities and long gaps can leave several feasible alignments. Forcing one match per point can create false precision |
| Anchors with abstention | High-confidence proximity and directional agreement | Coverage can be low and favour easier observations. Report coverage and confidence separately from accuracy |
| Route-level metrics | The intact ordered reference and a predicted route | Measures a different target from pointwise accuracy and candidate miss rate |

Tune any alignment or abstention rule on permitted training folds. Track label coverage, ambiguity, unmatched observations and alignment provenance before interpreting Top-1 accuracy or log loss. Measure candidate recall on qualified labels while separately reporting how many observations could be labelled. Ask the organisers whether an observation-level reference mapping exists or whether a disclosed alignment proxy is acceptable for the required pointwise metrics.

Sources: organiser's *Dataset Exploration* and *Models and Training Guidelines*, plus the [generated summary](../reports/eda/summary.md) and [local route validation CSV](../reports/eda/reference_routes.csv).
