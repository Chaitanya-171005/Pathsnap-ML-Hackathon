# PathSnap Phase 5 handoff

Phase 5 uses the existing 50 m radius, Top-8 candidates and 15-feature classifier pipeline. The fixed Phase 3 Histogram Gradient Boosting artifact is the default. Phase 4 can be selected explicitly. The inference loader opens the GPS track, nodes and arcs. Offline evaluation imports the reference loader and Phase 2B qualification separately.

## Run decoding

```bash
source .venv/bin/activate
python -m src.decoding --partition train --trip-id 00000026 --model phase3
```

Pass several IDs after `--trip-id` to decode a selection, or use `--all-trips` for the partition. The CLI streams trips sequentially. `--data-root`, `--output-dir`, `--model`, `--model-path` and `--batch-size` control inputs and outputs. The decoder options are `--distance-scale-m`, `--transition-weight`, `--position-slack-m`, `--restart-cost`, `--routing-budget` and `--routing-cache-size`. Use only trusted local joblib artifacts. See [the output contract](PHASE5_OUTPUT_CONTRACT.md) for the schema and Folium/FastAPI integration.

To rerun the measured training checks and the reused ten-trip development holdout with the frozen defaults:

```bash
python -m src.decoding --partition train \
  --trip-id 00000019 00000026 00000054 00000083 \
  --evaluate --student-holdout
python -m pytest -q
```

`--evaluate` uses the frozen Phase 2B policy, verifies its identity against the classifier manifest, and writes aggregate reports under `reports/phase5/` by default. `--student-holdout` writes the configuration and model hash before opening the student-testing inputs, then checks the original, qualified and eligible denominators against Phase 3. Only the two student development partitions are accepted.

## Directed graph and inference

The state at a GPS fix is a directed arc ID and its projected fraction along that arc. The road graph stores outgoing original arc IDs per node and uses Haversine endpoint lengths. Reverse and parallel arcs retain separate identities. A zero-length self-loop or invalid geometry cannot supply a usable directed fraction or connector. Positive-length directed cycles remain routable.

For state $i$ at observation $t$, the emission is the log of the clipped positive-class probability:

$$e_t(i) = \log\left(\operatorname{clip}(p_t(i), 10^{-12}, 1-10^{-12})\right).$$

This monotone transform uses raw per-candidate evidence, with no across-candidate normalization or learned calibration. The HMM excludes invalid state evidence. The existing classifier API raises an error for a malformed probability batch.

Let $g$ be the Haversine displacement between adjacent GPS fixes, $\Delta t$ the elapsed seconds, and $d$ the candidate-to-candidate road distance. For a forward move on one arc, $d$ is the arc length times the change in projected fraction. Otherwise, $d$ is the remaining length on the first arc, the directed connector length, and the traversed fraction of the next arc. Moving backward on a directed arc requires a real directed cycle. Dijkstra stops when the target is settled or the remaining distance bound or work budget runs out. Its cache stores settled exact paths and proven radius failures in a trip-local LRU. Interrupted searches are recorded and left out of the cache. The default budget permits 4,096 queue pops and outgoing-arc scans combined per search, and the cache holds at most 4,096 ordered node pairs.

The feasible total distance is bounded by $v_{\max}\Delta t + s$. With $v_{\max}=55$ m/s and position slack $s=15$ m, the transition score is:

$$a_t(j,i)=-\frac{w}{b}\left(\lvert d-g\rvert+\max(0,d-v_{\max}\Delta t)\right),\qquad
w=0.5,\quad b=25\;\mathrm{m}.$$

For a connected step, Viterbi uses $V_t(i)=e_t(i)+\max_j[V_{t-1}(j)+a_t(j,i)]$. This is a log-space score decoder with unnormalized transition potentials. If a particular state pair has no feasible directed link, its alternative is a restart at cost 4 log-score units. A winning restart ends the preceding route segment and starts another without connector geometry. If every pair fails, the previous chain finishes and the next fix starts a fresh chain. Ties prefer a feasible link over an equally scored restart, then use the stable order of distance rank and arc ID within each class. Shortest path ties are deterministic under node heap and original arc ordering.

Empty or invalid-GPS rows terminate the chain and stay unmatched. Missing timestamps, nonpositive deltas, GPS gaps over 30 seconds, and observed speeds over 55 m/s terminate continuity. A candidate with invalid time may be emitted as an isolated match. No reference alignment or label is consulted in these decisions. Selected transitions retain the exact connector arcs, which are inserted in route order. Consecutive forward fixes on the same arc merge into one occurrence. Actual repeats through cycles and separately decoded segments stay visible.

## Measuring quality and performance

[Measured Phase 5 results](../reports/phase5/summary.md) include pooled and per-trip Top-1, route metrics, difficult-case checks and stage timings. Training checks use a classifier fitted on those trips. The ten student trips were evaluated with the fixed decoder configuration as a reused development holdout.

Paired Top-1 uses qualified Phase 2B observation labels present in Top-8. Qualified candidate misses are separate. Abstentions on eligible observations count as incorrect for HMM. Coverage is measured against all original fixes. Route precision and recall use unique directed arc IDs. Sequence agreement is occurrence-count F1 over directed adjacent arc pairs, excluding boundaries between decoded segments and invalid or disconnected reference pairs. The raw classifier-only route diagnostic collapses consecutive repeated Top-1 arc IDs, adds no connector arcs and counts non-adjacent jumps between available consecutive choices. Its bigram score counts connected arc pairs. Lengths are occurrence-based and include full selected and connector arcs. The separate snapped-distance field measures travel between the projected positions inside each decoded segment.

Runtime separates raw reads, spatial indexing and candidate generation, classifier scoring, graph construction, routing and transition scoring, Viterbi, reconstruction and offline evaluation. The per-trip RSS values are process high-water marks, so later rows may include peaks from earlier trips. The total inference time includes bookkeeping outside the individual stages.

Projected GPS jitter, near-parallel roads and short sampling intervals can force explicit restarts or create route detours. Full-arc geometry can overstate the length of highly fragmented trips. Search exhaustion and reference discontinuities are reported explicitly. Decoding never creates a link between disconnected segments. See the per-trip reports for cases where pointwise agreement and route-level sequence agreement move in different directions.
