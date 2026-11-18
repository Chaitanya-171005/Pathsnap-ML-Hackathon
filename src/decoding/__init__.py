"""PathSnap Phase 5 inference API. Reference routes belong to offline evaluation."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from time import perf_counter

from src.candidates.pipeline import CandidateConfig, candidate_batches
from src.candidates.spatial import RoadIndex
from src.models.inference import load_model, predict_trip
from src.preprocessing.geometry import haversine_m
from src.preprocessing.loaders import PARTITIONS

from .graph import RoadGraph
from .hmm import DecoderConfig, State, emission, transition
from .reconstruct import reconstruct
from .viterbi import Viterbi


MODELS = {
    phase: Path("artifacts/models") / phase / "gradient_boosting.joblib"
    for phase in ("phase3", "phase4")
}
DEFAULT_MODEL = "phase3"
CANDIDATES = CandidateConfig()


@dataclass(slots=True)
class DecodeRun:
    result: dict
    candidate_ids: list[tuple[int, ...]]
    graph: RoadGraph
    timings: dict


def model_bundle(model=DEFAULT_MODEL, path=None):
    path = Path(path) if path is not None else MODELS[model]
    bundle = load_model(path)
    if (bundle["model_id"] != "gradient_boosting"
            or bundle["manifest"].get("candidate_config") != asdict(CANDIDATES)):
        raise ValueError("Artifact is incompatible with the fixed Phase 5 candidate contract")
    return bundle


def _groups(trip, index, bundle, batch_size, timings):
    """Score Phase 2A batches with their original projection fractions in scope."""
    pending = []
    total = 0
    source = iter(candidate_batches(trip, CANDIDATES, index=index))
    while True:
        start = perf_counter()
        item = next(source, None)
        timings["candidates_s"] += perf_counter() - start
        if item is not None:
            pending.append(item)
            total += max(1, len(item[1]))
            if total < batch_size:
                continue
        if pending:
            start = perf_counter()
            scored = list(predict_trip(bundle["pipeline"], bundle["model_id"], pending, batch_size))
            timings["classifier_s"] += perf_counter() - start
            for (observation, rows), (_, predictions) in zip(pending, scored, strict=True):
                yield observation, rows, predictions
            pending.clear()
            total = 0
        if item is None:
            return


def _states(rows, predictions, graph):
    fractions = {row.arc_id: row.projection_fraction for row in rows
                 if row.projection_valid and 0 <= row.projection_fraction <= 1}
    states = []
    for prediction in sorted(predictions, key=lambda item: (item.candidate_rank, item.arc_id)):
        score = emission(prediction.probability)
        if (score is not None and prediction.arc_id in graph.arcs
                and prediction.arc_id in fractions):
            states.append(State(prediction.arc_id, prediction.candidate_rank,
                                prediction.probability, fractions[prediction.arc_id], score))
    return states


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    scale = 1024 ** (2 if sys.platform == "darwin" else 1)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def decode_trip(trip, bundle, *, model=DEFAULT_MODEL, config=DecoderConfig(), batch_size=8192):
    """Decode one loaded inference trip. The returned result is JSON-ready."""
    if trip.partition not in PARTITIONS:
        raise ValueError("Only train and student_test are supported in Phase 5")
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    started = perf_counter()
    timings = dict(candidates_s=0.0, classifier_s=0.0, routing_s=0.0, viterbi_s=0.0)
    index = RoadIndex(trip.nodes, trip.arcs)
    timings["candidates_s"] += perf_counter() - started
    graph_started = perf_counter()
    graph = RoadGraph(trip.nodes, index, search_budget=config.search_budget,
                      cache_size=config.cache_size)
    timings["graph_s"] = perf_counter() - graph_started
    viterbi = Viterbi()
    matches, candidate_ids, decoded = [], [], []
    segment_reason = pending_break = None

    def finish_segment():
        start = perf_counter()
        choices = viterbi.finish()
        timings["viterbi_s"] += perf_counter() - start
        for i, (_, _, _, reason) in enumerate(choices):
            if i == 0 or reason is not None:
                decoded.append((segment_reason if i == 0 else reason, []))
            decoded[-1][1].append(choices[i][:3])

    for observation, rows, predictions in _groups(trip, index, bundle, batch_size, timings):
        obs_id = observation.observation_id
        candidate_ids.append(tuple(p.arc_id for p in predictions))
        match = {
            "partition": trip.partition, "trip_id": trip.trip_id, "observation_id": obs_id,
            "arc_id": None, "candidate_rank": None, "model_score": None, "segment_id": None,
            "classifier_arc_id": predictions[0].arc_id if predictions else None,
            "candidate_count": len(rows), "status": "unmatched", "break_reason": None,
        }
        matches.append(match)
        states = _states(rows, predictions, graph) if observation.gps_valid else []
        if not states:
            finish_segment()
            reason = observation.candidate_status if not rows else "no_usable_states"
            pending_break = match["break_reason"] = reason
            continue

        motion = observation.motion_status
        reason = pending_break
        if motion in ("invalid_timestamp_pair", "nonpositive_time_step", "gps_gap",
                      "invalid_gps_pair", "speed_above_limit"):
            reason = reason or motion
        if not observation.timestamp_valid:
            reason = "invalid_timestamp"
        if reason:
            finish_segment()

        if viterbi.columns:
            route_started = perf_counter()
            delta_s = observation.delta_t_s
            observed_m = float(haversine_m(trip.track[obs_id - 1, :2], trip.track[obs_id, :2]))
            links, resets = [], []
            for i, first in viterbi.active():
                for j, second in enumerate(states):
                    link, failure = transition(graph, first, second, observed_m, delta_s,
                                               CANDIDATES.max_speed_mps, config)
                    if link is None:
                        if failure == "no_directed_path_within_limit":
                            failure = "no_feasible_transition"
                        resets.append((i, j, failure))
                    else:
                        links.append((i, j, link))
            timings["routing_s"] += perf_counter() - route_started
            start = perf_counter()
            connected = viterbi.advance(obs_id, states, links, resets, config.restart_cost)
            timings["viterbi_s"] += perf_counter() - start
            if connected:
                continue
            finish_segment()
            failures = {failure for _, _, failure in resets}
            if "routing_budget_exceeded" in failures:
                reason = "routing_budget_exceeded"
            elif failures == {"movement_limit"}:
                reason = "movement_limit"
            else:
                reason = "no_feasible_transition"
        segment_reason = reason
        pending_break = None
        start = perf_counter()
        viterbi.start(obs_id, states)
        timings["viterbi_s"] += perf_counter() - start

    finish_segment()
    start = perf_counter()
    for segment_id, (reason, choices) in enumerate(decoded):
        for offset, (obs_id, state, _) in enumerate(choices):
            matches[obs_id].update(arc_id=state.arc_id, candidate_rank=state.rank,
                                   model_score=state.probability, segment_id=segment_id,
                                   status="matched", break_reason=None if offset else reason)
    model_id = f"{model}/{bundle['model_id']}"
    segments, geojson = reconstruct(trip, model_id, graph, decoded)
    status = ("empty" if not segments else "disconnected" if len(segments) > 1 else
              "partial" if len(decoded[0][1]) != len(matches) else "connected")
    result = {
        "schema_version": "1.0", "partition": trip.partition, "trip_id": trip.trip_id,
        "model_id": model_id, "observation_count": len(matches), "status": status,
        "decoder_config": asdict(config), "matches": matches, "route_segments": segments,
        "route_geojson": geojson,
        "break_counts": dict(Counter(m["break_reason"] for m in matches if m["break_reason"])),
        "routing": {"searches": graph.searches, "cache_hits": graph.cache_hits,
                    "budget_exhaustions": graph.budget_exhaustions},
    }
    timings["reconstruction_s"] = perf_counter() - start
    timings["inference_s"] = perf_counter() - started
    timings["peak_rss_mib"] = peak_rss_mib()
    result["processing_time_s"] = timings["inference_s"]
    return DecodeRun(result, candidate_ids, graph, timings)
