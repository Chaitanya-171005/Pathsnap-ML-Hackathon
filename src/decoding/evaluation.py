"""Offline-only Phase 2B proxy and supplied-route evaluation."""

from collections import Counter
from itertools import pairwise

import numpy as np

from src.alignment.core import align_trip
from src.alignment.targets import targets_for_arc_ids
from src.preprocessing.geometry import haversine_m
from src.preprocessing.loaders import Trip, load_route


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def observation_metrics(result, candidate_ids, labels):
    counts = Counter()
    for match, candidates, label in zip(result["matches"], candidate_ids, labels, strict=True):
        key = (label.partition, label.trip_id, label.observation_id)
        if key != (match["partition"], match["trip_id"], match["observation_id"]):
            raise ValueError("Proxy label and observation keys differ")
        counts["original"] += 1
        counts["matched"] += match["arc_id"] is not None
        counts["with_candidates"] += bool(candidates)
        outcome, _ = targets_for_arc_ids(label, candidates)
        counts[outcome] += 1
        if outcome == "included":
            counts["classifier_correct"] += match["classifier_arc_id"] == label.arc_id
            counts["hmm_correct"] += match["arc_id"] == label.arc_id
    return {
        **{key: counts[key] for key in ("original", "matched", "with_candidates",
                                       "classifier_correct", "hmm_correct")},
        "qualified": counts["included"] + counts["candidate_miss"],
        "eligible": counts["included"], "candidate_misses": counts["candidate_miss"],
    }


def add_rates(row):
    row["match_coverage"] = ratio(row["matched"], row["original"])
    row["candidate_coverage"] = ratio(row["with_candidates"], row["original"])
    row["eligible_coverage"] = ratio(row["eligible"], row["original"])
    row["qualified_candidate_recall"] = ratio(row["eligible"], row["qualified"])
    row["classifier_top1"] = ratio(row["classifier_correct"], row["eligible"])
    row["hmm_top1"] = ratio(row["hmm_correct"], row["eligible"])
    row["correct_delta"] = row["hmm_correct"] - row["classifier_correct"]
    row["top1_delta_pp"] = ratio(100 * row["correct_delta"], row["eligible"])
    return row


def route_metrics(result, graph, reference):
    segments = result["route_segments"]
    predicted = [arc_id for segment in segments for arc_id in segment["arc_ids"]]
    arcs = graph.arcs
    reference = list(map(int, reference))
    valid = [arc_id for arc_id in reference if arc_id in arcs]
    overlap = set(predicted) & set(valid)
    predicted_pairs = Counter(pair for s in segments for pair in pairwise(s["arc_ids"]))
    reference_pairs = Counter()
    disconnected = uncheckable = 0
    for a, b in pairwise(reference):
        if a not in arcs or b not in arcs:
            uncheckable += 1
        elif arcs[a].end != arcs[b].start:
            disconnected += 1
        else:
            reference_pairs[a, b] += 1
    internal_disconnections = sum(arcs[a].end != arcs[b].start for s in segments
                                  for a, b in pairwise(s["arc_ids"]))
    baseline_ids = []
    baseline_pairs = Counter()
    baseline_disconnections = 0
    previous = None
    for match in result["matches"]:
        arc_id = match["classifier_arc_id"]
        if arc_id not in arcs:
            previous = None
            continue
        if arc_id == previous:
            continue
        baseline_ids.append(arc_id)
        if previous is not None:
            if arcs[previous].end == arcs[arc_id].start:
                baseline_pairs[previous, arc_id] += 1
            else:
                baseline_disconnections += 1
        previous = arc_id
    baseline_unique = set(baseline_ids)
    baseline_overlap = len(baseline_unique & set(valid))
    predicted_length = sum(arcs[arc_id].length_m for arc_id in predicted)
    reference_length = sum(arcs[arc_id].length_m for arc_id in valid)
    return {
        "route_status": result["status"], "segments": len(segments),
        "predicted_arc_occurrences": len(predicted), "reference_arc_occurrences": len(reference),
        "reference_invalid_occurrences": len(reference) - len(valid),
        "predicted_unique_arcs": len(set(predicted)), "reference_valid_unique_arcs": len(set(valid)),
        "overlap_unique_arcs": len(overlap), "arc_precision": ratio(len(overlap), len(set(predicted))),
        "arc_recall": ratio(len(overlap), len(set(valid))),
        "directed_bigram_f1": ratio(2 * (predicted_pairs & reference_pairs).total(),
                                    predicted_pairs.total() + reference_pairs.total()),
        "predicted_segment_breaks": max(0, len(segments) - 1),
        "predicted_internal_disconnections": internal_disconnections,
        "no_feasible_transitions": sum(result["break_counts"].get(reason, 0) for reason in
                                       ("no_feasible_transition", "movement_limit",
                                        "routing_budget_exceeded")),
        "reference_disconnected_transitions": disconnected,
        "reference_uncheckable_transitions": uncheckable,
        "predicted_full_arc_length_m": predicted_length,
        "predicted_matched_distance_m": sum(s["matched_distance_m"] for s in segments),
        "reference_valid_arc_length_m": reference_length,
        "full_arc_length_delta_m": predicted_length - reference_length,
        "classifier_arc_precision": ratio(baseline_overlap, len(baseline_unique)),
        "classifier_arc_recall": ratio(baseline_overlap, len(set(valid))),
        "classifier_directed_bigram_f1": ratio(2 * (baseline_pairs & reference_pairs).total(),
                                              baseline_pairs.total() + reference_pairs.total()),
        "classifier_disconnected_transitions": baseline_disconnections,
    }


def evaluate_trip(files, trip, run, policy):
    reference, issues = load_route(files.path("route"))
    if any(issue.code in ("missing_file", "unreadable_file") for issue in issues):
        raise ValueError(f"Reference route unavailable for {files.partition}/{files.trip_id}")
    source = Trip(files, trip.track, trip.nodes, trip.arcs, reference, list(trip.issues) + issues)
    labels = align_trip(source, policy)["qualified"]
    key = dict(partition=trip.partition, trip_id=trip.trip_id)
    observations = add_rates({**key, **observation_metrics(run.result, run.candidate_ids, labels)})
    route = {**key, **route_metrics(run.result, run.graph, reference)}
    difficult_details = {"break_counts": run.result["break_counts"]}
    if (trip.partition, trip.trip_id) == ("train", "00000054") and len(reference):
        end_arc = run.graph.arcs.get(int(reference[-1]))
        if end_arc is not None and len(trip.track):
            # Match Phase 2A's offline endpoint-based tail definition.
            distance = haversine_m(trip.track[:, :2], trip.nodes[end_arc.end])
            if np.isfinite(distance).any():
                tail_start = int(np.nanargmin(distance)) + 1
                tail = run.result["matches"][tail_start:]
                difficult_details.update(
                    tail_start_observation_id=tail_start, tail_observations=len(tail),
                    tail_no_candidates=sum(m["candidate_count"] == 0 for m in tail),
                    tail_matched=sum(m["arc_id"] is not None for m in tail),
                    tail_matches_without_candidates=sum(
                        m["arc_id"] is not None and m["candidate_count"] == 0 for m in tail),
                )
    return observations, route, difficult_details
