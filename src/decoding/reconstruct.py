"""Build segmented directed arc occurrences and compact GeoJSON."""

from itertools import chain


def reconstruct(trip, model_id, graph, decoded):
    segments = []
    features = []
    for segment_id, (reason, choices) in enumerate(decoded):
        occurrences = []
        previous = None
        distance = 0.0
        for observation_id, state, link in choices:
            if link is not None:
                distance += link.path.distance_m
                if link.same_arc_forward:
                    previous["last_observation_id"] = observation_id
                    continue
                occurrences.extend({"arc_id": arc_id, "role": "connector"}
                                   for arc_id in link.path.arcs)
            previous = {
                "arc_id": state.arc_id, "role": "selected",
                "first_observation_id": observation_id, "last_observation_id": observation_id,
            }
            occurrences.append(previous)

        arc_ids = [entry["arc_id"] for entry in occurrences]
        arcs = [graph.arcs[arc_id] for arc_id in arc_ids]
        segments.append({
            "segment_id": segment_id, "break_reason": reason,
            "first_observation_id": choices[0][0], "last_observation_id": choices[-1][0],
            "arc_ids": arc_ids, "occurrences": occurrences,
            "matched_distance_m": distance,
            "full_arc_length_m": sum(arc.length_m for arc in arcs),
        })
        node_ids = chain((arcs[0].start,), (arc.end for arc in arcs))
        features.append({
            "type": "Feature", "id": segment_id,
            "properties": {"segment_id": segment_id, "break_reason": reason, "model_id": model_id,
                           "partition": trip.partition, "trip_id": trip.trip_id,
                           "arc_count": len(arcs)},
            "geometry": {"type": "LineString",
                         "coordinates": [trip.nodes[node].tolist() for node in node_ids]},
        })
    return segments, {"type": "FeatureCollection", "features": features}
