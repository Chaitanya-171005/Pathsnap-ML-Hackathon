"""Inference-only loading and deterministic candidate and observation records."""

from collections.abc import Iterator
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from src.preprocessing.geometry import haversine_m
from src.preprocessing.loaders import ParseIssue, TripFiles, load_arcs, load_nodes, load_track
from src.preprocessing.validation import valid_coordinates

from .spatial import RoadIndex


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    top_k: int = 8
    max_gap_s: float = 30.0
    max_speed_mps: float = 55.0
    min_heading_step_m: float = 2.0

    def __post_init__(self):
        thresholds = (self.radius_m, self.max_gap_s, self.max_speed_mps, self.min_heading_step_m)
        if not all(math.isfinite(value) and value > 0 for value in thresholds):
            raise ValueError("Distance, time and speed thresholds must be positive and finite")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")


@dataclass(frozen=True)
class InferenceTrip:
    partition: str
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    issues: tuple[ParseIssue, ...] = ()


def load_inference_trip(files: TripFiles) -> InferenceTrip:
    track, track_issues = load_track(files.path("track"))
    nodes, node_issues = load_nodes(files.path("nodes"))
    arcs, arc_issues = load_arcs(files.path("arcs"))
    return InferenceTrip(files.partition, files.trip_id, track, nodes, arcs,
                         tuple(track_issues + node_issues + arc_issues))


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    partition: str
    trip_id: str
    observation_id: int
    arc_id: int
    distance_m: float
    arc_length_m: float
    distance_rank: int
    delta_t_s: float | None
    speed_mps: float | None
    heading_difference_deg: float | None
    projection_fraction: float | None
    from_in_degree: int
    from_out_degree: int
    to_in_degree: int
    to_out_degree: int
    time_step_valid: bool
    speed_valid: bool
    heading_difference_valid: bool
    projection_valid: bool


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    partition: str
    trip_id: str
    observation_id: int
    gps_valid: bool
    timestamp_valid: bool
    candidate_status: str
    candidates_within_radius: int
    candidate_count: int
    delta_t_s: float | None
    speed_mps: float | None
    heading_deg: float | None
    motion_status: str


def _motion(track, gps_valid, timestamp_valid, config):
    """Yield past-only, adjacent-row motion data without bridging unavailable rows."""
    for i, (lon, lat, timestamp) in enumerate(track):
        if i == 0:
            yield None, None, None, "first_observation"
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            delta = float(timestamp - track[i - 1, 2])
        if not (timestamp_valid[i - 1] and timestamp_valid[i] and math.isfinite(delta)):
            yield None, None, None, "invalid_timestamp_pair"
        elif delta <= 0:
            yield delta, None, None, "nonpositive_time_step"
        elif delta > config.max_gap_s:
            yield delta, None, None, "gps_gap"
        elif not (gps_valid[i - 1] and gps_valid[i]):
            yield delta, None, None, "invalid_gps_pair"
        else:
            start_lon, start_lat = track[i - 1, :2]
            step = float(haversine_m([start_lon, start_lat], [lon, lat]))
            speed = step / delta
            if speed > config.max_speed_mps:
                yield delta, None, None, "speed_above_limit"
            elif step < config.min_heading_step_m:
                yield delta, speed, None, "short_step"
            else:
                lat1, lat2, dlon = np.radians([start_lat, lat, lon - start_lon])
                east = np.sin(dlon) * np.cos(lat2)
                north = (np.cos(lat1) * np.sin(lat2)
                         - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
                heading = float(np.degrees(np.arctan2(east, north)) % 360)
                yield delta, speed, heading, "ok"


def candidate_batches(trip: InferenceTrip, config: CandidateConfig,
                      index: RoadIndex | None = None
                      ) -> Iterator[tuple[ObservationRecord, tuple[CandidateRecord, ...]]]:
    """Yield one observation record and its ordered candidates for each source row.

    If supplied, the index must have been built from this trip's nodes and arcs.
    """
    if index is None:
        index = RoadIndex(trip.nodes, trip.arcs)
    gps_valid = valid_coordinates(trip.track[:, :2])
    timestamp_valid = np.isfinite(trip.track[:, 2])
    points = index.projection.project(trip.track[gps_valid, :2])
    nearby = index.nearby(points, config.radius_m)
    for i, (delta, speed, heading, motion_status) in enumerate(
        _motion(trip.track, gps_valid, timestamp_valid, config)
    ):
        key = (trip.partition, trip.trip_id, i)
        if not gps_valid[i]:
            yield ObservationRecord(*key, False, bool(timestamp_valid[i]), "invalid_gps",
                                    0, 0, delta, speed, heading, motion_status), ()
            continue
        indices, distances, fractions = next(nearby)
        candidates = []
        for rank, (j, distance, fraction) in enumerate(zip(
            indices[:config.top_k], distances[:config.top_k], fractions[:config.top_k],
            strict=True
        ), start=1):
            start, end = index.endpoints[j]
            projection_valid = bool(np.isfinite(fraction))
            difference = None
            if heading is not None and projection_valid:
                difference = float(abs((heading - index.heading_deg[j] + 180) % 360 - 180))
            candidates.append(CandidateRecord(
                *key, int(index.arc_ids[j]), float(distance), float(index.length_m[j]), rank,
                delta, speed, difference, float(fraction) if projection_valid else None,
                int(index.in_degree[start]), int(index.out_degree[start]),
                int(index.in_degree[end]), int(index.out_degree[end]),
                delta is not None and 0 < delta <= config.max_gap_s,
                speed is not None, difference is not None, projection_valid,
            ))
        status = "ok" if candidates else "no_candidates"
        yield ObservationRecord(*key, True, bool(timestamp_valid[i]), status, len(indices),
                                len(candidates), delta, speed, heading, motion_status), tuple(candidates)
