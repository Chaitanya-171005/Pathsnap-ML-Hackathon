"""Reference-only proposals and offline proxy-label qualification."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from time import perf_counter

import numpy as np

from src.candidates.pipeline import CandidateConfig, _motion
from src.candidates.spatial import RoadIndex
from src.preprocessing.loaders import Trip
from src.preprocessing.validation import validate_trip

from .sequence import monotonic_path


METHODS = ("nearest", "monotonic", "qualified")
STATUSES = ("accepted", "ambiguous", "unmatched", "invalid_observation", "disconnected_reference")
PROVENANCE = "ordered_reference_proxy_v1"


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    proposal_radius_m: float = 45.0
    qualified_distance_m: float = 20.0
    tie_margin_m: float = 0.05
    anchor_margin: float = 3.0
    qualified_margin: float = 3.0
    max_heading_difference_deg: float = 75.0
    heading_penalty: float = 18.0
    match_reward: float = 32.0
    skip_cost: float = 0.6
    break_cost: float = 12.0
    break_buffer_m: float = 12.0
    max_gap_s: float = 30.0
    max_speed_mps: float = 55.0
    min_heading_step_m: float = 2.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.qualified_distance_m > self.proposal_radius_m:
            raise ValueError("qualified_distance_m must be within the proposal radius")
        if self.max_heading_difference_deg > 180:
            raise ValueError("max_heading_difference_deg must be at most 180")

    @property
    def policy_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelRecord:
    partition: str
    trip_id: str
    observation_id: int
    method: str
    arc_id: int | None
    occurrence_index: int | None
    section_index: int | None
    distance_m: float | None
    projection_fraction: float | None
    heading_difference_deg: float | None
    local_cost: float | None
    alternative_margin: float | None
    plausible_alternatives: int
    distance_ties: int
    duplicate_directed_geometry: bool
    boundary_risk: bool
    gap_before: bool
    motion_status: str
    status: str
    reason: str
    policy_id: str
    provenance: str = PROVENANCE


@dataclass(frozen=True, slots=True)
class Options:
    occurrences: np.ndarray
    distances: np.ndarray
    fractions: np.ndarray
    heading_differences: np.ndarray
    costs: np.ndarray


def _duplicate_reference_geometry(trip: Trip, arc_geometry: np.ndarray) -> np.ndarray:
    """Flag exact directed endpoint geometry shared by different road arc IDs."""
    duplicate = np.zeros(len(trip.arcs), dtype=bool)
    coords = trip.nodes[trip.arcs[arc_geometry]].reshape(-1, 4)
    if len(coords):
        for column in (0, 2):
            coords[coords[:, column] == 180, column] = -180
        _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
        duplicate[arc_geometry] = counts[inverse] > 1
    return duplicate


def _sections(valid):
    drawable = valid.route_geometry
    breaks = (valid.route_disconnections | valid.route_uncheckable
              | ~drawable[:-1] | ~drawable[1:])
    sections = np.cumsum(np.r_[False, breaks])[:len(drawable)]
    return sections, np.r_[False, breaks][:len(drawable)], np.r_[breaks, False][:len(drawable)]


def qualification(record: LabelRecord, policy: AlignmentPolicy) -> tuple[str, str]:
    """Score the accept/abstain decision independently of model predictions or Top-K."""
    status, reason = record.status, record.reason
    if record.arc_id is not None and status in ("accepted", "ambiguous"):
        if record.duplicate_directed_geometry or record.projection_fraction is None:
            pass  # Both are structural ambiguity independent of diagnostic thresholds.
        elif record.distance_m > policy.qualified_distance_m:
            status, reason = "unmatched", "distance_above_qualification_limit"
        elif record.gap_before:
            status, reason = "ambiguous", "large_timestamp_gap"
        elif record.motion_status in ("invalid_timestamp_pair", "nonpositive_time_step"):
            status, reason = "ambiguous", "unreliable_timestamp_step"
        elif (record.heading_difference_deg is not None
              and record.heading_difference_deg > policy.max_heading_difference_deg):
            status, reason = "ambiguous", "heading_disagreement"
        elif record.alternative_margin is not None and record.alternative_margin <= policy.qualified_margin:
            status, reason = "ambiguous", "small_sequence_feasible_margin"
    return status, reason


def qualify(record: LabelRecord, policy: AlignmentPolicy, policy_id: str) -> LabelRecord:
    status, reason = qualification(record, policy)
    if status == "accepted":
        reason = "passes_qualification"
    return replace(record, method="qualified", status=status, reason=reason, policy_id=policy_id)


def align_trip(trip: Trip, policy: AlignmentPolicy, timings: dict | None = None
               ) -> dict[str, list[LabelRecord]]:
    """Produce all three methods for every original GPS row in trip order."""
    if trip.files.partition not in ("train", "student_test"):
        raise ValueError("Alignment accepts only training and student-testing trips")
    started = perf_counter()
    policy_id = policy.policy_id
    valid = validate_trip(trip)
    usable = valid.gps & valid.timestamps
    n = len(trip.track)
    sections, break_before, break_after = _sections(valid)
    reference_arcs = np.full((len(trip.route), 2), -1, dtype=np.int64)
    reference_arcs[valid.route_geometry] = trip.arcs[trip.route[valid.route_geometry]]
    index = RoadIndex(trip.nodes, reference_arcs)
    duplicate = _duplicate_reference_geometry(trip, valid.arc_geometry)
    gaps = np.zeros(n, dtype=bool)
    gaps[1:] = valid.timestamp_deltas > policy.max_gap_s
    motion_config = CandidateConfig(max_gap_s=policy.max_gap_s, max_speed_mps=policy.max_speed_mps,
                                   min_heading_step_m=policy.min_heading_step_m)
    # Identical reliability screening to inference, applied to observation data alone.
    motion = list(_motion(trip.track, valid.gps, valid.timestamps, motion_config))
    empty = np.empty(0, dtype=int)
    options = [Options(empty, empty, empty, empty, empty)] * n
    points = index.projection.project(trip.track[usable, :2])
    nearby = index.nearby(points, policy.proposal_radius_m)
    for row, (hits, distances, fractions) in zip(np.flatnonzero(usable), nearby, strict=True):
        heading = motion[row][2]
        difference = np.full(len(hits), np.nan) if heading is None else abs(
            (heading - index.heading_deg[hits] + 180) % 360 - 180)
        costs = distances + np.nan_to_num(difference) * policy.heading_penalty / 180.0
        options[row] = Options(index.arc_ids[hits], distances, fractions, difference, costs)

    proposals_ready = perf_counter()
    path = monotonic_path(options, gaps, sections, match_reward=policy.match_reward,
                          skip_cost=policy.skip_cost, break_cost=policy.break_cost)
    sequence_ready = perf_counter()
    # A chosen occurrence can bound its neighbours only when its own local
    # evidence favours it. A run of geometrically tied states cannot endorse
    # itself by assigning every row to one occurrence.
    anchors = []
    for row in sorted(path):
        choices = options[row]
        ix = path[row]
        cost = choices.costs[ix]
        rival_cost = np.min(choices.costs[choices.occurrences != choices.occurrences[ix]],
                            initial=np.inf)
        if not gaps[row] and cost < policy.match_reward and rival_cost - cost > policy.anchor_margin:
            anchors.append(row)
    anchor_rows = set(anchors)
    result = {method: [] for method in METHODS}
    before = None
    anchor_before = 0
    cursor = 0

    for row, choices in enumerate(options):
        if row in anchor_rows:
            cursor += 1
        next_row = anchors[cursor] if cursor < len(anchors) else None
        following = (int(options[next_row].occurrences[path[next_row]])
                     if next_row is not None else len(trip.route))

        def record(method: str, ix: int | None) -> LabelRecord:
            key = (trip.files.partition, trip.files.trip_id, row, method)
            flags = {"gap_before": bool(gaps[row]), "motion_status": motion[row][3],
                     "policy_id": policy_id}
            if ix is None:
                invalid = not usable[row]
                reason = ("invalid_gps" if not valid.gps[row] else "invalid_timestamp") if invalid else (
                    "sequence_skipped" if len(choices.occurrences) else (
                        "outside_proposal_radius" if len(index.arc_ids) else "no_drawable_reference"))
                return LabelRecord(*key, None, None, None, None, None, None, None, None, 0, 0,
                                   False, False, **flags,
                                   status="invalid_observation" if invalid else "unmatched", reason=reason)

            occ = int(choices.occurrences[ix])
            arc = int(trip.route[occ])
            fraction = float(choices.fractions[ix]) if np.isfinite(choices.fractions[ix]) else None
            heading = (float(choices.heading_differences[ix])
                       if np.isfinite(choices.heading_differences[ix]) else None)
            length = index.length_m[np.searchsorted(index.arc_ids, occ)]
            near_break = bool(
                (break_before[occ] and (fraction is None or fraction * length <= policy.break_buffer_m))
                or (break_after[occ] and (fraction is None or (1 - fraction) * length <= policy.break_buffer_m)))
            crossed = method == "monotonic" and before is not None and sections[occ] != sections[before]
            alternatives = choices.occurrences != occ
            scores = choices.distances if method == "nearest" else choices.costs
            if method == "monotonic":
                alternatives &= ((choices.occurrences >= anchor_before)
                                 & (choices.occurrences <= following))
            margin = float(scores[alternatives].min() - scores[ix]) if alternatives.any() else None
            ties = int((alternatives & (abs(choices.distances - choices.distances[ix])
                                        <= policy.tie_margin_m)).sum())
            if near_break or crossed:
                status, reason = "disconnected_reference", "reference_section_boundary"
            elif duplicate[arc]:
                status, reason = "ambiguous", "duplicate_directed_geometry"
            elif fraction is None:
                status, reason = "ambiguous", "zero_length_reference_arc"
            elif margin is not None and margin <= policy.tie_margin_m:
                status, reason = "ambiguous", "near_tie_or_better_alternative"
            else:
                status, reason = "accepted", "nearest_distance" if method == "nearest" else "monotonic_chain"
            return LabelRecord(*key, arc, occ, int(sections[occ]), float(choices.distances[ix]),
                               fraction, heading, float(scores[ix]), margin, int(alternatives.sum()), ties,
                               bool(duplicate[arc]), bool(near_break or crossed), **flags,
                               status=status, reason=reason)

        nearest = record("nearest", 0 if len(choices.occurrences) else None)
        monotonic = record("monotonic", path.get(row))
        result["nearest"].append(nearest)
        result["monotonic"].append(monotonic)
        if monotonic.occurrence_index is not None:
            before = monotonic.occurrence_index
            if row in anchor_rows:
                anchor_before = before

    diagnostics_ready = perf_counter()
    result["qualified"] = [qualify(row, policy, policy_id) for row in result["monotonic"]]
    if timings is not None:
        timings.update(proposals_s=proposals_ready - started,
                       sequence_s=sequence_ready - proposals_ready,
                       baseline_and_sequence_records_s=diagnostics_ready - sequence_ready,
                       qualification_s=perf_counter() - diagnostics_ready)
    return result
