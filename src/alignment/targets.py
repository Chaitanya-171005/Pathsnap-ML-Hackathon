"""Offline joins from qualified proxy labels to existing inference candidate rows."""

from collections.abc import Iterable
from dataclasses import dataclass

from src.candidates.pipeline import CandidateRecord

from .core import LabelRecord


@dataclass(frozen=True, slots=True)
class CandidateTarget:
    partition: str
    trip_id: str
    observation_id: int
    arc_id: int
    target: int
    method: str
    provenance: str
    policy_id: str


def label_candidates(label: LabelRecord, candidates: Iterable[CandidateRecord]) -> tuple[
    str, tuple[CandidateTarget, ...]
]:
    """Validate join keys and emit labels only for a qualified, present target."""
    key = (label.partition, label.trip_id, label.observation_id)
    rows = tuple(candidates)
    if any((c.partition, c.trip_id, c.observation_id) != key for c in rows):
        raise ValueError("Candidate observation key does not match the alignment key")
    return targets_for_arc_ids(label, (row.arc_id for row in rows))


def targets_for_arc_ids(label: LabelRecord, arc_ids: Iterable[int]) -> tuple[
    str, tuple[CandidateTarget, ...]
]:
    """Lower-level join for IDs supplied by the same observation's road query."""
    if label.method != "qualified" or label.status != "accepted":
        return "unqualified", ()
    if label.arc_id is None:
        raise ValueError("An accepted proxy label must have a target arc ID")
    ids = tuple(arc_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate arc ID in an observation group")
    if label.arc_id not in ids:
        return "candidate_miss", ()
    return "included", tuple(
        CandidateTarget(label.partition, label.trip_id, label.observation_id, arc_id,
                        int(arc_id == label.arc_id), label.method, label.provenance, label.policy_id)
        for arc_id in ids
    )
