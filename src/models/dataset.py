"""Offline supervised data construction. Reference access is confined here."""

from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
from itertools import groupby
from pathlib import Path

import numpy as np

from src.alignment.core import PROVENANCE, AlignmentPolicy, align_trip
from src.alignment.targets import label_candidates
from src.candidates.pipeline import CandidateConfig, candidate_batches, load_inference_trip
from src.preprocessing.loaders import PARTITIONS, discover_trips, load_trip

from .inference import FEATURES, feature_values


CANDIDATE_CONFIG = CandidateConfig()
EXPECTED = {
    "train": (208417, 121855, 121721, 650978),
    "student_test": (19160, 13335, 13325, None),
}


@dataclass
class Dataset:
    partition: str
    trips: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    trip_codes: np.ndarray
    observation_ids: np.ndarray
    arc_ids: np.ndarray
    coverage: list[dict]
    exclusions: list[dict]

    def digest(self):
        digest = sha256(repr((self.partition, self.trips, FEATURES)).encode())
        for values in (self.X, self.y, self.trip_codes, self.observation_ids, self.arc_ids):
            digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
        return digest.hexdigest()


def _key(record):
    return record.partition, record.trip_id, record.observation_id


def join_targets(observation, candidates, targets):
    """Return binary labels in Phase 2A order, validating the complete key set."""
    expected = _key(observation)
    if any(_key(row) != expected for row in candidates):
        raise ValueError(f"Candidate observation key mismatch: {expected}")
    arcs = tuple(row.arc_id for row in candidates)
    if len(set(arcs)) != len(arcs):
        raise ValueError(f"Duplicate feature key: {expected}")
    if not targets:
        return ()
    keys = {(p, t, obs, arc): value for p, t, obs, arc, value in targets}
    wanted = [(*expected, arc) for arc in arcs]
    if len(keys) != len(targets) or set(keys) != set(wanted):
        raise ValueError(f"Duplicate or incompatible candidate-target keys: {expected}")
    y = tuple(keys[key] for key in wanted)
    if any(value not in (0, 1) for value in y) or sum(y) != 1:
        raise ValueError(f"Expected exactly one positive in candidate group: {expected}")
    return y


def _training_rows(audits, targets, policy, partition, trip_id, observations):
    policy_id = policy.policy_id
    target_group = next(targets, None)
    for observation, candidates in observations:
        audit = next(audits, None)
        key = _key(observation)
        if audit is None or (audit["partition"], audit["trip_id"], int(audit["observation_id"])) != key:
            raise ValueError(f"Missing or incorrect Phase 2B observation audit: {key}")
        if (audit["policy_id"] != policy_id or float(audit["radius_m"]) != CANDIDATE_CONFIG.radius_m
                or int(audit["top_k"]) != CANDIDATE_CONFIG.top_k
                or int(audit["candidate_count"]) != len(candidates)):
            raise ValueError(f"Incompatible policy, candidate configuration or candidate count: {key}")
        outcome, reason = audit["outcome"], audit["reason"]
        rows = []
        if target_group is not None and target_group[0] == (trip_id, observation.observation_id):
            for row in target_group[1]:
                if (row["partition"], row["method"], row["provenance"], row["policy_id"]) != (
                    partition, "qualified", PROVENANCE, policy_id
                ):
                    raise ValueError(f"Incompatible proxy target provenance: {key}")
                rows.append((row["partition"], row["trip_id"], int(row["observation_id"]),
                             int(row["arc_id"]), int(row["target"])))
            target_group = next(targets, None)
        if outcome not in ("included", "unqualified", "candidate_miss") or bool(rows) != (outcome == "included"):
            raise ValueError(f"Incompatible audit and targets: {key}")
        y = join_targets(observation, candidates, rows)
        if y and candidates[y.index(1)].distance_rank != int(audit["target_rank_within_radius"]):
            raise ValueError(f"Proxy target rank disagrees with Phase 2B audit: {key}")
        yield observation, candidates, y, outcome, reason
    if next(audits, None) is not None or target_group is not None:
        raise ValueError(f"Unexpected leftover Phase 2B export rows for {trip_id}")


def _student_rows(files, observations, policy):
    reference = load_trip(files)
    labels = align_trip(reference, policy)["qualified"]
    for (observation, candidates), label in zip(observations, labels, strict=True):
        if _key(observation) != _key(label):
            raise ValueError("Student evaluation proposal and candidate keys differ")
        outcome, targets = label_candidates(label, candidates)
        reason = label.reason if outcome == "unqualified" else (
            "target_present" if outcome == "included" else
            "target_absent_from_top_k" if candidates else "empty_candidate_group")
        y = join_targets(observation, candidates, [
            (*_key(row), row.arc_id, row.target) for row in targets])
        yield observation, candidates, y, outcome, reason


def _collect(files, policy, training_exports=None):
    partition = files[0].partition
    arrays = [[] for _ in range(5)]
    coverage, exclusions = [], []
    for trip_code, file in enumerate(files):
        trip = load_inference_trip(file)
        if trip.issues:
            raise ValueError(f"Candidate sources have parse issues for {file.trip_id}: {trip.issues[:3]}")
        candidates = candidate_batches(trip, CANDIDATE_CONFIG)
        if training_exports is None:
            records = _student_rows(file, candidates, policy)
        else:
            audits, targets = training_exports
            audit_rows = next(audits, ("", iter(())))
            target_rows = next(targets, ("", iter(())))
            if audit_rows[0] != file.trip_id or target_rows[0] != file.trip_id:
                raise ValueError(f"Phase 2B exports do not match training trip {file.trip_id}")
            records = _training_rows(audit_rows[1], groupby(target_rows[1], lambda row: (
                row["trip_id"], int(row["observation_id"]))), policy, partition, file.trip_id, candidates)

        features, outcomes, observations, arc_ids = [], [], [], []
        counts, reasons = Counter(), Counter()
        for obs, group, y, outcome, reason in records:
            counts[outcome] += 1
            if not y:
                reasons[outcome, reason] += 1
                continue
            features.extend(map(feature_values, group))
            outcomes.extend(y)
            observations.extend([obs.observation_id] * len(group))
            arc_ids.extend(row.arc_id for row in group)
        for chunks, values in zip(arrays, (
            np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURES)),
            np.asarray(outcomes, dtype=np.int8), np.full(len(outcomes), trip_code, dtype=np.int16),
            np.asarray(observations, dtype=np.int32), np.asarray(arc_ids, dtype=np.int32),
        ), strict=True):
            chunks.append(values)
        original = counts.total()
        qualified = counts["included"] + counts["candidate_miss"]
        coverage.append(dict(partition=partition, trip_id=file.trip_id, original=original,
                             qualified=qualified, eligible=counts["included"], misses=counts["candidate_miss"],
                             unqualified=counts["unqualified"], candidate_rows=len(outcomes),
                             positives=sum(outcomes), negatives=len(outcomes) - sum(outcomes)))
        exclusions.extend(dict(partition=partition, trip_id=file.trip_id, outcome=outcome,
                               reason=reason, observations=n) for (outcome, reason), n in sorted(reasons.items()))
    return Dataset(partition, tuple(f.trip_id for f in files), *(np.concatenate(x) for x in arrays),
                   coverage, exclusions)


def build_dataset(data_dir: Path, partition: str, policy: AlignmentPolicy,
                  alignment_dir: Path = Path("data/alignments")) -> Dataset:
    if partition not in PARTITIONS:
        raise ValueError("Select one of the two student development partitions")
    files = [file for file in discover_trips(data_dir) if file.partition == partition]
    if len(files) != PARTITIONS[partition]:
        raise ValueError(f"Expected {PARTITIONS[partition]} {partition} trips, found {len(files)}")
    if partition == "train":
        root = alignment_dir / "train"
        with ((root / "candidate_join_status.csv").open(newline="") as audit_file,
              (root / "candidate_targets.csv").open(newline="") as target_file):
            audits, targets = (groupby(csv.DictReader(stream), lambda row: row["trip_id"])
                               for stream in (audit_file, target_file))
            dataset = _collect(files, policy, (audits, targets))
            if next(audits, None) is not None or next(targets, None) is not None:
                raise ValueError("Unexpected trips remain in Phase 2B training exports")
    else:
        dataset = _collect(files, policy)
    actual = tuple(sum(row[field] for row in dataset.coverage)
                   for field in ("original", "qualified", "eligible", "candidate_rows"))
    if any(expected is not None and value != expected
           for value, expected in zip(actual, EXPECTED[partition], strict=True)):
        raise ValueError(f"{partition} counts disagree with the frozen Phase 2B data: {actual}")
    return dataset
