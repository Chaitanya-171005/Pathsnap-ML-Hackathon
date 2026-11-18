"""Local development-trip access and inference shared by the HTTP endpoints."""

from functools import lru_cache
import logging
from math import isfinite
import re
from time import perf_counter

import numpy as np

from src.candidates.pipeline import candidate_batches, load_inference_trip
from src.decoding import CANDIDATES, decode_trip, model_bundle
from src.models.inference import predict_trip
from src.preprocessing.loaders import PARTITIONS, TripFiles
from src.preprocessing.validation import valid_coordinates


LOGGER = logging.getLogger(__name__)
TRIP_ID = re.compile(r"[A-Za-z0-9_-]{1,80}", re.ASCII)
INFERENCE_FILES = ("track", "nodes", "arcs")
MODES = {
    "phase3_fixed": ("Phase 3 Fixed Classifier", "gradient_boosting", "phase3"),
    "hmm_viterbi": ("HMM / Viterbi Route Reconstruction", "hmm_viterbi", "phase3"),
    "phase4_tuned": ("Phase 4 Tuned Classifier (experimental)", "gradient_boosting_tuned", "phase4"),
}


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@lru_cache(maxsize=192)
def _row_count(path, modified_ns, size):
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _read(files, _metadata):
    trip = load_inference_trip(files)
    if any(issue.code in ("missing_file", "unreadable_file") for issue in trip.issues):
        raise ApiError(503, "data_unavailable", "Trip files are temporarily unavailable")
    return trip


class InferenceService:
    def __init__(self, root, model_paths):
        self.root = root.resolve()
        if "evaluator_test" in (*root.parts, *self.root.parts):
            raise ValueError("PATHSNAP_DATA_ROOT must point to the development dataset root")
        self.bundles = {}
        self.phase4_present = model_paths["phase4"].is_file()
        for phase, path in model_paths.items():
            if not path.is_file():
                continue
            try:
                self.bundles[phase] = model_bundle(phase, path)
            except Exception:
                LOGGER.exception("Failed to load configured %s classifier", phase)
        self.read = lru_cache(maxsize=2)(_read)

    def partition_dir(self, partition):
        if partition not in PARTITIONS:
            raise ApiError(422, "invalid_partition", "Partition must be train or student_test")
        directory = self.root / partition
        if directory.is_symlink() or not directory.is_dir():
            raise ApiError(503, "data_unavailable", "Partition data is unavailable")
        return directory

    def partitions(self):
        return [p for p in PARTITIONS if not (self.root / p).is_symlink()
                and (self.root / p).is_dir()]

    def capabilities(self):
        modes = []
        for mode, (label, model_id, phase) in MODES.items():
            if phase == "phase4" and not self.phase4_present:
                continue
            ready = phase in self.bundles
            modes.append(dict(id=mode, label=label, model_id=model_id, available=ready,
                              reason=None if ready else "Local classifier artifact is unavailable"))
        return dict(api_version="v1", partitions=self.partitions(), modes=modes,
                    default_mode="phase3_fixed", synthetic=False)

    def files(self, partition, trip_id):
        if not isinstance(trip_id, str) or not TRIP_ID.fullmatch(trip_id):
            raise ApiError(422, "invalid_trip_id", "Invalid trip ID")
        directory = self.partition_dir(partition) / trip_id
        files = TripFiles(partition, trip_id, directory)
        if directory.is_symlink() or not directory.is_dir():
            raise ApiError(404, "trip_not_found", "Trip unavailable")
        for extension in INFERENCE_FILES:
            path = files.path(extension)
            if path.is_symlink() or not path.is_file():
                raise ApiError(404, "trip_not_found", "Trip unavailable")
        return files

    def summary(self, files):
        track = files.path("track")
        stat = track.stat()
        return dict(partition=files.partition, trip_id=files.trip_id,
                    observation_count=_row_count(track, stat.st_mtime_ns, stat.st_size),
                    description="Development trip")

    def trips(self, partition):
        summaries = []
        for directory in sorted(self.partition_dir(partition).iterdir()):
            if not TRIP_ID.fullmatch(directory.name):
                continue
            try:
                files = self.files(partition, directory.name)
            except ApiError as error:
                if error.status == 404:
                    continue
                raise
            summaries.append(self.summary(files))
        return summaries

    def trip(self, files):
        metadata = tuple((stat.st_mtime_ns, stat.st_size)
                         for stat in (files.path(ext).stat() for ext in INFERENCE_FILES))
        return self.read(files, metadata)

    def geometry(self, files):
        trip = self.trip(files)
        valid_times = np.flatnonzero(np.isfinite(trip.track[:, 2]))
        origin = float(trip.track[valid_times[0], 2]) if len(valid_times) else None
        observations = []
        for row in np.flatnonzero(valid_coordinates(trip.track[:, :2])):
            lon, lat, timestamp = map(float, trip.track[row])
            relative = timestamp - origin if origin is not None else None
            observations.append(dict(observation_id=int(row), coordinate=[lon, lat],
                                     relative_timestamp_s=(relative if relative is not None
                                                           and isfinite(relative) else None)))
        valid_nodes = valid_coordinates(trip.nodes)
        endpoints = ((trip.arcs >= 0) & (trip.arcs < len(trip.nodes))).all(axis=1)
        endpoints[endpoints] = valid_nodes[trip.arcs[endpoints]].all(axis=1)
        arcs = []
        for arc_id in np.flatnonzero(endpoints):
            start, end = map(int, trip.arcs[arc_id])
            arcs.append(dict(arc_id=int(arc_id), coordinates=trip.nodes[[start, end]].tolist(),
                             from_node=start, to_node=end))
        return dict(partition=files.partition, trip_id=files.trip_id, synthetic=False,
                    observations=observations, arcs=arcs)

    def predict(self, files, mode, result_id):
        _, model_id, phase = MODES[mode]
        bundle = self.bundles.get(phase)
        if bundle is None:
            raise ApiError(503, "model_unavailable", "Requested classifier artifact is unavailable")
        started = perf_counter()
        trip = self.trip(files)
        if mode == "hmm_viterbi":
            return decode_trip(trip, bundle, model=phase).result
        selected, empty = [], []
        for observation, predictions in predict_trip(
            bundle["pipeline"], bundle["model_id"], candidate_batches(trip, CANDIDATES)
        ):
            if predictions:
                top = predictions[0]
                selected.append(dict(observation_id=observation.observation_id, arc_id=top.arc_id,
                                     candidate_rank=top.candidate_rank, probability=top.probability))
            else:
                empty.append(observation.observation_id)
        return dict(result_id=result_id, status="completed", error=None, result=dict(
            result_id=result_id, partition=files.partition, trip_id=files.trip_id, mode=mode,
            model_id=model_id, observation_count=len(trip.track), selections=selected,
            empty_observation_ids=empty, decoded_route=None, runtime_s=perf_counter() - started,
            continuity_metrics={}, evaluation_metrics={}, synthetic=False,
        ))
