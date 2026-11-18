"""Atomic, fingerprinted local artifacts for one sequential experiment."""

from contextlib import contextmanager
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

from src.models.inference import load_model, positive_scores


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@contextmanager
def atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_record(path, contract, payload):
    record = dict(contract=contract, payload=payload)
    write_json(path, dict(**record, sha256=fingerprint(record)))


def read_record(path, contract):
    record = read_json(path)
    digest = record.pop("sha256")
    if record["contract"] != contract or fingerprint(record) != digest:
        raise ValueError(f"Incompatible or corrupt checkpoint: {path}")
    return record["payload"]


def bind_directory(directory, manifest):
    path = directory / "optimization_manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Artifact directory is bound to different code, data or settings")
    else:
        write_json(path, manifest)


def score_checkpoint(path, contract, size, compute):
    if not path.exists():
        scores, timing = compute()
        with atomic_path(path) as temporary, temporary.open("wb") as stream:
            np.savez(stream, scores=scores, timing=canonical(timing), contract=contract,
                     sha256=sha256(scores.tobytes() + canonical(timing).encode()).hexdigest())
    with np.load(path, allow_pickle=False) as stored:
        scores = stored["scores"]
        timing = str(stored["timing"])
        if (str(stored["contract"]) != contract or scores.shape != (size,) or scores.dtype != np.float64
                or not (np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all())
                or sha256(scores.tobytes() + timing.encode()).hexdigest() != str(stored["sha256"])):
            raise ValueError(f"Invalid score checkpoint: {path}")
    return scores, json.loads(timing)


def freeze_model(directory, manifest, selection, fit, probe, args):
    binary = directory / "gradient_boosting.joblib"
    freeze = directory / "frozen_model.json"
    contract = fingerprint(dict(manifest=manifest, selection=selection))
    if freeze.exists():
        record = read_record(freeze, contract)
        if file_hash(binary) != record["sha256"]:
            raise ValueError("Frozen model hash mismatch")
    else:
        pipeline, fit_s = fit()
        scores = positive_scores(pipeline, probe, args.batch_size, args.workers)
        with atomic_path(binary) as temporary:
            joblib.dump(dict(model_id="gradient_boosting", features=manifest["features"], pipeline=pipeline,
                             manifest=manifest, selection=selection), temporary, compress=3)
            reloaded = load_model(temporary)
            np.testing.assert_array_equal(scores, positive_scores(reloaded["pipeline"], probe,
                                                                  args.batch_size, args.workers))
        record = dict(sha256=file_hash(binary), model_bytes=binary.stat().st_size, fit_s=fit_s,
                      reload_probe_sha256=sha256(scores.tobytes()).hexdigest(), reload_rows=len(probe))
        save_record(freeze, contract, record)
    bundle = load_model(binary)
    if bundle["manifest"] != manifest or bundle["selection"] != selection:
        raise ValueError("Frozen model metadata mismatch")
    scores = positive_scores(bundle["pipeline"], probe, args.batch_size, args.workers)
    if sha256(scores.tobytes()).hexdigest() != record["reload_probe_sha256"]:
        raise ValueError("Frozen model probe mismatch")
    return bundle["pipeline"], record
