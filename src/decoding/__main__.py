"""Decode selected trips and optionally score them against offline references."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from src.candidates.pipeline import load_inference_trip
from src.preprocessing.loaders import PARTITIONS, TripFiles

from . import DEFAULT_MODEL, MODELS, decode_trip, model_bundle
from .hmm import DecoderConfig


DEFAULT_DATA = Path("data/raw")
DEFAULT_REPORTS = Path("reports/phase5")
DEFAULT_POLICY = Path("reports/phase2b/train/policy.json")


def _safe_input(path):
    if "evaluator_test" in path.parts or "evaluator_test" in path.resolve().parts:
        raise ValueError("Phase 5 accepts only student development data")
    return path


def _output_directory(path, data_root):
    resolved = _safe_input(path).resolve()
    protected = (data_root.resolve(), DEFAULT_DATA.resolve(), Path("artifacts/models").resolve())
    if any(resolved.is_relative_to(parent) for parent in protected):
        raise ValueError("Output must be outside raw data and model artifacts")
    reports = Path("reports").resolve()
    if resolved == reports or (resolved.is_relative_to(reports)
                               and not resolved.is_relative_to(DEFAULT_REPORTS.resolve())):
        raise ValueError("Use reports/phase5 or an output directory outside reports")
    return resolved


def _json(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False,
                            encoding="utf-8") as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, allow_nan=False, separators=(",", ":"))
            output.write("\n")
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _files(root, partition, trip_ids, evaluate):
    folder = _safe_input(root / partition)
    ids = sorted(set(trip_ids)) if trip_ids else sorted(
        item.name for item in folder.iterdir()
        if item.name.isascii() and item.name.isdecimal() and _safe_input(item).is_dir())
    if not ids:
        raise ValueError(f"No trips found in {folder}")
    files = []
    for trip_id in ids:
        if not (trip_id.isascii() and trip_id.isdecimal()):
            raise ValueError(f"Invalid numeric trip ID {trip_id!r}")
        trip = TripFiles(partition, trip_id, folder / trip_id)
        for extension in ("track", "nodes", "arcs") + (("route",) if evaluate else ()):
            if not _safe_input(trip.path(extension)).is_file():
                raise ValueError(f"Missing input {trip.path(extension)}")
        files.append(trip)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--partition", choices=PARTITIONS, default="train")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--trip-id", nargs="+", action="extend")
    selection.add_argument("--all-trips", action="store_true")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/decoding"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--student-holdout", action="store_true",
                        help="After selected training trips, evaluate all 10 student trips")
    defaults = DecoderConfig()
    parser.add_argument("--distance-scale-m", type=float, default=defaults.distance_scale_m)
    parser.add_argument("--transition-weight", type=float, default=defaults.transition_weight)
    parser.add_argument("--position-slack-m", type=float, default=defaults.position_slack_m)
    parser.add_argument("--routing-budget", type=int, default=defaults.search_budget)
    parser.add_argument("--routing-cache-size", type=int, default=defaults.cache_size)
    parser.add_argument("--restart-cost", type=float, default=defaults.restart_cost)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.student_holdout and (not args.evaluate or args.partition != "train"):
        parser.error("--student-holdout requires --partition train --evaluate")

    try:
        config = DecoderConfig(args.distance_scale_m, args.transition_weight, args.position_slack_m,
                               args.routing_budget, args.routing_cache_size, args.restart_cost)
        root = _safe_input(args.data_root)
        output = _output_directory(args.output_dir, root)
        model_path = _safe_input(args.model_path or MODELS[args.model])
        model_start = perf_counter()
        bundle = model_bundle(args.model, model_path)
        model_load_s = perf_counter() - model_start
        artifact_hash = sha256(model_path.read_bytes()).hexdigest()
        model_id = f"{args.model}/{bundle['model_id']}"
        if args.evaluate:
            from src.alignment.core import AlignmentPolicy
            from src.models.dataset import EXPECTED

            from .evaluation import evaluate_trip
            from .report import DIFFICULT, write_report

            report_dir = _output_directory(args.report_dir, root)
            if output == report_dir:
                raise ValueError("Use different prediction and report output directories")
            policy = AlignmentPolicy(**json.loads(_safe_input(args.policy).read_text()))
            if policy.policy_id != bundle["manifest"]["policy_id"]:
                raise ValueError("Evaluation policy does not match the model's frozen Phase 2B policy")

        def freeze():
            report_dir.mkdir(parents=True, exist_ok=True)
            _json(report_dir / "decoder_freeze.json", {
                "decoder_config": asdict(config), "model_id": model_id, "artifact_sha256": artifact_hash,
                "policy_id": policy.policy_id, "candidate_config": bundle["manifest"]["candidate_config"],
            })

        def process(files_to_run):
            for file in files_to_run:
                started = perf_counter()
                trip = load_inference_trip(file)
                load_s = perf_counter() - started
                run = decode_trip(trip, bundle, model=args.model, config=config, batch_size=args.batch_size)
                directory = _output_directory(output / file.partition, root)
                directory.mkdir(parents=True, exist_ok=True)
                _json(directory / f"{file.trip_id}.pathsnap.json", run.result)
                _json(directory / f"{file.trip_id}.pathsnap.geojson", run.result["route_geojson"])
                if args.evaluate:
                    evaluation_started = perf_counter()
                    point, route, details = evaluate_trip(file, trip, run, policy)
                    points.append(point)
                    routes.append(route)
                    runtime.append(dict(partition=file.partition, trip_id=file.trip_id, load_s=load_s,
                                        **run.timings, evaluation_s=perf_counter()-evaluation_started,
                                        trip_total_s=perf_counter()-started,
                                        model_load_s=model_load_s if not runtime else 0.0,
                                        **run.result["routing"]))
                    if file.partition == "train" and file.trip_id in DIFFICULT:
                        details["break_counts"] = json.dumps(details["break_counts"], sort_keys=True)
                        difficult[file.trip_id] = details
                print(f"{file.partition}/{file.trip_id} {run.result['status']}: "
                      f"{len(run.result['route_segments'])} segments, "
                      f"{run.result['observation_count']} fixes, {run.timings['inference_s']:.3f}s",
                      flush=True)

        points, routes, runtime, difficult = [], [], [], {}
        if args.evaluate and args.partition == "student_test":
            freeze()
        files = _files(root, args.partition, args.trip_id, args.evaluate)
        process(files)
        if args.student_holdout:
            freeze()
            holdout = _files(root, "student_test", None, True)
            if len(holdout) != PARTITIONS["student_test"]:
                raise ValueError(f"Expected 10 student-testing trips, found {len(holdout)}")
            process(holdout)
            totals = tuple(sum(row[k] for row in points if row["partition"] == "student_test")
                           for k in ("original", "qualified", "eligible"))
            if totals != EXPECTED["student_test"][:3]:
                raise ValueError(f"Student holdout counts differ from the frozen Phase 3 data: {totals}")
        if args.evaluate:
            if not (args.student_holdout or args.partition == "student_test"):
                freeze()
            write_report(report_dir, points, routes, runtime, model_id, artifact_hash, config, difficult)
    except (ValueError, OSError) as error:
        parser.exit(2, f"PathSnap decoding: {error}\n")


if __name__ == "__main__":
    main()
