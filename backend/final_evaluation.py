"""Explicitly enabled, authenticated evaluation over a private LAN."""

from dataclasses import asdict, dataclass, field
from hashlib import file_digest
import json
import logging
import os
from pathlib import Path
from secrets import compare_digest

from frontend.config import private_evaluation_host
from frontend.models.schemas import FINAL_PARTITION, PredictionJob
from src.alignment.core import AlignmentPolicy, _align_trip
from src.decoding import CANDIDATES, DecoderConfig, _decode_trip
from src.decoding.evaluation import route_continuity, score_reference
from src.preprocessing.loaders import PARTITIONS, load_route

from .service import ApiError, TRIP_ID, inference_files


LOGGER = logging.getLogger("uvicorn.error")
DOCUMENTED_SCORING = "confirmed_phase5_proxy"
MODEL_ID = "phase3/gradient_boosting"


@dataclass(frozen=True)
class FinalSettings:
    root: Path
    token: str = field(repr=False)
    scoring: str

    @classmethod
    def from_env(cls):
        if os.getenv("PATHSNAP_FINAL_EVALUATION") != "1":
            return None
        root = os.getenv("PATHSNAP_FINAL_ROOT")
        token = os.getenv("PATHSNAP_FINAL_TOKEN", "")
        scoring = os.getenv("PATHSNAP_FINAL_SCORING", "pending")
        if not root or not Path(root).is_absolute():
            raise ValueError("Final evaluation requires a separately configured absolute data path")
        if len(token) < 32 or not token.isascii() or not token.isalnum():
            raise ValueError("Final evaluation requires an alphanumeric token of at least 32 characters")
        if scoring not in ("pending", DOCUMENTED_SCORING):
            raise ValueError("Unsupported final scoring protocol")
        return cls(Path(root), token, scoring)

    def allows(self, request):
        if not (request.client and private_evaluation_host(request.client.host)
                and private_evaluation_host(request.url.hostname)):
            return False
        if any(name in request.headers for name in (
            "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
        )):
            return False
        authorization = request.headers.get("authorization", "").encode("utf-8")
        return compare_digest(authorization, f"Bearer {self.token}".encode("ascii"))


def verify_freeze(repo, model_path):
    freeze = json.loads((repo / "reports/phase5/decoder_freeze.json").read_text(encoding="utf-8"))
    policy = AlignmentPolicy(**json.loads(
        (repo / "reports/phase2b/train/policy.json").read_text(encoding="utf-8")
    ))
    config = DecoderConfig()
    with model_path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if (freeze["model_id"] != MODEL_ID or freeze["artifact_sha256"] != digest
            or freeze["candidate_config"] != asdict(CANDIDATES)
            or freeze["decoder_config"] != asdict(config) or freeze["policy_id"] != policy.policy_id):
        raise ValueError("Final evaluation requires the exact frozen Phase 3/5 model and configuration")
    return config, policy


class FinalEvaluation:
    def __init__(self, settings, service, config, policy):
        self.settings = settings
        self.service = service
        self.config = config
        self.policy = policy
        self.bundle = service.bundles.get("phase3")
        if self.bundle is None or self.bundle["manifest"].get("policy_id") != policy.policy_id:
            raise ValueError("Final evaluation requires a compatible frozen classifier")
        root = settings.root
        resolved = root.resolve()
        if (root.is_symlink() or service.root.is_relative_to(resolved)
                or any(resolved.is_relative_to(service.root / partition) for partition in PARTITIONS)):
            raise ValueError("Final evaluation requires a separate trip catalog")
        directories = sorted(path for path in resolved.iterdir()
                             if TRIP_ID.fullmatch(path.name) and path.is_dir())
        if len(directories) != 10:
            raise ValueError("Final evaluation requires exactly ten trip directories")
        try:
            self.catalog = {path.name: inference_files(FINAL_PARTITION, path) for path in directories}
        except ApiError as error:
            raise ValueError("Final evaluation catalog contains an incomplete or linked trip") from error

    def files(self, trip_id):
        if trip_id not in self.catalog:
            raise ApiError(404, "trip_not_found", "Trip unavailable")
        return inference_files(FINAL_PARTITION, self.catalog[trip_id].directory)

    def summary(self, files):
        return self.service.summary(files, description="Final evaluation trip")

    def trips(self):
        return [self.summary(self.files(trip_id)) for trip_id in self.catalog]

    def evaluation_metrics(self, files, trip, run):
        if self.settings.scoring == "pending":
            return {}, "pending_protocol"

        path = files.path("route")
        if path.is_symlink():
            return {}, "reference_unavailable"
        reference, issues = load_route(path)
        if issues or not len(reference) or not any(arc in run.graph.arcs for arc in reference):
            return {}, "reference_unavailable"
        points, route = score_reference(files, trip, run, reference, issues, self.policy, align=_align_trip)
        return {
            "qualified_proxy_observations_count": points["qualified"],
            "eligible_proxy_observations_count": points["eligible"],
            "qualified_candidate_misses_count": points["candidate_misses"],
            "classifier_proxy_top1_fraction": points["classifier_top1"],
            "hmm_proxy_top1_fraction": points["hmm_top1"],
            "hmm_directed_arc_precision_fraction": route["arc_precision"],
            "hmm_directed_arc_recall_fraction": route["arc_recall"],
            "hmm_directed_arc_pair_f1_fraction": route["directed_bigram_f1"],
        }, "scored"

    def predict(self, files, result_id):
        trip = self.service.trip(files)
        run = _decode_trip(trip, self.bundle, config=self.config)
        metrics, scoring_status = self.evaluation_metrics(files, trip, run)
        result = run.result
        job = PredictionJob.model_validate({
            "result_id": result_id, "status": "completed",
            "result": {
                "result_id": result_id, "partition": files.partition, "trip_id": files.trip_id,
                "mode": "hmm_viterbi", "model_id": result["model_id"],
                "observation_count": result["observation_count"], "phase5": result,
                "empty_observation_ids": [match["observation_id"] for match in result["matches"]
                                          if not match["candidate_count"]],
                "runtime_s": result["processing_time_s"],
                "continuity_metrics": route_continuity(result, run.graph),
                "evaluation_metrics": metrics,
            },
        })
        LOGGER.info("PathSnap final inference trip_id=%s request_id=%s inference_s=%.3f scoring=%s",
                    files.trip_id, result_id, result["processing_time_s"], scoring_status)
        return job.model_dump(mode="json")
