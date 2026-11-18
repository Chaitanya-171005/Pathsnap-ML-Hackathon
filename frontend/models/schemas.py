"""Proposed v1 wire schemas. Coordinates always use longitude, latitude."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from frontend.models.base import Coordinate, NonNegative, Probability, Schema, TripID
from frontend.models.phase5 import Phase5Result

ModeID = Literal["phase3_fixed", "phase4_tuned", "hmm_viterbi", "demo_route"]


class Health(Schema):
    status: Literal["ok", "degraded"]
    version: str


class PredictionMode(Schema):
    id: ModeID
    label: str
    model_id: str
    available: bool
    reason: str | None = None


class Capabilities(Schema):
    api_version: Literal["v1"] = "v1"
    partitions: list[str]
    modes: list[PredictionMode]
    default_mode: ModeID = "phase3_fixed"
    synthetic: bool = False

    @model_validator(mode="after")
    def check_modes(self):
        ids = [mode.id for mode in self.modes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate prediction mode")
        if self.default_mode not in ids:
            raise ValueError("Default mode must be advertised")
        if not self.synthetic and any(m.id == "demo_route" and m.available for m in self.modes):
            raise ValueError("demo_route is only permitted for synthetic clients")
        return self


class TripSummary(Schema):
    partition: str
    trip_id: TripID
    observation_count: NonNegative
    description: str = ""


class TripList(Schema):
    trips: list[TripSummary]


class Observation(Schema):
    observation_id: NonNegative
    coordinate: Coordinate
    relative_timestamp_s: float | None = None


class Arc(Schema):
    arc_id: int
    coordinates: list[Coordinate] = Field(min_length=2)
    from_node: int | None = None
    to_node: int | None = None


class Geometry(Schema):
    partition: str
    trip_id: TripID
    observations: list[Observation]
    arcs: list[Arc]
    synthetic: bool = False

    @model_validator(mode="after")
    def unique_ids(self):
        for records, field in ((self.observations, "observation_id"), (self.arcs, "arc_id")):
            ids = [getattr(row, field) for row in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {field}")
        return self


class PredictionRequest(Schema):
    partition: str
    trip_id: TripID
    mode: ModeID = "phase3_fixed"


class CandidateSelection(Schema):
    observation_id: NonNegative
    arc_id: int
    candidate_rank: Annotated[int, Field(ge=1)] | None = None
    probability: Probability | None = None


class DecodedRoute(Schema):
    provenance: Literal["hmm_viterbi", "synthetic_demo"]
    arc_ids: list[int]
    coordinates: list[Coordinate] = Field(min_length=2)


class PredictionResult(Schema):
    result_id: str
    partition: str
    trip_id: TripID
    mode: ModeID
    model_id: str
    observation_count: NonNegative
    selections: list[CandidateSelection] = Field(default_factory=list)
    empty_observation_ids: list[NonNegative] = Field(default_factory=list)
    decoded_route: DecodedRoute | None = None
    phase5: Phase5Result | None = None
    runtime_s: Annotated[float, Field(ge=0)] | None = None
    continuity_metrics: dict[str, float | None] = Field(default_factory=dict)
    evaluation_metrics: dict[str, float | None] = Field(default_factory=dict)
    synthetic: bool = False

    @model_validator(mode="after")
    def consistent_result(self):
        ids = [s.observation_id for s in self.selections]
        empty = self.empty_observation_ids
        if len(ids) != len(set(ids)) or len(empty) != len(set(empty)) or set(ids) & set(empty):
            raise ValueError("Selected and empty observation groups must be unique and disjoint")
        if self.phase5:
            decoded = self.phase5
            if self.mode != "hmm_viterbi" or self.decoded_route is not None or self.selections:
                raise ValueError("Phase 5 results require HMM mode and their own segmented geometry/matches")
            if (self.partition, self.trip_id, self.model_id, self.observation_count) != (decoded.partition, decoded.trip_id, decoded.model_id, decoded.observation_count):
                raise ValueError("Phase 5 result identity does not match its UI envelope")
            if self.runtime_s != decoded.processing_time_s:
                raise ValueError("Phase 5 runtime must preserve processing_time_s")
            if self.empty_observation_ids != [m.observation_id for m in decoded.matches if m.candidate_count == 0]:
                raise ValueError("Empty groups must reflect candidate_count, not unmatched status")
        elif len(ids) + len(empty) != self.observation_count:
            raise ValueError("Every observation requires a selection or explicit empty group")
        if self.mode == "demo_route" and not self.synthetic:
            raise ValueError("Demo mode requires synthetic data")
        if self.decoded_route:
            if self.decoded_route.provenance == "synthetic_demo" and not self.synthetic:
                raise ValueError("Synthetic route must be labelled synthetic")
            if self.decoded_route.provenance == "hmm_viterbi" and self.mode != "hmm_viterbi":
                raise ValueError("Genuine decoded route requires HMM mode")
        return self


class PredictionJob(Schema):
    result_id: str
    status: Literal["queued", "running", "completed", "failed"]
    error: str | None = None
    result: PredictionResult | None = None

    @model_validator(mode="after")
    def completed_result(self):
        if self.status == "completed" and (self.result is None or self.result.result_id != self.result_id):
            raise ValueError("Completed job must contain its matching result")
        if self.status != "completed" and self.result is not None:
            raise ValueError("Only completed jobs may contain results")
        return self


class ErrorResponse(Schema):
    code: str
    message: str
    request_id: str | None = None
