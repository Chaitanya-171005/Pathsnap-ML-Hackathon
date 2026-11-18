"""Presentation-only validation of docs/PHASE5_OUTPUT_CONTRACT.md (schema 1.0).

No decoder, road graph, model artifact or reference-route dependencies.
"""
from collections import Counter
from typing import Any, Literal
from pydantic import Field, model_validator
from frontend.models.base import Coordinate, NonNegative, Probability, Schema, TripID


class ObservationMatch(Schema):
    partition: str
    trip_id: TripID
    observation_id: NonNegative
    arc_id: NonNegative | None
    candidate_rank: int | None = Field(ge=1)
    model_score: Probability | None
    segment_id: NonNegative | None
    classifier_arc_id: NonNegative | None
    candidate_count: NonNegative
    status: Literal["matched", "unmatched"]
    break_reason: str | None

    @model_validator(mode="after")
    def check_match(self):
        selected = (self.arc_id, self.candidate_rank, self.model_score, self.segment_id)
        if self.status == "matched":
            if any(value is None for value in selected):
                raise ValueError("Matched observations require arc, rank, score and segment")
            if self.candidate_rank > self.candidate_count:
                raise ValueError("Candidate rank exceeds candidate count")
        elif any(value is not None for value in selected):
            raise ValueError("Unmatched observations must retain null selected values")
        if self.candidate_count == 0 and self.classifier_arc_id is not None:
            raise ValueError("Empty candidate groups cannot have a classifier selection")
        return self


class ArcOccurrence(Schema):
    arc_id: NonNegative
    role: Literal["selected", "connector"]
    first_observation_id: NonNegative | None = None
    last_observation_id: NonNegative | None = None

    @model_validator(mode="after")
    def check_range(self):
        if self.role == "selected":
            if self.first_observation_id is None or self.last_observation_id is None:
                raise ValueError("Selected occurrences require observation bounds")
            if self.first_observation_id > self.last_observation_id:
                raise ValueError("Invalid occurrence observation range")
        return self


class RouteSegment(Schema):
    segment_id: NonNegative
    break_reason: str | None
    first_observation_id: NonNegative
    last_observation_id: NonNegative
    arc_ids: list[NonNegative] = Field(min_length=1)
    occurrences: list[ArcOccurrence] = Field(min_length=1)
    matched_distance_m: float = Field(ge=0)
    full_arc_length_m: float = Field(ge=0)

    @model_validator(mode="after")
    def check_occurrences(self):
        if self.arc_ids != [occurrence.arc_id for occurrence in self.occurrences]:
            raise ValueError("Directed arc occurrences must retain segment order and repetitions")
        if self.first_observation_id > self.last_observation_id:
            raise ValueError("Invalid segment observation range")
        return self


class LineString(Schema):
    type: Literal["LineString"]
    coordinates: list[Coordinate] = Field(min_length=2)


class SegmentProperties(Schema):
    segment_id: NonNegative
    break_reason: str | None
    partition: str
    trip_id: TripID
    model_id: str
    arc_count: NonNegative


class SegmentFeature(Schema):
    type: Literal["Feature"]
    id: NonNegative
    properties: SegmentProperties
    geometry: LineString


class RouteGeoJSON(Schema):
    type: Literal["FeatureCollection"]
    features: list[SegmentFeature]


class RoutingStats(Schema):
    searches: NonNegative
    cache_hits: NonNegative
    budget_exhaustions: NonNegative


class Phase5Result(Schema):
    schema_version: Literal["1.0"]
    partition: str
    trip_id: TripID
    model_id: str
    observation_count: NonNegative
    status: Literal["connected", "partial", "disconnected", "empty"]
    decoder_config: dict[str, Any]
    matches: list[ObservationMatch]
    route_segments: list[RouteSegment]
    route_geojson: RouteGeoJSON
    break_counts: dict[str, NonNegative]
    routing: RoutingStats
    processing_time_s: float = Field(ge=0)

    @model_validator(mode="after")
    def check_contract(self):
        if len(self.matches) != self.observation_count or [m.observation_id for m in self.matches] != list(range(self.observation_count)):
            raise ValueError("matches must retain one record per source row in original order")
        if any((m.partition, m.trip_id) != (self.partition, self.trip_id) for m in self.matches):
            raise ValueError("Match identity differs from the result")
        if [s.segment_id for s in self.route_segments] != list(range(len(self.route_segments))):
            raise ValueError("Segment IDs must be zero-based in route order")
        if len(self.route_geojson.features) != len(self.route_segments):
            raise ValueError("GeoJSON must contain one LineString per segment")
        matched = [m for m in self.matches if m.status == "matched"]
        groups = {s.segment_id: [] for s in self.route_segments}
        for match in matched:
            if match.segment_id not in groups:
                raise ValueError("Match references an unknown segment")
            groups[match.segment_id].append(match.observation_id)
        for segment, feature in zip(self.route_segments, self.route_geojson.features):
            props = feature.properties
            ids = groups[segment.segment_id]
            if not ids or (ids[0], ids[-1]) != (segment.first_observation_id, segment.last_observation_id):
                raise ValueError("Segment observation bounds differ from matches")
            if feature.id != segment.segment_id or props.segment_id != segment.segment_id:
                raise ValueError("GeoJSON segment IDs differ from route segments")
            if (props.partition, props.trip_id, props.model_id) != (self.partition, self.trip_id, self.model_id):
                raise ValueError("GeoJSON identity differs from the result")
            if props.arc_count != len(segment.arc_ids) or props.break_reason != segment.break_reason:
                raise ValueError("GeoJSON segment metadata differs from route segments")
        expected = ("empty" if not self.route_segments else "disconnected" if len(self.route_segments) > 1
                    else "partial" if len(matched) != self.observation_count else "connected")
        if self.status != expected:
            raise ValueError("Route status disagrees with segment and match counts")
        if self.break_counts != dict(Counter(m.break_reason for m in self.matches if m.break_reason is not None)):
            raise ValueError("break_counts must count annotated match records")
        return self
