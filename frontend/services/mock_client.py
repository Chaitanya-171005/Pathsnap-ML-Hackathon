"""Deterministic invented street grid. No organiser data, network or model files."""
import math
from frontend.models.schemas import (
    Arc, Capabilities, CandidateSelection, DecodedRoute, Geometry, Health, Observation,
    PredictionJob, PredictionMode, PredictionResult, TripList, TripSummary,
)
from frontend.services.api_client import ClientError, RequestStatus


class MockPathSnapClient:
    IDS = ("00000083", "00000026", "00000054")

    def __init__(self):
        self.last_request = RequestStatus()
        self.jobs = {}

    def _record(self, endpoint):
        self.last_request = RequestStatus(endpoint, "Synthetic demo · local", 0.0)

    def health(self):
        self._record("demo/health")
        return Health(status="ok", version="synthetic-1")

    def capabilities(self):
        self._record("demo/capabilities")
        return Capabilities(partitions=["synthetic_city"], synthetic=True, modes=[
            PredictionMode(id="phase3_fixed", label="Phase 3 fixed · simulated selections", model_id="gradient_boosting", available=True),
            PredictionMode(id="phase4_tuned", label="Phase 4 tuned · experimental", model_id="gradient_boosting_tuned", available=False, reason="Requires a live backend with the tuned model"),
            PredictionMode(id="hmm_viterbi", label="HMM / Viterbi", model_id="hmm_viterbi", available=False, reason="Decoder complete; awaiting live FastAPI connection"),
            PredictionMode(id="demo_route", label="Synthetic route example · not HMM output", model_id="synthetic_example", available=True),
        ])

    def trips(self, partition):
        if partition != "synthetic_city":
            raise ClientError("Unknown demo partition")
        self._record("demo/trips")
        return TripList(trips=[TripSummary(partition=partition, trip_id=key, observation_count=180 + i * 60,
            description=f"Invented city grid · scenario {i + 1}") for i, key in enumerate(self.IDS)])

    def trip(self, partition, trip_id):
        for trip in self.trips(partition).trips:
            if trip.trip_id == trip_id:
                return trip
        raise ClientError("Unknown demo trip")

    def geometry(self, partition, trip_id):
        trip = self.trip(partition, trip_id)
        index = self.IDS.index(trip_id)
        origin = (77.59 + index * .008, 12.96 + index * .003)
        def point(x, y):
            return origin[0] + x * .0015, origin[1] + y * .0015
        arcs = []
        for y in range(7):
            for x in range(7):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx <= 6 and y + dy <= 6:
                        a, b = y * 7 + x, (y + dy) * 7 + x + dx
                        coords = [point(x, y), point(x + dx, y + dy)]
                        arcs.extend([Arc(arc_id=len(arcs), coordinates=coords, from_node=a, to_node=b),
                                     Arc(arc_id=len(arcs) + 1, coordinates=coords[::-1], from_node=b, to_node=a)])
        observations = []
        for i in range(trip.observation_count):
            x, y = route_position(i / (trip.observation_count - 1))
            lon, lat = point(x, y)
            observations.append(Observation(observation_id=i,
                coordinate=(lon + .00013 * math.sin(i * 1.7 + index), lat + .00011 * math.cos(i * 1.3)),
                relative_timestamp_s=float(i * 2)))
        self._record("demo/geometry")
        return Geometry(partition=partition, trip_id=trip_id, observations=observations, arcs=arcs, synthetic=True)

    def predict(self, request):
        if not any(m.id == request.mode and m.available for m in self.capabilities().modes):
            raise ClientError("This prediction mode is unavailable in Demo mode.")
        geometry = self.geometry(request.partition, request.trip_id)
        selections, empty = [], []
        route_arcs = []
        for arc in geometry.arcs:
            # Explicit synthetic continuous path: east, north, then east.
            a, b = arc.from_node, arc.to_node
            if (a // 7 == b // 7 == 1 and b == a + 1 and a % 7 < 4) or (a % 7 == b % 7 == 4 and b == a + 7 and 1 <= a // 7 < 5) or (a // 7 == b // 7 == 5 and b == a + 1 and 4 <= a % 7 < 6):
                route_arcs.append(arc)
        route_arcs.sort(key=lambda a: (0 if a.from_node // 7 == 1 and a.to_node == a.from_node + 1 else 1 if a.to_node == a.from_node + 7 else 2, a.from_node))
        for obs in geometry.observations:
            if obs.observation_id % 43 == 0:
                empty.append(obs.observation_id)
                continue
            segment = min(int(obs.observation_id / (len(geometry.observations) - 1) * 10), 9)
            arc = route_arcs[segment]
            selections.append(CandidateSelection(observation_id=obs.observation_id, arc_id=arc.arc_id,
                candidate_rank=1 + (obs.observation_id % 3), probability=round(.78 + .19 * abs(math.sin(obs.observation_id)), 5)))
        result_id = f"demo-{request.trip_id}-{request.mode}"
        route = None
        if request.mode == "demo_route":
            route = DecodedRoute(provenance="synthetic_demo", arc_ids=[a.arc_id for a in route_arcs],
                coordinates=[route_arcs[0].coordinates[0]] + [a.coordinates[-1] for a in route_arcs])
        result = PredictionResult(result_id=result_id, partition=request.partition, trip_id=request.trip_id,
            mode=request.mode, model_id="synthetic_example" if route else "gradient_boosting (simulated)",
            observation_count=len(geometry.observations), selections=selections, empty_observation_ids=empty,
            decoded_route=route, synthetic=True)
        job = PredictionJob(result_id=result_id, status="completed", result=result)
        self.jobs[result_id] = job
        self._record("demo/predictions")
        return job

    def result(self, result_id):
        if result_id not in self.jobs:
            raise ClientError("Demo result not found; run prediction again.")
        self._record("demo/results")
        return self.jobs[result_id]


def route_position(t):
    distance = t * 10
    if distance <= 4:
        return distance, 1
    if distance <= 8:
        return 4, 1 + distance - 4
    return 4 + distance - 8, 5
