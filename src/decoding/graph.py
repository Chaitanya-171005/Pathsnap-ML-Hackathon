"""Directed shortest paths over original road arc IDs."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf

from src.preprocessing.geometry import haversine_m


@dataclass(frozen=True, slots=True)
class Arc:
    start: int
    end: int
    length_m: float


@dataclass(frozen=True, slots=True)
class Path:
    distance_m: float
    arcs: tuple[int, ...]


class RoadGraph:
    def __init__(self, nodes, index, *, search_budget=4096, cache_size=4096):
        if search_budget < 1 or cache_size < 0:
            raise ValueError("Routing search budget must be positive and cache size nonnegative")
        self.arcs = {}
        self.outgoing = defaultdict(list)
        self.search_budget = search_budget
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.searches = self.cache_hits = self.budget_exhaustions = 0
        starts, ends = index.endpoints.T
        lengths = haversine_m(nodes[starts], nodes[ends])
        for arc_id, start, end, length, planar_length in zip(
            index.arc_ids, starts, ends, lengths, index.length_m, strict=True
        ):
            if length <= 1e-6 or planar_length <= 0:
                continue
            arc_id = int(arc_id)
            self.arcs[arc_id] = Arc(int(start), int(end), float(length))
            self.outgoing[int(start)].append(arc_id)

    def shortest(self, source, target, limit_m):
        """Return a path and optional failure reason under distance and work limits."""
        if limit_m < 0:
            return None, "movement_limit"
        if source == target:
            return Path(0.0, ()), None
        key = (source, target)
        cached = self.cache.get(key)
        if cached is not None:
            radius, path = cached
            if path is not None or limit_m <= radius:
                self.cache_hits += 1
                self.cache.move_to_end(key)
                if path is not None and path.distance_m <= limit_m:
                    return path, None
                return None, "no_directed_path_within_limit"

        self.searches += 1
        path, reason, radius = self._search(source, target, limit_m)
        if reason == "routing_budget_exceeded":
            self.budget_exhaustions += 1
        elif self.cache_size:
            self.cache[key] = (radius, path)
            self.cache.move_to_end(key)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return path, reason

    def _search(self, source, target, limit_m):
        queue = [(0.0, source)]
        distances = {source: 0.0}
        previous = {}
        work_left = self.search_budget
        pruned = False
        while queue:
            distance, node = heappop(queue)
            work_left -= 1
            if work_left < 0:
                return None, "routing_budget_exceeded", 0
            if distance != distances[node]:
                continue
            if node == target:
                arcs = []
                while node != source:
                    arc_id = previous[node]
                    arcs.append(arc_id)
                    node = self.arcs[arc_id].start
                return Path(distance, tuple(reversed(arcs))), None, inf
            for arc_id in self.outgoing.get(node, ()):
                work_left -= 1
                if work_left < 0:
                    return None, "routing_budget_exceeded", 0
                arc = self.arcs[arc_id]
                cost = distance + arc.length_m
                if cost >= distances.get(arc.end, inf):
                    continue
                if cost > limit_m:
                    pruned = True
                    continue
                distances[arc.end] = cost
                previous[arc.end] = arc_id
                heappush(queue, (cost, arc.end))
        return None, "no_directed_path_within_limit", limit_m if pruned else inf

    def connect(self, first, second, limit_m):
        """Use directed fractional positions and retain connector arc IDs."""
        left, right = self.arcs[first.arc_id], self.arcs[second.arc_id]
        if first.arc_id == second.arc_id and second.fraction >= first.fraction:
            distance = (second.fraction - first.fraction) * left.length_m
            if distance <= limit_m:
                return Path(distance, ()), None
            return None, "movement_limit"
        ends_m = (1 - first.fraction) * left.length_m + second.fraction * right.length_m
        path, reason = self.shortest(left.end, right.start, limit_m - ends_m)
        if path is None:
            return None, reason
        return Path(ends_m + path.distance_m, path.arcs), None
