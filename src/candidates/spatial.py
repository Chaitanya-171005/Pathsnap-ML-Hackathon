"""Trip-local projected road geometry and radius search for directed arc IDs."""

from collections.abc import Iterator
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

from src.preprocessing.geometry import EARTH_RADIUS_M
from src.preprocessing.validation import valid_coordinates


@dataclass(frozen=True)
class LocalProjection:
    longitude_origin: float
    latitude_origin: float

    @classmethod
    def from_nodes(cls, nodes: np.ndarray) -> "LocalProjection":
        if not len(nodes):
            return cls(0.0, 0.0)
        return cls(float(nodes[0, 0]), float(np.mean(nodes[:, 1])))

    def project(self, coordinates: np.ndarray) -> np.ndarray:
        longitude, latitude = coordinates.T
        delta = longitude - self.longitude_origin
        delta = np.where(delta >= 180, delta - 360, np.where(delta < -180, delta + 360, delta))
        scale = EARTH_RADIUS_M * np.pi / 180
        return np.column_stack((delta * scale * np.cos(np.radians(self.latitude_origin)),
                                (latitude - self.latitude_origin) * scale))


def project_onto_segments(point: np.ndarray, starts: np.ndarray,
                          ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clamped distances in metres and directed fractions for planar segments.

    Degenerate segments are points, with a NaN fraction. Use a common endpoint
    order for distance arithmetic so reverse arcs have exactly the same distance.
    """
    reverse = (starts[:, 0] > ends[:, 0]) | (
        (starts[:, 0] == ends[:, 0]) & (starts[:, 1] > ends[:, 1]))
    first = np.where(reverse[:, None], ends, starts)
    last = np.where(reverse[:, None], starts, ends)
    delta = last - first
    offset = point - first
    squared_length = np.einsum("ij,ij->i", delta, delta)
    nonzero = squared_length > 0
    fraction = np.divide(np.einsum("ij,ij->i", offset, delta), squared_length,
                         out=np.zeros(len(starts)), where=nonzero)
    np.clip(fraction, 0.0, 1.0, out=fraction)
    residual = offset - fraction[:, None] * delta
    # Use the original endpoint difference for endpoint clamps.
    residual[fraction == 1.0] = point - last[fraction == 1.0]
    directed = np.where(nonzero, np.where(reverse, 1.0 - fraction, fraction), np.nan)
    return np.hypot(residual[:, 0], residual[:, 1]), directed


@dataclass(frozen=True)
class MidpointGroup:
    tree: cKDTree
    indices: np.ndarray
    max_half_length: float


class RoadIndex:
    """Midpoint trees grouped by half-length to bound exact segment search."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        valid_nodes = valid_coordinates(nodes)
        self.projection = LocalProjection.from_nodes(nodes[valid_nodes])
        xy = self.projection.project(nodes)
        half_world_m = np.pi * EARTH_RADIUS_M * np.cos(np.radians(self.projection.latitude_origin))
        if valid_nodes.any() and np.ptp(xy[valid_nodes, 0]) > half_world_m:
            raise ValueError("Road network is too wide for the trip-local projection")

        endpoints_valid = ((arcs >= 0) & (arcs < len(nodes))).all(axis=1)
        self.in_degree = np.bincount(arcs[endpoints_valid, 1], minlength=len(nodes))
        self.out_degree = np.bincount(arcs[endpoints_valid, 0], minlength=len(nodes))
        has_geometry = endpoints_valid.copy()
        has_geometry[endpoints_valid] = valid_nodes[arcs[endpoints_valid]].all(axis=1)
        self.arc_ids = np.flatnonzero(has_geometry)
        self.endpoints = arcs[has_geometry]
        self.start, self.end = (xy[self.endpoints[:, 0]], xy[self.endpoints[:, 1]])
        direction = self.end - self.start
        self.length_m = np.hypot(direction[:, 0], direction[:, 1])
        self.heading_deg = np.degrees(np.arctan2(direction[:, 0], direction[:, 1])) % 360
        self.heading_deg[self.length_m == 0] = np.nan

        half_length = self.length_m / 2
        midpoints = self.start + direction / 2
        exponents = np.frexp(np.maximum(1.0, half_length))[1]
        self.groups = []
        for exponent in np.unique(exponents):
            indices = np.flatnonzero(exponents == exponent)
            self.groups.append(MidpointGroup(cKDTree(midpoints[indices]), indices,
                                             float(half_length[indices].max())))

    def nearby(self, points: np.ndarray, radius_m: float
               ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield internal indices, distances and fractions within radius in input order.

        A segment within the radius has a midpoint no farther away than the radius
        plus half its length. Bulk tree queries are bounded to a batch of points.
        """
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("radius_m must be positive and finite")
        for offset in range(0, len(points), 128):
            batch = points[offset:offset + 128]
            found = [[] for _ in batch]
            for group in self.groups:
                search_radius = radius_m + group.max_half_length + 1e-7
                hits = group.tree.query_ball_point(batch, search_radius, return_sorted=False)
                for indices, local_hits in zip(found, hits, strict=True):
                    if local_hits:
                        indices.extend(group.indices[local_hits])
            for point, indices in zip(batch, found, strict=True):
                indices = np.asarray(indices, dtype=np.intp)
                distances, fractions = project_onto_segments(point, self.start[indices],
                                                            self.end[indices])
                within = distances <= radius_m
                indices, distances, fractions = indices[within], distances[within], fractions[within]
                order = np.lexsort((self.arc_ids[indices], distances))
                yield indices[order], distances[order], fractions[order]
