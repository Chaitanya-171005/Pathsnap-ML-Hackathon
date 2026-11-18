"""Validity masks preserve raw row identity and exclude unsafe indexing."""

from dataclasses import dataclass

import numpy as np

from .loaders import Trip


def valid_coordinates(coordinates: np.ndarray) -> np.ndarray:
    return (np.isfinite(coordinates).all(axis=1)
            & (np.abs(coordinates) <= [180, 90]).all(axis=1))


@dataclass(frozen=True)
class TripValidity:
    gps: np.ndarray
    timestamps: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    arc_geometry: np.ndarray
    route_ids: np.ndarray
    route_geometry: np.ndarray
    route_disconnections: np.ndarray
    route_uncheckable: np.ndarray
    timestamp_deltas: np.ndarray


def validate_trip(trip: Trip) -> TripValidity:
    gps = valid_coordinates(trip.track[:, :2])
    timestamps = np.isfinite(trip.track[:, 2])
    nodes = valid_coordinates(trip.nodes)
    arcs = ((trip.arcs >= 0) & (trip.arcs < len(nodes))).all(axis=1)
    arc_geometry = arcs.copy()
    arc_geometry[arcs] = nodes[trip.arcs[arcs]].all(axis=1)
    route_ids = (trip.route >= 0) & (trip.route < len(arcs))
    route_geometry = route_ids.copy()
    route_geometry[route_ids] = arc_geometry[trip.route[route_ids]]
    usable = route_ids.copy()
    usable[route_ids] = arcs[trip.route[route_ids]]
    checkable = usable[:-1] & usable[1:]
    disconnected = np.zeros(len(checkable), dtype=bool)
    disconnected[checkable] = (trip.arcs[trip.route[:-1][checkable], 1]
                               != trip.arcs[trip.route[1:][checkable], 0])
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = np.diff(trip.track[:, 2])
    deltas[~(timestamps[:-1] & timestamps[1:] & np.isfinite(deltas))] = np.nan
    return TripValidity(gps, timestamps, nodes, arcs, arc_geometry, route_ids,
                        route_geometry, disconnected, ~checkable, deltas)
