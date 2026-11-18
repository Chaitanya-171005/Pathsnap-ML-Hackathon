"""Parse headerless PathSnap files without sorting, renumbering or dropping rows."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


PARTITIONS = {"train": 80, "student_test": 10}
EXTENSIONS = ("track", "nodes", "arcs", "route")
INTEGER = re.compile(r"[+-]?[0-9]+")


@dataclass(frozen=True)
class TripFiles:
    partition: str
    trip_id: str
    directory: Path

    def path(self, extension: str) -> Path:
        return self.directory / f"{self.trip_id}.{extension}"


@dataclass(frozen=True)
class ParseIssue:
    file: str
    code: str
    line: int | None = None
    column: int | None = None


@dataclass
class Trip:
    files: TripFiles
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray
    issues: list[ParseIssue]


def discover_trips(data_dir: Path) -> list[TripFiles]:
    """Discover ASCII-numeric trip directory names in student partitions only."""
    trips = []
    for partition in PARTITIONS:
        directory = data_dir / partition
        if not directory.is_dir():
            continue
        trips.extend(
            TripFiles(partition, path.name, path)
            for path in sorted(directory.iterdir())
            if path.is_dir() and path.name.isascii() and path.name.isdecimal()
        )
    return trips


def _read(path: Path, columns: int, integers: bool) -> tuple[np.ndarray, list[ParseIssue]]:
    issues = []
    rows = []
    sentinel = -1 if integers else np.nan
    dtype = np.int64 if integers else np.float64
    try:
        # The formats are ASCII. Replacing undecodable bytes makes bad tokens
        # visible as parse issues without losing the rest of the file.
        with path.open(encoding="ascii", errors="replace") as source:
            for line, raw in enumerate(source, 1):
                tokens = raw.split()
                if len(tokens) != columns:
                    issues.append(ParseIssue(path.suffix, "column_count", line))
                if len(tokens) > columns:
                    rows.append([sentinel] * columns)
                    continue
                row = []
                for column, token in enumerate(tokens, 1):
                    try:
                        if integers:
                            if not INTEGER.fullmatch(token):
                                raise ValueError
                            value = int(token)
                            if not -(2**63) <= value < 2**63:
                                raise ValueError
                        else:
                            value = float(token)
                    except ValueError:
                        issues.append(ParseIssue(path.suffix, "invalid_integer" if integers else
                                                 "invalid_number", line, column))
                        value = sentinel
                    row.append(value)
                rows.append(row + [sentinel] * (columns - len(row)))
    except FileNotFoundError:
        issues.append(ParseIssue(path.suffix, "missing_file"))
        return np.empty((0, columns), dtype=dtype), issues
    except OSError:
        issues.append(ParseIssue(path.suffix, "unreadable_file"))
        return np.empty((0, columns), dtype=dtype), issues
    if not rows:
        issues.append(ParseIssue(path.suffix, "empty_file"))
    return np.asarray(rows, dtype=dtype).reshape(-1, columns), issues


def load_track(path: Path) -> tuple[np.ndarray, list[ParseIssue]]:
    return _read(path, 3, False)


def load_nodes(path: Path) -> tuple[np.ndarray, list[ParseIssue]]:
    return _read(path, 2, False)


def load_arcs(path: Path) -> tuple[np.ndarray, list[ParseIssue]]:
    return _read(path, 2, True)


def load_route(path: Path) -> tuple[np.ndarray, list[ParseIssue]]:
    rows, issues = _read(path, 1, True)
    return rows[:, 0], issues


def load_trip(files: TripFiles) -> Trip:
    track, track_issues = load_track(files.path("track"))
    nodes, node_issues = load_nodes(files.path("nodes"))
    arcs, arc_issues = load_arcs(files.path("arcs"))
    route, route_issues = load_route(files.path("route"))
    return Trip(files, track, nodes, arcs, route,
                track_issues + node_issues + arc_issues + route_issues)
