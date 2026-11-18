"""Distances take (longitude, latitude) in degrees and return metres."""

import numpy as np


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start_lon, start_lat = np.moveaxis(np.radians(start), -1, 0)
    end_lon, end_lat = np.moveaxis(np.radians(end), -1, 0)
    a = (np.sin((end_lat - start_lat) / 2) ** 2 + np.cos(start_lat) * np.cos(end_lat)
         * np.sin((end_lon - start_lon) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _wrap_longitude(delta: np.ndarray) -> np.ndarray:
    return (delta + 180) % 360 - 180


def nearest_segment_m(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Approximate distance to the nearest segment in each point's local tangent plane.

    Each segment is projected using the point's latitude and the shortest longitude
    difference. Work is batched to bound temporary arrays even for long routes.
    """
    if not len(segments):
        return np.full(len(points), np.nan)
    distances = np.empty(len(points))
    metres_per_degree = EARTH_RADIUS_M * np.pi / 180
    start, end = segments[:, 0], segments[:, 1]
    batch_size = max(1, 200_000 // len(segments))
    for offset in range(0, len(points), batch_size):
        longitude, latitude = points[offset:offset + batch_size].T[:, :, None]
        lon_scale = metres_per_degree * np.cos(np.radians(latitude))
        x = _wrap_longitude(start[:, 0] - longitude) * lon_scale
        y = (start[:, 1] - latitude) * metres_per_degree
        dx = _wrap_longitude(end[:, 0] - start[:, 0]) * lon_scale
        dy = (end[:, 1] - start[:, 1]) * metres_per_degree
        length_squared = dx**2 + dy**2
        fraction = np.divide(-(x * dx + y * dy), length_squared,
                             out=np.zeros_like(x), where=length_squared > 0)
        fraction = np.clip(fraction, 0, 1)
        distances[offset:offset + len(x)] = np.sqrt(
            ((x + fraction * dx)**2 + (y + fraction * dy)**2).min(axis=1)
        )
    return distances
