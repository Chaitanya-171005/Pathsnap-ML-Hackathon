"""Log-space emission and directed transition evidence."""

from dataclasses import dataclass
from math import isfinite, log

from .graph import Path


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    distance_scale_m: float = 25.0
    transition_weight: float = 0.5
    position_slack_m: float = 15.0
    search_budget: int = 4096
    cache_size: int = 4096
    restart_cost: float = 4.0

    def __post_init__(self):
        if not (isfinite(self.distance_scale_m) and self.distance_scale_m > 0):
            raise ValueError("Distance scale must be positive and finite")
        if not all(isfinite(value) and value >= 0
                   for value in (self.transition_weight, self.position_slack_m, self.restart_cost)):
            raise ValueError("Transition weight, slack and restart cost must be finite and nonnegative")
        if type(self.search_budget) is not int or self.search_budget < 1:
            raise ValueError("Routing search budget must be a positive integer")
        if type(self.cache_size) is not int or self.cache_size < 0:
            raise ValueError("Routing cache size must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class State:
    arc_id: int
    rank: int
    probability: float
    fraction: float
    emission: float


def emission(probability):
    """Use clipped positive-class log scores. Invalid scores remove a state."""
    if probability is None or not isfinite(probability) or not 0 <= probability <= 1:
        return None
    return log(min(max(probability, 1e-12), 1 - 1e-12))


@dataclass(frozen=True, slots=True)
class Transition:
    score: float
    path: Path
    same_arc_forward: bool


def transition(graph, first, second, observed_m, delta_s, max_speed_mps, config):
    limit = max_speed_mps * delta_s + config.position_slack_m
    path, reason = graph.connect(first, second, limit)
    if path is None:
        return None, reason
    mismatch = abs(path.distance_m - observed_m)
    speed_excess = max(0.0, path.distance_m - max_speed_mps * delta_s)
    penalty = config.transition_weight * (mismatch + speed_excess) / config.distance_scale_m
    if not isfinite(penalty):
        return None, "invalid_transition_score"
    same_arc_forward = first.arc_id == second.arc_id and second.fraction >= first.fraction
    return Transition(-penalty, path, same_arc_forward), None
