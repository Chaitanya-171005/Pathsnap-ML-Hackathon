"""Viterbi backpointers, with deterministic state ordering and segment resets."""

from dataclasses import dataclass
from math import inf

from .hmm import Transition


@dataclass(slots=True)
class Cell:
    score: float
    previous: int | None
    transition: Transition | None
    break_reason: str | None = None


class Viterbi:
    """Maintain one chain of active columns and backtrack at a sequence break."""

    def __init__(self):
        self.columns = []

    def start(self, observation_id, states):
        self.columns.append((observation_id, states, [
            Cell(state.emission, None, None) for state in states
        ]))

    def active(self):
        _, states, cells = self.columns[-1]
        return [(i, state) for i, (state, cell) in enumerate(zip(states, cells, strict=True))
                if cell.score != -inf]

    def advance(self, observation_id, states, links, resets=(), restart_cost=0):
        previous = self.columns[-1][2]
        cells = [Cell(-inf, None, None) for _ in states]
        connected = False
        for i, j, link in links:
            score = previous[i].score + link.score + states[j].emission
            # Links arrive in ascending previous-state order.
            if score > cells[j].score:
                cells[j] = Cell(score, i, link)
                connected = True
        if connected:
            # A restart competes only for a pair without a feasible directed link.
            for i, j, reason in resets:
                score = previous[i].score - restart_cost + states[j].emission
                if score > cells[j].score:
                    cells[j] = Cell(score, i, None, reason)
            self.columns.append((observation_id, states, cells))
        return connected

    def finish(self):
        if not self.columns:
            return []
        cells = self.columns[-1][2]
        index = max(range(len(cells)), key=lambda i: cells[i].score)
        decoded = []
        for obs_id, states, cells in reversed(self.columns):
            cell = cells[index]
            decoded.append((obs_id, states[index], cell.transition, cell.break_reason))
            index = cell.previous
        self.columns.clear()
        return list(reversed(decoded))
