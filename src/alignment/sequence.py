"""Sparse monotonic alignment over reference-route occurrences."""

import numpy as np


class PrefixBest:
    """Fenwick maximum over route occurrences, with stable backpointer ties."""

    def __init__(self, size: int):
        self.tree = [(-float("inf"), 0)] * (size + 1)

    def before(self, occurrence: int) -> tuple[float, int]:
        best = (-float("inf"), 0)
        while occurrence:
            best = max(best, self.tree[occurrence])
            occurrence -= occurrence & -occurrence
        return best

    def add(self, occurrence: int, value: tuple[float, int]) -> None:
        position = occurrence + 1
        while position < len(self.tree):
            self.tree[position] = max(value, self.tree[position])
            position += position & -position


def monotonic_path(options, gaps: np.ndarray, sections: np.ndarray, *,
                   match_reward: float, skip_cost: float, break_cost: float) -> dict[int, int]:
    """Return observation-to-option indices for a maximum-scoring sparse chain.

    Each match contributes reward minus local distance/heading cost. Advancing
    pays per skipped occurrence and per crossed section. At a timestamp gap the
    skipped-occurrence charge is waived. Rows can be skipped at zero reward.
    """
    regular = PrefixBest(len(sections))
    gap_tree = PrefixBest(len(sections))
    same = [(-float("inf"), 0)] * len(sections)
    nodes = [None]  # Backpointer 0 starts a chain.
    best = (0.0, 0)
    gap_since = False

    for obs, choices in enumerate(options):
        gap_since |= bool(gaps[obs])
        if not len(choices.occurrences):
            continue
        tree, skip = (gap_tree, 0.0) if gap_since else (regular, skip_cost)
        gap_since = False
        pending = []
        for ix, (occurrence, local_cost) in enumerate(zip(choices.occurrences, choices.costs,
                                                         strict=True)):
            occ = int(occurrence)
            section_cost = break_cost * int(sections[occ])
            prefix_score, prefix_node = tree.before(occ)
            predecessor_score, predecessor = max(
                (0.0, 0), same[occ],
                (prefix_score - skip * (occ - 1) - section_cost, prefix_node),
            )
            value = predecessor_score + match_reward - float(local_cost)
            node_id = len(nodes)
            nodes.append((obs, ix, -predecessor))
            best = max(best, (value, -node_id))
            pending.append((occ, value, -node_id, section_cost))

        # Commit only after scoring the row. One observation contributes at most once.
        for occ, value, backpointer, section_cost in pending:
            same[occ] = max(same[occ], (value, backpointer))
            regular.add(occ, (value + skip_cost * occ + section_cost, backpointer))
            gap_tree.add(occ, (value + section_cost, backpointer))

    path = {}
    node_id = -best[1]
    while node_id:
        obs, ix, node_id = nodes[node_id]
        path[obs] = ix
    return path
