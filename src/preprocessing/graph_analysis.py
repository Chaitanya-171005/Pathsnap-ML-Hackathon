"""Connectivity via sparse directed adjacency, with trip-local node indices."""

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def topology(node_count: int, arcs: np.ndarray) -> dict:
    starts, ends = arcs.T
    graph = csr_matrix((np.ones(len(arcs), dtype=np.int64), (starts, ends)),
                       shape=(node_count, node_count))
    incident = np.bincount(arcs.ravel(), minlength=node_count)
    result = {
        "duplicate_directed_edges": len(arcs) - graph.nnz,
        "self_loops": int((starts == ends).sum()),
        "isolated_nodes": int((incident == 0).sum()),
    }
    for connection in ("weak", "strong"):
        count, labels = connected_components(graph, directed=True, connection=connection)
        sizes = np.bincount(labels)
        result[f"{connection}_components"] = count
        result[f"{connection}_largest_nodes"] = int(sizes.max(initial=0))
        result[f"{connection}_size_frequency"] = dict(sorted(Counter(sizes.tolist()).items()))
    return result
