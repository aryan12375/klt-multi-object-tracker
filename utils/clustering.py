"""
utils/clustering.py — DBSCAN-based point grouping
===================================================
Groups raw feature points into per-object clusters.
No sklearn dependency — pure NumPy DBSCAN implementation
for portability in embedded/lab environments.
"""

import numpy as np
from typing import List


def cluster_points_to_objects(
    points: np.ndarray,        # (N, 2) float32
    eps: float = 60.0,
    min_samples: int = 3,
) -> List[np.ndarray]:
    """
    Run DBSCAN on a flat (N,2) point array.
    Returns a list of arrays, each being the points for one cluster.
    Noise points (label=-1) are discarded.
    """
    if len(points) == 0:
        return []

    labels = _dbscan(points, eps, min_samples)
    unique = set(labels) - {-1}
    clusters = []
    for lbl in unique:
        mask = labels == lbl
        clusters.append(points[mask])
    return clusters


def _dbscan(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
    Pure NumPy DBSCAN implementation.
    Returns array of integer labels, -1 = noise.
    Complexity: O(N²) — fine for typical feature counts (<500 pts).
    """
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # Precompute pairwise squared distances
    diff = X[:, None, :] - X[None, :, :]         # (N,N,2)
    dist2 = (diff ** 2).sum(axis=-1)              # (N,N)
    in_eps = dist2 <= eps ** 2                    # (N,N) bool

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = np.where(in_eps[i])[0]

        if len(neighbours) < min_pts:
            labels[i] = -1   # noise for now
            continue

        # Start a new cluster
        labels[i] = cluster_id
        seed_set = list(neighbours)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = np.where(in_eps[q])[0]
                if len(q_neighbours) >= min_pts:
                    for nb in q_neighbours:
                        if nb not in seed_set:
                            seed_set.append(nb)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels
