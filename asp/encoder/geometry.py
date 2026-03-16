"""
Latent Space Geometry Operations

Pure functions for computing distances and similarities in the
embedding space.  No side effects, no state.

These operations define what "close to an attack manifold" means
geometrically.  The key insight: adversarial prompts that bypass
keyword filters still cluster in embedding space because their
semantic INTENT is preserved even when surface form changes.

Cite: adversarial ML literature on latent space attack manifolds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cosine similarity between two vectors.

    Both vectors are assumed to be L2-normalized (output of EmbeddingAdapter).
    In that case, cosine sim = dot product.
    """
    dot = float(np.dot(a, b))
    # Clamp to [-1, 1] to handle floating point drift
    return max(-1.0, min(1.0, dot))


def cosine_similarity_matrix(
    query: NDArray[np.float64], db: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Cosine similarities between a query vector and a matrix of vectors.

    Args:
        query: shape (dim,)
        db: shape (n, dim)

    Returns:
        shape (n,) array of similarity scores.
    """
    # For L2-normalized vectors: sim = query @ db.T
    return db @ query


def find_nearest(
    query: NDArray[np.float64], db: NDArray[np.float64], ids: list[str]
) -> tuple[str, float]:
    """Find the nearest vector in db to query.

    Returns:
        (id, similarity) of the nearest match.
    """
    if len(ids) == 0:
        return ("", -1.0)

    sims = cosine_similarity_matrix(query, db)
    best_idx = int(np.argmax(sims))
    return ids[best_idx], float(sims[best_idx])


def manifold_distance(
    point: NDArray[np.float64], manifold_points: NDArray[np.float64], k: int = 3
) -> float:
    """Average distance to the k nearest points on a manifold.

    A more robust measure than single-nearest-neighbor: reduces
    sensitivity to outlier signatures in the attack DB.
    """
    if manifold_points.shape[0] == 0:
        return float("inf")

    sims = cosine_similarity_matrix(point, manifold_points)
    k = min(k, len(sims))
    top_k = np.sort(sims)[-k:]
    # Convert similarity to distance: distance = 1 - similarity
    return float(1.0 - np.mean(top_k))
