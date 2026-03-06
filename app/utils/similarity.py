"""
Cosine similarity utility for normalized vectors.

Since all embeddings are L2-normalized, cosine similarity
simplifies to the dot product, avoiding division overhead.
"""

import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two unit-normalized vectors.

    For normalized vectors: cos_sim(a, b) = dot(a, b).

    Args:
        v1: First normalized vector.
        v2: Second normalized vector.

    Returns:
        Similarity score in [-1, 1].
    """
    return float(np.dot(v1, v2))
