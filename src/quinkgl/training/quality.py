"""
Global Model Quality Assessment.

Measures inter-peer model similarity using cosine similarity
between weight vectors to detect convergence across nodes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_weight_fingerprint(weights: Any) -> Dict[str, Any]:
    """
    Compute a lightweight fingerprint of model weights.

    Includes norm and top-k values for quick comparison
    without transmitting full weights.

    Args:
        weights: Model weights (numpy array or dict).

    Returns:
        Dict with fingerprint data.
    """
    if isinstance(weights, np.ndarray):
        flat = weights.flatten().astype(np.float64)
        norm = float(np.linalg.norm(flat))
        top_k = min(5, len(flat))
        top_indices = np.argpartition(np.abs(flat), -top_k)[-top_k:]
        return {
            "norm": norm,
            "top_values": flat[top_indices].tolist(),
            "total_elements": int(flat.size),
            "dtype": str(weights.dtype),
        }
    elif isinstance(weights, dict):
        total_norm = 0.0
        total_elements = 0
        for v in weights.values():
            if isinstance(v, np.ndarray):
                flat = v.flatten().astype(np.float64)
                total_norm += float(np.linalg.norm(flat)) ** 2
                total_elements += int(flat.size)
        return {
            "norm": float(np.sqrt(total_norm)),
            "total_elements": total_elements,
            "layer_count": len(weights),
        }
    return {}


def cosine_similarity_weights(a: Any, b: Any) -> float:
    """
    Compute cosine similarity between two sets of weights.

    Args:
        a: First set of weights.
        b: Second set of weights.

    Returns:
        Cosine similarity in [-1, 1]. 1.0 = identical direction.
    """
    a_flat = _flatten_weights(a)
    b_flat = _flatten_weights(b)

    if a_flat is None or b_flat is None:
        return 0.0

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compute_peer_similarity(
    updates: List[Any],
) -> Dict[str, float]:
    """
    Compute average pairwise cosine similarity across model updates.

    Args:
        updates: List of model weights from different peers.

    Returns:
        Dict with mean, min, max similarity and peer count.
    """
    if len(updates) < 2:
        return {"mean_similarity": 1.0, "min_similarity": 1.0, "max_similarity": 1.0, "peer_count": len(updates)}

    flat_weights = []
    for w in updates:
        flat = _flatten_weights(w)
        if flat is not None:
            flat_weights.append(flat)

    if len(flat_weights) < 2:
        return {"mean_similarity": 1.0, "min_similarity": 1.0, "max_similarity": 1.0, "peer_count": len(flat_weights)}

    similarities = []
    for i in range(len(flat_weights)):
        for j in range(i + 1, len(flat_weights)):
            sim = cosine_similarity_weights(flat_weights[i], flat_weights[j])
            similarities.append(sim)

    if not similarities:
        return {"mean_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0, "peer_count": len(flat_weights)}

    return {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "peer_count": len(flat_weights),
    }


def _flatten_weights(weights: Any) -> Optional[np.ndarray]:
    """Flatten weights into a single 1D numpy array."""
    if isinstance(weights, np.ndarray):
        return weights.flatten().astype(np.float64)
    elif isinstance(weights, dict):
        parts = []
        for key in sorted(weights.keys()):
            v = weights[key]
            if isinstance(v, np.ndarray):
                parts.append(v.flatten().astype(np.float64))
        if parts:
            return np.concatenate(parts)
    return None
