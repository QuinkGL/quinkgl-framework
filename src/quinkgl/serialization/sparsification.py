"""
Gradient Sparsification and Delta Compression.

Implements Deep Gradient Compression (Lin et al., 2018) for
reducing bandwidth by sending only the most significant gradients,
and delta compression for sending only changed weights.

References:
    Deep Gradient Compression (DGC): Lin et al., ICLR 2018
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SparsificationConfig:
    top_k_ratio: float = 0.01
    method: str = "top_k"


@dataclass
class DeltaCompressionConfig:
    enabled: bool = True


@dataclass
class SparseUpdate:
    indices: np.ndarray
    values: np.ndarray
    shape: tuple
    original_dtype: str


def sparsify_weights(
    weights: Any,
    config: Optional[SparsificationConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Sparsify model weights by keeping only top-k significant values.

    For each weight tensor, only the top-k% of values (by magnitude)
    are retained. The rest are set to zero and not transmitted.

    Args:
        weights: Model weights (numpy array or dict of arrays).
        config: Sparsification configuration.

    Returns:
        Tuple of (sparse_weights, metadata) where metadata contains
        information needed for desparsification.
    """
    config = config or SparsificationConfig()

    if isinstance(weights, np.ndarray):
        return _sparsify_array(weights, config)
    elif isinstance(weights, dict):
        sparse_dict = {}
        meta_dict = {}
        for key, value in weights.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                s, m = _sparsify_array(value, config)
                sparse_dict[key] = s
                meta_dict[key] = m
            else:
                sparse_dict[key] = value
                meta_dict[key] = None
        return sparse_dict, meta_dict
    else:
        return weights, {}


def desparsify_weights(
    sparse_weights: Any,
    metadata: Dict[str, Any],
    base_weights: Optional[Any] = None,
) -> Any:
    """
    Reconstruct full weights from sparse update.

    If base_weights is provided, the sparse values are applied on top
    of the base (delta reconstruction). Otherwise, zeros fill the
    non-sparse positions.

    Args:
        sparse_weights: Sparse weights (dict with indices/values or full array).
        metadata: Metadata from sparsification.
        base_weights: Optional base weights for delta reconstruction.

    Returns:
        Full reconstructed weights.
    """
    if isinstance(sparse_weights, dict) and sparse_weights.get("__sparse_weight__"):
        # S7b: single tensor in new sparse format (was a numpy array before sparsification).
        base = base_weights if isinstance(base_weights, np.ndarray) else None
        return _desparsify_array(sparse_weights, metadata, base)
    elif isinstance(sparse_weights, dict):
        # Dict of weight tensors — each value may be sparse dict or plain array.
        result = {}
        for key, value in sparse_weights.items():
            meta = metadata.get(key)
            if meta is not None and isinstance(meta, dict) and "sparse" in meta:
                base = None
                if base_weights is not None and isinstance(base_weights, dict):
                    base = base_weights.get(key)
                result[key] = _desparsify_array(value, meta, base)
            else:
                if base_weights is not None and isinstance(base_weights, dict):
                    result[key] = base_weights.get(key, value)
                else:
                    result[key] = value
        return result
    elif isinstance(sparse_weights, np.ndarray) and metadata:
        # Legacy dense sparse array.
        base = base_weights if isinstance(base_weights, np.ndarray) else None
        return _desparsify_array(sparse_weights, metadata, base)
    else:
        return sparse_weights


def compute_delta(
    current_weights: Any,
    base_weights: Any,
) -> Any:
    """
    Compute the delta between current and base weights.

    Args:
        current_weights: Current model weights.
        base_weights: Previous (reference) weights.

    Returns:
        Delta weights (current - base).
    """
    if isinstance(current_weights, np.ndarray):
        return current_weights.astype(np.float64) - base_weights.astype(np.float64)
    elif isinstance(current_weights, dict):
        delta = {}
        for key in current_weights:
            if key in base_weights and isinstance(current_weights[key], np.ndarray):
                delta[key] = (
                    current_weights[key].astype(np.float64)
                    - base_weights[key].astype(np.float64)
                )
            else:
                delta[key] = current_weights[key]
        # S6a: mark keys removed from current_weights with a None tombstone so
        # apply_delta does not silently restore them from base_weights.
        for key in base_weights:
            if key not in current_weights:
                delta[key] = None
        return delta
    else:
        return current_weights


def apply_delta(
    base_weights: Any,
    delta: Any,
) -> Any:
    """
    Apply delta to base weights to reconstruct current weights.

    Args:
        base_weights: Base weights.
        delta: Delta to apply.

    Returns:
        Reconstructed weights (base + delta).
    """
    if isinstance(base_weights, np.ndarray):
        result = base_weights.astype(np.float64) + delta.astype(np.float64)
        return result.astype(base_weights.dtype)
    elif isinstance(base_weights, dict):
        result = {}
        for key in base_weights:
            if key in delta:
                if delta[key] is None:
                    # S6b: tombstone — key was deleted from current_weights; skip restoration.
                    continue
                elif isinstance(delta[key], np.ndarray):
                    result[key] = (
                        base_weights[key].astype(np.float64) + delta[key].astype(np.float64)
                    ).astype(base_weights[key].dtype)
                else:
                    result[key] = base_weights[key]
            else:
                result[key] = base_weights[key]
        for key in delta:
            if key not in base_weights and delta[key] is not None:
                result[key] = delta[key]
        return result
    else:
        return base_weights


def _sparsify_array(
    arr: np.ndarray,
    config: SparsificationConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Sparsify a single numpy array.

    S7a: Returns a sparse dict ``{"__sparse_weight__": True, "indices": int32_arr,
    "values": float_arr}`` instead of a full dense array, providing real bandwidth
    savings proportional to ``top_k_ratio``.
    """
    if not np.issubdtype(arr.dtype, np.floating):
        return arr, None

    flat = arr.flatten().astype(np.float64)
    total = len(flat)
    k = max(1, int(total * config.top_k_ratio))

    abs_flat = np.abs(flat)

    if config.method == "top_k":
        if k >= total:
            return arr, {
                "sparse": False,
                "original_shape": list(arr.shape),
                "original_dtype": str(arr.dtype),
            }

        threshold_idx = np.argpartition(abs_flat, -k)[-k:]
        threshold_idx = np.sort(threshold_idx)

        # S7a: sparse representation — indices + values only, not a full dense array.
        sparse_repr = {
            "__sparse_weight__": True,
            "indices": threshold_idx.astype(np.int32),
            "values": flat[threshold_idx].astype(arr.dtype),
        }
        meta = {
            "sparse": True,
            "format": "indices_values",
            "top_k_ratio": config.top_k_ratio,
            "k": k,
            "total": total,
            "non_zero_count": k,
            "original_shape": list(arr.shape),
            "original_dtype": str(arr.dtype),
        }
        return sparse_repr, meta
    else:
        raise ValueError(f"Unknown sparsification method: {config.method}")


def _desparsify_array(
    sparse_arr: Any,
    meta: Dict[str, Any],
    base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Desparsify a single numpy array.

    Handles both the new sparse-dict format (S7b) and the legacy dense format.
    S3a: emits a warning when base=None and sparse=True so callers are alerted
    to lossy reconstruction (non-sparse positions will be zeroed).
    """
    if not meta.get("sparse", False):
        if base is not None:
            return base
        return sparse_arr

    original_dtype = np.dtype(meta.get("original_dtype", meta.get("dtype", "float32")))
    original_shape = tuple(meta.get("original_shape", meta.get("shape", [])))
    total = meta.get("total", int(np.prod(original_shape)))

    if isinstance(sparse_arr, dict) and sparse_arr.get("__sparse_weight__"):
        # S7b: new sparse format — reconstruct from indices + values.
        if base is None:
            logger.warning(
                "desparsify: base_weights=None with sparse=True — non-sparse positions "
                "will be filled with zeros. This is correct only when sparsification was "
                "applied to a delta (not absolute weights). For absolute weight "
                "reconstruction, provide base_weights."
            )
        indices = sparse_arr["indices"]
        values = sparse_arr["values"]

        if base is not None:
            result = base.flatten().astype(np.float64)
        else:
            result = np.zeros(total, dtype=np.float64)

        result[indices] = values.astype(np.float64)
    else:
        # Legacy dense sparse format.
        if base is None:
            logger.warning(
                "desparsify: base_weights=None with sparse=True — non-sparse positions "
                "will be filled with zeros. This is correct only when sparsification was "
                "applied to a delta (not absolute weights). For absolute weight "
                "reconstruction, provide base_weights."
            )
        if base is not None:
            result = base.astype(np.float64) + sparse_arr.astype(np.float64)
        else:
            result = sparse_arr.astype(np.float64)

    return result.reshape(original_shape).astype(original_dtype)
