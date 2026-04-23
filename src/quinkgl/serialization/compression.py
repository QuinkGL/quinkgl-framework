"""
Compression Pipeline Configuration.

Combines quantization, sparsification, delta compression,
and zlib into a single configurable pipeline for model
weight transmission.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from quinkgl.serialization.quantization import (
    QuantizationConfig,
    quantize_weights,
    dequantize_weights,
)
from quinkgl.serialization.sparsification import (
    SparsificationConfig,
    DeltaCompressionConfig,
    sparsify_weights,
    desparsify_weights,
    compute_delta,
    apply_delta,
)
from quinkgl.serialization.error_feedback import ErrorFeedbackState

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    quantization: Optional[QuantizationConfig] = None
    sparsification: Optional[SparsificationConfig] = None
    delta_compression: DeltaCompressionConfig = field(default_factory=DeltaCompressionConfig)
    zlib_compression: bool = True
    zlib_threshold_bytes: int = 10240
    error_feedback: bool = False
    # TASK-092: Downcast weights to half/bfloat16 before transmission.
    # Set to "float16", "bfloat16", or None (no downcast).
    weight_dtype: Optional[str] = None


def compress_weights(
    weights: Any,
    config: CompressionConfig,
    base_weights: Optional[Any] = None,
    ef_state: Optional[ErrorFeedbackState] = None,
) -> tuple:
    """
    Apply the full compression pipeline to weights.

    Pipeline order: Delta → Sparsify → Quantize → Serialize → Zlib

    Args:
        weights: Current model weights.
        config: Compression configuration.
        base_weights: Previous weights for delta computation.

    Returns:
        Tuple of (compressed_data, compression_meta) where
        compression_meta is needed for decompression.
    """
    meta: Dict[str, Any] = {
        "pipeline_version": 1,
        "steps": [],
        "original_size": _estimate_weight_size(weights),
    }
    processed = weights

    # TASK-092: Step 0 — Downcast weights to half/bfloat16 before the rest
    # of the pipeline.  This reduces wire size by ~50% with minimal accuracy
    # loss for well-conditioned models.
    if config.weight_dtype is not None:
        original_dtypes = _record_dtypes(processed)
        processed = _downcast_weights(processed, config.weight_dtype)
        meta["steps"].append("downcast")
        meta["original_dtypes"] = original_dtypes
        meta["weight_dtype"] = config.weight_dtype

    # Step 1: Delta compression
    if config.delta_compression.enabled and base_weights is not None:
        processed = compute_delta(processed, base_weights)
        meta["steps"].append("delta")
        meta["has_delta"] = True
    else:
        meta["has_delta"] = False

    # Step 2: Error feedback — inject residual before sparsification
    pre_sparse = None
    if config.error_feedback and ef_state is not None and config.sparsification is not None:
        pre_sparse = processed  # save uncorrected delta for residual computation
        processed = ef_state.apply(processed)

    # Step 3: Sparsification
    if config.sparsification is not None:
        processed, sparse_meta = sparsify_weights(processed, config.sparsification)
        meta["steps"].append("sparsify")
        meta["sparse_meta"] = sparse_meta

    # Step 4: Error feedback — store new residual
    if config.error_feedback and ef_state is not None and pre_sparse is not None:
        # S1a: pass pre_sparse (uncorrected delta), not ef_state.apply(pre_sparse),
        # to avoid double-applying the residual and compounding corrections each round.
        #
        # The EF residual = corrected − compressed (both must be dense numpy arrays).
        # After S7, `processed` may be a sparse dict; reconstruct the dense form so
        # the residual computation in _update_array works correctly.
        compressed_for_ef = processed
        if isinstance(processed, dict):
            compressed_for_ef = desparsify_weights(
                processed,
                meta.get("sparse_meta", {}),
                base_weights=None,
            )
        ef_state.update(pre_sparse, compressed_for_ef)
        meta["steps"].append("error_feedback")
        meta["ef_residual_norm"] = ef_state.total_residual_norm

    # Step 5: Quantization
    if config.quantization is not None:
        processed, quant_meta = quantize_weights(processed, config.quantization)
        meta["steps"].append("quantize")
        meta["quant_meta"] = quant_meta

    # Step 6: Serialize
    from quinkgl.network.model_serializer import serialize_model
    serialized = serialize_model(processed, enable_compression=False)

    # Step 7: Zlib compression
    if config.zlib_compression and len(serialized) > config.zlib_threshold_bytes:
        import zlib
        compressed = zlib.compress(serialized, level=6)
        ratio = (1 - len(compressed) / len(serialized)) * 100
        logger.debug(
            f"Zlib compression: {len(serialized)} -> {len(compressed)} bytes "
            f"({ratio:.1f}% reduction)"
        )
        serialized = compressed
        meta["steps"].append("zlib")

    meta["compressed_size"] = len(serialized)
    return serialized, meta


def compress_decompress_roundtrip(
    weights: Any,
    config: CompressionConfig,
    base_weights: Optional[Any] = None,
    ef_state: Optional[ErrorFeedbackState] = None,
) -> Any:
    """T-12: Enforce the full compress→decompress pipeline in a single call.

    This function compresses the weights and immediately decompresses them,
    guaranteeing that the pipeline is applied as a unit and no step is
    bypassed.  Use this for testing, validation, or when you need the
    reconstructed weights after a full round-trip through the pipeline.

    Args:
        weights: Current model weights.
        config: Compression configuration.
        base_weights: Previous weights for delta computation.
        ef_state: Optional error-feedback state.

    Returns:
        Reconstructed weights after a full compress→decompress cycle.
    """
    compressed_data, meta = compress_weights(weights, config, base_weights, ef_state)
    return decompress_weights(compressed_data, meta, base_weights)


def decompress_weights(
    data: bytes,
    meta: Dict[str, Any],
    base_weights: Optional[Any] = None,
) -> Any:
    """
    Apply the decompression pipeline in reverse order.

    Pipeline order: Zlib → Deserialize → Dequantize → Desparsify → Apply delta

    Args:
        data: Compressed data.
        meta: Compression metadata from compress_weights.
        base_weights: Base weights for delta reconstruction.

    Returns:
        Reconstructed weights.
    """
    # S9a: validate pipeline_version so corrupted or mismatched metadata is detected early.
    version = meta.get("pipeline_version")
    if version is not None and version != 1:
        raise ValueError(
            f"Unsupported compression pipeline_version={version}. Expected 1."
        )

    processed_data = data

    # Step 1: Zlib decompression with streaming bytes-budget guard (NET-023/024)
    if "zlib" in meta.get("steps", []):
        import zlib
        # S-07: Streaming decompression with bytes-budget guard
        # Prevents decompression bomb attacks by checking budget incrementally
        max_expansion = len(processed_data) * 100
        decomp = zlib.decompressobj()
        chunks: list[bytes] = []
        total_bytes = 0
        chunk_size = 64 * 1024  # 64KB streaming chunks
        offset = 0
        while offset < len(processed_data):
            end = min(offset + chunk_size, len(processed_data))
            out = decomp.decompress(processed_data[offset:end], max_length=max_expansion - total_bytes)
            total_bytes += len(out)
            if total_bytes > max_expansion:
                raise ValueError(
                    f"Decompressed data exceeds budget: {total_bytes} > {max_expansion} bytes "
                    f"(compression ratio > 100x)"
                )
            chunks.append(out)
            offset = end
        # Flush any remaining data
        out = decomp.flush()
        total_bytes += len(out)
        if total_bytes > max_expansion:
            raise ValueError(
                f"Decompressed data exceeds budget: {total_bytes} > {max_expansion} bytes "
                f"(compression ratio > 100x)"
            )
        chunks.append(out)
        processed_data = b"".join(chunks)

    # Step 2: Deserialize
    from quinkgl.network.model_serializer import deserialize_model
    weights = deserialize_model(processed_data)

    # Step 3: Dequantize
    if "quantize" in meta.get("steps", []):
        quant_meta = meta.get("quant_meta")
        # S9b: raise instead of silently skipping when expected metadata is missing.
        if quant_meta is None:
            raise ValueError(
                "decompress_weights: 'quantize' step in pipeline but quant_meta is None. "
                "Metadata may be corrupted or truncated."
            )
        weights = dequantize_weights(weights, quant_meta)

    # Step 4: Desparsify
    if "sparsify" in meta.get("steps", []):
        sparse_meta = meta.get("sparse_meta")
        # S9b: raise instead of silently skipping.
        if sparse_meta is None:
            raise ValueError(
                "decompress_weights: 'sparsify' step in pipeline but sparse_meta is None. "
                "Metadata may be corrupted or truncated."
            )
        # S3b: when sparsification operates on a delta, base is not needed for
        # desparsify (zeros are correct for the delta itself). When operating on
        # absolute weights, base is mandatory for correct reconstruction.
        base = base_weights if not meta.get("has_delta", False) else None
        weights = desparsify_weights(weights, sparse_meta, base)

    # Step 5: Apply delta
    if meta.get("has_delta", False) and base_weights is not None:
        weights = apply_delta(base_weights, weights)

    # TASK-092: Step 6 — Upcast weights back to original dtype
    if "downcast" in meta.get("steps", []):
        original_dtypes = meta.get("original_dtypes")
        if original_dtypes is not None:
            weights = _upcast_weights(weights, original_dtypes)

    return weights


def _record_dtypes(weights: Any) -> Dict[str, str]:
    """TASK-092: Record original dtypes of weight arrays for upcast on decompress."""
    if isinstance(weights, dict):
        return {k: str(v.dtype) for k, v in weights.items() if hasattr(v, 'dtype')}
    elif hasattr(weights, 'dtype'):
        return {'__single__': str(weights.dtype)}
    return {}


def _downcast_weights(weights: Any, target_dtype: str) -> Any:
    """TASK-092: Downcast weight arrays to float16 or bfloat16."""
    import numpy as np
    np_dtype = np.dtype(target_dtype)
    if isinstance(weights, dict):
        return {k: v.astype(np_dtype) if hasattr(v, 'astype') else v for k, v in weights.items()}
    elif hasattr(weights, 'astype'):
        return weights.astype(np_dtype)
    return weights


def _upcast_weights(weights: Any, original_dtypes: Dict[str, str]) -> Any:
    """TASK-092: Upcast weight arrays back to their original dtypes."""
    import numpy as np
    if isinstance(weights, dict):
        result = {}
        for k, v in weights.items():
            if k in original_dtypes and hasattr(v, 'astype'):
                result[k] = v.astype(np.dtype(original_dtypes[k]))
            else:
                result[k] = v
        return result
    elif hasattr(weights, 'astype'):
        orig = original_dtypes.get('__single__', 'float32')
        return weights.astype(np.dtype(orig))
    return weights


def _estimate_weight_size(weights: Any) -> int:
    """Estimate the byte size of weights."""
    total = 0
    if isinstance(weights, dict):
        for v in weights.values():
            if hasattr(v, 'nbytes'):
                total += v.nbytes
            elif hasattr(v, 'size'):
                total += v.size * 4
    elif hasattr(weights, 'nbytes'):
        total = weights.nbytes
    elif hasattr(weights, 'size'):
        total = weights.size * 4
    return total
