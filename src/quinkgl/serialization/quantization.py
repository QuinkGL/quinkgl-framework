"""
Weight Quantization for Bandwidth Optimization.

Implements QSGD-style quantization (Alistarh et al., 2017) for
reducing the size of model updates transmitted over the network.

References:
    QSGD: Communication-Efficient SGD via Gradient Quantization
    with Encoding (Alistarh et al., NeurIPS 2017)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    bits: int = 8
    method: str = "linear"
    seed: int = 42  # RNG seed for reproducibility

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.bits not in [4, 8, 16, 32]:
            raise ValueError(
                f"Invalid bits value: {self.bits}. "
                f"Must be one of [4, 8, 16, 32]."
            )
        if self.method not in ["linear", "stochastic"]:
            raise ValueError(
                f"Invalid quantization method: {self.method}. "
                f"Must be 'linear' or 'stochastic'."
            )


@dataclass
class QuantizationResult:
    quantized: Any
    scale: Any
    zero_point: Any
    original_dtype: Any
    original_shape: Any


def quantize_weights(
    weights: Any,
    config: Optional[QuantizationConfig] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Quantize model weights to reduced bit-width.

    Args:
        weights: Model weights (numpy array or dict of arrays).
        config: Quantization configuration.

    Returns:
        Tuple of (quantized_weights, metadata) where metadata
        contains scale factors needed for dequantization.
    """
    config = config or QuantizationConfig()

    if isinstance(weights, np.ndarray):
        q, meta = _quantize_array(weights, config)
        return q, meta
    elif isinstance(weights, dict):
        quantized_dict = {}
        meta_dict = {}
        for key, value in weights.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                q, m = _quantize_array(value, config)
                quantized_dict[key] = q
                meta_dict[key] = m
            else:
                quantized_dict[key] = value
                meta_dict[key] = None
        return quantized_dict, meta_dict
    else:
        return weights, {}


def dequantize_weights(
    quantized: Any,
    metadata: Dict[str, Any],
) -> Any:
    """
    Dequantize weights back to float32 using stored scale factors.

    Args:
        quantized: Quantized weights (numpy array or dict).
        metadata: Scale factors from quantization.

    Returns:
        Dequantized weights in original format.
    """
    if isinstance(quantized, np.ndarray) and metadata:
        return _dequantize_array(quantized, metadata)
    elif isinstance(quantized, dict):
        result = {}
        for key, value in quantized.items():
            meta = metadata.get(key)
            if meta is not None and isinstance(value, np.ndarray):
                result[key] = _dequantize_array(value, meta)
            else:
                result[key] = value
        return result
    else:
        return quantized


def _quantize_array(
    arr: np.ndarray,
    config: QuantizationConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quantize a single numpy array."""
    if not np.issubdtype(arr.dtype, np.floating):
        return arr, None

    flat = arr.flatten().astype(np.float64)
    num_levels = 2 ** config.bits

    if config.method == "linear":
        arr_min = flat.min()
        arr_max = flat.max()

        if arr_max == arr_min:
            # S4a: use scale=0.0 as a sentinel for constant-value tensors so
            # dequantization can reconstruct the constant instead of returning zeros.
            quantized = np.zeros_like(arr, dtype=np.uint8)
            meta = {
                "scale": 0.0,
                "zero_point": float(arr_min),
                "original_dtype": str(arr.dtype),
                "original_shape": list(arr.shape),
                "method": "linear",
                "bits": config.bits,
            }
            return quantized, meta

        scale = (arr_max - arr_min) / (num_levels - 1)
        zero_point = arr_min

        normalized = (arr.astype(np.float64) - zero_point) / scale
        quantized = np.clip(np.round(normalized), 0, num_levels - 1).astype(
            np.uint8 if config.bits <= 8 else np.uint16
        )

        meta = {
            "scale": float(scale),
            "zero_point": float(zero_point),
            "original_dtype": str(arr.dtype),
            "original_shape": list(arr.shape),
            "method": "linear",
            "bits": config.bits,
        }
        return quantized, meta

    elif config.method == "qsgd":
        norm = np.linalg.norm(flat)
        if norm == 0:
            quantized = np.zeros_like(arr, dtype=np.uint8)
            meta = {
                "norm": 0.0,
                "original_dtype": str(arr.dtype),
                "original_shape": list(arr.shape),
                "method": "qsgd",
                "bits": config.bits,
            }
            return quantized, meta

        sign = np.sign(arr.astype(np.float64))
        normalized = np.abs(arr.astype(np.float64)) / norm * num_levels

        noise = np.random.uniform(0, 1, size=normalized.shape)
        quantized = np.floor(normalized + noise).clip(0, num_levels - 1).astype(
            np.uint8 if config.bits <= 8 else np.uint16
        )

        # S2a: encode sign as a packed bit-vector (1 bit per element) so the
        # metadata is msgpack-safe and ~32x smaller than a float32 array.
        # Convention: bit=1 → value was ≥ 0 (positive or zero), bit=0 → negative.
        sign_flat = sign.flatten()
        sign_bits = np.packbits((sign_flat >= 0).astype(np.uint8))
        meta = {
            "norm": float(norm),
            "sign_bits": sign_bits.tobytes(),
            "original_dtype": str(arr.dtype),
            "original_shape": list(arr.shape),
            "method": "qsgd",
            "bits": config.bits,
        }
        return quantized, meta

    else:
        raise ValueError(f"Unknown quantization method: {config.method}")


def _dequantize_array(
    quantized: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """Dequantize a single numpy array."""
    method = meta.get("method", "linear")
    original_dtype = np.dtype(meta["original_dtype"])
    original_shape = tuple(meta["original_shape"])

    if method == "linear":
        scale = meta["scale"]
        zero_point = meta["zero_point"]
        if scale == 0.0:
            # S4b: constant-value tensor — reconstruct to the stored constant.
            result = np.full(original_shape, zero_point, dtype=np.float64)
        else:
            result = quantized.astype(np.float64) * scale + zero_point

    elif method == "qsgd":
        norm = meta["norm"]
        num_levels = 2 ** meta["bits"]
        total = int(np.prod(original_shape))

        # S2b: decode sign from packed bits (new format) or fall back to legacy array.
        sign_bits_data = meta.get("sign_bits")
        legacy_sign = meta.get("sign")
        if sign_bits_data is not None:
            # New format: packed bit-vector
            sign_bits = np.frombuffer(sign_bits_data, dtype=np.uint8)
            sign_unpacked = np.unpackbits(sign_bits)[:total].astype(np.float64)
            sign = np.where(sign_unpacked > 0, 1.0, -1.0)
        elif isinstance(legacy_sign, np.ndarray):
            sign = legacy_sign.flatten().astype(np.float64)
        else:
            sign = np.ones(total, dtype=np.float64)

        flat_q = quantized.flatten().astype(np.float64)
        # S8a: validate that sign and quantized arrays have the same length.
        if len(sign) != len(flat_q):
            raise ValueError(
                f"QSGD sign/quantized length mismatch: "
                f"sign={len(sign)}, quantized={len(flat_q)}"
            )
        flat_result = flat_q / num_levels * norm * sign
        result = flat_result

    else:
        raise ValueError(f"Unknown quantization method: {method}")

    return result.reshape(original_shape).astype(original_dtype)
