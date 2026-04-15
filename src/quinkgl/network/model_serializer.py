"""
Model Serialization Utilities

Handles serialization and deserialization of model weights
for P2P transmission.

SECURITY: This module uses safe serialization (msgpack + numpy)
instead of pickle to prevent arbitrary code execution vulnerabilities.
"""

import base64
import io
import logging
import struct
from typing import Any, Dict, Union

import msgpack
import numpy as np

logger = logging.getLogger(__name__)

# Maximum size for serialized models (100 MB) to prevent DoS
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024


def _serialize_numpy_array(arr: np.ndarray) -> bytes:
    """
    Safely serialize a numpy array to bytes.

    Uses numpy's native save format which is safe and efficient.
    """
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def _deserialize_numpy_array(data: bytes) -> np.ndarray:
    """
    Safely deserialize bytes to a numpy array.
    """
    buffer = io.BytesIO(data)
    return np.load(buffer, allow_pickle=False)


def serialize_model(weights: Any, enable_compression: bool = False) -> bytes:
    """
    Serialize model weights to bytes for transmission.

    Uses msgpack for structured data and numpy's native format for arrays.
    This is safe from arbitrary code execution vulnerabilities.

    Args:
        weights: Model weights (dict, numpy array, or list)
        enable_compression: Whether to compress the output

    Returns:
        Serialized bytes (base64 encoded)

    Raises:
        ValueError: If serialization fails or model is too large
    """
    try:
        if isinstance(weights, dict):
            # Serialize each value appropriately
            serializable = {}
            for key, value in weights.items():
                if isinstance(value, np.ndarray):
                    # Use numpy's binary format for arrays
                    serializable[key] = {
                        "__type__": "numpy.ndarray",
                        "__data__": base64.b64encode(_serialize_numpy_array(value)).decode('utf-8'),
                        "dtype": str(value.dtype),
                        "shape": value.shape
                    }
                elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                    serializable[key] = value
                else:
                    # Try to convert to numpy array
                    try:
                        arr = np.array(value)
                        serializable[key] = {
                            "__type__": "numpy.ndarray",
                            "__data__": base64.b64encode(_serialize_numpy_array(arr)).decode('utf-8'),
                            "dtype": str(arr.dtype),
                            "shape": arr.shape
                        }
                    except Exception:
                        serializable[key] = str(value)
            data = msgpack.packb(serializable, use_bin_type=True)

        elif isinstance(weights, np.ndarray):
            # Single numpy array - wrap in a dict structure
            wrapped = {
                "__type__": "numpy.ndarray",
                "__data__": base64.b64encode(_serialize_numpy_array(weights)).decode('utf-8'),
                "dtype": str(weights.dtype),
                "shape": weights.shape
            }
            data = msgpack.packb(wrapped, use_bin_type=True)

        elif isinstance(weights, (list, tuple)):
            # Convert list/tuple to numpy array for efficiency
            arr = np.array(weights)
            wrapped = {
                "__type__": "numpy.ndarray",
                "__data__": base64.b64encode(_serialize_numpy_array(arr)).decode('utf-8'),
                "dtype": str(arr.dtype),
                "shape": arr.shape
            }
            data = msgpack.packb(wrapped, use_bin_type=True)

        else:
            # Simple types - serialize directly
            data = msgpack.packb(weights, use_bin_type=True)

        # Check size before returning
        if len(data) > MAX_MODEL_SIZE_BYTES:
            raise ValueError(
                f"Model size ({len(data) / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({MAX_MODEL_SIZE_BYTES / 1024 / 1024:.2f} MB)"
            )

        # Base64 encode for safe transmission
        result = base64.b64encode(data)

        # Optional compression
        if enable_compression and len(result) > 10240:  # Only compress if > 10KB
            import zlib
            compressed = zlib.compress(result, level=6)
            logger.debug(f"Compressed model: {len(result)} -> {len(compressed)} bytes")
            return base64.b64encode(b"ZLIB" + compressed)

        return result

    except Exception as e:
        logger.error(f"Failed to serialize model: {e}")
        raise ValueError(f"Model serialization failed: {e}")


def deserialize_model(data: bytes) -> Any:
    """
    Deserialize model weights from bytes.

    Uses msgpack for structured data and numpy's native format for arrays.
    This is safe from arbitrary code execution vulnerabilities.

    Args:
        data: Serialized bytes (base64 encoded)

    Returns:
        Model weights (original format)

    Raises:
        ValueError: If deserialization fails or data is malformed
    """
    try:
        # Base64 decode
        decoded = base64.b64decode(data)

        # Check for compression marker
        if decoded[:4] == b"ZLIB":
            import zlib
            decoded = zlib.decompress(decoded[4:])
            logger.debug("Decompressed model data")

        # Check size limit before unpacking
        if len(decoded) > MAX_MODEL_SIZE_BYTES:
            raise ValueError(
                f"Data size ({len(decoded) / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({MAX_MODEL_SIZE_BYTES / 1024 / 1024:.2f} MB)"
            )

        # Unpack using msgpack
        unpacked = msgpack.unpackb(decoded, raw=False)

        # Helper to deserialize numpy array from wrapped format
        def deserialize_array(wrapped: dict) -> np.ndarray:
            """Deserialize a wrapped numpy array."""
            if wrapped.get("__type__") != "numpy.ndarray":
                raise ValueError(f"Invalid type marker: {wrapped.get('__type__')}")
            array_bytes = base64.b64decode(wrapped["__data__"])
            return _deserialize_numpy_array(array_bytes)

        # Handle different return types
        if isinstance(unpacked, dict):
            # Check if this is a single wrapped array
            if "__type__" in unpacked and unpacked["__type__"] == "numpy.ndarray":
                return deserialize_array(unpacked)

            # Otherwise, it's a dict of values
            result = {}
            for key, value in unpacked.items():
                if isinstance(value, dict) and value.get("__type__") == "numpy.ndarray":
                    result[key] = deserialize_array(value)
                elif isinstance(value, list):
                    # Could be a serialized array or just a list
                    result[key] = value
                else:
                    result[key] = value
            return result

        elif isinstance(unpacked, list):
            return np.array(unpacked)

        return unpacked

    except Exception as e:
        logger.error(f"Failed to deserialize model: {e}")
        raise ValueError(f"Model deserialization failed: {e}")


def get_model_size_info(weights: Any) -> Dict[str, Any]:
    """
    Get size information about model weights.

    Args:
        weights: Model weights

    Returns:
        Dict with size information
    """
    # Count parameters
    param_count = 0
    if isinstance(weights, dict):
        for value in weights.values():
            if hasattr(value, 'size'):
                param_count += value.size
            elif isinstance(value, list):
                param_count += len(value)
    elif hasattr(weights, 'size'):
        param_count = weights.size
    elif isinstance(weights, list):
        param_count = len(weights)

    # Get actual serialized size
    serialized = serialize_model(weights, enable_compression=False)
    size_bytes = len(serialized)
    size_mb = size_bytes / (1024 * 1024)

    return {
        "parameter_count": param_count,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "size_human": f"{size_mb:.2f} MB"
    }


def compress_weights(weights: Any, compression_level: int = 6) -> bytes:
    """
    Compress model weights using zlib.

    Args:
        weights: Model weights
        compression_level: Compression level (0-9)

    Returns:
        Compressed bytes (base64 encoded)
    """
    import zlib

    # Use the new safe serialization with compression enabled
    serialized = serialize_model(weights, enable_compression=False)

    compressed = zlib.compress(serialized, level=compression_level)

    original_size = len(serialized)
    compressed_size = len(compressed)
    ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

    logger.debug(
        f"Compression: {original_size} -> {compressed_size} bytes "
        f"({ratio:.1f}% reduction)"
    )

    return base64.b64encode(b"ZLIB" + compressed)


def decompress_weights(data: bytes) -> Any:
    """
    Decompress model weights.

    Args:
        data: Compressed bytes (base64 encoded)

    Returns:
        Model weights
    """
    # The deserialize_model function now handles ZLIB-compressed data
    return deserialize_model(data)
