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
from typing import Any, Dict

import msgpack
import numpy as np

logger = logging.getLogger(__name__)

# S-06: Wire format version byte for forward compatibility
WIRE_FORMAT_VERSION = 1

# Maximum size for serialized models (unified with MAX_INCOMING_MESSAGE_SIZE)
# Imported from gossip_community to avoid duplication
try:
    from quinkgl.network.gossip_community import MAX_INCOMING_MESSAGE_SIZE as MAX_MODEL_SIZE_BYTES
except ImportError:
    # Fallback if gossip_community not available
    MAX_MODEL_SIZE_BYTES = 150 * 1024 * 1024


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


def _to_serializable(value: Any) -> Any:
    """Recursively convert a value to a msgpack-safe structure.

    S5a: numpy arrays are stored as raw bytes (no inner base64 encoding).
    S7c: nested dicts containing numpy arrays are handled recursively, enabling
         the sparse weight format ``{"__sparse_weight__": True, "indices": ...,
         "values": ...}`` to round-trip through the serialization pipeline.
    """
    if isinstance(value, np.ndarray):
        return {
            "__type__": "numpy.ndarray",
            "__data__": _serialize_numpy_array(value),  # raw bytes — no base64
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    elif isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        # Attempt homogeneous numeric conversion to numpy for efficiency.
        try:
            arr = np.array(value)
            if arr.dtype != object:
                return {
                    "__type__": "numpy.ndarray",
                    "__data__": _serialize_numpy_array(arr),
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                }
        except Exception:
            pass
        return [_to_serializable(v) for v in value]
    else:
        # Last resort: try numpy conversion, then string fallback.
        try:
            arr = np.array(value)
            if arr.dtype != object:
                return {
                    "__type__": "numpy.ndarray",
                    "__data__": _serialize_numpy_array(arr),
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                }
        except Exception:
            pass
        return str(value)


def serialize_model(weights: Any, enable_compression: bool = False) -> bytes:
    """
    Serialize model weights to bytes for transmission.

    Uses msgpack for structured data and numpy's native format for arrays.
    This is safe from arbitrary code execution vulnerabilities.

    S5a: numpy arrays are stored as raw bytes inside msgpack (no inner base64),
    reducing wire size by ~33% per array compared to the legacy format.
    Outer base64 encoding is retained for transport-layer compatibility.

    S-06: Wire format versioning - version byte prepended for forward compatibility.

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
            serializable = {str(k): _to_serializable(v) for k, v in weights.items()}
        elif isinstance(weights, np.ndarray):
            serializable = _to_serializable(weights)
        elif isinstance(weights, (list, tuple)):
            arr = np.array(weights)
            serializable = _to_serializable(arr)
        else:
            serializable = weights

        data = msgpack.packb(serializable, use_bin_type=True)

        # Check size before returning
        if len(data) > MAX_MODEL_SIZE_BYTES:
            raise ValueError(
                f"Model size ({len(data) / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({MAX_MODEL_SIZE_BYTES / 1024 / 1024:.2f} MB)"
            )

        # S-06: Prepend wire format version byte
        versioned_data = bytes([WIRE_FORMAT_VERSION]) + data

        # Base64 encode for safe transmission
        result = base64.b64encode(versioned_data)

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

    S-06: Wire format versioning - validates version byte for forward compatibility.

    Args:
        data: Serialized bytes (base64 encoded)

    Returns:
        Model weights (original format)

    Raises:
        ValueError: If deserialization fails, data is malformed, or version mismatch
    """
    try:
        # Base64 decode
        decoded = base64.b64decode(data)

        # Check for compression marker
        if decoded[:4] == b"ZLIB":
            import zlib
            # NET-023/024: Streaming decompression with bytes-budget guard
            compressed = decoded[4:]
            max_expansion = len(compressed) * 100
            decomp = zlib.decompressobj()
            chunks: list[bytes] = []
            total_bytes = 0
            chunk_size = 64 * 1024
            offset = 0
            while offset < len(compressed):
                end = min(offset + chunk_size, len(compressed))
                out = decomp.decompress(compressed[offset:end], max_length=max_expansion - total_bytes)
                total_bytes += len(out)
                if total_bytes > max_expansion:
                    raise ValueError(
                        f"Decompressed data exceeds budget: {total_bytes} > {max_expansion} bytes"
                    )
                chunks.append(out)
                offset = end
            out = decomp.flush(max_length=max_expansion - total_bytes)
            total_bytes += len(out)
            if total_bytes > max_expansion:
                raise ValueError(
                    f"Decompressed data exceeds budget: {total_bytes} > {max_expansion} bytes"
                )
            chunks.append(out)
            decoded = b"".join(chunks)
            logger.debug("Decompressed model data")

        # S-06: Validate and strip wire format version byte
        if len(decoded) < 1:
            raise ValueError("Data too short to contain version byte")
        version = decoded[0]
        if version != WIRE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported wire format version: {version}. "
                f"Expected {WIRE_FORMAT_VERSION}. "
                f"QuinkGL versions may be incompatible."
            )
        decoded = decoded[1:]  # Strip version byte

        # Check size limit before unpacking
        if len(decoded) > MAX_MODEL_SIZE_BYTES:
            raise ValueError(
                f"Data size ({len(decoded) / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({MAX_MODEL_SIZE_BYTES / 1024 / 1024:.2f} MB)"
            )

        # Unpack using msgpack
        unpacked = msgpack.unpackb(decoded, raw=False)

        def _from_serializable(value: Any) -> Any:
            """Recursively convert msgpack-unpacked value back to numpy/Python types.

            S5a: supports both legacy (inner base64 string) and new (raw bytes) formats.
            S7c: handles nested dicts so sparse weight representations round-trip correctly.
            """
            if isinstance(value, dict):
                if value.get("__type__") == "numpy.ndarray":
                    raw = value["__data__"]
                    if isinstance(raw, str):
                        # Legacy format: inner base64-encoded string.
                        array_bytes = base64.b64decode(raw)
                    else:
                        # New format: raw bytes stored natively by msgpack.
                        array_bytes = raw
                    return _deserialize_numpy_array(array_bytes)
                else:
                    # Generic nested dict (e.g. sparse weight representation).
                    return {k: _from_serializable(v) for k, v in value.items()}
            elif isinstance(value, list):
                return value
            else:
                return value

        return _from_serializable(unpacked)

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


# S-08: Removed duplicate compress_weights/decompress_weights - use serialization/compression.py pipeline
# These functions were unused and duplicated the serialization/compression.py API
