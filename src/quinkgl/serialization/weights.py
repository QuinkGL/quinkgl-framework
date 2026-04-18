"""Serialization helpers for lists of numpy weight arrays.

S10: This module uses ``np.savez_compressed`` and is *not* part of the main
gossip pipeline, which uses the msgpack path in
``quinkgl.network.model_serializer``.  The two formats are incompatible.

Use ``serialize_numpy_weights`` / ``deserialize_numpy_weights`` explicitly
when you need the NumPy compressed format (e.g. for local checkpointing).
The legacy ``serialize`` / ``deserialize`` aliases are deprecated and will
emit a ``DeprecationWarning``.
"""

import io
import re
import warnings
from typing import List

import numpy as np


def serialize_numpy_weights(weights: List[np.ndarray]) -> bytes:
    """Serialize a list of numpy arrays into compressed bytes via np.savez_compressed."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *weights)
    return buffer.getvalue()


def deserialize_numpy_weights(data: bytes) -> List[np.ndarray]:
    """Deserialize compressed bytes back into a list of numpy arrays."""
    buffer = io.BytesIO(data)
    with np.load(buffer, allow_pickle=False) as loaded_data:
        def key_index(key: str) -> int:
            match = re.fullmatch(r"arr_(\d+)", key)
            if match is None:
                raise ValueError(f"Unexpected serialized array key: {key}")
            return int(match.group(1))

        return [loaded_data[key] for key in sorted(loaded_data.files, key=key_index)]


# ---------------------------------------------------------------------------
# Deprecated aliases
# ---------------------------------------------------------------------------

def serialize(weights: List[np.ndarray]) -> bytes:
    """Deprecated — use ``serialize_numpy_weights`` instead."""
    warnings.warn(
        "quinkgl.serialization.weights.serialize() is deprecated; "
        "use serialize_numpy_weights() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return serialize_numpy_weights(weights)


def deserialize(data: bytes) -> List[np.ndarray]:
    """Deprecated — use ``deserialize_numpy_weights`` instead."""
    warnings.warn(
        "quinkgl.serialization.weights.deserialize() is deprecated; "
        "use deserialize_numpy_weights() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return deserialize_numpy_weights(data)
