"""Serialization helpers for lists of numpy weight arrays."""

import io
import re
from typing import List

import numpy as np


def serialize(weights: List[np.ndarray]) -> bytes:
    """Serialize a list of numpy arrays into compressed bytes."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, *weights)
    return buffer.getvalue()


def deserialize(data: bytes) -> List[np.ndarray]:
    """Deserialize compressed bytes back into a list of numpy arrays."""
    buffer = io.BytesIO(data)
    with np.load(buffer, allow_pickle=False) as loaded_data:
        def key_index(key: str) -> int:
            match = re.fullmatch(r"arr_(\d+)", key)
            if match is None:
                raise ValueError(f"Unexpected serialized array key: {key}")
            return int(match.group(1))

        return [loaded_data[key] for key in sorted(loaded_data.files, key=key_index)]
