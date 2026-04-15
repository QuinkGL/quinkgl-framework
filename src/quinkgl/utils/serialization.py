"""Backward-compatible shim for weight serialization helpers."""

from quinkgl.serialization.weights import deserialize, serialize

__all__ = ["serialize", "deserialize"]
