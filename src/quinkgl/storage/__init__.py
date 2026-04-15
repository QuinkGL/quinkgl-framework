"""
Storage Module

Model versioning and persistence for gossip learning.
"""

from quinkgl.storage.model_store import ModelStore, ModelCheckpoint

__all__ = [
    "ModelStore",
    "ModelCheckpoint",
]
