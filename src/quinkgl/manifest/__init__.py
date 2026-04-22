"""Swarm Manifest — schema and data policy definitions."""

from quinkgl.manifest.arch_hash import compute_arch_hash
from quinkgl.manifest.loader import load_manifest
from quinkgl.manifest.magnet import MagnetLink, format_magnet, parse_magnet
from quinkgl.manifest.schema import (
    MANIFEST_SCHEMA_VERSION,
    ByzantineSpec,
    CollaborationPolicy,
    DataPolicy,
    ModelSpec,
    PersonalizationPolicy,
    PrototypePolicy,
    SwarmManifest,
    TaskSpec,
)

__all__ = [
    "ByzantineSpec",
    "CollaborationPolicy",
    "DataPolicy",
    "MANIFEST_SCHEMA_VERSION",
    "MagnetLink",
    "ModelSpec",
    "PersonalizationPolicy",
    "PrototypePolicy",
    "SwarmManifest",
    "TaskSpec",
    "compute_arch_hash",
    "format_magnet",
    "load_manifest",
    "parse_magnet",
]
