"""Swarm Manifest — schema and data policy definitions."""

from quinkgl.manifest.arch_hash import compute_arch_hash
from quinkgl.manifest.errors import ERR_MANIFEST_SCHEMA_VERSION
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
from quinkgl.manifest.signing import keygen, sign_manifest, verify_manifest


def check_compatibility(manifest: SwarmManifest) -> None:
    """Verify that the current ``quinkgl`` build can load ``manifest`` (§10.6.5).

    Raises :class:`ValueError` tagged with ``ERR_MANIFEST_SCHEMA_VERSION``
    when the manifest declares a newer ``schema_version`` than this build
    supports — the only remediation is upgrading the library.  Older
    manifests are accepted silently (forward-compat of older fixtures is a
    first-class requirement: peers SHOULD continue to understand manifests
    produced by N-1 releases).
    """
    if manifest is None:
        raise TypeError("check_compatibility requires a SwarmManifest instance")
    version = getattr(manifest, "schema_version", None)
    if not isinstance(version, int):
        raise ValueError(
            ERR_MANIFEST_SCHEMA_VERSION,
            {
                "detail": "manifest is missing an integer schema_version",
                "got": version,
            },
        )
    if version > MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            ERR_MANIFEST_SCHEMA_VERSION,
            {
                "detail": (
                    f"manifest requires schema_version={version}, but this "
                    f"quinkgl build supports up to "
                    f"{MANIFEST_SCHEMA_VERSION}; upgrade `quinkgl` to load it"
                ),
                "manifest_schema_version": version,
                "supported_schema_version": MANIFEST_SCHEMA_VERSION,
            },
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
    "check_compatibility",
    "compute_arch_hash",
    "format_magnet",
    "keygen",
    "load_manifest",
    "parse_magnet",
    "sign_manifest",
    "verify_manifest",
]
