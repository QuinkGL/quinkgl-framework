"""Structured error codes for the manifest / wire / node surface (spec §19).

Every constant is a `str` whose value equals its own identifier so that it can
be used interchangeably as an exception tag, a telemetry event suffix, and a
CLI exit-code key (see §11.11).

Canonical raise form::

    from quinkgl.manifest.errors import ERR_MANIFEST_INVALID_JSON
    raise ValueError(ERR_MANIFEST_INVALID_JSON, {"path": str(path)})

Call-sites SHOULD also emit a matching observability event (e.g.
`manifest.invalid_json`) so that operators can correlate raised errors with
telemetry streams.

The authoritative list lives in spec §19.  `tests/manifest/test_error_codes.py`
enforces that every constant named there exists here and that `__all__` is in
sync.  Do not add codes here without also updating §19 (requires spec version
bump per §20.3).
"""

from __future__ import annotations

# --- §19.1 Manifest ---------------------------------------------------------

ERR_MANIFEST_INVALID_JSON: str = "ERR_MANIFEST_INVALID_JSON"
ERR_MANIFEST_NOT_OBJECT: str = "ERR_MANIFEST_NOT_OBJECT"
ERR_MANIFEST_SCHEMA_VERSION: str = "ERR_MANIFEST_SCHEMA_VERSION"
ERR_MANIFEST_UNKNOWN_KEYS: str = "ERR_MANIFEST_UNKNOWN_KEYS"
ERR_MANIFEST_MISSING_KEYS: str = "ERR_MANIFEST_MISSING_KEYS"
ERR_MANIFEST_FIELD_INVALID: str = "ERR_MANIFEST_FIELD_INVALID"
ERR_MANIFEST_EXPIRED: str = "ERR_MANIFEST_EXPIRED"
ERR_MANIFEST_DATA_POLICY: str = "ERR_MANIFEST_DATA_POLICY"
ERR_MANIFEST_HASH_MISMATCH: str = "ERR_MANIFEST_HASH_MISMATCH"
ERR_MANIFEST_FETCH_REQUIRED: str = "ERR_MANIFEST_FETCH_REQUIRED"

# --- §19.2 Magnet -----------------------------------------------------------

ERR_MAGNET_SCHEME: str = "ERR_MAGNET_SCHEME"
ERR_MAGNET_XT: str = "ERR_MAGNET_XT"
ERR_MAGNET_DUPLICATE: str = "ERR_MAGNET_DUPLICATE"

# --- §19.3 Node -------------------------------------------------------------

ERR_NODE_NO_MANIFEST: str = "ERR_NODE_NO_MANIFEST"
ERR_NODE_AGGREGATION_MISMATCH: str = "ERR_NODE_AGGREGATION_MISMATCH"
ERR_NODE_TOPOLOGY_MISMATCH: str = "ERR_NODE_TOPOLOGY_MISMATCH"
ERR_NODE_UNSIGNED_MANIFEST_REJECTED: str = "ERR_NODE_UNSIGNED_MANIFEST_REJECTED"
ERR_NODE_ARCH_MISMATCH: str = "ERR_NODE_ARCH_MISMATCH"
ERR_NODE_DATA_SHAPE_MISMATCH: str = "ERR_NODE_DATA_SHAPE_MISMATCH"
ERR_RUN_NO_STANDARD_MODEL: str = "ERR_RUN_NO_STANDARD_MODEL"
ERR_SCRIPT_CALLABLES_MISSING: str = "ERR_SCRIPT_CALLABLES_MISSING"

# --- §19.4 Trust / Signing (Phase 2 consumers; surface ready in Phase 1) ----

ERR_TRUST_POLICY_VIOLATION: str = "ERR_TRUST_POLICY_VIOLATION"
ERR_TRUST_TOFU_CONFLICT: str = "ERR_TRUST_TOFU_CONFLICT"
ERR_SIGNING_UNAVAILABLE: str = "ERR_SIGNING_UNAVAILABLE"
ERR_SIGNATURE_INVALID: str = "ERR_SIGNATURE_INVALID"
ERR_CREATOR_NOT_TRUSTED: str = "ERR_CREATOR_NOT_TRUSTED"

# --- §19.5 Wire -------------------------------------------------------------

ERR_WIRE_UNKNOWN_SWARM: str = "ERR_WIRE_UNKNOWN_SWARM"
ERR_WIRE_RATE_LIMITED: str = "ERR_WIRE_RATE_LIMITED"
ERR_WIRE_TIMEOUT: str = "ERR_WIRE_TIMEOUT"
ERR_WIRE_CHUNK_INCONSISTENT: str = "ERR_WIRE_CHUNK_INCONSISTENT"


__all__ = [
    # §19.1
    "ERR_MANIFEST_INVALID_JSON",
    "ERR_MANIFEST_NOT_OBJECT",
    "ERR_MANIFEST_SCHEMA_VERSION",
    "ERR_MANIFEST_UNKNOWN_KEYS",
    "ERR_MANIFEST_MISSING_KEYS",
    "ERR_MANIFEST_FIELD_INVALID",
    "ERR_MANIFEST_EXPIRED",
    "ERR_MANIFEST_DATA_POLICY",
    "ERR_MANIFEST_HASH_MISMATCH",
    "ERR_MANIFEST_FETCH_REQUIRED",
    # §19.2
    "ERR_MAGNET_SCHEME",
    "ERR_MAGNET_XT",
    "ERR_MAGNET_DUPLICATE",
    # §19.3
    "ERR_NODE_NO_MANIFEST",
    "ERR_NODE_AGGREGATION_MISMATCH",
    "ERR_NODE_TOPOLOGY_MISMATCH",
    "ERR_NODE_UNSIGNED_MANIFEST_REJECTED",
    "ERR_NODE_ARCH_MISMATCH",
    "ERR_NODE_DATA_SHAPE_MISMATCH",
    "ERR_RUN_NO_STANDARD_MODEL",
    "ERR_SCRIPT_CALLABLES_MISSING",
    # §19.4
    "ERR_TRUST_POLICY_VIOLATION",
    "ERR_TRUST_TOFU_CONFLICT",
    "ERR_SIGNING_UNAVAILABLE",
    "ERR_SIGNATURE_INVALID",
    "ERR_CREATOR_NOT_TRUSTED",
    # §19.5
    "ERR_WIRE_UNKNOWN_SWARM",
    "ERR_WIRE_RATE_LIMITED",
    "ERR_WIRE_TIMEOUT",
    "ERR_WIRE_CHUNK_INCONSISTENT",
]
