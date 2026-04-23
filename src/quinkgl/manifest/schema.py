"""Swarm Manifest — data-policy + Phase 1 schema v3 extensions.

The manifest is the canonical commitment that binds every in-scope field to a
single SHA-256 hash; peers that disagree on any field produce different
community IDs.  Spec:

* §4.1 — CURRENT top-level keys (``schema_version``, ``model_arch_fingerprint``,
  ``data_schema_hash``, ``aggregation``, ``topology``, ``compression``,
  ``data_policy``).
* §4.7 — Phase 1 additions: ``name``, ``description``, ``created_at``,
  ``expires_at``, ``task``, ``model``, ``byzantine``, ``round_limit``,
  ``bootstrap_peers``, ``tracker_urls``, ``creator_pubkey``, ``signature``.
* §4.7.8 — validation order; each failure raises
  ``ValueError(ERR_CODE, context_dict)`` using constants from
  :mod:`quinkgl.manifest.errors`.
* §5 — canonical encoding: deterministic sorted JSON, ``signature`` popped
  per §5.3.

Section references to ``DOMAIN_AWARE_COLLABORATION_DESIGN.md §8`` describe
the data-policy sub-schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from quinkgl.manifest.errors import (
    ERR_MANIFEST_EXPIRED,
    ERR_MANIFEST_FIELD_INVALID,
    ERR_MANIFEST_INVALID_JSON,
    ERR_MANIFEST_MISSING_KEYS,
    ERR_MANIFEST_NOT_OBJECT,
    ERR_MANIFEST_SCHEMA_VERSION,
    ERR_MANIFEST_UNKNOWN_KEYS,
)

# `.qgl` files are small on-disk descriptors (§7).  A 1 MiB ceiling protects
# naive loaders from pathological inputs without meaningfully constraining
# legitimate manifests (real ones are well under 10 KiB).
_QGL_MAX_BYTES = 1 * 1024 * 1024


# Phase 1 bumps the manifest schema from 2 to 3.  Peers MUST reject unknown
# versions via ``ERR_MANIFEST_SCHEMA_VERSION`` (§20.3); ``strict=False`` on
# :meth:`SwarmManifest.from_dict` still accepts legacy v2 payloads for
# backward-compat loading (§20.1).
MANIFEST_SCHEMA_VERSION = 3
_SUPPORTED_LEGACY_VERSIONS = frozenset({2})

_VALID_NOISE_MECHANISMS = {"gaussian", "laplace", "none"}

# RFC 3339 UTC profile: `YYYY-MM-DDThh:mm:ss[.fff]Z`.  Only UTC (`Z`) accepted
# so that canonical bytes do not depend on local offsets.
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_ARCH_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ED25519_PUBKEY_RE = re.compile(r"^ed25519:[0-9a-f]{64}$")
_ED25519_SIG_RE = re.compile(r"^ed25519:[0-9a-f]{128}$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_PEER_ID_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

_TASK_TYPES = frozenset({"classification", "regression", "segmentation", "detection"})
_LABEL_TYPES = frozenset({"integer", "float", "binary", "multiclass", "multilabel"})
_FRAMEWORKS = frozenset({"pytorch", "tensorflow", "custom"})
_PEER_KINDS = frozenset({"ipv8", "tunnel"})

_NAME_MAX_LEN = 128
_DESCRIPTION_MAX_LEN = 1024
_TAGS_MAX = 16


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _raise(code: str, **ctx: Any) -> None:
    raise ValueError(code, ctx)


def _ensure_dict(data: Any, context: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"{context} must be a dict, got {type(data).__name__}"
        )


def _ensure_dict_v3(data: Any, context: str) -> None:
    """Phase-1 variant that emits a structured :data:`ERR_MANIFEST_NOT_OBJECT`
    for the top-level object, preserving the legacy error shape elsewhere."""
    if not isinstance(data, dict):
        _raise(ERR_MANIFEST_NOT_OBJECT, context=context, got=type(data).__name__)


def _check_allowed_keys(data: Dict[str, Any], allowed: set[str], context: str) -> None:
    extra = set(data.keys()) - allowed
    if extra:
        raise ValueError(f"{context} contains unknown fields: {sorted(extra)}")


def _check_required_keys(data: Dict[str, Any], required: set[str], context: str) -> None:
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{context} is missing required fields: {sorted(missing)}")


def _require_schema_version(data: Dict[str, Any], expected: int, context: str) -> None:
    version = data.get("schema_version")
    if version != expected:
        raise ValueError(
            f"{context}.schema_version must be {expected}, got {version}"
        )


def _normalize_buckets(
    buckets: Any,
    context: str,
) -> List[Tuple[Any, Any, Any]]:
    if not isinstance(buckets, list):
        raise ValueError(f"{context} must be a list, got {type(buckets).__name__}")
    normalized: List[Tuple[Any, Any, Any]] = []
    for entry in buckets:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(f"{context} entries must be 3-item lists/tuples")
        normalized.append((entry[0], entry[1], entry[2]))
    return normalized


def _validate_bucket_spec(
    buckets: List[Tuple[Any, Any, Any]],
    context: str,
    numeric_type: type,
) -> None:
    if not buckets:
        raise ValueError(f"{context} must not be empty")
    for name, low, high in buckets:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} bucket names must be non-empty strings")
        if not isinstance(low, numeric_type) or not isinstance(high, numeric_type):
            raise ValueError(
                f"{context} bucket bounds must be {numeric_type.__name__} values"
            )
        if low >= high:
            raise ValueError(f"{context} bucket lower bound must be < upper bound")


def _in_01(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _parse_rfc3339(value: str, field_name: str) -> datetime:
    """Parse an RFC 3339 UTC timestamp.  Raises ``ERR_MANIFEST_FIELD_INVALID``
    on malformed input so the exact failure code is stable for callers."""
    if not isinstance(value, str) or not _RFC3339_RE.match(value):
        _raise(
            ERR_MANIFEST_FIELD_INVALID,
            field=field_name,
            detail="not RFC 3339 UTC (expected ...Z)",
            value=value,
        )
    try:
        # ``fromisoformat`` accepts the trailing ``Z`` only on Python 3.11+.
        # Normalise to ``+00:00`` for older interpreters.
        normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(normalised).astimezone(timezone.utc)
    except ValueError:
        _raise(
            ERR_MANIFEST_FIELD_INVALID,
            field=field_name,
            detail="unparseable RFC 3339 timestamp",
            value=value,
        )
        raise  # unreachable, helps type-checkers


# ---------------------------------------------------------------------------
# Data-policy sub-classes (CURRENT, unchanged apart from the shared constant)
# ---------------------------------------------------------------------------


@dataclass
class CollaborationPolicy:
    """Collaboration policy for swarm membership.

    T-03: Added AffinityWeights for multi-signal affinity computation.
    """
    version: int = 1
    mode: str = "personalized"
    exploration_initial: float = 0.8
    exploration_decay: float = 0.95
    exploration_min: float = 0.1
    affinity_weights: Optional[Dict[str, float]] = None
    ema_alpha: float = 0.2
    edge_decay_factor: float = 0.95
    eviction_min_weight: float = 0.05
    cold_start_rounds: int = 3
    affinity_label_w: float = 0.4
    affinity_feature_w: float = 0.3
    affinity_gradient_w: float = 0.15
    affinity_history_w: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "exploration_initial": self.exploration_initial,
            "exploration_decay": self.exploration_decay,
            "exploration_min": self.exploration_min,
            "ema_alpha": self.ema_alpha,
            "edge_decay_factor": self.edge_decay_factor,
            "eviction_min_weight": self.eviction_min_weight,
            "cold_start_rounds": self.cold_start_rounds,
            "affinity_label_w": self.affinity_label_w,
            "affinity_feature_w": self.affinity_feature_w,
            "affinity_gradient_w": self.affinity_gradient_w,
            "affinity_history_w": self.affinity_history_w,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict: bool = True,
    ) -> "CollaborationPolicy":
        _ensure_dict(data, "CollaborationPolicy")
        allowed = {
            "version",
            "mode",
            "exploration_initial",
            "exploration_decay",
            "exploration_min",
            "ema_alpha",
            "edge_decay_factor",
            "eviction_min_weight",
            "cold_start_rounds",
            "affinity_label_w",
            "affinity_feature_w",
            "affinity_gradient_w",
            "affinity_history_w",
        }
        if strict:
            _check_allowed_keys(data, allowed, "CollaborationPolicy")
            _check_required_keys(data, allowed, "CollaborationPolicy")
        return cls(
            version=data.get("version", 1),
            mode=data.get("mode", "personalized"),
            exploration_initial=data.get("exploration_initial", 0.8),
            exploration_decay=data.get("exploration_decay", 0.95),
            exploration_min=data.get("exploration_min", 0.1),
            ema_alpha=data.get("ema_alpha", 0.2),
            edge_decay_factor=data.get("edge_decay_factor", 0.95),
            eviction_min_weight=data.get("eviction_min_weight", 0.05),
            cold_start_rounds=data.get("cold_start_rounds", 3),
            affinity_label_w=data.get("affinity_label_w", 0.4),
            affinity_feature_w=data.get("affinity_feature_w", 0.3),
            affinity_gradient_w=data.get("affinity_gradient_w", 0.15),
            affinity_history_w=data.get("affinity_history_w", 0.15),
        )

    def validate(self) -> None:
        _in_01(self.exploration_initial, "exploration_initial")
        _in_01(self.exploration_decay, "exploration_decay")
        _in_01(self.exploration_min, "exploration_min")
        _in_01(self.ema_alpha, "ema_alpha")
        _in_01(self.edge_decay_factor, "edge_decay_factor")
        _in_01(self.eviction_min_weight, "eviction_min_weight")
        _in_01(self.affinity_label_w, "affinity_label_w")
        _in_01(self.affinity_feature_w, "affinity_feature_w")
        _in_01(self.affinity_gradient_w, "affinity_gradient_w")
        _in_01(self.affinity_history_w, "affinity_history_w")
        total_affinity = (
            self.affinity_label_w
            + self.affinity_feature_w
            + self.affinity_gradient_w
            + self.affinity_history_w
        )
        if not (0.9 <= total_affinity <= 1.1):
            raise ValueError(
                f"Affinity weights must sum to approximately 1.0, got {total_affinity}"
            )
        if self.exploration_min > self.exploration_initial:
            raise ValueError(
                f"exploration_min ({self.exploration_min}) must be <= "
                f"exploration_initial ({self.exploration_initial})"
            )
        if self.cold_start_rounds < 1:
            raise ValueError(f"cold_start_rounds must be >= 1, got {self.cold_start_rounds}")
        if self.mode not in ("personalized", "standard", "agnostic"):
            raise ValueError(f"mode must be personalized|standard|agnostic, got '{self.mode}'")


@dataclass
class PersonalizationPolicy:
    version: int = 1
    model_split: str = "auto"
    apfl_enabled: bool = True
    apfl_initial_alpha: float = 0.5
    fedbn_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_split": self.model_split,
            "apfl_enabled": self.apfl_enabled,
            "apfl_initial_alpha": self.apfl_initial_alpha,
            "fedbn_enabled": self.fedbn_enabled,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict: bool = True,
    ) -> "PersonalizationPolicy":
        _ensure_dict(data, "PersonalizationPolicy")
        allowed = {
            "version",
            "model_split",
            "apfl_enabled",
            "apfl_initial_alpha",
            "fedbn_enabled",
        }
        if strict:
            _check_allowed_keys(data, allowed, "PersonalizationPolicy")
            _check_required_keys(data, allowed, "PersonalizationPolicy")
        return cls(
            version=data.get("version", 1),
            model_split=data.get("model_split", "auto"),
            apfl_enabled=data.get("apfl_enabled", True),
            apfl_initial_alpha=data.get("apfl_initial_alpha", 0.5),
            fedbn_enabled=data.get("fedbn_enabled", True),
        )

    def validate(self) -> None:
        if self.model_split not in ("auto", "manual"):
            raise ValueError(f"model_split must be auto|manual, got '{self.model_split}'")
        _in_01(self.apfl_initial_alpha, "apfl_initial_alpha")


@dataclass
class PrototypePolicy:
    version: int = 1
    enabled: bool = False
    alignment_weight: float = 0.1
    fedpac_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "alignment_weight": self.alignment_weight,
            "fedpac_enabled": self.fedpac_enabled,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict: bool = True,
    ) -> "PrototypePolicy":
        _ensure_dict(data, "PrototypePolicy")
        allowed = {"version", "enabled", "alignment_weight", "fedpac_enabled"}
        if strict:
            _check_allowed_keys(data, allowed, "PrototypePolicy")
            _check_required_keys(data, allowed, "PrototypePolicy")
        return cls(
            version=data.get("version", 1),
            enabled=data.get("enabled", False),
            alignment_weight=data.get("alignment_weight", 0.1),
            fedpac_enabled=data.get("fedpac_enabled", False),
        )

    def validate(self) -> None:
        if self.alignment_weight < 0.0:
            raise ValueError(f"alignment_weight must be >= 0, got {self.alignment_weight}")
        if self.fedpac_enabled and not self.enabled:
            raise ValueError("fedpac_enabled requires prototypes.enabled = true")


@dataclass
class DataPolicy:
    """Data policy configuration for swarm membership.

    T-08: Added version field for policy versioning.
    """
    version: int = 1
    schema_version: int = MANIFEST_SCHEMA_VERSION
    fingerprint_enabled: bool = True
    min_affinity: float = 0.3
    privacy_level: str = "standard"
    label_granularity: str = "bucket"
    label_buckets: List[Tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("low", 0.0, 0.2),
            ("medium", 0.2, 0.5),
            ("high", 0.5, 1.0),
        ]
    )
    feature_noise_sigma: float = 0.1
    feature_clip_norm: float = 5.0
    feature_dp_epsilon: Optional[float] = None
    feature_dp_delta: float = 1e-5
    feature_sensitivity: Optional[float] = None
    feature_noise_mechanism: str = "gaussian"
    sample_count_buckets: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [
            ("0-100", 0, 100),
            ("100-1k", 100, 1000),
            ("1k-10k", 1000, 10000),
            ("10k-100k", 10000, 100000),
            ("100k+", 100000, 10**9),
        ]
    )
    gradient_fingerprint: bool = False
    gradient_noise_sigma: float = 0.05
    gradient_dp_epsilon: Optional[float] = None
    gradient_dp_delta: float = 1e-5
    gradient_sensitivity: Optional[float] = None
    gradient_noise_mechanism: str = "gaussian"
    class_count_buckets: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [
            ("sparse", 0, 2),
            ("small", 2, 6),
            ("medium", 6, 11),
            ("large", 11, 10**6),
        ]
    )
    min_classes_to_reveal: int = 2
    hash_label_keys: bool = True
    label_key_hash_length: int = 16
    collaboration: CollaborationPolicy = field(default_factory=CollaborationPolicy)
    personalization: PersonalizationPolicy = field(default_factory=PersonalizationPolicy)
    prototypes: PrototypePolicy = field(default_factory=PrototypePolicy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint_enabled": self.fingerprint_enabled,
            "min_affinity": self.min_affinity,
            "privacy_level": self.privacy_level,
            "label_granularity": self.label_granularity,
            "label_buckets": [list(bucket) for bucket in self.label_buckets],
            "feature_noise_sigma": self.feature_noise_sigma,
            "feature_clip_norm": self.feature_clip_norm,
            "feature_dp_epsilon": self.feature_dp_epsilon,
            "feature_dp_delta": self.feature_dp_delta,
            "feature_sensitivity": self.feature_sensitivity,
            "feature_noise_mechanism": self.feature_noise_mechanism,
            "sample_count_buckets": [list(bucket) for bucket in self.sample_count_buckets],
            "gradient_fingerprint": self.gradient_fingerprint,
            "gradient_noise_sigma": self.gradient_noise_sigma,
            "gradient_dp_epsilon": self.gradient_dp_epsilon,
            "gradient_dp_delta": self.gradient_dp_delta,
            "gradient_sensitivity": self.gradient_sensitivity,
            "gradient_noise_mechanism": self.gradient_noise_mechanism,
            "class_count_buckets": [list(bucket) for bucket in self.class_count_buckets],
            "min_classes_to_reveal": self.min_classes_to_reveal,
            "hash_label_keys": self.hash_label_keys,
            "label_key_hash_length": self.label_key_hash_length,
            "collaboration": self.collaboration.to_dict(),
            "personalization": self.personalization.to_dict(),
            "prototypes": self.prototypes.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        strict: bool = True,
    ) -> "DataPolicy":
        _ensure_dict(data, "DataPolicy")
        allowed = {
            "schema_version",
            "fingerprint_enabled",
            "min_affinity",
            "privacy_level",
            "label_granularity",
            "label_buckets",
            "feature_noise_sigma",
            "feature_clip_norm",
            "feature_dp_epsilon",
            "feature_dp_delta",
            "feature_sensitivity",
            "feature_noise_mechanism",
            "sample_count_buckets",
            "gradient_fingerprint",
            "gradient_noise_sigma",
            "gradient_dp_epsilon",
            "gradient_dp_delta",
            "gradient_sensitivity",
            "gradient_noise_mechanism",
            "class_count_buckets",
            "min_classes_to_reveal",
            "hash_label_keys",
            "label_key_hash_length",
            "collaboration",
            "personalization",
            "prototypes",
        }
        if strict:
            _check_allowed_keys(data, allowed, "DataPolicy")
            _check_required_keys(data, allowed, "DataPolicy")
            _require_schema_version(data, MANIFEST_SCHEMA_VERSION, "DataPolicy")
        schema_version = data.get("schema_version", MANIFEST_SCHEMA_VERSION)
        collab_data = data.get("collaboration", CollaborationPolicy().to_dict())
        pers_data = data.get("personalization", PersonalizationPolicy().to_dict())
        proto_data = data.get("prototypes", PrototypePolicy().to_dict())
        label_buckets = _normalize_buckets(
            data.get("label_buckets", cls().label_buckets),
            "DataPolicy.label_buckets",
        )
        sample_count_buckets = _normalize_buckets(
            data.get("sample_count_buckets", cls().sample_count_buckets),
            "DataPolicy.sample_count_buckets",
        )
        class_count_buckets = _normalize_buckets(
            data.get("class_count_buckets", cls().class_count_buckets),
            "DataPolicy.class_count_buckets",
        )
        return cls(
            schema_version=schema_version,
            fingerprint_enabled=data.get("fingerprint_enabled", True),
            min_affinity=data.get("min_affinity", 0.3),
            privacy_level=data.get("privacy_level", "standard"),
            label_granularity=data.get("label_granularity", "bucket"),
            label_buckets=label_buckets,
            feature_noise_sigma=data.get("feature_noise_sigma", 0.1),
            feature_clip_norm=data.get("feature_clip_norm", 5.0),
            feature_dp_epsilon=data.get("feature_dp_epsilon"),
            feature_dp_delta=data.get("feature_dp_delta", 1e-5),
            feature_sensitivity=data.get("feature_sensitivity"),
            feature_noise_mechanism=data.get("feature_noise_mechanism", "gaussian"),
            sample_count_buckets=sample_count_buckets,
            gradient_fingerprint=data.get("gradient_fingerprint", False),
            gradient_noise_sigma=data.get("gradient_noise_sigma", 0.05),
            gradient_dp_epsilon=data.get("gradient_dp_epsilon"),
            gradient_dp_delta=data.get("gradient_dp_delta", 1e-5),
            gradient_sensitivity=data.get("gradient_sensitivity"),
            gradient_noise_mechanism=data.get("gradient_noise_mechanism", "gaussian"),
            class_count_buckets=class_count_buckets,
            min_classes_to_reveal=data.get("min_classes_to_reveal", 2),
            hash_label_keys=data.get("hash_label_keys", True),
            label_key_hash_length=data.get("label_key_hash_length", 16),
            collaboration=CollaborationPolicy.from_dict(collab_data, strict=strict),
            personalization=PersonalizationPolicy.from_dict(pers_data, strict=strict),
            prototypes=PrototypePolicy.from_dict(proto_data, strict=strict),
        )

    def validate(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {MANIFEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        _in_01(self.min_affinity, "min_affinity")
        if self.feature_noise_sigma < 0.0:
            raise ValueError(f"feature_noise_sigma must be >= 0, got {self.feature_noise_sigma}")
        if self.feature_clip_norm <= 0.0:
            raise ValueError(f"feature_clip_norm must be > 0, got {self.feature_clip_norm}")
        if self.privacy_level not in ("strict", "standard", "relaxed"):
            raise ValueError(
                f"privacy_level must be strict|standard|relaxed, got '{self.privacy_level}'"
            )
        if self.label_granularity not in ("exact", "bucket", "coarse"):
            raise ValueError(
                f"label_granularity must be exact|bucket|coarse, got '{self.label_granularity}'"
            )
        if self.feature_noise_mechanism not in _VALID_NOISE_MECHANISMS:
            raise ValueError(
                f"feature_noise_mechanism must be one of {_VALID_NOISE_MECHANISMS}, got '{self.feature_noise_mechanism}'"
            )
        if self.gradient_noise_mechanism not in _VALID_NOISE_MECHANISMS:
            raise ValueError(
                f"gradient_noise_mechanism must be one of {_VALID_NOISE_MECHANISMS}, got '{self.gradient_noise_mechanism}'"
            )
        if self.feature_dp_epsilon is not None and self.feature_dp_epsilon <= 0.0:
            raise ValueError(f"feature_dp_epsilon must be > 0, got {self.feature_dp_epsilon}")
        if self.gradient_dp_epsilon is not None and self.gradient_dp_epsilon <= 0.0:
            raise ValueError(f"gradient_dp_epsilon must be > 0, got {self.gradient_dp_epsilon}")
        if not (0.0 < self.feature_dp_delta < 1.0):
            raise ValueError(f"feature_dp_delta must be in (0, 1), got {self.feature_dp_delta}")
        if not (0.0 < self.gradient_dp_delta < 1.0):
            raise ValueError(f"gradient_dp_delta must be in (0, 1), got {self.gradient_dp_delta}")
        if self.feature_sensitivity is not None and self.feature_sensitivity <= 0.0:
            raise ValueError(
                f"feature_sensitivity must be > 0, got {self.feature_sensitivity}"
            )
        if self.gradient_sensitivity is not None and self.gradient_sensitivity <= 0.0:
            raise ValueError(
                f"gradient_sensitivity must be > 0, got {self.gradient_sensitivity}"
            )
        if self.gradient_noise_sigma < 0.0:
            raise ValueError(f"gradient_noise_sigma must be >= 0, got {self.gradient_noise_sigma}")
        if self.min_classes_to_reveal < 0:
            raise ValueError(
                f"min_classes_to_reveal must be >= 0, got {self.min_classes_to_reveal}"
            )
        if self.label_key_hash_length < 1:
            raise ValueError(
                f"label_key_hash_length must be >= 1, got {self.label_key_hash_length}"
            )
        _validate_bucket_spec(self.label_buckets, "label_buckets", float)
        _validate_bucket_spec(self.sample_count_buckets, "sample_count_buckets", int)
        _validate_bucket_spec(self.class_count_buckets, "class_count_buckets", int)
        self.collaboration.validate()
        self.personalization.validate()
        self.prototypes.validate()

    def apply_join_policy(self) -> None:
        """Validate and apply policy when joining a swarm (§8.2 steps 2–6).

        Steps:
          2. Validate all individual fields (delegated to ``validate``).
          3. Enforce privacy-level constraints on noise and granularity.
          4. Enforce fingerprint requirements for personalized collaboration.
          5. Validate DP parameter consistency (epsilon/delta/sigma).
          6. Enforce collaboration-mode constraints on affinity weights.
        """
        # Step 2: Basic field validation
        self.validate()

        # Step 3: Privacy-level constraints
        if self.privacy_level == "strict":
            if self.label_granularity == "exact":
                raise ValueError(
                    "privacy_level='strict' requires label_granularity != 'exact'; "
                    "use 'bucket' or 'coarse'"
                )
            if self.feature_noise_sigma < 0.5:
                raise ValueError(
                    f"privacy_level='strict' requires feature_noise_sigma >= 0.5, "
                    f"got {self.feature_noise_sigma}"
                )
            if self.gradient_noise_sigma < 0.2:
                raise ValueError(
                    f"privacy_level='strict' requires gradient_noise_sigma >= 0.2, "
                    f"got {self.gradient_noise_sigma}"
                )
            if not self.hash_label_keys:
                raise ValueError(
                    "privacy_level='strict' requires hash_label_keys=True"
                )
        elif self.privacy_level == "relaxed":
            if self.label_granularity == "coarse":
                raise ValueError(
                    "privacy_level='relaxed' should not use label_granularity='coarse'; "
                    "use 'exact' or 'bucket'"
                )

        # Step 4: Fingerprint requirements for personalized collaboration
        if self.collaboration.mode == "personalized" and not self.fingerprint_enabled:
            raise ValueError(
                "collaboration.mode='personalized' requires fingerprint_enabled=True "
                "for affinity-based peer selection"
            )

        # Step 5: DP parameter consistency
        if self.feature_dp_epsilon is not None:
            if self.feature_noise_sigma <= 0.0:
                raise ValueError(
                    "feature_dp_epsilon is set but feature_noise_sigma=0; "
                    "DP requires non-zero noise"
                )
        if self.gradient_dp_epsilon is not None:
            if self.gradient_noise_sigma <= 0.0:
                raise ValueError(
                    "gradient_dp_epsilon is set but gradient_noise_sigma=0; "
                    "DP requires non-zero noise"
                )
            if not self.gradient_fingerprint:
                raise ValueError(
                    "gradient_dp_epsilon is set but gradient_fingerprint=False; "
                    "gradient DP requires gradient fingerprinting enabled"
                )

        # Step 6: Collaboration-mode constraints on affinity weights
        if self.collaboration.mode == "personalized":
            total_w = (
                self.collaboration.affinity_label_w
                + self.collaboration.affinity_feature_w
                + self.collaboration.affinity_gradient_w
                + self.collaboration.affinity_history_w
            )
            if total_w <= 0.0:
                raise ValueError(
                    "collaboration.mode='personalized' requires at least one "
                    f"non-zero affinity weight (total={total_w})"
                )

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def manifest_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1 — v3 sub-schemas (§4.7.1, §4.7.2, §4.7.3)
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Task contract (§4.7.1)."""

    type: str = "classification"
    input_shape: List[int] = field(default_factory=lambda: [1])
    output_shape: List[int] = field(default_factory=lambda: [1])
    label_type: str = "integer"
    tags: List[str] = field(default_factory=list)

    _ALLOWED_KEYS = {"type", "input_shape", "output_shape", "label_type", "tags"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "label_type": self.label_type,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Any, strict: bool = True) -> "TaskSpec":
        if not isinstance(data, dict):
            _raise(ERR_MANIFEST_FIELD_INVALID, field="task", detail="must be object")
        if strict:
            extra = set(data.keys()) - cls._ALLOWED_KEYS
            if extra:
                _raise(
                    ERR_MANIFEST_UNKNOWN_KEYS,
                    field="task",
                    unknown=sorted(extra),
                )
            missing = cls._ALLOWED_KEYS - set(data.keys())
            if missing:
                _raise(
                    ERR_MANIFEST_MISSING_KEYS,
                    field="task",
                    missing=sorted(missing),
                )
        return cls(
            type=data.get("type", "classification"),
            input_shape=list(data.get("input_shape", [1])),
            output_shape=list(data.get("output_shape", [1])),
            label_type=data.get("label_type", "integer"),
            tags=list(data.get("tags", [])),
        )

    def validate(self) -> None:
        if self.type not in _TASK_TYPES:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="task.type",
                detail=f"must be one of {sorted(_TASK_TYPES)}",
                value=self.type,
            )
        if self.label_type not in _LABEL_TYPES:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="task.label_type",
                detail=f"must be one of {sorted(_LABEL_TYPES)}",
                value=self.label_type,
            )
        for shape_name, shape in (
            ("input_shape", self.input_shape),
            ("output_shape", self.output_shape),
        ):
            if not isinstance(shape, list) or not shape:
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"task.{shape_name}",
                    detail="must be a non-empty list of positive ints",
                )
            for dim in shape:
                # ``bool`` is a subclass of ``int``; reject it explicitly to
                # avoid "True == 1" masquerade.
                if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                    _raise(
                        ERR_MANIFEST_FIELD_INVALID,
                        field=f"task.{shape_name}",
                        detail="every dimension must be a positive int",
                        value=dim,
                    )
        if not isinstance(self.tags, list) or len(self.tags) > _TAGS_MAX:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="task.tags",
                detail=f"must be a list of at most {_TAGS_MAX} strings",
            )
        for tag in self.tags:
            if not isinstance(tag, str) or not _TAG_RE.match(tag):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field="task.tags",
                    detail="each tag must match ^[a-z0-9][a-z0-9-]{0,31}$",
                    value=tag,
                )


def _default_arch_hash() -> str:
    """Placeholder `sha256:` value used when no model is supplied.

    It is syntactically valid so that a freshly constructed
    :class:`SwarmManifest` passes `validate()`, but it is obviously not a real
    architecture digest — callers building production manifests MUST replace
    it via :func:`quinkgl.manifest.compute_arch_hash`.
    """
    return "sha256:" + "0" * 64


@dataclass
class ModelSpec:
    """Model binding (§4.7.2)."""

    framework: str = "custom"
    arch_hash: str = field(default_factory=_default_arch_hash)
    arch_spec: Optional[Dict[str, Any]] = None
    genesis_weights_hash: Optional[str] = None
    genesis_weights_url: Optional[str] = None

    _ALLOWED_KEYS = {
        "framework",
        "arch_hash",
        "arch_spec",
        "genesis_weights_hash",
        "genesis_weights_url",
    }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "arch_hash": self.arch_hash,
            "arch_spec": self.arch_spec,
            "genesis_weights_hash": self.genesis_weights_hash,
            "genesis_weights_url": self.genesis_weights_url,
        }

    @classmethod
    def from_dict(cls, data: Any, strict: bool = True) -> "ModelSpec":
        if not isinstance(data, dict):
            _raise(ERR_MANIFEST_FIELD_INVALID, field="model", detail="must be object")
        if strict:
            extra = set(data.keys()) - cls._ALLOWED_KEYS
            if extra:
                _raise(ERR_MANIFEST_UNKNOWN_KEYS, field="model", unknown=sorted(extra))
            # `arch_spec`, genesis_* may be null but keys MUST be present.
            missing = cls._ALLOWED_KEYS - set(data.keys())
            if missing:
                _raise(
                    ERR_MANIFEST_MISSING_KEYS,
                    field="model",
                    missing=sorted(missing),
                )
        return cls(
            framework=data.get("framework", "custom"),
            arch_hash=data.get("arch_hash", _default_arch_hash()),
            arch_spec=data.get("arch_spec"),
            genesis_weights_hash=data.get("genesis_weights_hash"),
            genesis_weights_url=data.get("genesis_weights_url"),
        )

    def validate(self) -> None:
        if self.framework not in _FRAMEWORKS:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="model.framework",
                detail=f"must be one of {sorted(_FRAMEWORKS)}",
                value=self.framework,
            )
        if not isinstance(self.arch_hash, str) or not _ARCH_HASH_RE.match(self.arch_hash):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="model.arch_hash",
                detail="must match ^sha256:[0-9a-f]{64}$",
                value=self.arch_hash,
            )
        if self.arch_spec is not None and not isinstance(self.arch_spec, dict):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="model.arch_spec",
                detail="must be object or null",
            )
        if self.genesis_weights_hash is not None:
            if not isinstance(self.genesis_weights_hash, str) or not _ARCH_HASH_RE.match(
                self.genesis_weights_hash
            ):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field="model.genesis_weights_hash",
                    detail="must be null or match ^sha256:[0-9a-f]{64}$",
                )
        if self.genesis_weights_url is not None and not isinstance(
            self.genesis_weights_url, str
        ):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="model.genesis_weights_url",
                detail="must be string or null",
            )


@dataclass
class ByzantineSpec:
    """Byzantine tolerance parameters (§4.7.3)."""

    f: int = 0
    enforce_n_gt_2f_plus_2: bool = False

    _ALLOWED_KEYS = {"f", "enforce_n_gt_2f_plus_2"}

    def to_dict(self) -> Dict[str, Any]:
        return {"f": self.f, "enforce_n_gt_2f_plus_2": self.enforce_n_gt_2f_plus_2}

    @classmethod
    def from_dict(cls, data: Any, strict: bool = True) -> "ByzantineSpec":
        if not isinstance(data, dict):
            _raise(ERR_MANIFEST_FIELD_INVALID, field="byzantine", detail="must be object")
        if strict:
            extra = set(data.keys()) - cls._ALLOWED_KEYS
            if extra:
                _raise(
                    ERR_MANIFEST_UNKNOWN_KEYS,
                    field="byzantine",
                    unknown=sorted(extra),
                )
        return cls(
            f=int(data.get("f", 0)),
            enforce_n_gt_2f_plus_2=bool(data.get("enforce_n_gt_2f_plus_2", False)),
        )

    def validate(self) -> None:
        if isinstance(self.f, bool) or not isinstance(self.f, int) or self.f < 0:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="byzantine.f",
                detail="must be non-negative int",
                value=self.f,
            )
        if not isinstance(self.enforce_n_gt_2f_plus_2, bool):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="byzantine.enforce_n_gt_2f_plus_2",
                detail="must be bool",
            )


# ---------------------------------------------------------------------------
# SwarmManifest — v3 top-level
# ---------------------------------------------------------------------------


_V3_ALLOWED_TOP_KEYS = {
    # §4.1 legacy keys
    "schema_version",
    "model_arch_fingerprint",
    "data_schema_hash",
    "aggregation",
    "topology",
    "compression",
    "data_policy",
    # §4.7 additions
    "name",
    "description",
    "created_at",
    "expires_at",
    "task",
    "model",
    "byzantine",
    "round_limit",
    "bootstrap_peers",
    "tracker_urls",
    "creator_pubkey",
    "signature",
}

_V3_REQUIRED_TOP_KEYS = {
    "schema_version",
    "model_arch_fingerprint",
    "data_schema_hash",
    "aggregation",
    "topology",
    "compression",
    "data_policy",
    "name",
    "created_at",
    "task",
    "model",
}


@dataclass
class SwarmManifest:
    """Complete swarm manifest binding every in-scope policy field (§4)."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    model_arch_fingerprint: str = ""  # Hash of model architecture (legacy).
    data_schema_hash: str = ""
    aggregation_name: str = "FedAvg"
    aggregation_params: Dict[str, Any] = field(default_factory=dict)
    topology_name: str = "Random"
    topology_params: Dict[str, Any] = field(default_factory=dict)
    compression_enabled: bool = False
    compression_params: Dict[str, Any] = field(default_factory=dict)
    data_policy: DataPolicy = field(default_factory=DataPolicy)

    # --- §4.7 Phase 1 additions ------------------------------------------------
    name: str = "unnamed"
    description: str = ""
    created_at: str = "1970-01-01T00:00:00Z"
    expires_at: Optional[str] = None
    task: TaskSpec = field(default_factory=TaskSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    byzantine: ByzantineSpec = field(default_factory=ByzantineSpec)
    round_limit: Optional[int] = None
    bootstrap_peers: List[Dict[str, Any]] = field(default_factory=list)
    tracker_urls: List[List[str]] = field(default_factory=list)
    creator_pubkey: Optional[str] = None  # §4.7.6 (Phase 2 signing target)
    signature: Optional[str] = None  # §4.7.7 — EXCLUDED from canonical bytes

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_arch_fingerprint": self.model_arch_fingerprint,
            "data_schema_hash": self.data_schema_hash,
            "aggregation": {
                "name": self.aggregation_name,
                "params": self.aggregation_params,
            },
            "topology": {
                "name": self.topology_name,
                "params": self.topology_params,
            },
            "compression": {
                "enabled": self.compression_enabled,
                "params": self.compression_params,
            },
            "data_policy": self.data_policy.to_dict(),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "task": self.task.to_dict(),
            "model": self.model.to_dict(),
            "byzantine": self.byzantine.to_dict(),
            "round_limit": self.round_limit,
            "bootstrap_peers": [dict(p) for p in self.bootstrap_peers],
            "tracker_urls": [list(tier) for tier in self.tracker_urls],
            "creator_pubkey": self.creator_pubkey,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = True) -> "SwarmManifest":
        """Load a manifest dict.

        When ``strict=True`` enforces the §4.7.8 validation order and emits
        structured ``ERR_*`` codes on failure.  When ``strict=False`` also
        accepts a legacy ``schema_version=2`` payload, filling any missing v3
        keys with safe defaults (see §20.1, §20.2).
        """
        _ensure_dict_v3(data, "SwarmManifest")

        version = data.get("schema_version")
        if strict:
            if version != MANIFEST_SCHEMA_VERSION:
                _raise(
                    ERR_MANIFEST_SCHEMA_VERSION,
                    expected=MANIFEST_SCHEMA_VERSION,
                    got=version,
                )
            extra = set(data.keys()) - _V3_ALLOWED_TOP_KEYS
            if extra:
                _raise(ERR_MANIFEST_UNKNOWN_KEYS, unknown=sorted(extra))
            missing = _V3_REQUIRED_TOP_KEYS - set(data.keys())
            if missing:
                _raise(ERR_MANIFEST_MISSING_KEYS, missing=sorted(missing))
        else:
            if version not in (MANIFEST_SCHEMA_VERSION, *_SUPPORTED_LEGACY_VERSIONS):
                _raise(
                    ERR_MANIFEST_SCHEMA_VERSION,
                    expected=MANIFEST_SCHEMA_VERSION,
                    supported=sorted({MANIFEST_SCHEMA_VERSION, *_SUPPORTED_LEGACY_VERSIONS}),
                    got=version,
                )

        agg_data = data.get("aggregation", {}) or {}
        topo_data = data.get("topology", {}) or {}
        comp_data = data.get("compression", {}) or {}
        policy_data = data.get("data_policy", DataPolicy().to_dict())

        # In ``strict=False`` mode, fall back to default instances for any
        # missing v3 sub-blocks so that legacy v2 payloads load cleanly.
        task_data = data.get("task", TaskSpec().to_dict())
        model_data = data.get("model", ModelSpec().to_dict())
        byz_data = data.get("byzantine", ByzantineSpec().to_dict())

        instance = cls(
            schema_version=data.get("schema_version", MANIFEST_SCHEMA_VERSION),
            model_arch_fingerprint=data.get("model_arch_fingerprint", ""),
            data_schema_hash=data.get("data_schema_hash", ""),
            aggregation_name=agg_data.get("name", "FedAvg"),
            aggregation_params=agg_data.get("params", {}),
            topology_name=topo_data.get("name", "Random"),
            topology_params=topo_data.get("params", {}),
            compression_enabled=comp_data.get("enabled", False),
            compression_params=comp_data.get("params", {}),
            data_policy=DataPolicy.from_dict(
                policy_data, strict=strict and version == MANIFEST_SCHEMA_VERSION
            ),
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            created_at=data.get("created_at", "1970-01-01T00:00:00Z"),
            expires_at=data.get("expires_at"),
            task=TaskSpec.from_dict(task_data, strict=strict),
            model=ModelSpec.from_dict(model_data, strict=strict),
            byzantine=ByzantineSpec.from_dict(byz_data, strict=strict),
            round_limit=data.get("round_limit"),
            bootstrap_peers=list(data.get("bootstrap_peers", [])),
            tracker_urls=list(data.get("tracker_urls", [])),
            creator_pubkey=data.get("creator_pubkey"),
            signature=data.get("signature"),
        )

        if strict:
            instance._validate_v3_fields()

        return instance

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Full v3 validation — used by publishers and the strict join path."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {MANIFEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        self._validate_v3_fields()
        # Legacy non-empty checks: kept for backward compatibility with the
        # audit suite that inspects these specific messages.
        if not self.model_arch_fingerprint:
            raise ValueError("model_arch_fingerprint must be non-empty")
        if not self.data_schema_hash:
            raise ValueError("data_schema_hash must be non-empty")
        self.data_policy.validate()
        # T-12: Cap manifest size to prevent DoS
        manifest_bytes = self.canonical_bytes()
        MAX_MANIFEST_SIZE = 10240  # 10KB
        if len(manifest_bytes) > MAX_MANIFEST_SIZE:
            raise ValueError(
                f"manifest size {len(manifest_bytes)} bytes exceeds maximum {MAX_MANIFEST_SIZE} bytes"
            )

    def _validate_v3_fields(self) -> None:
        """Run the §4.7.8 per-field checks — used by strict loading and by
        :meth:`validate`.  The legacy ``model_arch_fingerprint`` /
        ``data_schema_hash`` non-empty checks stay in :meth:`validate`."""

        # name
        if (
            not isinstance(self.name, str)
            or not (1 <= len(self.name) <= _NAME_MAX_LEN)
            or any(ord(c) < 0x20 for c in self.name)
        ):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="name",
                detail=f"1-{_NAME_MAX_LEN} UTF-8 chars, no control chars",
                value=self.name,
            )
        # description
        if not isinstance(self.description, str) or len(self.description) > _DESCRIPTION_MAX_LEN:
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="description",
                detail=f"0-{_DESCRIPTION_MAX_LEN} chars string",
            )
        # created_at / expires_at
        created = _parse_rfc3339(self.created_at, "created_at")
        expires = None
        if self.expires_at is not None:
            expires = _parse_rfc3339(self.expires_at, "expires_at")
            if expires <= created:
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field="expires_at",
                    detail="must be strictly after created_at",
                )
            if expires < datetime.now(timezone.utc):
                _raise(ERR_MANIFEST_EXPIRED, expires_at=self.expires_at)

        # round_limit
        if self.round_limit is not None and (
            isinstance(self.round_limit, bool)
            or not isinstance(self.round_limit, int)
            or self.round_limit < 0
        ):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="round_limit",
                detail="must be null or non-negative int",
                value=self.round_limit,
            )

        # creator_pubkey / signature (Phase 2 fields, syntactically checked now)
        if self.creator_pubkey is not None:
            if not isinstance(self.creator_pubkey, str) or not _ED25519_PUBKEY_RE.match(
                self.creator_pubkey
            ):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field="creator_pubkey",
                    detail="must match ^ed25519:[0-9a-f]{64}$",
                )
        if self.signature is not None:
            if not isinstance(self.signature, str) or not _ED25519_SIG_RE.match(self.signature):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field="signature",
                    detail="must match ^ed25519:[0-9a-f]{128}$",
                )

        # bootstrap_peers
        if not isinstance(self.bootstrap_peers, list):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="bootstrap_peers",
                detail="must be a list",
            )
        for i, peer in enumerate(self.bootstrap_peers):
            if not isinstance(peer, dict):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"bootstrap_peers[{i}]",
                    detail="must be an object",
                )
            if peer.get("kind") not in _PEER_KINDS:
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"bootstrap_peers[{i}].kind",
                    detail=f"must be one of {sorted(_PEER_KINDS)}",
                    value=peer.get("kind"),
                )
            peer_id = peer.get("peer_id")
            if not isinstance(peer_id, str) or not _PEER_ID_HEX_RE.match(peer_id):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"bootstrap_peers[{i}].peer_id",
                    detail="must be hex string",
                )
            addr = peer.get("address")
            if not isinstance(addr, str) or ":" not in addr:
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"bootstrap_peers[{i}].address",
                    detail="must be 'host:port'",
                )

        # tracker_urls — two-level array per §4.7.5.
        if not isinstance(self.tracker_urls, list):
            _raise(
                ERR_MANIFEST_FIELD_INVALID,
                field="tracker_urls",
                detail="must be a two-level list (tiers -> urls)",
            )
        for i, tier in enumerate(self.tracker_urls):
            if not isinstance(tier, list):
                _raise(
                    ERR_MANIFEST_FIELD_INVALID,
                    field=f"tracker_urls[{i}]",
                    detail="each tier must be a list of URL strings",
                )
            for j, url in enumerate(tier):
                if not isinstance(url, str) or not url:
                    _raise(
                        ERR_MANIFEST_FIELD_INVALID,
                        field=f"tracker_urls[{i}][{j}]",
                        detail="must be a non-empty string",
                    )

        # nested v3 blocks
        self.task.validate()
        self.model.validate()
        self.byzantine.validate()

    # ------------------------------------------------------------------
    # Canonical encoding / hashing (§5)
    # ------------------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """Deterministic SHA-256-safe serialization (§5.1).

        Per the §5.1 Phase 1 delta, the ``signature`` field is popped BEFORE
        hashing — it is the value being signed so it cannot be included in
        the input.  ``creator_pubkey`` IS retained (§5.3).
        """
        d = self.to_dict()
        d.pop("signature", None)
        return json.dumps(
            d,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def manifest_hash(self) -> str:
        """SHA-256 hex of :meth:`canonical_bytes` (see §6)."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    # ------------------------------------------------------------------
    # `.qgl` file format (§7, §10.2)
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: Any,
        *,
        strict: bool = True,
    ) -> "SwarmManifest":
        """Load a manifest from a UTF-8 `.qgl` file.

        The file MUST be ≤ 1 MiB and MUST NOT begin with a UTF-8 BOM
        (§7).  Malformed JSON raises ``ValueError(ERR_MANIFEST_INVALID_JSON,
        …)``.  Missing files raise :class:`FileNotFoundError` — that is the
        standard Python signal for "not there" and does not need its own
        manifest-specific code.
        """
        # Import here to avoid a hard ``pathlib`` dependency at module load
        # time on restricted runtimes.
        from pathlib import Path as _Path

        p = _Path(path)
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            raise

        if size > _QGL_MAX_BYTES:
            _raise(
                ERR_MANIFEST_INVALID_JSON,
                path=str(p),
                detail=f"file exceeds {_QGL_MAX_BYTES} byte limit",
                size=size,
            )

        raw = p.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            _raise(
                ERR_MANIFEST_INVALID_JSON,
                path=str(p),
                detail="UTF-8 BOM not permitted in .qgl files",
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            _raise(
                ERR_MANIFEST_INVALID_JSON,
                path=str(p),
                detail=f"not valid UTF-8: {exc}",
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            _raise(
                ERR_MANIFEST_INVALID_JSON,
                path=str(p),
                detail=str(exc),
                line=exc.lineno,
                col=exc.colno,
            )

        instance = cls.from_dict(data, strict=strict)
        # `from_dict(strict=True)` already runs `_validate_v3_fields`; re-run
        # full `validate()` on the strict path so legacy non-empty checks are
        # enforced too.
        if strict:
            instance.validate()
        return instance

    def to_magnet(
        self,
        *,
        trackers: Optional[List[str]] = None,
        bootstrap_peers: Optional[List[str]] = None,
        protocol_version: int = 1,
    ) -> str:
        """Render this manifest as a canonical ``quinkgl:?…`` magnet URI.

        ``swarm_id`` is :meth:`manifest_hash` decoded to 32 bytes; the
        display name defaults to :attr:`name` when it differs from the
        scaffold placeholder ``"unnamed"``.  ``trackers`` and
        ``bootstrap_peers`` override (rather than extend) the manifest's
        ``tracker_urls`` / ``bootstrap_peers`` when supplied — callers that
        want the manifest's built-in lists can pass ``None`` and the method
        will fall back to them.
        """
        # Local import avoids a circular module-level dependency: magnet.py
        # imports error codes from errors.py and does not touch schema.
        from quinkgl.manifest.magnet import MagnetLink, format_magnet

        swarm_id_hex = self.manifest_hash()
        swarm_id = bytes.fromhex(swarm_id_hex)

        if trackers is None:
            # Flatten the two-level tracker_urls tier list into a single
            # ordered tier for the magnet URI (tiers are a BEP-12 concept
            # and we don't encode them in the magnet payload).
            trackers = [url for tier in self.tracker_urls for url in tier]
        if bootstrap_peers is None:
            bootstrap_peers = [
                p.get("address", "") for p in self.bootstrap_peers if p.get("address")
            ]

        display_name = self.name if self.name and self.name != "unnamed" else None

        link = MagnetLink(
            swarm_id=swarm_id,
            display_name=display_name,
            keywords=[],
            trackers=list(trackers),
            bootstrap_peers=list(bootstrap_peers),
            protocol_version=protocol_version,
        )
        return format_magnet(link)

    def to_file(
        self,
        path: Any,
        *,
        pretty: bool = True,
    ) -> None:
        """Atomically write the manifest as a `.qgl` file.

        Uses the tmp-file + ``os.replace`` pattern so an interrupted write
        cannot leave the destination in a partial state.  Writes UTF-8
        without a BOM and without a trailing newline (except in pretty
        mode, where ``json.dumps(indent=2)`` emits one naturally — we keep
        the trailing newline for human-readability).

        ``pretty=True`` emits 2-space indent + per-line newlines; consumers
        MUST re-canonicalise before hashing (§7).

        ``pretty=False`` emits the exact bytes produced by
        :meth:`canonical_bytes` — byte-identical to the hash input.
        """
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if pretty:
            payload = json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        else:
            payload = self.canonical_bytes()

        # Atomic write: same-directory tmp file guarantees ``os.replace`` is
        # on the same filesystem (rename is atomic only within a filesystem).
        tmp = p.with_name(p.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:  # pragma: no cover — best-effort on exotic FS
                    pass
            os.replace(tmp, p)
        except Exception:
            # Clean up the partial tmp file so callers never see stale
            # ``.qgl.tmp`` fragments after a failed write.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
