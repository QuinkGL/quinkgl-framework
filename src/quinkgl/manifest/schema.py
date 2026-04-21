"""
Swarm Manifest — Data Policy Schema.

The manifest stores *policy*, not data.  Fingerprints are exchanged at
runtime between peers.  This module defines the data_policy sub-schema
that controls Layers 1–4 of the domain-aware collaboration system.

Section references: DOMAIN_AWARE_COLLABORATION_DESIGN.md §8.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Schema version for manifest canonicalization. Incremented when the
# canonical serialization contract changes in a backwards-incompatible way.
MANIFEST_SCHEMA_VERSION = 2
_VALID_NOISE_MECHANISMS = {"gaussian", "laplace", "none"}


def _ensure_dict(data: Dict[str, Any], context: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a dict, got {type(data).__name__}")


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


@dataclass
class CollaborationPolicy:
    version: int = 1
    mode: str = "personalized"
    exploration_initial: float = 0.8
    exploration_decay: float = 0.95
    exploration_min: float = 0.1
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
            self.affinity_label_w +
            self.affinity_feature_w +
            self.affinity_gradient_w +
            self.affinity_history_w
        )
        if not (0.9 <= total_affinity <= 1.1):  # Allow small floating-point tolerance
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
        """Validate and apply policy when joining a swarm (§8.2 steps 2–6)."""
        self.validate()

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization suitable for hashing.

        Guarantees:
          - Keys sorted recursively (``sort_keys=True``).
          - No insignificant whitespace (fixed separators).
          - Stable float repr via JSON's ``repr()`` formatting.
          - Schema version prefixed so that hashes across schema versions
            cannot collide.

        Two semantically-equal policies produce identical bytes regardless
        of how their sub-dicts were originally ordered.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def manifest_hash(self) -> str:
        """SHA-256 of the canonical manifest bytes, hex-encoded.

        This commitment binds every in-scope policy field to a single hash.
        Peers that disagree on any field produce different hashes and
        therefore different community IDs (see ``generate_community_id``).
        """
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _in_01(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class SwarmManifest:
    """Complete swarm manifest binding all collaboration policies.

    This is the canonical commitment that peers use to derive community IDs.
    Any difference in any field produces a different manifest hash and thus
    a different overlay network.
    """
    schema_version: int = MANIFEST_SCHEMA_VERSION
    model_arch_fingerprint: str = ""  # Hash of model architecture
    data_schema_hash: str = ""  # Hash of data schema
    aggregation_name: str = "FedAvg"
    aggregation_params: Dict[str, Any] = field(default_factory=dict)
    topology_name: str = "Random"
    topology_params: Dict[str, Any] = field(default_factory=dict)
    compression_enabled: bool = False
    compression_params: Dict[str, Any] = field(default_factory=dict)
    data_policy: DataPolicy = field(default_factory=DataPolicy)

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], strict: bool = True) -> "SwarmManifest":
        _ensure_dict(data, "SwarmManifest")
        allowed = {
            "schema_version",
            "model_arch_fingerprint",
            "data_schema_hash",
            "aggregation",
            "topology",
            "compression",
            "data_policy",
        }
        if strict:
            _check_allowed_keys(data, allowed, "SwarmManifest")
            _check_required_keys(data, allowed, "SwarmManifest")
            _require_schema_version(data, MANIFEST_SCHEMA_VERSION, "SwarmManifest")

        agg_data = data.get("aggregation", {})
        topo_data = data.get("topology", {})
        comp_data = data.get("compression", {})
        policy_data = data.get("data_policy", DataPolicy().to_dict())

        return cls(
            schema_version=data.get("schema_version", MANIFEST_SCHEMA_VERSION),
            model_arch_fingerprint=data.get("model_arch_fingerprint", ""),
            data_schema_hash=data.get("data_schema_hash", ""),
            aggregation_name=agg_data.get("name", "FedAvg"),
            aggregation_params=agg_data.get("params", {}),
            topology_name=topo_data.get("name", "Random"),
            topology_params=topo_data.get("params", {}),
            compression_enabled=comp_data.get("enabled", False),
            compression_params=comp_data.get("params", {}),
            data_policy=DataPolicy.from_dict(policy_data, strict=strict),
        )

    def validate(self) -> None:
        """Validate all manifest fields."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {MANIFEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if not self.model_arch_fingerprint:
            raise ValueError("model_arch_fingerprint must be non-empty")
        if not self.data_schema_hash:
            raise ValueError("data_schema_hash must be non-empty")
        self.data_policy.validate()

    def canonical_bytes(self) -> bytes:
        """Deterministic serialization suitable for hashing."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def manifest_hash(self) -> str:
        """SHA-256 of the canonical manifest bytes, hex-encoded."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
