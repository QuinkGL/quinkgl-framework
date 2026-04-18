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
from typing import Any, Dict, Optional


# Schema version for manifest canonicalization. Incremented when the
# canonical serialization contract changes in a backwards-incompatible way.
MANIFEST_SCHEMA_VERSION = 1


@dataclass
class CollaborationPolicy:
    mode: str = "personalized"
    exploration_initial: float = 0.8
    exploration_decay: float = 0.95
    exploration_min: float = 0.1
    ema_alpha: float = 0.2
    edge_decay_factor: float = 0.95
    eviction_min_weight: float = 0.05
    cold_start_rounds: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "exploration_initial": self.exploration_initial,
            "exploration_decay": self.exploration_decay,
            "exploration_min": self.exploration_min,
            "ema_alpha": self.ema_alpha,
            "edge_decay_factor": self.edge_decay_factor,
            "eviction_min_weight": self.eviction_min_weight,
            "cold_start_rounds": self.cold_start_rounds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborationPolicy":
        return cls(
            mode=data.get("mode", "personalized"),
            exploration_initial=data.get("exploration_initial", 0.8),
            exploration_decay=data.get("exploration_decay", 0.95),
            exploration_min=data.get("exploration_min", 0.1),
            ema_alpha=data.get("ema_alpha", 0.2),
            edge_decay_factor=data.get("edge_decay_factor", 0.95),
            eviction_min_weight=data.get("eviction_min_weight", 0.05),
            cold_start_rounds=data.get("cold_start_rounds", 3),
        )

    def validate(self) -> None:
        _in_01(self.exploration_initial, "exploration_initial")
        _in_01(self.exploration_decay, "exploration_decay")
        _in_01(self.exploration_min, "exploration_min")
        _in_01(self.ema_alpha, "ema_alpha")
        _in_01(self.edge_decay_factor, "edge_decay_factor")
        _in_01(self.eviction_min_weight, "eviction_min_weight")
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
    model_split: str = "auto"
    apfl_enabled: bool = True
    apfl_initial_alpha: float = 0.5
    fedbn_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_split": self.model_split,
            "apfl_enabled": self.apfl_enabled,
            "apfl_initial_alpha": self.apfl_initial_alpha,
            "fedbn_enabled": self.fedbn_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalizationPolicy":
        return cls(
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
    enabled: bool = False
    alignment_weight: float = 0.1
    fedpac_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alignment_weight": self.alignment_weight,
            "fedpac_enabled": self.fedpac_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrototypePolicy":
        return cls(
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
    fingerprint_enabled: bool = True
    min_affinity: float = 0.3
    privacy_level: str = "standard"
    label_granularity: str = "bucket"
    feature_noise_sigma: float = 0.1
    gradient_fingerprint: bool = False
    collaboration: CollaborationPolicy = field(default_factory=CollaborationPolicy)
    personalization: PersonalizationPolicy = field(default_factory=PersonalizationPolicy)
    prototypes: PrototypePolicy = field(default_factory=PrototypePolicy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint_enabled": self.fingerprint_enabled,
            "min_affinity": self.min_affinity,
            "privacy_level": self.privacy_level,
            "label_granularity": self.label_granularity,
            "feature_noise_sigma": self.feature_noise_sigma,
            "gradient_fingerprint": self.gradient_fingerprint,
            "collaboration": self.collaboration.to_dict(),
            "personalization": self.personalization.to_dict(),
            "prototypes": self.prototypes.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPolicy":
        collab_data = data.get("collaboration", {})
        pers_data = data.get("personalization", {})
        proto_data = data.get("prototypes", {})
        return cls(
            fingerprint_enabled=data.get("fingerprint_enabled", True),
            min_affinity=data.get("min_affinity", 0.3),
            privacy_level=data.get("privacy_level", "standard"),
            label_granularity=data.get("label_granularity", "bucket"),
            feature_noise_sigma=data.get("feature_noise_sigma", 0.1),
            gradient_fingerprint=data.get("gradient_fingerprint", False),
            collaboration=CollaborationPolicy.from_dict(collab_data),
            personalization=PersonalizationPolicy.from_dict(pers_data),
            prototypes=PrototypePolicy.from_dict(proto_data),
        )

    def validate(self) -> None:
        _in_01(self.min_affinity, "min_affinity")
        if self.feature_noise_sigma < 0.0:
            raise ValueError(f"feature_noise_sigma must be >= 0, got {self.feature_noise_sigma}")
        if self.privacy_level not in ("strict", "standard", "relaxed"):
            raise ValueError(
                f"privacy_level must be strict|standard|relaxed, got '{self.privacy_level}'"
            )
        if self.label_granularity not in ("exact", "bucket", "coarse"):
            raise ValueError(
                f"label_granularity must be exact|bucket|coarse, got '{self.label_granularity}'"
            )
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
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "data_policy": self.to_dict(),
        }
        return json.dumps(
            payload,
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
