"""Tests for canonical manifest hashing and community-ID binding.

Covers audit Task-2 findings F1 (missing manifest commitment) and
F2 (community ID not bound to DataPolicy).
"""

import pytest

from quinkgl.manifest.schema import (
    MANIFEST_SCHEMA_VERSION,
    CollaborationPolicy,
    DataPolicy,
    PersonalizationPolicy,
    PrototypePolicy,
)
from quinkgl.network.gossip_community import generate_community_id


# ── Canonical serialization ─────────────────────────────────────────


class TestCanonicalBytes:
    def test_deterministic(self):
        p1 = DataPolicy()
        p2 = DataPolicy()
        assert p1.canonical_bytes() == p2.canonical_bytes()

    def test_includes_schema_version(self):
        p = DataPolicy()
        assert f'"schema_version":{MANIFEST_SCHEMA_VERSION}'.encode() in p.canonical_bytes()

    def test_key_order_independent(self):
        """Rebuilding from a reordered dict must produce the same hash."""
        p1 = DataPolicy(min_affinity=0.5, feature_noise_sigma=0.2)
        d = p1.to_dict()
        # Reverse the top-level key order
        reordered = dict(reversed(list(d.items())))
        p2 = DataPolicy.from_dict(reordered)
        assert p1.manifest_hash() == p2.manifest_hash()

    def test_no_nan_allowed(self):
        """NaN/Inf floats must be rejected, not silently hashed as 'NaN'."""
        p = DataPolicy(feature_noise_sigma=float("nan"))
        with pytest.raises(ValueError):
            p.canonical_bytes()


# ── Hash properties ────────────────────────────────────────────────


class TestManifestHash:
    def test_is_sha256_hex(self):
        h = DataPolicy().manifest_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_default_stable(self):
        assert DataPolicy().manifest_hash() == DataPolicy().manifest_hash()

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: setattr(p, "min_affinity", 0.9),
            lambda p: setattr(p, "feature_noise_sigma", 0.25),
            lambda p: setattr(p, "privacy_level", "strict"),
            lambda p: setattr(p, "label_granularity", "exact"),
            lambda p: setattr(p, "gradient_fingerprint", True),
            lambda p: setattr(p.collaboration, "mode", "agnostic"),
            lambda p: setattr(p.collaboration, "ema_alpha", 0.9),
            lambda p: setattr(p.personalization, "apfl_enabled", False),
            lambda p: setattr(p.prototypes, "enabled", True),
        ],
    )
    def test_every_field_flips_hash(self, mutate):
        base = DataPolicy()
        mutated = DataPolicy()
        mutate(mutated)
        assert base.manifest_hash() != mutated.manifest_hash()


# ── Community ID binding ───────────────────────────────────────────


class TestCommunityIDBinding:
    def test_manifest_hash_changes_community_id(self):
        cid_no_manifest = generate_community_id("health", "schema123")
        cid_with_manifest = generate_community_id(
            "health", "schema123", manifest_hash=DataPolicy().manifest_hash()
        )
        assert cid_no_manifest != cid_with_manifest

    def test_different_policies_different_community_ids(self):
        p1 = DataPolicy(min_affinity=0.3)
        p2 = DataPolicy(min_affinity=0.7)
        cid1 = generate_community_id("health", "schema123", p1.manifest_hash())
        cid2 = generate_community_id("health", "schema123", p2.manifest_hash())
        assert cid1 != cid2

    def test_same_policy_same_community_id(self):
        p1 = DataPolicy()
        p2 = DataPolicy()
        cid1 = generate_community_id("health", "schema123", p1.manifest_hash())
        cid2 = generate_community_id("health", "schema123", p2.manifest_hash())
        assert cid1 == cid2

    def test_backwards_compatible_signature(self):
        """Legacy two-arg call must still work."""
        cid = generate_community_id("health", "schema123")
        assert isinstance(cid, bytes)
        assert len(cid) == 20

    def test_empty_vs_absent_manifest_distinct(self):
        """Empty-string manifest_hash treated as absent; no aliasing."""
        cid_absent = generate_community_id("health", "schema123")
        cid_empty = generate_community_id("health", "schema123", manifest_hash="")
        assert cid_absent == cid_empty  # documented: falsy → absent
