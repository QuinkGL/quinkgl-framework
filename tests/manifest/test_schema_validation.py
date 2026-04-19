"""Strict schema-validation tests for manifest payloads."""

import pytest

from quinkgl.manifest.schema import (
    MANIFEST_SCHEMA_VERSION,
    CollaborationPolicy,
    DataPolicy,
    PersonalizationPolicy,
    PrototypePolicy,
)


class TestNestedStrictParsing:
    def test_collaboration_unknown_field_rejected(self):
        payload = CollaborationPolicy().to_dict() | {"unexpected": True}
        with pytest.raises(ValueError, match="unknown fields"):
            CollaborationPolicy.from_dict(payload)

    def test_personalization_missing_field_rejected(self):
        payload = PersonalizationPolicy().to_dict()
        payload.pop("fedbn_enabled")
        with pytest.raises(ValueError, match="missing required fields"):
            PersonalizationPolicy.from_dict(payload)

    def test_prototype_unknown_field_rejected(self):
        payload = PrototypePolicy().to_dict() | {"unexpected": True}
        with pytest.raises(ValueError, match="unknown fields"):
            PrototypePolicy.from_dict(payload)

    def test_nested_non_strict_still_allows_partial(self):
        payload = {"mode": "standard"}
        parsed = CollaborationPolicy.from_dict(payload, strict=False)
        assert parsed.mode == "standard"


class TestDataPolicyStrictParsing:
    def test_missing_schema_version_rejected(self):
        payload = DataPolicy().to_dict()
        payload.pop("schema_version")
        with pytest.raises(ValueError, match="schema_version"):
            DataPolicy.from_dict(payload)

    def test_wrong_schema_version_rejected(self):
        payload = DataPolicy().to_dict()
        payload["schema_version"] = MANIFEST_SCHEMA_VERSION - 1
        with pytest.raises(ValueError, match="schema_version"):
            DataPolicy.from_dict(payload)

    def test_unknown_field_rejected(self):
        payload = DataPolicy().to_dict() | {"unexpected": True}
        with pytest.raises(ValueError, match="unknown fields"):
            DataPolicy.from_dict(payload)

    def test_missing_required_field_rejected(self):
        payload = DataPolicy().to_dict()
        payload.pop("feature_noise_sigma")
        with pytest.raises(ValueError, match="missing required fields"):
            DataPolicy.from_dict(payload)

    def test_nested_unknown_field_rejected(self):
        payload = DataPolicy().to_dict()
        payload["collaboration"]["unexpected"] = True
        with pytest.raises(ValueError, match="unknown fields"):
            DataPolicy.from_dict(payload)

    def test_invalid_bucket_shape_rejected(self):
        payload = DataPolicy().to_dict()
        payload["label_buckets"] = [["low", 0.0]]
        with pytest.raises(ValueError, match="3-item"):
            DataPolicy.from_dict(payload)

    def test_strict_false_preserves_legacy_defaulting(self):
        parsed = DataPolicy.from_dict({"fingerprint_enabled": False}, strict=False)
        assert parsed.fingerprint_enabled is False
        assert parsed.schema_version == MANIFEST_SCHEMA_VERSION

    def test_new_privacy_fields_are_hash_bound(self):
        base = DataPolicy()
        mutated = DataPolicy(feature_dp_epsilon=0.5)
        assert base.manifest_hash() != mutated.manifest_hash()

    def test_hash_changes_when_bucket_policy_changes(self):
        base = DataPolicy()
        mutated = DataPolicy(min_classes_to_reveal=3)
        assert base.manifest_hash() != mutated.manifest_hash()
