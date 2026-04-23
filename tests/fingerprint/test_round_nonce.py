"""Tests for per-round fingerprint rotation (audit Task-2 finding F6)."""

import pytest

from quinkgl.fingerprint.computer import FingerprintComputer
from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    FINGERPRINT_SCHEMA_VERSION,
    FingerprintPrivacyConfig,
)


class TestRoundNonceDerivation:
    def test_none_round_keeps_legacy_mode(self):
        comp = FingerprintComputer()
        assert comp._derive_round_nonce(None) is None

    def test_round_nonce_is_deterministic_per_round(self):
        comp = FingerprintComputer()
        assert comp._derive_round_nonce(7) == comp._derive_round_nonce(7)

    def test_round_nonce_differs_across_rounds(self):
        comp = FingerprintComputer()
        assert comp._derive_round_nonce(7) != comp._derive_round_nonce(8)

    @pytest.mark.parametrize("bad_round", [-1, 1.5, "7"])
    def test_invalid_round_rejected(self, bad_round):
        comp = FingerprintComputer()
        with pytest.raises(ValueError, match="round_number"):
            comp._derive_round_nonce(bad_round)


class TestRoundBoundFingerprint:
    def test_round_nonce_emitted_on_wire(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=3)
        assert fp.round_nonce is not None
        payload = fp.to_dict()
        assert payload["schema_version"] == FINGERPRINT_SCHEMA_VERSION
        assert payload["round_nonce"] == fp.round_nonce

    def test_round_nonce_roundtrip(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=4)
        fp2 = DataFingerprint.from_dict(fp.to_dict())
        assert fp2.round_nonce == fp.round_nonce

    def test_same_round_same_labels_same_hashed_keys(self):
        comp = FingerprintComputer()
        fp1 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=5)
        fp2 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=5)
        assert fp1.round_nonce == fp2.round_nonce
        assert fp1.label_buckets == fp2.label_buckets

    def test_different_rounds_rotate_hashed_keys(self):
        comp = FingerprintComputer()
        fp1 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=5)
        fp2 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=6)
        assert fp1.round_nonce != fp2.round_nonce
        assert fp1.label_buckets != fp2.label_buckets
        assert set(fp1.label_buckets.values()) == set(fp2.label_buckets.values())

    def test_legacy_mode_without_round_keeps_no_nonce(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20})
        assert fp.round_nonce is None

    def test_opt_out_raw_labels_do_not_rotate(self):
        cfg = FingerprintPrivacyConfig(hash_label_keys=False)
        comp = FingerprintComputer(cfg)
        fp1 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=5)
        fp2 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=6)
        assert fp1.label_buckets == fp2.label_buckets == {"cat": "medium", "dog": "high"}
        assert fp1.round_nonce != fp2.round_nonce

    def test_hmac_hashing_still_rotates_by_round(self):
        cfg = FingerprintPrivacyConfig(label_key_secret=b"swarm-secret")
        comp = FingerprintComputer(cfg)
        fp1 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=5)
        fp2 = comp.compute_from_label_counts({"cat": 10, "dog": 20}, round_number=6)
        assert fp1.label_buckets != fp2.label_buckets


class TestStrictParsing:
    def test_missing_round_nonce_rejected(self):
        fp = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={},
            sample_bucket="0-100",
            num_classes=0,
            class_count_bucket="small",
            round_nonce="abcd1234",
        )
        payload = fp.to_dict()
        payload.pop("round_nonce")
        with pytest.raises(ValueError, match="missing required fields"):
            DataFingerprint.from_dict(payload)

    def test_empty_round_nonce_rejected(self):
        fp = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={},
            sample_bucket="0-100",
            num_classes=0,
            class_count_bucket="small",
            round_nonce="abcd1234",
        )
        payload = fp.to_dict()
        payload["round_nonce"] = ""
        with pytest.raises(ValueError, match="round_nonce"):
            DataFingerprint.from_dict(payload)
