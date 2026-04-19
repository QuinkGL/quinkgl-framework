"""Tests for minority-class disclosure mitigations.

Covers:
  - class_count_bucket replaces raw num_classes on the wire
  - single-class dataset is indistinguishable from an empty dataset
  - label keys are obfuscated by default (SHA-256), optionally HMAC-keyed
  - two peers with identical labels produce matching hashes → affinity still works
  - two peers with different HMAC secrets produce different hashes for same label
  - opt-out path (hash_label_keys=False) preserves raw labels
"""

import pytest

from quinkgl.fingerprint.computer import FingerprintComputer
from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    FingerprintPrivacyConfig,
)


# ── class_count_bucket ─────────────────────────────────────────────


class TestClassCountBucket:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, "sparse"),
            (1, "sparse"),
            (2, "small"),
            (5, "small"),
            (6, "medium"),
            (10, "medium"),
            (11, "large"),
            (999, "large"),
        ],
    )
    def test_bucketing(self, n, expected):
        comp = FingerprintComputer()
        assert comp._bucket_class_count(n) == expected

    def test_raw_num_classes_suppressed_under_default_privacy(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"a": 10, "b": 20, "c": 30, "d": 40})
        # With hashing (default) we must not leak the raw integer class count.
        assert fp.num_classes == 0
        assert fp.class_count_bucket == "small"

    def test_raw_num_classes_preserved_when_hashing_disabled(self):
        cfg = FingerprintPrivacyConfig(hash_label_keys=False)
        comp = FingerprintComputer(cfg)
        fp = comp.compute_from_label_counts({"a": 10, "b": 20, "c": 30, "d": 40})
        assert fp.num_classes == 4


# ── Single-class indistinguishability ──────────────────────────────


class TestSingleClassMasking:
    def test_single_class_looks_like_no_data(self):
        comp = FingerprintComputer()
        fp_single = comp.compute_from_label_counts({"only": 50})
        fp_empty = comp.compute_from_label_counts({})
        assert fp_single.label_buckets == fp_empty.label_buckets == {}
        assert fp_single.class_count_bucket == fp_empty.class_count_bucket == "sparse"
        assert fp_single.num_classes == fp_empty.num_classes == 0

    def test_two_class_dataset_is_revealed(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"a": 10, "b": 20})
        # At the default threshold, two classes is sufficient to reveal
        # the bucket mapping.
        assert len(fp.label_buckets) == 2
        assert fp.class_count_bucket == "small"

    def test_custom_reveal_threshold(self):
        cfg = FingerprintPrivacyConfig(min_classes_to_reveal=3)
        comp = FingerprintComputer(cfg)
        fp = comp.compute_from_label_counts({"a": 10, "b": 20})
        # Two classes now fall below the threshold → suppressed.
        assert fp.label_buckets == {}


# ── Label key hashing ──────────────────────────────────────────────


class TestLabelKeyHashing:
    def test_hashed_keys_by_default(self):
        comp = FingerprintComputer()
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20, "bird": 30})
        # Raw label names must not appear as keys.
        assert "cat" not in fp.label_buckets
        assert "dog" not in fp.label_buckets
        # Hashed keys are hex strings of the configured length.
        for k in fp.label_buckets:
            assert len(k) == 16
            assert all(c in "0123456789abcdef" for c in k)

    def test_same_label_same_hash_cross_peer(self):
        comp_a = FingerprintComputer()
        comp_b = FingerprintComputer()
        fp_a = comp_a.compute_from_label_counts({"cat": 10, "dog": 20})
        fp_b = comp_b.compute_from_label_counts({"cat": 10, "dog": 20})
        assert set(fp_a.label_buckets.keys()) == set(fp_b.label_buckets.keys())

    def test_hmac_secret_isolates_swarms(self):
        """Two swarms with different HMAC secrets get different hashes for
        the same raw label, preventing cross-swarm linkage."""
        cfg_s1 = FingerprintPrivacyConfig(label_key_secret=b"swarm-1-secret")
        cfg_s2 = FingerprintPrivacyConfig(label_key_secret=b"swarm-2-secret")
        comp_s1 = FingerprintComputer(cfg_s1)
        comp_s2 = FingerprintComputer(cfg_s2)
        h1 = comp_s1._hash_label_key("shared_label")
        h2 = comp_s2._hash_label_key("shared_label")
        assert h1 != h2

    def test_hmac_secret_stable_within_swarm(self):
        cfg = FingerprintPrivacyConfig(label_key_secret=b"swarm-secret")
        comp1 = FingerprintComputer(cfg)
        comp2 = FingerprintComputer(cfg)
        assert comp1._hash_label_key("cat") == comp2._hash_label_key("cat")

    def test_opt_out_preserves_raw_labels(self):
        cfg = FingerprintPrivacyConfig(hash_label_keys=False)
        comp = FingerprintComputer(cfg)
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20})
        assert set(fp.label_buckets.keys()) == {"cat", "dog"}

    def test_hash_length_is_configurable(self):
        cfg = FingerprintPrivacyConfig(label_key_hash_length=8)
        comp = FingerprintComputer(cfg)
        fp = comp.compute_from_label_counts({"cat": 10, "dog": 20})
        for k in fp.label_buckets:
            assert len(k) == 8


# ── Affinity still works with hashed keys ──────────────────────────


class TestAffinityStillFunctional:
    def test_same_raw_labels_yield_perfect_label_similarity(self):
        comp = FingerprintComputer()
        fp_a = comp.compute_from_label_counts({"cat": 30, "dog": 70})
        fp_b = comp.compute_from_label_counts({"cat": 30, "dog": 70})
        assert fp_a._label_similarity(fp_b) == pytest.approx(1.0)

    def test_disjoint_labels_yield_zero_similarity(self):
        comp = FingerprintComputer()
        fp_a = comp.compute_from_label_counts({"cat": 30, "dog": 70})
        fp_b = comp.compute_from_label_counts({"wolf": 30, "bear": 70})
        # All keys are disjoint; no pairwise matches, so score = 0.
        assert fp_a._label_similarity(fp_b) == 0.0


# ── Serialization roundtrip preserves new field ────────────────────


class TestSerialization:
    def test_class_count_bucket_roundtrips(self):
        fp = DataFingerprint(
            label_buckets={"aa": "high"},
            noised_moments={},
            sample_bucket="100-1k",
            num_classes=0,
            class_count_bucket="medium",
        )
        fp2 = DataFingerprint.from_dict(fp.to_dict())
        assert fp2.class_count_bucket == "medium"

    def test_legacy_dict_without_schema_version_rejected(self):
        legacy = {
            "label_buckets": {"a": "high"},
            "noised_moments": {},
            "sample_bucket": "0-100",
            "num_classes": 1,
        }
        with pytest.raises(ValueError, match="schema_version"):
            DataFingerprint.from_dict(legacy)
