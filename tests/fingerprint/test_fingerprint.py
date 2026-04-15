"""Tests for privacy-preserving data fingerprinting."""

import json

import numpy as np
import pytest

from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    AffinityWeights,
    FingerprintPrivacyConfig,
    _BUCKET_ORDER,
    _adjacent_bucket,
)
from quinkgl.fingerprint.computer import FingerprintComputer
from quinkgl.topology.base import PeerInfo, SelectionContext, is_version_compatible


class TestLabelQuantization:
    def test_low_bucket(self):
        computer = FingerprintComputer()
        result = computer._quantize_labels({"a": 0.1, "b": 0.05})
        assert result["a"] == "low"
        assert result["b"] == "low"

    def test_medium_bucket(self):
        computer = FingerprintComputer()
        result = computer._quantize_labels({"a": 0.3, "b": 0.45})
        assert result["a"] == "medium"
        assert result["b"] == "medium"

    def test_high_bucket(self):
        computer = FingerprintComputer()
        result = computer._quantize_labels({"a": 0.6, "b": 0.9})
        assert result["a"] == "high"
        assert result["b"] == "high"

    def test_edge_case_single_class(self):
        computer = FingerprintComputer()
        result = computer._quantize_labels({"only": 1.0})
        assert result["only"] == "high"

    def test_empty_proportions(self):
        computer = FingerprintComputer()
        result = computer._quantize_labels({})
        assert result == {}


class TestFeatureNoise:
    def test_noise_added(self):
        np.random.seed(42)
        computer = FingerprintComputer(FingerprintPrivacyConfig(feature_noise_sigma=0.1))
        moments = {"layer1": (1.0, 0.5)}
        noised = computer._add_feature_noise(moments)
        assert noised["layer1"][0] != 1.0
        assert noised["layer1"][1] != 0.5

    def test_variance_non_negative(self):
        computer = FingerprintComputer(FingerprintPrivacyConfig(feature_noise_sigma=1.0))
        moments = {"layer1": (0.0, 0.01)}
        for _ in range(100):
            noised = computer._add_feature_noise(moments)
            assert noised["layer1"][1] >= 0.0

    def test_clipping_applied(self):
        computer = FingerprintComputer(FingerprintPrivacyConfig(
            feature_noise_sigma=0.0, feature_clip_norm=1.0
        ))
        moments = {"layer1": (100.0, 50.0)}
        noised = computer._add_feature_noise(moments)
        assert noised["layer1"][0] == 1.0
        assert noised["layer1"][1] == 1.0

    def test_gradient_noise_disabled_by_default(self):
        computer = FingerprintComputer()
        grad_moments = {"layer1": (1.0, 0.5)}
        result = computer._add_gradient_noise(grad_moments)
        assert result["layer1"][0] != 1.0


class TestSampleBucketing:
    def test_tiny_dataset(self):
        computer = FingerprintComputer()
        assert computer._bucket_sample_count(50) == "0-100"

    def test_medium_dataset(self):
        computer = FingerprintComputer()
        assert computer._bucket_sample_count(500) == "100-1k"

    def test_large_dataset(self):
        computer = FingerprintComputer()
        assert computer._bucket_sample_count(5000) == "1k-10k"

    def test_very_large_dataset(self):
        computer = FingerprintComputer()
        assert computer._bucket_sample_count(50000) == "10k-100k"

    def test_enormous_dataset(self):
        computer = FingerprintComputer()
        assert computer._bucket_sample_count(200000) == "100k+"


class TestFingerprintComputer:
    def test_compute_from_label_counts(self):
        computer = FingerprintComputer()
        fp = computer.compute_from_label_counts(
            label_counts={"0": 50, "1": 30, "2": 20},
        )
        assert fp.label_buckets["0"] == "high"
        assert fp.label_buckets["1"] == "medium"
        assert fp.label_buckets["2"] == "medium"
        assert fp.num_classes == 3
        assert fp.sample_bucket == "100-1k"

    def test_compute_with_feature_moments(self):
        computer = FingerprintComputer(FingerprintPrivacyConfig(feature_noise_sigma=0.0))
        fp = computer.compute_from_label_counts(
            label_counts={"a": 100},
            feature_moments={"conv1": (0.5, 0.1)},
        )
        assert fp.noised_moments["conv1"] == (0.5, 0.1)

    def test_gradient_disabled_by_default(self):
        computer = FingerprintComputer()
        fp = computer.compute_from_label_counts(
            label_counts={"a": 10},
            gradient_moments={"layer1": (1.0, 0.5)},
        )
        assert fp.gradient_moments is None

    def test_gradient_enabled_explicitly(self):
        computer = FingerprintComputer(FingerprintPrivacyConfig(gradient_enabled=True))
        fp = computer.compute_from_label_counts(
            label_counts={"a": 10},
            gradient_moments={"layer1": (1.0, 0.5)},
        )
        assert fp.gradient_moments is not None
        assert "layer1" in fp.gradient_moments

    def test_extract_bn_moments(self):
        weights = {
            "bn1.running_mean": np.array([0.5, 0.6]),
            "bn1.running_var": np.array([0.1, 0.2]),
            "conv1.weight": np.array([1.0, 2.0]),
        }
        moments = FingerprintComputer.extract_bn_moments(weights)
        assert "bn1" in moments
        assert moments["bn1"][0] == pytest.approx(0.55)
        assert moments["bn1"][1] == pytest.approx(0.15)


class TestLabelSimilarity:
    def test_identical_buckets(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high", "b": "low"},
            noised_moments={}, sample_bucket="0-100", num_classes=2,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "high", "b": "low"},
            noised_moments={}, sample_bucket="0-100", num_classes=2,
        )
        assert fp1._label_similarity(fp2) == 1.0

    def test_adjacent_buckets(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={}, sample_bucket="0-100", num_classes=1,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "medium"},
            noised_moments={}, sample_bucket="0-100", num_classes=1,
        )
        assert fp1._label_similarity(fp2) == 0.5

    def test_opposite_buckets(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={}, sample_bucket="0-100", num_classes=1,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "low"},
            noised_moments={}, sample_bucket="0-100", num_classes=1,
        )
        assert fp1._label_similarity(fp2) == 0.0

    def test_partial_overlap(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high", "b": "low"},
            noised_moments={}, sample_bucket="0-100", num_classes=2,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "high", "b": "medium"},
            noised_moments={}, sample_bucket="0-100", num_classes=2,
        )
        sim = fp1._label_similarity(fp2)
        assert sim == pytest.approx(0.75)


class TestFeatureSimilarity:
    def test_identical_moments(self):
        fp1 = DataFingerprint(
            label_buckets={}, noised_moments={"l": (1.0, 0.5)},
            sample_bucket="0-100", num_classes=0,
        )
        fp2 = DataFingerprint(
            label_buckets={}, noised_moments={"l": (1.0, 0.5)},
            sample_bucket="0-100", num_classes=0,
        )
        assert fp1._feature_similarity(fp2) == pytest.approx(1.0)

    def test_orthogonal_moments(self):
        fp1 = DataFingerprint(
            label_buckets={}, noised_moments={"l": (1.0, 0.0)},
            sample_bucket="0-100", num_classes=0,
        )
        fp2 = DataFingerprint(
            label_buckets={}, noised_moments={"l": (0.0, 1.0)},
            sample_bucket="0-100", num_classes=0,
        )
        sim = fp1._feature_similarity(fp2)
        assert sim == pytest.approx(0.0, abs=1e-7)

    def test_empty_moments_returns_neutral(self):
        fp1 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
        )
        fp2 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
        )
        assert fp1._feature_similarity(fp2) == 0.5


class TestAffinityScore:
    def test_identical_fingerprints(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k", num_classes=1,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k", num_classes=1,
        )
        assert fp1.affinity_score(fp2) == pytest.approx(1.0)

    def test_dissimilar_fingerprints(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high", "b": "low"},
            noised_moments={"l": (1.0, 0.0)},
            sample_bucket="1k-10k", num_classes=2,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "low", "b": "high"},
            noised_moments={"l": (0.0, 1.0)},
            sample_bucket="1k-10k", num_classes=2,
        )
        score = fp1.affinity_score(fp2)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_external_history_score_increases_affinity(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k", num_classes=1,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "medium"},
            noised_moments={"l": (0.5, 0.3)},
            sample_bucket="1k-10k", num_classes=1,
        )
        w_no_hist = AffinityWeights()
        w_hist = AffinityWeights(external_history_score=0.8)
        score_no_hist = fp1.affinity_score(fp2, w_no_hist)
        score_hist = fp1.affinity_score(fp2, w_hist)
        assert score_hist > score_no_hist

    def test_affinity_bounded_0_1(self):
        fp1 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (10.0, 5.0)},
            sample_bucket="1k-10k", num_classes=1,
        )
        fp2 = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (10.0, 5.0)},
            sample_bucket="1k-10k", num_classes=1,
        )
        w = AffinityWeights(external_history_score=2.0)
        score = fp1.affinity_score(fp2, w)
        assert 0.0 <= score <= 1.0


class TestGradientSimilarity:
    def test_gradient_similarity_computed(self):
        fp1 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
            gradient_moments={"l": (1.0, 0.5)},
        )
        fp2 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
            gradient_moments={"l": (1.0, 0.5)},
        )
        assert fp1._gradient_similarity(fp2) == pytest.approx(1.0)

    def test_gradient_missing_returns_zero(self):
        fp1 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
        )
        fp2 = DataFingerprint(
            label_buckets={}, noised_moments={},
            sample_bucket="0-100", num_classes=0,
            gradient_moments={"l": (1.0, 0.5)},
        )
        assert fp1._gradient_similarity(fp2) == 0.0


class TestSerialization:
    def test_roundtrip(self):
        fp = DataFingerprint(
            label_buckets={"a": "high", "b": "low"},
            noised_moments={"conv1": (0.5, 0.1)},
            sample_bucket="1k-10k",
            num_classes=2,
        )
        d = fp.to_dict()
        fp2 = DataFingerprint.from_dict(d)
        assert fp2.label_buckets == fp.label_buckets
        assert fp2.noised_moments == fp.noised_moments
        assert fp2.sample_bucket == fp.sample_bucket
        assert fp2.num_classes == fp.num_classes
        assert fp2.gradient_moments is None

    def test_roundtrip_with_gradient(self):
        fp = DataFingerprint(
            label_buckets={"a": "medium"},
            noised_moments={"conv1": (0.3, 0.05)},
            sample_bucket="100-1k",
            num_classes=1,
            gradient_moments={"layer3": (0.8, 0.2)},
        )
        d = fp.to_dict()
        fp2 = DataFingerprint.from_dict(d)
        assert fp2.gradient_moments is not None
        assert fp2.gradient_moments["layer3"] == (0.8, 0.2)

    def test_json_serializable(self):
        fp = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k",
            num_classes=1,
        )
        d = fp.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        fp2 = DataFingerprint.from_dict(d2)
        assert fp2.label_buckets == fp.label_buckets


class TestBucketHelpers:
    def test_adjacent_bucket_low_medium(self):
        assert _adjacent_bucket("low", "medium") is True

    def test_adjacent_bucket_medium_high(self):
        assert _adjacent_bucket("medium", "high") is True

    def test_not_adjacent_low_high(self):
        assert _adjacent_bucket("low", "high") is False

    def test_same_bucket_not_adjacent(self):
        assert _adjacent_bucket("high", "high") is False


class TestPeerInfoIntegration:
    def test_peerinfo_with_fingerprint(self):
        fp = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k",
            num_classes=1,
        )
        peer = PeerInfo(
            peer_id="peer-1",
            domain="health",
            data_schema_hash="abc123",
            data_fingerprint=fp,
        )
        assert peer.data_fingerprint is not None
        assert peer.data_fingerprint.label_buckets == {"a": "high"}

    def test_peerinfo_without_fingerprint(self):
        peer = PeerInfo(
            peer_id="peer-2",
            domain="health",
            data_schema_hash="abc123",
        )
        assert peer.data_fingerprint is None

    def test_peerinfo_manifest_id(self):
        peer = PeerInfo(
            peer_id="peer-3",
            domain="health",
            data_schema_hash="abc123",
            manifest_id=b"\x01\x02\x03" * 10,
        )
        assert peer.manifest_id is not None
        assert len(peer.manifest_id) == 30


class TestSelectionContextManifestId:
    def test_manifest_id_primary_match(self):
        mid = b"\xaa\xbb" * 15
        peer1 = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", manifest_id=mid)
        peer2 = PeerInfo(peer_id="p2", domain="d", data_schema_hash="s", manifest_id=b"\x00" * 30)
        peer3 = PeerInfo(peer_id="p3", domain="d", data_schema_hash="s")
        ctx = SelectionContext(
            my_peer_id="me",
            my_domain="d",
            my_data_schema_hash="s",
            my_manifest_id=mid,
            known_peers=[peer1, peer2, peer3],
            my_model_version="1.0.0",
        )
        compatible = ctx.get_compatible_peers()
        assert len(compatible) == 1
        assert compatible[0].peer_id == "p1"

    def test_manifest_id_mismatch_uses_legacy(self):
        peer = PeerInfo(
            peer_id="p1", domain="d", data_schema_hash="s",
            model_version="1.0.0",
        )
        ctx = SelectionContext(
            my_peer_id="me",
            my_domain="d",
            my_data_schema_hash="s",
            my_manifest_id=b"\xaa" * 30,
            known_peers=[peer],
            my_model_version="1.0.0",
        )
        compatible = ctx.get_compatible_peers()
        assert len(compatible) == 1

    def test_no_manifest_id_uses_legacy(self):
        peer = PeerInfo(
            peer_id="p1", domain="d", data_schema_hash="s",
            model_version="1.0.0",
        )
        ctx = SelectionContext(
            my_peer_id="me",
            my_domain="d",
            my_data_schema_hash="s",
            known_peers=[peer],
            my_model_version="1.0.0",
        )
        compatible = ctx.get_compatible_peers()
        assert len(compatible) == 1

    def test_fingerprint_in_context(self):
        fp = DataFingerprint(
            label_buckets={"a": "high"},
            noised_moments={"l": (1.0, 0.5)},
            sample_bucket="1k-10k", num_classes=1,
        )
        ctx = SelectionContext(
            my_peer_id="me",
            my_domain="d",
            my_data_schema_hash="s",
            known_peers=[],
            my_fingerprint=fp,
        )
        assert ctx.my_fingerprint is not None
        assert ctx.my_fingerprint.label_buckets == {"a": "high"}
