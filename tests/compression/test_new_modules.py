"""Tests for version compatibility, consensus, quantization, and quality."""

import pytest
import numpy as np

from quinkgl.topology.base import is_version_compatible, SelectionContext, PeerInfo
from quinkgl.gossip.consensus import ConsensusTracker, PeerCheckpoint
from quinkgl.serialization.quantization import QuantizationConfig, quantize_weights, dequantize_weights
from quinkgl.serialization.sparsification import SparsificationConfig, compute_delta, apply_delta
from quinkgl.training.quality import compute_weight_fingerprint, cosine_similarity_weights, compute_peer_similarity


class TestVersionCompatibility:
    def test_same_major_compatible(self):
        assert is_version_compatible("1.0.0", "1.9.9") is True

    def test_different_major_incompatible(self):
        assert is_version_compatible("1.0.0", "2.0.0") is False

    def test_zero_major_incompatible_with_one(self):
        assert is_version_compatible("0.1.0", "1.0.0") is False

    def test_same_version_compatible(self):
        assert is_version_compatible("3.2.1", "3.2.1") is True

    def test_selection_context_filters_by_version(self):
        ctx = SelectionContext(
            my_peer_id="me",
            my_domain="health",
            my_data_schema_hash="abc",
            known_peers=[
                PeerInfo("p1", "health", "abc", model_version="1.0.0"),
                PeerInfo("p2", "health", "abc", model_version="2.0.0"),
                PeerInfo("p3", "health", "abc", model_version="1.5.0"),
            ],
            my_model_version="1.0.0",
        )
        compatible = ctx.get_compatible_peers()
        peer_ids = {p.peer_id for p in compatible}
        assert "p1" in peer_ids
        assert "p2" not in peer_ids
        assert "p3" in peer_ids


class TestConsensusTracker:
    def test_empty_tracker_no_consensus(self):
        tracker = ConsensusTracker()
        result = tracker.check_consensus()
        assert result is None

    def test_consensus_reached(self):
        tracker = ConsensusTracker(consensus_threshold=0.5, loss_tolerance=0.5)
        tracker.record_checkpoint(PeerCheckpoint("n1", 10, 0.5, 0.9))
        tracker.record_checkpoint(PeerCheckpoint("n2", 10, 0.52, 0.88))
        tracker.record_checkpoint(PeerCheckpoint("n3", 10, 1.0, 0.7))
        result = tracker.check_consensus(10)
        assert result.reached is True
        assert result.total_peers == 3

    def test_consensus_not_reached(self):
        tracker = ConsensusTracker(consensus_threshold=0.8, loss_tolerance=0.05)
        tracker.record_checkpoint(PeerCheckpoint("n1", 10, 0.5, 0.9))
        tracker.record_checkpoint(PeerCheckpoint("n2", 10, 1.5, 0.3))
        result = tracker.check_consensus(10)
        assert result.reached is False

    def test_should_checkpoint(self):
        tracker = ConsensusTracker(checkpoint_interval=5)
        assert tracker.should_checkpoint(5) is True
        tracker._last_local_checkpoint_round = 5
        assert tracker.should_checkpoint(10) is True
        assert tracker.should_checkpoint(7) is False


class TestQuantization:
    def test_8bit_quantization_roundtrip(self):
        weights = np.random.randn(100).astype(np.float32)
        q, meta = quantize_weights(weights, QuantizationConfig(bits=8))
        restored = dequantize_weights(q, meta)
        error = np.abs(weights - restored).mean()
        assert error < 0.15

    def test_dict_quantization_roundtrip(self):
        weights = {
            "layer1": np.random.randn(50).astype(np.float32),
            "layer2": np.random.randn(30).astype(np.float32),
        }
        q, meta = quantize_weights(weights, QuantizationConfig(bits=8))
        restored = dequantize_weights(q, meta)
        for key in weights:
            error = np.abs(weights[key] - restored[key]).mean()
            assert error < 0.15

    def test_non_float_weights_unchanged(self):
        weights = np.array([1, 2, 3], dtype=np.int32)
        q, meta = quantize_weights(weights, QuantizationConfig(bits=8))
        assert meta is None

    def test_quantization_size_reduction(self):
        weights = np.random.randn(1000).astype(np.float32)
        q, meta = quantize_weights(weights, QuantizationConfig(bits=8))
        assert q.nbytes < weights.nbytes


class TestDeltaCompression:
    def test_delta_roundtrip_numpy(self):
        base = np.random.randn(50).astype(np.float32)
        current = base + np.random.randn(50).astype(np.float32) * 0.01
        delta = compute_delta(current, base)
        reconstructed = apply_delta(base, delta)
        np.testing.assert_allclose(current, reconstructed, atol=1e-5)

    def test_delta_roundtrip_dict(self):
        base = {"l1": np.random.randn(20).astype(np.float32)}
        current = {"l1": base["l1"] + np.random.randn(20).astype(np.float32) * 0.01}
        delta = compute_delta(current, base)
        reconstructed = apply_delta(base, delta)
        np.testing.assert_allclose(current["l1"], reconstructed["l1"], atol=1e-5)

    def test_zero_delta(self):
        base = np.random.randn(50).astype(np.float32)
        delta = compute_delta(base, base)
        np.testing.assert_allclose(delta, np.zeros_like(base), atol=1e-10)


class TestQualityAssessment:
    def test_identical_weights_high_similarity(self):
        w = np.random.randn(100).astype(np.float32)
        sim = cosine_similarity_weights(w, w)
        assert abs(sim - 1.0) < 1e-5

    def test_opposite_weights_negative_similarity(self):
        w = np.random.randn(100).astype(np.float32)
        sim = cosine_similarity_weights(w, -w)
        assert sim < -0.9

    def test_similar_weights_high_similarity(self):
        a = np.random.randn(100).astype(np.float32)
        b = a + np.random.randn(100).astype(np.float32) * 0.01
        sim = cosine_similarity_weights(a, b)
        assert sim > 0.9

    def test_weight_fingerprint_has_norm(self):
        w = np.random.randn(100).astype(np.float32)
        fp = compute_weight_fingerprint(w)
        assert "norm" in fp
        assert fp["norm"] > 0

    def test_peer_similarity_single_update(self):
        w = np.random.randn(50).astype(np.float32)
        result = compute_peer_similarity([w])
        assert result["mean_similarity"] == 1.0

    def test_peer_similarity_multiple_updates(self):
        base = np.random.randn(100).astype(np.float32)
        updates = [base + np.random.randn(100).astype(np.float32) * 0.01 for _ in range(3)]
        result = compute_peer_similarity(updates)
        assert result["mean_similarity"] > 0.9
