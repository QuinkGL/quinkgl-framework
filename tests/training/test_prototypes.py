"""Tests for FedProto / FedPAC prototype alignment (Phase 6e)."""

import json
import numpy as np
import pytest

from quinkgl.training.prototypes import (
    ClassPrototype,
    PrototypeStore,
    FedPACCollaborator,
)


# ── ClassPrototype ──────────────────────────────────────────────────


class TestClassPrototype:
    def test_to_dict_basic(self):
        p = ClassPrototype(label="cat", embedding=np.array([1.0, 2.0]), sample_count=5)
        d = p.to_dict()
        assert d["label"] == "cat"
        assert d["sample_count"] == 5
        assert d["embedding"] == [1.0, 2.0]
        assert "variance" not in d

    def test_to_dict_with_variance(self):
        p = ClassPrototype(
            label="dog", embedding=np.array([3.0]), sample_count=10,
            variance=np.array([0.5]),
        )
        d = p.to_dict()
        assert d["variance"] == [0.5]

    def test_from_dict_roundtrip(self):
        p = ClassPrototype(
            label="bird", embedding=np.array([1.0, 2.0, 3.0]), sample_count=7,
            variance=np.array([0.1, 0.2, 0.3]),
        )
        d = p.to_dict()
        p2 = ClassPrototype.from_dict(d)
        assert p2.label == "bird"
        assert p2.sample_count == 7
        np.testing.assert_array_almost_equal(p2.embedding, p.embedding)
        np.testing.assert_array_almost_equal(p2.variance, p.variance)

    def test_from_dict_without_variance(self):
        d = {"label": "fish", "embedding": [4.0, 5.0], "sample_count": 3}
        p = ClassPrototype.from_dict(d)
        assert p.variance is None

    def test_to_json_roundtrip(self):
        p = ClassPrototype(label="x", embedding=np.array([1.5]), sample_count=2)
        j = p.to_json()
        p2 = ClassPrototype.from_json(j)
        assert p2.label == "x"
        np.testing.assert_array_almost_equal(p2.embedding, np.array([1.5]))

    def test_from_json_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            ClassPrototype.from_json("not json")


# ── PrototypeStore ──────────────────────────────────────────────────


class TestPrototypeStoreComputeLocal:
    def test_single_label_single_feature(self):
        store = PrototypeStore()
        features = {"a": [np.array([1.0, 2.0])]}
        protos = store.compute_local_prototypes(features)
        assert "a" in protos
        np.testing.assert_array_almost_equal(protos["a"].embedding, [1.0, 2.0])
        assert protos["a"].sample_count == 1

    def test_single_label_multiple_features(self):
        store = PrototypeStore()
        features = {"a": [np.array([2.0]), np.array([4.0]), np.array([6.0])]}
        protos = store.compute_local_prototypes(features)
        np.testing.assert_array_almost_equal(protos["a"].embedding, [4.0])
        assert protos["a"].sample_count == 3
        np.testing.assert_array_almost_equal(protos["a"].variance, [np.var([2.0, 4.0, 6.0])])

    def test_multiple_labels(self):
        store = PrototypeStore()
        features = {
            "cat": [np.array([1.0]), np.array([3.0])],
            "dog": [np.array([5.0]), np.array([7.0])],
        }
        protos = store.compute_local_prototypes(features)
        assert set(protos.keys()) == {"cat", "dog"}
        np.testing.assert_array_almost_equal(protos["cat"].embedding, [2.0])
        np.testing.assert_array_almost_equal(protos["dog"].embedding, [6.0])

    def test_empty_features_list_skipped(self):
        store = PrototypeStore()
        features = {"a": [], "b": [np.array([1.0])]}
        protos = store.compute_local_prototypes(features)
        assert "a" not in protos
        assert "b" in protos

    def test_empty_input(self):
        store = PrototypeStore()
        protos = store.compute_local_prototypes({})
        assert protos == {}


class TestPrototypeStoreMergeGlobal:
    def test_single_peer(self):
        store = PrototypeStore()
        peer_protos = {
            "peer1": [ClassPrototype("a", np.array([1.0]), 10)],
        }
        merged = store.merge_global_prototypes(peer_protos)
        np.testing.assert_array_almost_equal(merged["a"].embedding, [1.0])
        assert merged["a"].sample_count == 10

    def test_weighted_average_two_peers(self):
        store = PrototypeStore()
        peer_protos = {
            "p1": [ClassPrototype("a", np.array([2.0]), 10)],
            "p2": [ClassPrototype("a", np.array([6.0]), 30)],
        }
        merged = store.merge_global_prototypes(peer_protos)
        # weighted avg: (2*10 + 6*30) / 40 = 200/40 = 5.0
        np.testing.assert_array_almost_equal(merged["a"].embedding, [5.0])
        assert merged["a"].sample_count == 40

    def test_multiple_labels_merged(self):
        store = PrototypeStore()
        peer_protos = {
            "p1": [
                ClassPrototype("x", np.array([1.0]), 5),
                ClassPrototype("y", np.array([2.0]), 5),
            ],
            "p2": [
                ClassPrototype("x", np.array([3.0]), 5),
            ],
        }
        merged = store.merge_global_prototypes(peer_protos)
        np.testing.assert_array_almost_equal(merged["x"].embedding, [2.0])
        np.testing.assert_array_almost_equal(merged["y"].embedding, [2.0])

    def test_no_shared_labels(self):
        store = PrototypeStore()
        peer_protos = {
            "p1": [ClassPrototype("a", np.array([1.0]), 5)],
            "p2": [ClassPrototype("b", np.array([2.0]), 5)],
        }
        merged = store.merge_global_prototypes(peer_protos)
        assert "a" in merged
        assert "b" in merged

    def test_empty_peer_prototypes(self):
        store = PrototypeStore()
        merged = store.merge_global_prototypes({})
        assert merged == {}


class TestPrototypeStoreAlignmentLoss:
    def test_perfect_alignment_zero_loss(self):
        store = PrototypeStore()
        store.local_prototypes = {"a": ClassPrototype("a", np.array([1.0, 2.0]), 5)}
        store.global_prototypes = {"a": ClassPrototype("a", np.array([1.0, 2.0]), 10)}
        assert store.prototype_alignment_loss() == pytest.approx(0.0)

    def test_misalignment_nonzero_loss(self):
        store = PrototypeStore()
        store.local_prototypes = {"a": ClassPrototype("a", np.array([0.0]), 5)}
        store.global_prototypes = {"a": ClassPrototype("a", np.array([2.0]), 10)}
        loss = store.prototype_alignment_loss()
        assert loss == pytest.approx(4.0)

    def test_shared_labels_only(self):
        store = PrototypeStore()
        store.local_prototypes = {
            "a": ClassPrototype("a", np.array([0.0]), 5),
            "b": ClassPrototype("b", np.array([0.0]), 5),
        }
        store.global_prototypes = {
            "a": ClassPrototype("a", np.array([2.0]), 10),
        }
        # only "a" is shared
        assert store.prototype_alignment_loss() == pytest.approx(4.0)

    def test_no_shared_labels_returns_zero(self):
        store = PrototypeStore()
        store.local_prototypes = {"a": ClassPrototype("a", np.array([0.0]), 5)}
        store.global_prototypes = {"b": ClassPrototype("b", np.array([0.0]), 10)}
        assert store.prototype_alignment_loss() == pytest.approx(0.0)

    def test_empty_prototypes_returns_zero(self):
        store = PrototypeStore()
        assert store.prototype_alignment_loss() == pytest.approx(0.0)


class TestPrototypeStoreJson:
    def test_local_prototypes_to_json_roundtrip(self):
        store = PrototypeStore()
        store.local_prototypes = {
            "a": ClassPrototype("a", np.array([1.0, 2.0]), 5, variance=np.array([0.1, 0.2])),
            "b": ClassPrototype("b", np.array([3.0]), 10),
        }
        j = store.local_prototypes_to_json()
        data = json.loads(j)
        assert len(data) == 2
        labels = {d["label"] for d in data}
        assert labels == {"a", "b"}

    def test_parse_peer_prototypes(self):
        json_str = json.dumps([
            {"label": "cat", "embedding": [1.0, 2.0], "sample_count": 5},
            {"label": "dog", "embedding": [3.0, 4.0], "sample_count": 10},
        ])
        protos = PrototypeStore.parse_peer_prototypes("peer1", json_str)
        assert len(protos) == 2
        assert protos[0].label == "cat"
        assert protos[1].sample_count == 10

    def test_parse_peer_prototypes_empty_list(self):
        protos = PrototypeStore.parse_peer_prototypes("peer1", "[]")
        assert protos == []


# ── FedPACCollaborator ──────────────────────────────────────────────


class TestFedPACCollaborator:
    def test_compute_discrepancy_identical(self):
        collab = FedPACCollaborator()
        my = {"a": ClassPrototype("a", np.array([1.0, 1.0]), 5)}
        peers = {"p1": {"a": ClassPrototype("a", np.array([1.0, 1.0]), 10)}}
        disc = collab.compute_discrepancy(my, peers)
        assert disc["p1"] == pytest.approx(0.0)

    def test_compute_discrepancy_different(self):
        collab = FedPACCollaborator()
        my = {"a": ClassPrototype("a", np.array([0.0, 0.0]), 5)}
        peers = {"p1": {"a": ClassPrototype("a", np.array([3.0, 4.0]), 10)}}
        disc = collab.compute_discrepancy(my, peers)
        assert disc["p1"] == pytest.approx(5.0)  # L2 norm of (3,4)

    def test_compute_discrepancy_no_shared_labels(self):
        # Peers with zero label-overlap get ``inf`` so the combination-weight
        # softmax filters them out (exp(-inf) == 0).  A shared-label-free
        # peer would otherwise dominate the weighted average with a phantom
        # zero-distance and corrupt the aggregate.
        collab = FedPACCollaborator()
        my = {"a": ClassPrototype("a", np.array([1.0]), 5)}
        peers = {"p1": {"b": ClassPrototype("b", np.array([1.0]), 10)}}
        disc = collab.compute_discrepancy(my, peers)
        assert disc["p1"] == float("inf")

    def test_compute_discrepancy_multiple_peers(self):
        collab = FedPACCollaborator()
        my = {"a": ClassPrototype("a", np.array([0.0]), 5)}
        peers = {
            "p1": {"a": ClassPrototype("a", np.array([1.0]), 10)},
            "p2": {"a": ClassPrototype("a", np.array([2.0]), 10)},
        }
        disc = collab.compute_discrepancy(my, peers)
        assert disc["p1"] == pytest.approx(1.0)
        assert disc["p2"] == pytest.approx(2.0)

    def test_compute_combination_weights_similar_preferred(self):
        collab = FedPACCollaborator()
        disc = {"p1": 0.5, "p2": 2.0}
        weights = collab.compute_combination_weights(disc)
        assert weights["p1"] > weights["p2"]

    def test_compute_combination_weights_sum_to_one(self):
        collab = FedPACCollaborator()
        disc = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
        weights = collab.compute_combination_weights(disc)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_compute_combination_weights_zero_discrepancy(self):
        collab = FedPACCollaborator()
        disc = {"p1": 0.0, "p2": 1.0}
        weights = collab.compute_combination_weights(disc)
        assert weights["p1"] > weights["p2"]

    def test_compute_combination_weights_empty(self):
        collab = FedPACCollaborator()
        weights = collab.compute_combination_weights({})
        assert weights == {}

    def test_compute_combination_weights_uniform_fallback(self):
        collab = FedPACCollaborator()
        disc = {"p1": float("inf"), "p2": float("inf")}
        weights = collab.compute_combination_weights(disc, temperature=1.0)
        assert weights["p1"] == pytest.approx(0.5)
        assert weights["p2"] == pytest.approx(0.5)

    def test_compute_combination_weights_temperature(self):
        collab = FedPACCollaborator()
        disc = {"p1": 0.1, "p2": 1.0}
        w_low_t = collab.compute_combination_weights(disc, temperature=0.1)
        w_high_t = collab.compute_combination_weights(disc, temperature=10.0)
        # Lower temperature → sharper (more concentrated on best peer)
        diff_low = w_low_t["p1"] - w_low_t["p2"]
        diff_high = w_high_t["p1"] - w_high_t["p2"]
        assert diff_low > diff_high
