"""Tests for spectral analysis of gossip topology graphs.

Validates the SpectralAnalyzer's computation of algebraic connectivity,
spectral gap, and mixing matrix properties across known graph topologies.
"""

import math
import numpy as np
import pytest

from quinkgl.topology.spectral import (
    SpectralAnalyzer,
    SpectralReport,
    build_ring_adjacency,
    build_complete_adjacency,
    build_random_regular_adjacency,
)


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _star_adjacency(n: int) -> np.ndarray:
    """Star graph: node 0 connected to all others."""
    A = np.zeros((n, n))
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A


# ------------------------------------------------------------------ #
# Complete graph
# ------------------------------------------------------------------ #

class TestCompleteGraph:
    def test_complete_graph_is_connected(self):
        A = build_complete_adjacency(10)
        report = SpectralAnalyzer().analyze(A)
        assert report.is_connected is True

    def test_complete_graph_algebraic_connectivity(self):
        """For K_n, λ₂ of the Laplacian = n (all eigenvalues except λ₁=0 equal n)."""
        n = 10
        A = build_complete_adjacency(n)
        report = SpectralAnalyzer().analyze(A)
        assert report.algebraic_connectivity == pytest.approx(n, abs=1e-6)

    def test_complete_graph_has_best_spectral_gap(self):
        """Complete graph has the largest possible spectral gap."""
        n = 10
        A = build_complete_adjacency(n)
        report = SpectralAnalyzer().analyze(A)
        # For a complete graph the MH mixing matrix concentrates strongly
        assert report.spectral_gap > 0.8

    def test_complete_graph_edges(self):
        n = 10
        A = build_complete_adjacency(n)
        report = SpectralAnalyzer().analyze(A)
        assert report.num_edges == n * (n - 1) // 2


# ------------------------------------------------------------------ #
# Ring graph
# ------------------------------------------------------------------ #

class TestRingGraph:
    def test_ring_is_connected(self):
        A = build_ring_adjacency(10)
        report = SpectralAnalyzer().analyze(A)
        assert report.is_connected is True

    def test_ring_algebraic_connectivity(self):
        """Ring graph has known λ₂ = 2(1 - cos(2π/n))."""
        n = 10
        A = build_ring_adjacency(n)
        report = SpectralAnalyzer().analyze(A)
        expected = 2 * (1 - math.cos(2 * math.pi / n))
        assert report.algebraic_connectivity == pytest.approx(expected, abs=1e-6)

    def test_ring_worse_than_complete(self):
        """Ring graph should have worse spectral gap than complete graph."""
        n = 10
        ring = SpectralAnalyzer().analyze(build_ring_adjacency(n))
        complete = SpectralAnalyzer().analyze(build_complete_adjacency(n))
        assert ring.spectral_gap < complete.spectral_gap


# ------------------------------------------------------------------ #
# Disconnected graph
# ------------------------------------------------------------------ #

class TestDisconnectedGraph:
    def test_disconnected_graph(self):
        """Two isolated components → λ₂ = 0 → not connected."""
        A = np.zeros((6, 6))
        # Component 1: nodes 0-2
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1
        # Component 2: nodes 3-5
        A[3, 4] = A[4, 3] = 1
        A[4, 5] = A[5, 4] = 1

        report = SpectralAnalyzer().analyze(A)
        assert report.is_connected is False
        assert report.algebraic_connectivity == pytest.approx(0.0, abs=1e-6)

    def test_single_node(self):
        A = np.zeros((1, 1))
        report = SpectralAnalyzer().analyze(A)
        assert report.num_nodes == 1
        assert report.num_edges == 0

    def test_empty_graph(self):
        A = np.zeros((0, 0))
        report = SpectralAnalyzer().analyze(A)
        assert report.num_nodes == 0


# ------------------------------------------------------------------ #
# Star graph
# ------------------------------------------------------------------ #

class TestStarGraph:
    def test_star_is_connected(self):
        A = _star_adjacency(10)
        report = SpectralAnalyzer().analyze(A)
        assert report.is_connected is True

    def test_star_algebraic_connectivity_equals_1(self):
        """Star graph has λ₂ = 1 for n ≥ 3."""
        A = _star_adjacency(10)
        report = SpectralAnalyzer().analyze(A)
        assert report.algebraic_connectivity == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ #
# Mixing matrix properties
# ------------------------------------------------------------------ #

class TestMixingMatrix:
    def test_doubly_stochastic(self):
        """Metropolis–Hastings matrix should be doubly stochastic."""
        n = 8
        A = build_ring_adjacency(n)
        analyzer = SpectralAnalyzer()
        W = analyzer._metropolis_hastings(A, A.sum(axis=1), n)

        # Row sums = 1
        np.testing.assert_allclose(W.sum(axis=1), np.ones(n), atol=1e-12)
        # Column sums = 1 (doubly stochastic)
        np.testing.assert_allclose(W.sum(axis=0), np.ones(n), atol=1e-12)

    def test_eigenvalues_in_range(self):
        """All eigenvalues of W should be in [-1, 1]."""
        A = build_ring_adjacency(10)
        report = SpectralAnalyzer().analyze(A)
        assert np.all(report.mixing_matrix_eigenvalues >= -1.0 - 1e-10)
        assert np.all(report.mixing_matrix_eigenvalues <= 1.0 + 1e-10)

    def test_largest_eigenvalue_is_1(self):
        """Largest eigenvalue of doubly stochastic W should be 1."""
        A = build_complete_adjacency(5)
        report = SpectralAnalyzer().analyze(A)
        assert report.mixing_matrix_eigenvalues[-1] == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------------ #
# Peer-list interface
# ------------------------------------------------------------------ #

class TestPeerListInterface:
    def test_from_peer_lists(self):
        neighborhoods = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        report = SpectralAnalyzer().analyze_from_peer_lists(neighborhoods)
        assert report.num_nodes == 3
        assert report.num_edges == 3
        assert report.is_connected is True

    def test_from_peer_lists_disconnected(self):
        neighborhoods = {
            "A": ["B"],
            "B": ["A"],
            "C": ["D"],
            "D": ["C"],
        }
        report = SpectralAnalyzer().analyze_from_peer_lists(neighborhoods)
        assert report.is_connected is False


# ------------------------------------------------------------------ #
# Topology comparison
# ------------------------------------------------------------------ #

class TestTopologyComparison:
    def test_compare_ring_vs_complete(self):
        """compare_topologies should return one report per topology."""
        n = 10
        results = SpectralAnalyzer().compare_topologies({
            "ring": build_ring_adjacency(n),
            "complete": build_complete_adjacency(n),
        })
        assert "ring" in results
        assert "complete" in results
        assert results["ring"].spectral_gap < results["complete"].spectral_gap

    def test_summary_string(self):
        report = SpectralAnalyzer().analyze(build_ring_adjacency(6))
        summary = report.summary()
        assert "n=6" in summary
        assert "gap=" in summary


# ------------------------------------------------------------------ #
# Helper graph builders
# ------------------------------------------------------------------ #

class TestGraphBuilders:
    def test_ring_adjacency_shape(self):
        A = build_ring_adjacency(5)
        assert A.shape == (5, 5)
        assert A.sum() == 10  # 5 edges * 2

    def test_complete_adjacency_shape(self):
        A = build_complete_adjacency(4)
        assert A.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(A), np.zeros(4))

    def test_random_regular_degree(self):
        A = build_random_regular_adjacency(10, 4)
        degrees = A.sum(axis=1)
        # Should be close to target degree (±1 allowed due to heuristic)
        assert np.all(degrees >= 3)
        assert np.all(degrees <= 5)
