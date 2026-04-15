"""
Spectral Analysis for Gossip Topology Graphs.

Provides runtime measurement of the algebraic connectivity (Fiedler
value) and spectral gap for the communication topology.  In gossip
learning, convergence speed is directly proportional to the spectral
gap ``(1 − λ₂)`` of the mixing matrix.

References:
    - Koloskova et al. 2020 — "Unified Theory of Decentralized SGD with
      Changing Topology and Local Updates"
    - Boyd et al. 2006 — "Randomized Gossip Algorithms"
    - Lian et al. 2017 — "Can Decentralized Algorithms Outperform
      Centralized Algorithms?"

Usage:
    from quinkgl.topology.spectral import SpectralAnalyzer

    analyzer = SpectralAnalyzer()
    report = analyzer.analyze(adjacency)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectralReport:
    """Result of spectral analysis on the communication topology.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of undirected edges.
    algebraic_connectivity : float
        The second-smallest eigenvalue of the Laplacian (λ₂, Fiedler value).
        Positive ↔ graph is connected.
    spectral_gap : float
        ``1 − |λ₂(W)|`` where ``W`` is the doubly-stochastic mixing matrix.
        Larger gap → faster gossip convergence.
    mixing_time_upper : float
        Upper-bound estimate on mixing time: ``log(n) / spectral_gap``.
    is_connected : bool
        Whether the graph is connected (λ₂ > 0).
    laplacian_eigenvalues : np.ndarray
        Full sorted eigenvalue spectrum of the Laplacian.
    mixing_matrix_eigenvalues : np.ndarray
        Full sorted eigenvalue spectrum of the mixing matrix.
    degree_stats : Dict[str, float]
        Min, max, mean, std of node degrees.
    """

    num_nodes: int = 0
    num_edges: int = 0
    algebraic_connectivity: float = 0.0
    spectral_gap: float = 0.0
    mixing_time_upper: float = float("inf")
    is_connected: bool = False
    laplacian_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    mixing_matrix_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    degree_stats: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"n={self.num_nodes} e={self.num_edges} "
            f"λ₂={self.algebraic_connectivity:.4f} "
            f"gap={self.spectral_gap:.4f} "
            f"connected={self.is_connected} "
            f"mix_time≤{self.mixing_time_upper:.1f}"
        )


class SpectralAnalyzer:
    """Compute spectral properties of the gossip communication graph.

    The analyzer accepts an adjacency matrix (or an adjacency list) and
    computes the Laplacian eigenvalues, the Metropolis–Hastings mixing
    matrix, and derived quantities.

    Parameters
    ----------
    tolerance : float
        Threshold below which eigenvalues are considered zero.
    """

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = tolerance

    def analyze(self, adjacency: np.ndarray) -> SpectralReport:
        """Run full spectral analysis on an adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray
            Symmetric ``(n, n)`` adjacency matrix with 0/1 entries.
            Self-loops (diagonal) are ignored.

        Returns
        -------
        SpectralReport
        """
        A = np.asarray(adjacency, dtype=np.float64)
        n = A.shape[0]

        if n == 0:
            return SpectralReport()

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(A, 0.0)

        # Symmetrise (handle directed edges)
        A = np.maximum(A, A.T)

        num_edges = int(np.sum(A) / 2)
        degrees = A.sum(axis=1)
        degree_stats = {
            "min": float(np.min(degrees)),
            "max": float(np.max(degrees)),
            "mean": float(np.mean(degrees)),
            "std": float(np.std(degrees)),
        }

        # ---- Laplacian ----
        L = np.diag(degrees) - A
        lap_eigenvalues = np.sort(np.linalg.eigvalsh(L))

        # Algebraic connectivity = λ₂ (second smallest)
        if n < 2:
            algebraic_connectivity = 0.0
        else:
            algebraic_connectivity = max(float(lap_eigenvalues[1]), 0.0)

        is_connected = algebraic_connectivity > self.tolerance

        # ---- Metropolis–Hastings mixing matrix ----
        W = self._metropolis_hastings(A, degrees, n)
        mix_eigenvalues = np.sort(np.linalg.eigvalsh(W))

        # Spectral gap = 1 − |second largest eigenvalue of W|
        if n < 2:
            spectral_gap = 0.0
        else:
            # The eigenvalues of a doubly-stochastic matrix lie in [-1, 1]
            # with λ₁ = 1.  We want 1 - max(|λ₂|, |λ_n|).
            abs_eigs = np.abs(mix_eigenvalues)
            # Sort descending and take second largest |eigenvalue|
            abs_eigs_sorted = np.sort(abs_eigs)[::-1]
            second_largest_abs = abs_eigs_sorted[1] if len(abs_eigs_sorted) > 1 else 0.0
            spectral_gap = 1.0 - second_largest_abs

        # Mixing time upper bound
        if spectral_gap > self.tolerance:
            mixing_time_upper = np.log(n) / spectral_gap
        else:
            mixing_time_upper = float("inf")

        return SpectralReport(
            num_nodes=n,
            num_edges=num_edges,
            algebraic_connectivity=algebraic_connectivity,
            spectral_gap=spectral_gap,
            mixing_time_upper=mixing_time_upper,
            is_connected=is_connected,
            laplacian_eigenvalues=lap_eigenvalues,
            mixing_matrix_eigenvalues=mix_eigenvalues,
            degree_stats=degree_stats,
        )

    def analyze_from_peer_lists(
        self,
        peer_neighborhoods: Dict[str, List[str]],
    ) -> SpectralReport:
        """Convenience wrapper: build adjacency from peer-ID neighbor lists.

        Parameters
        ----------
        peer_neighborhoods : dict
            Mapping ``{node_id: [neighbor_id, ...]}`` for every node
            in the swarm.

        Returns
        -------
        SpectralReport
        """
        all_ids = sorted(set(peer_neighborhoods.keys()) |
                         {p for nbrs in peer_neighborhoods.values() for p in nbrs})
        id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
        n = len(all_ids)
        A = np.zeros((n, n), dtype=np.float64)

        for node_id, neighbors in peer_neighborhoods.items():
            i = id_to_idx[node_id]
            for nbr_id in neighbors:
                if nbr_id in id_to_idx:
                    j = id_to_idx[nbr_id]
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        return self.analyze(A)

    def compare_topologies(
        self,
        topologies: Dict[str, np.ndarray],
    ) -> Dict[str, SpectralReport]:
        """Analyze multiple adjacency matrices and return a comparison dict.

        Parameters
        ----------
        topologies : dict
            Mapping ``{name: adjacency_matrix}`` for each topology.

        Returns
        -------
        dict
            Mapping ``{name: SpectralReport}``.
        """
        return {name: self.analyze(adj) for name, adj in topologies.items()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _metropolis_hastings(A: np.ndarray, degrees: np.ndarray, n: int) -> np.ndarray:
        """Build the Metropolis–Hastings doubly-stochastic mixing matrix.

        W[i,j] = 1 / (1 + max(d_i, d_j))   if (i,j) is an edge
        W[i,i] = 1 − Σ_{j≠i} W[i,j]

        This is the standard weight rule for gossip algorithms that guarantees
        convergence to the average (Boyd et al. 2006).
        """
        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > 0:
                    w = 1.0 / (1.0 + max(degrees[i], degrees[j]))
                    W[i, j] = w
                    W[j, i] = w
        # Diagonal: ensure rows sum to 1
        for i in range(n):
            W[i, i] = 1.0 - np.sum(W[i, :])
        return W


def build_ring_adjacency(n: int) -> np.ndarray:
    """Create a ring graph adjacency matrix (useful for benchmarks)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


def build_complete_adjacency(n: int) -> np.ndarray:
    """Create a complete graph adjacency matrix."""
    A = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def build_random_regular_adjacency(n: int, degree: int, seed: int = 42) -> np.ndarray:
    """Create a random regular graph adjacency matrix.

    Uses a simple pairing algorithm; falls back to random edges if
    the pairing fails for odd ``n * degree``.
    """
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n), dtype=np.float64)

    if degree >= n:
        return build_complete_adjacency(n)

    # Simple heuristic: for each node, try to add `degree` random edges
    for _ in range(10):  # retry attempts
        A_trial = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            current_degree = int(A_trial[i].sum())
            needed = degree - current_degree
            if needed <= 0:
                continue
            candidates = [
                j for j in range(n)
                if j != i and A_trial[i, j] == 0 and int(A_trial[j].sum()) < degree
            ]
            if len(candidates) >= needed:
                chosen = rng.choice(candidates, size=needed, replace=False)
                for j in chosen:
                    A_trial[i, j] = 1.0
                    A_trial[j, i] = 1.0

        min_deg = A_trial.sum(axis=1).min()
        if min_deg >= degree - 1:
            return A_trial

    return A_trial
