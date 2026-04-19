"""
Krum aggregation strategies.
"""

from copy import deepcopy
from typing import List

import numpy as np

from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy, ModelUpdate
from quinkgl.aggregation.fedavg import FedAvg

__all__ = ["Krum", "MultiKrum"]


class Krum(AggregationStrategy):
    """
    Krum: Byzantine-resilient aggregation.

    Selects the update closest to the majority of updates,
    providing robustness against malicious peers.

    Reference: https://arxiv.org/abs/1703.02857
    """

    def __init__(self, num_byzantines: int = 1, **kwargs):
        """
        Initialize Krum aggregator.

        Args:
            num_byzantines: Expected number of malicious peers (f)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.num_byzantines = num_byzantines

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate using Krum - select the most central update.
        """
        self._validate_updates(updates)

        n = len(updates)
        if n < 2 * self.num_byzantines + 3:
            raise ValueError(
                f"Krum requires n >= 2f+3 updates (n={n}, f={self.num_byzantines})"
            )

        # Compute distances between all pairs of updates
        distances = self._compute_distances(updates)

        # Compute scores (sum of smallest n-2*f-1 distances for each update)
        n_closest = n - 2 * self.num_byzantines - 1
        scores = []
        for i in range(n):
            # Sort distances from update i to all others
            sorted_dist = np.sort(distances[i])
            # Sum of n-2*f-1 smallest distances
            scores.append(np.sum(sorted_dist[:n_closest]))

        # Select update with minimum score
        selected_idx = int(np.argmin(scores))
        selected_update = updates[selected_idx]

        return AggregatedModel(
            weights=deepcopy(selected_update.weights),
            contributing_peers=[selected_update.peer_id],
            total_samples=selected_update.sample_count,
            metadata={
                "aggregation_method": "krum",
                "num_byzantines": self.num_byzantines,
                "selected_peer": selected_update.peer_id
            },
            updates=[selected_update]
        )

    def _compute_distances(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Compute pairwise Euclidean distances between updates."""
        n = len(updates)
        distances = np.zeros((n, n))

        # Flatten all weights to vectors
        weight_vectors = []
        for update in updates:
            if isinstance(update.weights, np.ndarray):
                weight_vectors.append(update.weights.flatten())
            elif isinstance(update.weights, dict):
                # Concatenate all array values
                parts = []
                for key in sorted(update.weights.keys()):
                    val = update.weights[key]
                    if hasattr(val, '__array__'):
                        parts.append(val.flatten())
                weight_vectors.append(np.concatenate(parts))
            else:
                weight_vectors.append(np.array(update.weights).flatten())

        weight_vectors = [w.astype(np.float64) for w in weight_vectors]

        # Compute pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances


class MultiKrum(Krum):
    """
    Multi-Krum: Krum with averaging.

    Instead of selecting a single update, selects the n-2*f closest updates
    and averages them for better stability.
    """

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate using Multi-Krum - average the n-2*f most central updates.
        """
        self._validate_updates(updates)

        n = len(updates)
        if n < 2 * self.num_byzantines + 3:
            raise ValueError(
                f"MultiKrum requires n >= 2f+3 updates (n={n}, f={self.num_byzantines})"
            )

        # Compute distances
        distances = self._compute_distances(updates)

        # Compute scores
        n_closest = n - 2 * self.num_byzantines - 1
        scores = []
        for i in range(n):
            sorted_dist = np.sort(distances[i])
            scores.append(np.sum(sorted_dist[:n_closest]))

        # Select n-2*f updates with minimum scores
        num_selected = n - 2 * self.num_byzantines
        selected_indices = np.argsort(scores)[:num_selected].tolist()
        selected_updates = [updates[i] for i in selected_indices]

        # Average the selected updates
        fedavg = FedAvg()
        result = await fedavg.aggregate(selected_updates)

        result.metadata["aggregation_method"] = "multikrum"
        result.metadata["num_byzantines"] = self.num_byzantines
        result.metadata["selected_peers"] = [u.peer_id for u in selected_updates]

        return result
