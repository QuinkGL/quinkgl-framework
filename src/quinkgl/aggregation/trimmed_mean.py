"""
Trimmed Mean aggregation strategy.
"""

from typing import List

import numpy as np

from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy, ModelUpdate
from quinkgl.aggregation.fedavg import FedAvg

__all__ = ["TrimmedMean"]


class TrimmedMean(AggregationStrategy):
    """
    Trimmed Mean: Byzantine-resilient aggregation.

    Removes the smallest and largest values before averaging,
    providing robustness against malicious/faulty peers.

    Reference: https://arxiv.org/abs/1803.09877
    """

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        """
        Initialize TrimmedMean aggregator.

        Args:
            trim_ratio: Fraction of smallest/largest values to trim (0-0.5)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate using trimmed mean.
        """
        self._validate_updates(updates)

        if len(updates) < 3:
            # Not enough updates to trim, fall back to simple average
            fedavg = FedAvg()
            result = await fedavg.aggregate(updates)
            result.metadata["aggregation_method"] = "trimmed_mean_fallback"
            return result

        first_weights = updates[0].weights

        if isinstance(first_weights, np.ndarray):
            aggregated = self._trim_numpy(updates)
        elif isinstance(first_weights, dict):
            aggregated = self._trim_dict(updates)
        else:
            # Fallback
            fedavg = FedAvg()
            return await fedavg.aggregate(updates)

        return AggregatedModel(
            weights=aggregated,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={"aggregation_method": "trimmed_mean", "trim_ratio": self.trim_ratio},
            updates=updates
        )

    def _trim_numpy(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Apply trimmed mean to numpy arrays."""
        weights_matrix = np.stack([u.weights.flatten() for u in updates])
        n = len(updates)
        k = int(n * self.trim_ratio)

        if k == 0:
            # No trimming, use mean
            return np.mean(weights_matrix, axis=0).astype(updates[0].weights.dtype)

        # For each parameter, sort and trim
        trimmed_result = np.zeros_like(weights_matrix[0])
        for i in range(weights_matrix.shape[1]):
            values = weights_matrix[:, i]
            # Sort and trim k smallest and k largest
            sorted_values = np.sort(values)
            trimmed_values = sorted_values[k:n-k]
            trimmed_result[i] = np.mean(trimmed_values)

        return trimmed_result.reshape(updates[0].weights.shape).astype(updates[0].weights.dtype)

    def _trim_dict(self, updates: List[ModelUpdate]) -> dict:
        """Apply trimmed mean to dict weights."""
        result = {}
        all_keys = set()
        for update in updates:
            if isinstance(update.weights, dict):
                all_keys.update(update.weights.keys())

        for key in all_keys:
            values = []
            for update in updates:
                if isinstance(update.weights, dict) and key in update.weights:
                    val = update.weights[key]
                    if hasattr(val, '__array__'):
                        values.append(val.flatten())

            if values:
                weights_matrix = np.stack(values)
                n = len(values)
                k = int(n * self.trim_ratio)

                if k == 0:
                    trimmed = np.mean(weights_matrix, axis=0)
                else:
                    trimmed_result = np.zeros_like(weights_matrix[0])
                    for i in range(weights_matrix.shape[1]):
                        sorted_values = np.sort(weights_matrix[:, i])
                        trimmed_values = sorted_values[k:n-k]
                        trimmed_result[i] = np.mean(trimmed_values)
                    trimmed = trimmed_result

                result[key] = trimmed.reshape(values[0].shape).astype(values[0].dtype)
            else:
                # Use first available value
                for update in updates:
                    if isinstance(update.weights, dict) and key in update.weights:
                        result[key] = update.weights[key]
                        break

        return result
