"""
Trimmed Mean aggregation strategy.
"""

import logging
from typing import List, Dict, Any

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
            logging.error(f"TrimmedMean requires at least 3 updates, got {len(updates)}")
            raise ValueError("TrimmedMean requires at least 3 updates")

        self._get_trim_count(len(updates))

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

    def _get_trim_count(self, n: int) -> int:
        k = int(n * self.trim_ratio)
        if self.trim_ratio > 0 and k == 0:
            logging.warning(
                f"trim_ratio={self.trim_ratio} is too small for n={n}; no values will be trimmed, falling back to mean"
            )
        return k

    def _trim_numpy(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Apply trimmed mean to numpy arrays."""
        weights_matrix = np.stack([u.weights.flatten() for u in updates])
        n = len(updates)
        k = self._get_trim_count(n)

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
                k = self._get_trim_count(n)

                if k == 0:
                    logging.warning("k=0 in TrimmedMean, falling back to simple mean")
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

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable state for restart persistence (AGG-TASK-14)."""
        return {"config": dict(self.config), "trim_ratio": self.trim_ratio}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore mutable state from a snapshot (AGG-TASK-14)."""
        self.config = dict(state.get("config", {}))
        self.trim_ratio = float(state.get("trim_ratio", self.trim_ratio))
