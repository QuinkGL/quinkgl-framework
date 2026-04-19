"""
FedAvgM aggregation strategy.
"""

from copy import deepcopy
from typing import List

import numpy as np

from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy, ModelUpdate
from quinkgl.aggregation.fedavg import FedAvg

__all__ = ["FedAvgM"]


class FedAvgM(AggregationStrategy):
    """
    FedAvgM: Federated Averaging with Momentum.

    Uses server momentum to stabilize training and improve convergence.

    Reference: https://arxiv.org/abs/1909.03083
    """

    def __init__(self, server_momentum: float = 0.9, **kwargs):
        """
        Initialize FedAvgM aggregator.

        Args:
            server_momentum: Momentum coefficient (0-1, default 0.9)
            **kwargs: Additional configuration parameters
        """
        if not 0.0 <= server_momentum < 1.0:
            raise ValueError(
                f"server_momentum must be in [0, 1) — got {server_momentum}"
            )
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.momentum_buffer = None

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate with server momentum.

        Momentum buffer is updated as:
            buffer = momentum * buffer + (1 - momentum) * averaged_update
        """
        self._validate_updates(updates)

        # First, compute simple average of updates
        weights_list = [self.compute_weight(u) for u in updates]
        total_weight = sum(weights_list)

        if total_weight == 0:
            raise ValueError("Total weight is zero, cannot aggregate")

        first_weights = updates[0].weights

        if isinstance(first_weights, np.ndarray):
            averaged = self._average_numpy(updates, weights_list, total_weight)
        elif isinstance(first_weights, dict):
            averaged = self._average_dict(updates, weights_list, total_weight)
        else:
            # Fallback to FedAvg
            fedavg = FedAvg()
            return await fedavg.aggregate(updates)

        # Apply momentum
        if self.momentum_buffer is None:
            # First round: no momentum
            self.momentum_buffer = deepcopy(averaged)
        else:
            # Apply momentum: buffer = momentum * buffer + (1-momentum) * averaged
            self.momentum_buffer = self._apply_momentum(
                self.momentum_buffer, averaged
            )

        return AggregatedModel(
            weights=deepcopy(self.momentum_buffer),
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={"aggregation_method": "fedavgm", "momentum": self.server_momentum},
            updates=updates
        )

    def _apply_momentum(self, buffer, new_weights):
        """Apply momentum to the update."""
        if isinstance(buffer, np.ndarray):
            return (
                self.server_momentum * buffer + (1 - self.server_momentum) * new_weights
            ).astype(buffer.dtype)
        elif isinstance(buffer, dict):
            result = {}
            for key in buffer:
                if key in new_weights and hasattr(buffer[key], '__array__'):
                    result[key] = (
                        self.server_momentum * buffer[key] +
                        (1 - self.server_momentum) * new_weights[key]
                    ).astype(buffer[key].dtype)
                else:
                    result[key] = buffer[key]
            return result
        else:
            return buffer

    def _average_numpy(self, updates: List[ModelUpdate], weights_list: List[float], total_weight: float):
        """Average numpy array weights."""
        result = np.zeros_like(updates[0].weights, dtype=np.float64)
        for update, weight in zip(updates, weights_list):
            normalized_weight = weight / total_weight
            result += update.weights.astype(np.float64) * normalized_weight
        return result.astype(updates[0].weights.dtype)

    def _average_dict(self, updates: List[ModelUpdate], weights_list: List[float], total_weight: float):
        """Average dict weights."""
        result = {}
        all_keys = set()
        for update in updates:
            if isinstance(update.weights, dict):
                all_keys.update(update.weights.keys())

        for key in all_keys:
            values = []
            corresponding_weights = []
            for update, weight in zip(updates, weights_list):
                if isinstance(update.weights, dict) and key in update.weights:
                    values.append(update.weights[key])
                    corresponding_weights.append(weight)

            if values and hasattr(values[0], '__array__'):
                key_total = sum(corresponding_weights)
                averaged = np.zeros_like(values[0], dtype=np.float64)
                for val, w in zip(values, corresponding_weights):
                    averaged += val.astype(np.float64) * (w / key_total)
                result[key] = averaged.astype(values[0].dtype)
            elif values:
                result[key] = values[0]

        return result
