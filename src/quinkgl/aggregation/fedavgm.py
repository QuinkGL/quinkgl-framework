"""
FedAvgM aggregation strategy.
"""

from copy import deepcopy
from typing import Any, Dict, List

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
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.momentum_buffer = None
        self.global_weights = None

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate with server momentum.

        Momentum buffer is updated as:
            buffer = momentum * buffer + averaged_update
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
        if self.global_weights is None:
            self.global_weights = deepcopy(averaged)
            self.momentum_buffer = self._zeros_like(averaged)
        else:
            delta = self._compute_delta(self.global_weights, averaged)
            self.momentum_buffer = self._apply_momentum(
                self.momentum_buffer, delta
            )
            self.global_weights = self._apply_server_update(
                self.global_weights,
                self.momentum_buffer,
            )

        return AggregatedModel(
            weights=deepcopy(self.global_weights),
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={"aggregation_method": "fedavgm", "momentum": self.server_momentum},
            updates=updates
        )

    def _zeros_like(self, weights):
        if isinstance(weights, np.ndarray):
            return np.zeros_like(weights)
        elif isinstance(weights, dict):
            result = {}
            for key, value in weights.items():
                if hasattr(value, '__array__'):
                    result[key] = np.zeros_like(value)
                else:
                    result[key] = value
            return result
        else:
            return 0.0

    def _compute_delta(self, previous_global, averaged):
        if isinstance(previous_global, np.ndarray):
            return (previous_global.astype(np.float64) - averaged.astype(np.float64)).astype(previous_global.dtype)
        elif isinstance(previous_global, dict):
            result = {}
            for key, value in previous_global.items():
                if key in averaged and hasattr(value, '__array__'):
                    result[key] = (
                        value.astype(np.float64) - averaged[key].astype(np.float64)
                    ).astype(value.dtype)
                else:
                    result[key] = value
            return result
        else:
            return previous_global

    def _apply_momentum(self, buffer, delta):
        """Apply momentum to the update."""
        if isinstance(buffer, np.ndarray):
            return (
                self.server_momentum * buffer + delta
            ).astype(buffer.dtype)
        elif isinstance(buffer, dict):
            result = {}
            for key in buffer:
                if key in delta and hasattr(buffer[key], '__array__'):
                    result[key] = (
                        self.server_momentum * buffer[key] +
                        delta[key]
                    ).astype(buffer[key].dtype)
                else:
                    result[key] = buffer[key]
            return result
        else:
            return buffer

    def _apply_server_update(self, previous_global, momentum_buffer):
        if isinstance(previous_global, np.ndarray):
            return (
                previous_global.astype(np.float64) - momentum_buffer.astype(np.float64)
            ).astype(previous_global.dtype)
        elif isinstance(previous_global, dict):
            result = {}
            for key, value in previous_global.items():
                if key in momentum_buffer and hasattr(value, '__array__'):
                    result[key] = (
                        value.astype(np.float64) - momentum_buffer[key].astype(np.float64)
                    ).astype(value.dtype)
                else:
                    result[key] = value
            return result
        else:
            return previous_global

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable state for restart persistence."""
        state: Dict[str, Any] = {
            "config": dict(self.config),
            "server_momentum": self.server_momentum,
        }
        if self.global_weights is not None:
            if isinstance(self.global_weights, np.ndarray):
                state["global_weights"] = self.global_weights.tolist()
            elif isinstance(self.global_weights, dict):
                state["global_weights"] = {
                    k: v.tolist() for k, v in self.global_weights.items()
                }
        if self.momentum_buffer is not None:
            if isinstance(self.momentum_buffer, np.ndarray):
                state["momentum_buffer"] = self.momentum_buffer.tolist()
            elif isinstance(self.momentum_buffer, dict):
                state["momentum_buffer"] = {
                    k: v.tolist() for k, v in self.momentum_buffer.items()
                }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore mutable state from a snapshot."""
        self.config = dict(state.get("config", {}))
        self.server_momentum = float(state.get("server_momentum", self.server_momentum))
        gw_raw = state.get("global_weights")
        if gw_raw is not None:
            if isinstance(gw_raw, dict):
                self.global_weights = {
                    k: np.array(v, dtype=np.float64)
                    for k, v in gw_raw.items()
                }
            else:
                self.global_weights = np.array(gw_raw, dtype=np.float64)
        else:
            self.global_weights = None
        mb_raw = state.get("momentum_buffer")
        if mb_raw is not None:
            if isinstance(mb_raw, dict):
                self.momentum_buffer = {
                    k: np.array(v, dtype=np.float64)
                    for k, v in mb_raw.items()
                }
            else:
                self.momentum_buffer = np.array(mb_raw, dtype=np.float64)
        else:
            self.momentum_buffer = None

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
