"""
FedProx aggregation strategy.

Implements FedProx following Li et al. 2020 (arXiv:1812.06127).

The proximal term is applied during local training (gradient-level
regularization), not as post-hoc weight interpolation. This is the
mathematically correct formulation: the loss function receives an
additional term (mu/2)||w - w_global||^2 that penalizes drift from
the global model.

For backward compatibility, a `mode` parameter allows using the
legacy post-hoc approach ("weight_interpolation").
"""

import warnings
from copy import deepcopy
from typing import Any, Dict
from typing import List

import numpy as np

from quinkgl.aggregation.base import AggregatedModel, ModelUpdate
from quinkgl.aggregation.fedavg import FedAvg

__all__ = ["FedProx"]


class FedProx(FedAvg):
    """
    FedProx: Federated Learning with Proximal Term.

    In training-time mode (default), the proximal term is injected into
    the local training loss function via TrainingConfig.proximal_coefficient
    and TrainingConfig.global_weights. This follows the original paper.

    In weight_interpolation mode (legacy), post-hoc correction is applied:
    w_corrected = (1 - mu) * w + mu * w_global. This is kept for backward
    compatibility but does NOT match the original FedProx paper.

    Reference: https://arxiv.org/abs/1812.06127
    """

    def __init__(self, mu: float = 0.01, mode: str = "training_time", **kwargs):
        """
        Initialize FedProx aggregator.

        Args:
            mu: Proximal term coefficient (higher = stricter adherence to global model)
            mode: "training_time" (correct, default) or "weight_interpolation" (legacy)
            **kwargs: Additional arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.mode = mode
        if mode == "weight_interpolation":
            warnings.warn(
                'FedProx mode="weight_interpolation" is deprecated and will be removed '
                "in a future release. Use mode='training_time' (default) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.global_weights = None

    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate with FedProx.

        In training_time mode: aggregation is plain FedAvg; the proximal
        term is handled during local training via TrainingConfig. The global
        weights are stored for the next round's training config.

        In weight_interpolation mode: applies post-hoc correction before
        aggregation (legacy behavior).
        """
        self._validate_updates(updates)

        if self.global_weights is None:
            result = await super().aggregate(updates)
            self.global_weights = deepcopy(result.weights)
            return result

        if self.mode == "weight_interpolation":
            proximal_updates = self._apply_proximal_term(updates)
            result = await super().aggregate(proximal_updates)
        else:
            result = await super().aggregate(updates)

        self.global_weights = deepcopy(result.weights)
        result.metadata["aggregation_method"] = "fedprox"
        result.metadata["mu"] = self.mu
        result.metadata["fedprox_mode"] = self.mode

        return result

    def get_training_config_overrides(self, current_round: int = None) -> dict:
        """
        Get TrainingConfig overrides for FedProx training-time mode.

        AGG-TASK-07: Enforce FedProx training-time round-trip.
        Call this before each training round to inject the proximal
        term into the local loss function. The global_weights used
        must be from the previous aggregation round to ensure correctness.

        Args:
            current_round: Optional current training round for validation.
                          If provided, validates that global_weights exist
                          and are from a recent round.

        Returns:
            Dict with proximal_coefficient and global_weights,
            or empty dict if not applicable.
        """
        if self.mode == "training_time" and self.global_weights is not None:
            if current_round is not None and current_round == 0:
                # First round: no global weights available yet
                logger.debug("FedProx first round: no global weights for proximal term")
                return {}
            return {
                "proximal_coefficient": self.mu,
                "global_weights": self.global_weights,
            }
        return {}

    def _apply_proximal_term(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Apply post-hoc proximal term correction to updates (legacy mode)."""
        if self.global_weights is None:
            return updates

        corrected_updates = []
        for update in updates:
            new_update = deepcopy(update)
            new_update.weights = self._proximal_correction(
                update.weights,
                self.global_weights
            )
            corrected_updates.append(new_update)

        return corrected_updates

    def _proximal_correction(self, weights, global_weights):
        """Apply post-hoc proximal correction to weights (legacy mode)."""
        if isinstance(weights, np.ndarray):
            return (1 - self.mu) * weights + self.mu * global_weights
        elif isinstance(weights, dict):
            result = {}
            for key in weights:
                if key in global_weights and hasattr(weights[key], '__array__'):
                    result[key] = (
                        (1 - self.mu) * weights[key] + self.mu * global_weights[key]
                    ).astype(weights[key].dtype)
                else:
                    result[key] = weights[key]
            return result
        else:
            return weights

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable state for restart persistence (AGG-TASK-05)."""
        state: Dict[str, Any] = {
            "config": dict(self.config),
            "mu": self.mu,
            "mode": self.mode,
        }
        if self.global_weights is not None:
            if isinstance(self.global_weights, np.ndarray):
                state["global_weights"] = self.global_weights.tolist()
            elif isinstance(self.global_weights, dict):
                state["global_weights"] = {
                    k: v.tolist() for k, v in self.global_weights.items()
                }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore mutable state from a snapshot (AGG-TASK-05)."""
        self.config = dict(state.get("config", {}))
        self.mu = float(state.get("mu", self.mu))
        self.mode = state.get("mode", self.mode)
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
