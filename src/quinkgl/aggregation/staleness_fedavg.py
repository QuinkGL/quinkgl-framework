"""
Staleness-Weighted Federated Averaging.

Implements staleness-aware aggregation following Xie et al. 2019 —
Asynchronous Federated Optimization. Older updates (with higher staleness)
receive less weight in the aggregation.

Reference:
    "Asynchronous Federated Optimization" (Xie et al., 2019)
"""

from typing import List
import numpy as np

from quinkgl.aggregation.base import ModelUpdate, AggregatedModel
from quinkgl.aggregation.fedavg import FedAvg


class StalenessWeightedFedAvg(FedAvg):
    """
    Staleness-Weighted Federated Averaging.

    Extends FedAvg by adjusting update weights based on staleness
    (how many rounds old the update is). Stale updates are
    down-weighted to prevent them from dominating the aggregation.

    Staleness weight: 1 / (1 + staleness_coefficient * staleness)
    where staleness = current_round - update.round_number

    AGG-TASK-10: Round-number synchronisation contract:
        - All ModelUpdate instances must have a valid round_number attribute
        - The caller should provide current_round parameter that represents
          the current training round at the time of aggregation
        - If current_round is 0, it will be inferred from the maximum
          round_number in the updates (useful for first round)
        - Staleness is computed as: max(0, current_round - update.round_number)
        - Updates from future rounds (round_number > current_round) are
          treated as current (staleness = 0) to avoid negative staleness

    Attributes:
        staleness_coefficient: Controls how aggressively stale updates
            are down-weighted. Higher values penalize staleness more.
            Default: 0.1 (gentle penalty)
    """

    def __init__(
        self,
        staleness_coefficient: float = 0.1,
        weight_by: str = "data_size",
        **kwargs,
    ):
        """
        Initialize StalenessWeightedFedAvg.

        Args:
            staleness_coefficient: Penalty factor for staleness (default: 0.1).
            weight_by: Base weighting strategy ("data_size", "uniform", "inverse_loss").
            **kwargs: Additional arguments passed to FedAvg.
        """
        super().__init__(weight_by=weight_by, **kwargs)
        self.staleness_coefficient = staleness_coefficient

    def compute_staleness_weight(self, update: ModelUpdate, current_round: int) -> float:
        """
        Compute staleness-adjusted weight for a model update.

        Args:
            update: The model update.
            current_round: The current training round.

        Returns:
            Staleness-adjusted weight value.
        """
        staleness = max(0, current_round - update.round_number)
        staleness_factor = 1.0 / (1.0 + self.staleness_coefficient * staleness)
        base_weight = super().compute_weight(update)
        return base_weight * staleness_factor

    async def aggregate(
        self,
        updates: List[ModelUpdate],
        current_round: int = 0
    ) -> AggregatedModel:
        """
        Aggregate with staleness weighting.

        AGG-TASK-10: Round-number synchronisation contract:
            - If current_round is 0, it is inferred from updates (max round_number)
            - Staleness is computed as max(0, current_round - update.round_number)
            - Future updates (round_number > current_round) treated as current.

        If current_round is not provided, uses the maximum round_number
        from the updates as the reference round.

        Args:
            updates: List of model updates from peers.
            current_round: Current training round for staleness calculation.

        Returns:
            AggregatedModel with staleness-weighted weights.
        """
        self._validate_updates(updates)

        if current_round == 0:
            current_round = max(u.round_number for u in updates) if updates else 0

        weights_list = [
            self.compute_staleness_weight(u, current_round)
            for u in updates
        ]
        total_staleness_weight = sum(weights_list)

        # AGG-TASK-13: Unified total weight zero behaviour - use uniform weight fallback
        if total_staleness_weight == 0:
            total_staleness_weight = len(updates)
        if total_staleness_weight == 0:
            # Fallback to 1 to avoid division by zero (shouldn't happen with valid updates)
            total_staleness_weight = 1

        first_weights = updates[0].weights

        if isinstance(first_weights, np.ndarray):
            aggregated = self._aggregate_numpy(updates, weights_list, total_staleness_weight)
        elif isinstance(first_weights, dict):
            aggregated = self._aggregate_dict(updates, weights_list, total_staleness_weight)
        else:
            aggregated = self._aggregate_generic(updates, weights_list, total_staleness_weight)

        staleness_info = [
            {
                "peer_id": u.peer_id,
                "round": u.round_number,
                "staleness": max(0, current_round - u.round_number),
            }
            for u in updates
        ]

        return AggregatedModel(
            weights=aggregated,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={
                "aggregation_method": "staleness_weighted_fedavg",
                "staleness_coefficient": self.staleness_coefficient,
                "weight_by": self.weight_by,
                "current_round": current_round,
                "staleness_info": staleness_info,
            },
            updates=updates,
        )
