"""
Entropy-Weighted Aggregation Strategy (RNEP-inspired)

Weights peer model contributions by the Shannon entropy of their local
data distribution.  Peers with higher entropy (more diverse, IID-like data)
have more influence on the aggregated model, while peers with lower entropy
(skewed, non-IID data) contribute less.

Reference:
    Kang, J.-I.; Lee, S.-W. "RNEP: Random Node Entropy Pairing for
    Efficient Decentralized Training with Non-IID Local Data."
    Electronics 2024, 13, 4193.
    https://doi.org/10.3390/electronics13214193
"""

from typing import List, Dict, Any
import logging
import numpy as np

from quinkgl.aggregation.base import (
    AggregationStrategy,
    ModelUpdate,
    AggregatedModel,
)

logger = logging.getLogger(__name__)


def _shannon_entropy(distribution: dict) -> float:
    """Compute Shannon entropy from a label distribution dict.

    Args:
        distribution: Mapping of class label to sample count or proportion.

    Returns:
        Shannon entropy in nats (natural log).  Returns 0.0 for empty or
        degenerate distributions.
    """
    values = np.array(list(distribution.values()), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


class EntropyWeightedAvg(AggregationStrategy):
    """RNEP-style entropy-weighted federated averaging.

    Each peer's contribution to the aggregated model is proportional to the
    Shannon entropy of its local label distribution.  The entropy value is
    read from ``update.metadata["label_distribution"]`` — a dict mapping
    class labels to sample counts or proportions.

    If a peer does not provide a label distribution the strategy falls back
    to a configurable default weight so the update is not silently dropped.

    Parameters
    ----------
    fallback_weight : float
        Weight assigned to peers that do not provide label distribution
        metadata.  Defaults to ``1.0``.
    entropy_floor : float
        Minimum entropy value to prevent a peer from being completely
        ignored.  Defaults to ``0.01``.
    normalize : bool
        When *True* (default) the entropy values are normalized so that
        they sum to 1 — exactly matching the RNEP paper's
        ``NormalizeH`` procedure.  When *False* the raw entropy values
        are used as weights (still normalized by their sum for the
        weighted average, but not clipped to [0, 1] per-peer).
    """

    def __init__(
        self,
        fallback_weight: float = 1.0,
        entropy_floor: float = 0.01,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fallback_weight = fallback_weight
        self.entropy_floor = entropy_floor
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_entropy_weight(self, update: ModelUpdate) -> float:
        """Return the entropy-based weight for *update*.

        The label distribution is expected at
        ``update.metadata["label_distribution"]``.
        """
        dist = update.metadata.get("label_distribution")
        if dist is None or not isinstance(dist, dict) or len(dist) == 0:
            return self.fallback_weight

        entropy = _shannon_entropy(dist)
        return max(self.entropy_floor, entropy)

    def compute_weight(self, update: ModelUpdate) -> float:
        """Public override used by the base-class validation helpers."""
        return self._compute_entropy_weight(update)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    async def aggregate(self, updates: List[ModelUpdate]) -> AggregatedModel:
        """Aggregate model updates weighted by Shannon entropy.

        Follows the RNEP algorithm:
        1. Compute entropy for each peer's local data distribution.
        2. Normalize entropy values (sum = 1).
        3. Weighted-sum the model parameters.
        """
        self._validate_updates(updates)

        # Step 1 — compute raw entropy weights
        raw_weights = [self._compute_entropy_weight(u) for u in updates]

        # Step 2 — normalize (RNEP NormalizeH)
        total_w = sum(raw_weights)
        if total_w == 0:
            raise ValueError("Total entropy weight is zero, cannot aggregate")
        norm_weights = [w / total_w for w in raw_weights]

        # Log for observability
        for update, rw, nw in zip(updates, raw_weights, norm_weights):
            logger.debug(
                "EntropyWeightedAvg: peer=%s entropy=%.4f norm_weight=%.4f",
                update.peer_id,
                rw,
                nw,
            )

        # Step 3 — weighted sum of model parameters
        first_weights = updates[0].weights

        if isinstance(first_weights, np.ndarray):
            aggregated = self._agg_numpy(updates, norm_weights)
        elif isinstance(first_weights, dict):
            aggregated = self._agg_dict(updates, norm_weights)
        else:
            aggregated = self._agg_numpy_fallback(updates, norm_weights)

        return AggregatedModel(
            weights=aggregated,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={
                "aggregation_method": "entropy_weighted_avg",
                "entropy_weights": {
                    u.peer_id: round(nw, 6)
                    for u, nw in zip(updates, norm_weights)
                },
            },
            updates=updates,
        )

    # ------------------------------------------------------------------
    # Internal helpers (mirror FedAvg structure)
    # ------------------------------------------------------------------

    def _agg_numpy(
        self, updates: List[ModelUpdate], norm_weights: List[float]
    ) -> np.ndarray:
        result = np.zeros_like(updates[0].weights, dtype=np.float64)
        for update, w in zip(updates, norm_weights):
            result += update.weights.astype(np.float64) * w
        return result.astype(updates[0].weights.dtype)

    def _agg_dict(
        self, updates: List[ModelUpdate], norm_weights: List[float]
    ) -> dict:
        result = {}
        all_keys: set = set()
        for u in updates:
            if isinstance(u.weights, dict):
                all_keys.update(u.weights.keys())

        for key in all_keys:
            values, weights = [], []
            for u, w in zip(updates, norm_weights):
                if isinstance(u.weights, dict) and key in u.weights:
                    values.append(u.weights[key])
                    weights.append(w)

            if not values:
                continue

            first_val = values[0]
            if hasattr(first_val, "__array__"):
                key_total = sum(weights)
                arr = np.zeros_like(first_val, dtype=np.float64)
                for v, w in zip(values, weights):
                    arr += v.astype(np.float64) * (w / key_total)
                result[key] = arr.astype(first_val.dtype)
            else:
                result[key] = first_val

        return result

    def _agg_numpy_fallback(
        self, updates: List[ModelUpdate], norm_weights: List[float]
    ) -> np.ndarray:
        arrays = [np.array(u.weights, dtype=np.float64) for u in updates]
        result = np.zeros_like(arrays[0])
        for arr, w in zip(arrays, norm_weights):
            result += arr * w
        return result

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable state for restart persistence (AGG-TASK-14)."""
        return {"config": dict(self.config)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore mutable state from a snapshot (AGG-TASK-14)."""
        self.config = dict(state.get("config", {}))
