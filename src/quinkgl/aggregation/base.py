"""
Base Aggregation Strategy

Abstract base class for all aggregation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """
    Represents a model update from a peer.

    Framework agnostic - works with any model format that can
    be serialized to/from numpy arrays or bytes.
    """
    peer_id: str
    weights: Any  # numpy array, dict, or bytes depending on framework
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for weighted aggregation
    sample_count: int = 1
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    round_number: int = 0

    # Timestamp (UTC-aware to avoid naive datetime issues)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AggregatedModel:
    """
    Result of aggregating multiple model updates.
    """
    weights: Any
    contributing_peers: List[str]
    total_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Store the original updates for reference and advanced strategies
    updates: List[ModelUpdate] = field(default_factory=list)


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.

    An aggregation strategy combines multiple model updates
    into a single aggregated model.
    """

    def __init__(self, **kwargs):
        """Initialize the aggregation strategy with configuration."""
        self.config = kwargs

    @abstractmethod
    async def aggregate(
        self,
        updates: List[ModelUpdate]
    ) -> AggregatedModel:
        """
        Aggregate multiple model updates into one.

        Args:
            updates: List of model updates from peers

        Returns:
            AggregatedModel containing the combined weights
        """

    def compute_weight(self, update: ModelUpdate) -> float:
        """
        Compute the weight for a given model update.

        Default implementation weights by sample count.
        Override for custom weight strategies.

        Args:
            update: The model update to weight

        Returns:
            Float weight value (higher = more influence)
        """
        return float(update.sample_count)

    def _validate_updates(self, updates: List[ModelUpdate]) -> None:
        """
        Validate that updates can be aggregated.

        Args:
            updates: List of model updates to validate

        Raises:
            ValueError: If updates are invalid or incompatible
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates")

        # Reject duplicate peer_ids to prevent double-counting
        seen_peers: set = set()
        for update in updates:
            if update.peer_id in seen_peers:
                raise ValueError(
                    f"Duplicate peer_id in updates: {update.peer_id}"
                )
            seen_peers.add(update.peer_id)

        # Check for NaN/Inf in weights
        for update in updates:
            self._check_weights_valid(update)

        # Check for compatible shapes if using numpy arrays
        if len(updates) > 1:
            first_shape = self._get_shape(updates[0].weights)
            for update in updates[1:]:
                if self._get_shape(update.weights) != first_shape:
                    raise ValueError(
                        f"Incompatible weight shapes: "
                        f"{update.peer_id} has different shape"
                    )

    def _check_weights_valid(self, update: ModelUpdate) -> None:
        """
        Check if weights contain NaN or Inf values.

        Args:
            update: The model update to check

        Raises:
            ValueError: If weights contain NaN or Inf
        """
        weights = update.weights

        if isinstance(weights, np.ndarray):
            if np.isnan(weights).any():
                raise ValueError(f"Weights from {update.peer_id} contain NaN values")
            if np.isinf(weights).any():
                raise ValueError(f"Weights from {update.peer_id} contain Inf values")
        elif isinstance(weights, dict):
            for key, value in weights.items():
                if isinstance(value, np.ndarray):
                    if np.isnan(value).any():
                        raise ValueError(f"Weights[{key}] from {update.peer_id} contain NaN")
                    if np.isinf(value).any():
                        raise ValueError(f"Weights[{key}] from {update.peer_id} contain Inf")
                elif isinstance(value, dict):
                    self._check_weights_valid(
                        ModelUpdate(
                            peer_id=update.peer_id,
                            weights=value,
                            metadata={},
                        )
                    )

    def _get_shape(self, weights: Any) -> tuple:
        """
        Get the shape of weights if possible.

        Recursively walks nested dicts and lists to build a deterministic
        shape signature for cross-update compatibility checks.

        Args:
            weights: Weights object (numpy array, dict, etc.)

        Returns:
            Shape tuple or empty tuple if shape cannot be determined
        """
        if isinstance(weights, np.ndarray):
            return weights.shape
        elif isinstance(weights, dict):
            # Include per-value shapes so that same-key but different-shaped
            # tensors are caught before aggregation raises a broadcast error.
            items = []
            for key in sorted(weights.keys()):
                items.append((key, self._get_shape(weights[key])))
            return tuple(items)
        elif isinstance(weights, (list, tuple)):
            return tuple(self._get_shape(item) for item in weights)
        return ()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a serializable snapshot of strategy state.

        Subclasses should override this to persist learnable / mutable
        state (e.g. momentum buffers, control variates).
        """
        return {"config": dict(self.config)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state from a snapshot.

        Subclasses should override this alongside ``state_dict()``.
        """
        self.config = dict(state.get("config", {}))

    def _clip_value(self, value: float, min_val: float = 0.1, max_val: float = 10.0) -> float:
        """
        Clip a value to a specified range.

        Useful for preventing extreme weights in aggregation.

        Args:
            value: Value to clip
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clipped value
        """
        return max(min_val, min(max_val, value))
