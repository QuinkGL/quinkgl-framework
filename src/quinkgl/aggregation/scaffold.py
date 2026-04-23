"""
SCAFFOLD Aggregation Strategy (Gossip Variant).

Implements SCAFFOLD — Stochastic Controlled Averaging for Federated
Learning (Karimireddy et al., NeurIPS 2020), adapted for decentralized
gossip topology.

In the centralized formulation, each client maintains a *control variate*
that estimates its local gradient drift from the global gradient.  The
server aggregates both model updates and control-variate updates.

In our **gossip variant** (GT-SCAFFOLD-style), there is no central server.
Instead:
  - Each node maintains a local control variate ``c_i``.
  - The "global" control variate is approximated as the running average of
    control variates received from peers (``c_avg``).
  - During aggregation, we correct the averaged model by subtracting the
    estimated drift: ``Δ_corrected = Δ_peer − η(c_peer − c_avg)``.

References:
    Karimireddy et al. 2020 — "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning" (NeurIPS 2020)
    Li et al. 2022 — "GT-SCAFFOLD" (gossip topology variant)

Usage:
    from quinkgl.aggregation.scaffold import Scaffold

    strategy = Scaffold(learning_rate=0.01)
    result = await strategy.aggregate(updates)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from quinkgl.aggregation.base import (
    AggregatedModel,
    AggregationStrategy,
    ModelUpdate,
)

logger = logging.getLogger(__name__)

__all__ = ["Scaffold"]


class Scaffold(AggregationStrategy):
    """SCAFFOLD with control-variate drift correction for gossip FL.

    Each participating node is expected to attach its local control
    variate in ``update.metadata["control_variate"]`` (a dict of numpy
    arrays with the same keys as the model weights).  If a peer omits
    the control variate, its contribution is treated as vanilla FedAvg
    (zero correction).

    Parameters
    ----------
    learning_rate : float
        The local learning rate ``η`` used during training.  This is
        needed to scale the control-variate correction term.
    global_learning_rate : float
        Server-side (aggregation-side) learning rate for applying the
        corrected update.  Defaults to 1.0 (no scaling).
    control_momentum : float
        EMA factor for updating the running average of global control
        variates.  0.0 = full replacement, 0.9 = slow adaptation.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        global_learning_rate: float = 1.0,
        control_momentum: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.global_learning_rate = global_learning_rate
        self.control_momentum = control_momentum

        # Running estimate of the "global" control variate
        self._c_global: Optional[Dict[str, np.ndarray]] = None
        self._round: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def aggregate(self, updates: List[ModelUpdate]) -> AggregatedModel:
        """Aggregate model updates with SCAFFOLD drift correction.

        Steps:
          1. Separate model weights and control variates from each update.
          2. Compute the corrected weight for each peer.
          3. Average the corrected weights.
          4. Update the running estimate of the global control variate.

        AGG-TASK-09: Correction uses snapshotted global CV to ensure
        consistency across the aggregation round.
        """
        self._validate_updates(updates)
        self._round += 1

        # --- 1. Extract control variates ---
        peer_controls: List[Optional[Dict[str, np.ndarray]]] = []
        for u in updates:
            cv = u.metadata.get("control_variate")
            peer_controls.append(cv)

        # --- 2. Compute aggregated control variate from peers ---
        c_avg = self._average_control_variates(peer_controls, updates)

        # --- 3. Correct and aggregate model weights ---
        # AGG-TASK-09: Snapshot the global CV before correction to ensure consistency
        c_snapshot = deepcopy(self._c_global) if self._c_global is not None else None

        first_weights = updates[0].weights
        if isinstance(first_weights, dict):
            aggregated = self._aggregate_dict(updates, peer_controls, c_snapshot)
        elif isinstance(first_weights, np.ndarray):
            aggregated = self._aggregate_numpy(updates, peer_controls, c_snapshot)
        else:
            # Fallback: plain FedAvg without correction
            aggregated = self._plain_average(updates)

        # --- 4. Update running global control variate ---
        if c_avg is not None:
            self._update_global_control(c_avg)

        return AggregatedModel(
            weights=aggregated,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            metadata={
                "aggregation_method": "scaffold",
                "round": self._round,
                "learning_rate": self.learning_rate,
                "has_control_variates": any(c is not None for c in peer_controls),
            },
            updates=updates,
        )

    def get_local_control_variate(
        self,
        local_weights: Any,
        global_weights: Any,
        local_gradient: Optional[Any] = None,
        num_local_steps: int = 1,
        local_control_variate: Optional[Any] = None,
    ) -> Any:
        """Compute the updated local control variate after local training.

        This should be called by the training loop after local SGD steps
        and attached to the ModelUpdate metadata.

        SCAFFOLD Option II (recommended for communication efficiency):
            c_i^{new} = c_i − c + (w_global − w_local) / (K × η)

        Where:
            c_i = current local control variate
            c   = global control variate estimate
            K   = number of local steps
            η   = learning rate

        Note on "__single__" magic key:
            For numpy array weights (non-dict), control variates are stored
            using the special key "__single__" in dict metadata. This allows
            uniform handling of both dict and numpy weight formats. When
            local_weights is a numpy array, the returned control variate
            will also be a numpy array, but when stored in ModelUpdate.metadata
            it should use the "__single__" key for consistency with the
            aggregation logic.

        Parameters
        ----------
        local_weights : dict or np.ndarray
            Weights after local training.
        global_weights : dict or np.ndarray
            Weights at the start of the round (before local training).
        local_gradient : optional
            If provided, used directly as the local gradient estimate
            (Option I).
        num_local_steps : int
            Number of local SGD steps taken (K).

        Returns
        -------
        control_variate : same type as weights
            Updated local control variate to attach to ModelUpdate.
        """
        c_global = self._c_global

        if isinstance(local_weights, dict):
            cv = {}
            for key in local_weights:
                lw = local_weights[key].astype(np.float64)
                gw = global_weights[key].astype(np.float64) if key in global_weights else lw
                diff = (gw - lw) / (num_local_steps * self.learning_rate)

                if local_control_variate is not None and key in local_control_variate:
                    c_local = local_control_variate[key].astype(np.float64)
                else:
                    c_local = np.zeros_like(diff, dtype=np.float64)

                if c_global is not None and key in c_global:
                    c_ref = c_global[key].astype(np.float64)
                else:
                    c_ref = np.zeros_like(diff, dtype=np.float64)

                cv[key] = (c_local - c_ref + diff).astype(local_weights[key].dtype)
            return cv
        elif isinstance(local_weights, np.ndarray):
            lw = local_weights.astype(np.float64)
            gw = global_weights.astype(np.float64)
            diff = (gw - lw) / (num_local_steps * self.learning_rate)

            if local_control_variate is not None:
                if isinstance(local_control_variate, dict):
                    c_local = np.asarray(local_control_variate.get("__single__", 0.0), dtype=np.float64)
                else:
                    c_local = np.asarray(local_control_variate, dtype=np.float64)
            else:
                c_local = np.zeros_like(diff, dtype=np.float64)

            if c_global is not None:
                if isinstance(c_global, dict):
                    c_ref = np.asarray(c_global.get("__single__", 0.0), dtype=np.float64)
                else:
                    c_ref = np.asarray(c_global, dtype=np.float64)
            else:
                c_ref = np.zeros_like(diff, dtype=np.float64)

            return (c_local - c_ref + diff).astype(local_weights.dtype)
        return None

    @property
    def global_control_variate(self) -> Optional[Dict[str, np.ndarray]]:
        """Current estimate of the global control variate."""
        return self._c_global

    @property
    def round_number(self) -> int:
        return self._round

    def state_dict(self) -> Dict[str, Any]:
        """Serialize mutable state for restart persistence."""
        state: Dict[str, Any] = {
            "config": dict(self.config),
            "round": self._round,
        }
        if self._c_global is not None:
            state["c_global"] = {
                k: v.tolist() for k, v in self._c_global.items()
            }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore mutable state from a snapshot."""
        self.config = dict(state.get("config", {}))
        self._round = int(state.get("round", 0))
        c_global_raw = state.get("c_global")
        if c_global_raw is not None:
            self._c_global = {
                k: np.array(v, dtype=np.float64)
                for k, v in c_global_raw.items()
            }
        else:
            self._c_global = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _average_control_variates(
        self,
        peer_controls: List[Optional[Dict[str, np.ndarray]]],
        updates: List[ModelUpdate],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compute the sample-weighted average of peer control variates."""
        valid = [(cv, u) for cv, u in zip(peer_controls, updates) if cv is not None]
        if not valid:
            return self._c_global  # fallback to current estimate

        total_samples = sum(u.sample_count for _, u in valid)
        if total_samples == 0:
            total_samples = len(valid)
        if total_samples == 0:
            logger.warning("No samples and no valid peers for control variate averaging, using uniform weight")
            total_samples = 1

        result: Dict[str, np.ndarray] = {}
        for cv, u in valid:
            w = float(u.sample_count) / total_samples if total_samples > 0 else 1.0 / len(valid) if len(valid) > 0 else 1.0
            for key, val in cv.items():
                if key not in result:
                    result[key] = np.zeros_like(val, dtype=np.float64)
                result[key] += val.astype(np.float64) * w

        return result

    def _update_global_control(self, c_avg: Dict[str, np.ndarray]) -> None:
        beta = self.control_momentum
        if self._c_global is None or beta == 0.0:
            self._c_global = deepcopy(c_avg)
        else:
            for key in c_avg:
                if key in self._c_global:
                    self._c_global[key] = (
                        beta * self._c_global[key].astype(np.float64)
                        + (1 - beta) * c_avg[key].astype(np.float64)
                    )
                else:
                    self._c_global[key] = deepcopy(c_avg[key])

    def _aggregate_dict(
        self,
        updates: List[ModelUpdate],
        peer_controls: List[Optional[Dict[str, np.ndarray]]],
        c_avg: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate dict weights with SCAFFOLD correction."""
        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            total_samples = len(updates)
        if total_samples == 0:
            logger.warning("No samples and no updates for dict aggregation, using uniform weight")
            total_samples = 1

        all_keys: set = set()
        for u in updates:
            if isinstance(u.weights, dict):
                all_keys.update(u.weights.keys())

        result: Dict[str, np.ndarray] = {}
        for key in all_keys:
            # Find first non-None value for shape reference
            ref = None
            for u in updates:
                if isinstance(u.weights, dict) and key in u.weights:
                    ref = u.weights[key]
                    break
            if ref is None:
                continue

            acc = np.zeros_like(ref, dtype=np.float64)
            for u, cv in zip(updates, peer_controls):
                if not isinstance(u.weights, dict) or key not in u.weights:
                    continue
                w = float(u.sample_count) / total_samples
                peer_w = u.weights[key].astype(np.float64)

                # SCAFFOLD correction: subtract drift
                if cv is not None and key in cv and c_avg is not None and key in c_avg:
                    correction = self.learning_rate * (
                        cv[key].astype(np.float64) - c_avg[key].astype(np.float64)
                    )
                    peer_w = peer_w - correction

                acc += peer_w * w

            result[key] = acc.astype(ref.dtype)

        return result

    def _aggregate_numpy(
        self,
        updates: List[ModelUpdate],
        peer_controls: List[Optional[Dict[str, np.ndarray]]],
        c_avg: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Aggregate numpy weights with SCAFFOLD correction."""
        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            total_samples = len(updates)
        if total_samples == 0:
            logger.warning("No samples and no updates for numpy aggregation, using uniform weight")
            total_samples = 1

        acc = np.zeros_like(updates[0].weights, dtype=np.float64)
        for u, cv in zip(updates, peer_controls):
            w = float(u.sample_count) / total_samples
            peer_w = u.weights.astype(np.float64)

            # For numpy case, control variate is stored with key "__single__"
            if cv is not None:
                c_peer = cv.get("__single__", None) if isinstance(cv, dict) else cv
                c_g = c_avg.get("__single__", None) if isinstance(c_avg, dict) and c_avg else c_avg
                if c_peer is not None and c_g is not None:
                    correction = self.learning_rate * (
                        np.asarray(c_peer, dtype=np.float64) - np.asarray(c_g, dtype=np.float64)
                    )
                    peer_w = peer_w - correction

            acc += peer_w * w

        return acc.astype(updates[0].weights.dtype)

    def _plain_average(self, updates: List[ModelUpdate]) -> Any:
        """Fallback to plain weighted average (no correction)."""
        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            total_samples = len(updates)
        if total_samples == 0:
            logger.warning("No samples and no updates for plain average, using uniform weight")
            total_samples = 1

        if isinstance(updates[0].weights, np.ndarray):
            acc = np.zeros_like(updates[0].weights, dtype=np.float64)
            for u in updates:
                acc += u.weights.astype(np.float64) * (float(u.sample_count) / total_samples)
            return acc.astype(updates[0].weights.dtype)
        elif isinstance(updates[0].weights, dict):
            result = {}
            for key in updates[0].weights:
                acc = np.zeros_like(updates[0].weights[key], dtype=np.float64)
                for u in updates:
                    if isinstance(u.weights, dict) and key in u.weights:
                        acc += u.weights[key].astype(np.float64) * (
                            float(u.sample_count) / total_samples
                        )
                result[key] = acc.astype(updates[0].weights[key].dtype)
            return result
        return updates[0].weights
