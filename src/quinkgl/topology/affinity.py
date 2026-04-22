"""
Affinity Topology — Like-Attracts-Like with cold-start resilience.

Peer selection driven by data affinity (fingerprint similarity) plus
historical success.  New nodes start with high exploration and gradually
shift to exploitation as they learn the network.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from quinkgl.topology.base import (
    TopologyStrategy,
    SelectionContext,
    PeerInfo,
    is_version_compatible,
)
from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    AffinityWeights,
)


@dataclass
class CollaborationEdge:
    peer_id: str
    affinity: float = 0.0
    weight: float = 0.5
    rounds_since_update: int = 0
    successful_rounds: int = 0
    total_rounds: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return self.successful_rounds / self.total_rounds

    def update_affinity(self, new_affinity: float, ema_alpha: float = 0.2):
        self.affinity = new_affinity
        self.weight = (1 - ema_alpha) * self.weight + ema_alpha * new_affinity
        self.rounds_since_update = 0
        self.last_updated = datetime.now()

    def decay(self, decay_factor: float = 0.95):
        self.rounds_since_update += 1
        self.weight *= decay_factor ** self.rounds_since_update

    def record_collaboration(self, success: bool):
        self.total_rounds += 1
        if success:
            self.successful_rounds += 1


class CollaborationHistory:
    def __init__(self, max_peers: int = 100):
        self._history: Dict[str, CollaborationEdge] = {}
        self._max_peers = max_peers

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence (TOP-07)."""
        return {
            "history": {
                pid: {
                    "weight": edge.weight,
                    "rounds_since_update": edge.rounds_since_update,
                    "total_rounds": edge.total_rounds,
                    "successful_rounds": edge.successful_rounds,
                }
                for pid, edge in self._history.items()
            },
            "max_peers": self._max_peers,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dict for persistence (TOP-07)."""
        history_dict = state_dict.get("history", {})
        self._history = {}
        for pid, edge_data in history_dict.items():
            edge = CollaborationEdge()
            edge.weight = edge_data.get("weight", 0.0)
            edge.rounds_since_update = edge_data.get("rounds_since_update", 0)
            edge.total_rounds = edge_data.get("total_rounds", 0)
            edge.successful_rounds = edge_data.get("successful_rounds", 0)
            self._history[pid] = edge
        self._max_peers = state_dict.get("max_peers", 100)

    def get_edge(self, peer_id: str) -> Optional[CollaborationEdge]:
        return self._history.get(peer_id)

    def get_or_create_edge(self, peer_id: str) -> CollaborationEdge:
        if peer_id not in self._history:
            if len(self._history) >= self._max_peers:
                self._evict_weakest()
            self._history[peer_id] = CollaborationEdge(peer_id=peer_id)
        return self._history[peer_id]

    def get_history_score(self, peer_id: str) -> float:
        edge = self._history.get(peer_id)
        if edge is None or edge.total_rounds < 3:
            return 0.0
        return edge.success_rate

    def update_peer(self, peer_id: str, new_affinity: float, ema_alpha: float = 0.2):
        edge = self.get_or_create_edge(peer_id)
        edge.update_affinity(new_affinity, ema_alpha)

    def record_collaboration_result(self, peer_id: str, success: bool):
        edge = self.get_or_create_edge(peer_id)
        edge.record_collaboration(success)

    def decay_all(self, decay_factor: float = 0.95):
        for edge in self._history.values():
            edge.decay(decay_factor)

    def evict_dead_edges(self, min_weight: float = 0.05):
        dead = [pid for pid, e in self._history.items() if e.weight < min_weight]
        for pid in dead:
            del self._history[pid]

    def _evict_weakest(self):
        if not self._history:
            return
        weakest = min(self._history.items(), key=lambda x: x[1].weight)
        del self._history[weakest[0]]

    def get_top_peers(self, count: int) -> List[Tuple[str, float]]:
        sorted_peers = sorted(
            self._history.items(), key=lambda x: x[1].weight, reverse=True
        )
        return [(pid, edge.weight) for pid, edge in sorted_peers[:count]]

    @property
    def edge_count(self) -> int:
        return len(self._history)


class AffinityTopology(TopologyStrategy):
    """Like-Attracts-Like topology with cold-start resilience.

    Cold-start timeline is governed by ``cold_start_rounds`` (default 3)
    from the manifest data_policy.  The three phases are derived from
    this value:

    - Blind:   round 0 … cold_start_rounds
    - Learning: cold_start_rounds+1 … cold_start_rounds*3
    - Exploiting: > cold_start_rounds*3

    Parameters
    ----------
    min_affinity : float
        Minimum affinity to consider a peer for collaboration.
    exploration_initial : float
        Initial exploration ratio for new nodes (0.0–1.0).
    exploration_decay : float
        Multiplicative decay per round.
    exploration_min : float
        Floor for exploration ratio.
    ema_alpha : float
        EMA blending factor for affinity updates (0.0–1.0).
    edge_decay_factor : float
        Decay factor for stale edges per round.
    eviction_min_weight : float
        Edges below this weight are evicted.
    cold_start_rounds : int
        Governs cold-start phase transitions (from manifest policy).
    affinity_weights : AffinityWeights or None
        Weights for multi-signal affinity computation.
    """

    def __init__(
        self,
        min_affinity: float = 0.3,
        exploration_initial: float = 0.8,
        exploration_decay: float = 0.95,
        exploration_min: float = 0.1,
        ema_alpha: float = 0.2,
        edge_decay_factor: float = 0.95,
        eviction_min_weight: float = 0.05,
        cold_start_rounds: int = 3,
        affinity_weights: Optional[AffinityWeights] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_affinity = min_affinity
        self.exploration_ratio = exploration_initial
        self.exploration_initial = exploration_initial
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.ema_alpha = ema_alpha
        self.edge_decay_factor = edge_decay_factor
        self.eviction_min_weight = eviction_min_weight
        self.cold_start_rounds = cold_start_rounds
        self.affinity_weights = affinity_weights or AffinityWeights()
        self.history = CollaborationHistory()
        self._round_count = 0
        # TOP-12: Add lock for history mutations
        self._lock = asyncio.Lock()

    @property
    def cold_start_phase(self) -> str:
        if self._round_count <= self.cold_start_rounds:
            return "blind"
        if self._round_count <= self.cold_start_rounds * 3:
            return "learning"
        return "exploiting"

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence (TOP-07)."""
        return {
            "round_count": self._round_count,
            "exploration_ratio": self.exploration_ratio,
            "history": self.history.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dict for persistence (TOP-07)."""
        self._round_count = state_dict.get("round_count", 0)
        self.exploration_ratio = state_dict.get("exploration_ratio", self.exploration_initial)
        history_dict = state_dict.get("history", {})
        self.history.load_state_dict(history_dict)

    async def select_targets(
        self, context: SelectionContext, count: int = 3
    ) -> List[str]:
        compatible = context.get_compatible_peers()
        if not compatible:
            return []

        my_fp = context.my_fingerprint
        if my_fp is None:
            selected = random.sample(compatible, min(count, len(compatible)))
            return [p.peer_id for p in selected]

        scored: List[Tuple[str, float]] = []
        unscored: List[PeerInfo] = []

        for peer in compatible:
            if peer.data_fingerprint is not None:
                weights = AffinityWeights(
                    label=self.affinity_weights.label,
                    feature=self.affinity_weights.feature,
                    gradient=self.affinity_weights.gradient,
                    history=self.affinity_weights.history,
                    external_history_score=self.history.get_history_score(peer.peer_id),
                )
                aff = my_fp.affinity_score(peer.data_fingerprint, weights)
                scored.append((peer.peer_id, aff))
            else:
                unscored.append(peer)

        scored.sort(key=lambda x: x[1], reverse=True)
        scored_filtered = [(pid, aff) for pid, aff in scored if aff >= self.min_affinity]

        n_exploit = max(1, int(count * (1 - self.exploration_ratio)))
        n_explore = count - n_exploit

        targets = [pid for pid, _ in scored_filtered[:n_exploit]]

        explore_pool_ids = [
            p.peer_id
            for p in unscored
            if p.peer_id not in targets
        ]
        low_aff_ids = [
            pid for pid, aff in scored if aff < self.min_affinity and pid not in targets
        ]
        explore_pool_ids.extend(low_aff_ids)

        if explore_pool_ids and n_explore > 0:
            targets.extend(
                random.sample(explore_pool_ids, min(n_explore, len(explore_pool_ids)))
            )

        self._round_count += 1
        self.exploration_ratio = max(
            self.exploration_min,
            self.exploration_ratio * self.exploration_decay,
        )

        for pid, aff in scored:
            self.history.update_peer(pid, aff, self.ema_alpha)
        self.history.decay_all(self.edge_decay_factor)
        self.history.evict_dead_edges(self.eviction_min_weight)

        return targets[:count]

    async def should_accept_connection(
        self, context: SelectionContext, peer_info: PeerInfo
    ) -> bool:
        if context.my_manifest_id is not None and peer_info.manifest_id is not None:
            return peer_info.manifest_id == context.my_manifest_id
        if peer_info.domain != context.my_domain:
            return False
        if peer_info.data_schema_hash != context.my_data_schema_hash:
            return False
        if not is_version_compatible(context.my_model_version, peer_info.model_version):
            return False
        return True

    def record_round_result(self, peer_id: str, success: bool):
        self.history.record_collaboration_result(peer_id, success)

    def get_exploration_ratio(self) -> float:
        return self.exploration_ratio

    def get_collaboration_graph_summary(self) -> Dict[str, Any]:
        top = self.history.get_top_peers(20)
        return {
            "round": self._round_count,
            "phase": self.cold_start_phase,
            "exploration_ratio": self.exploration_ratio,
            "active_edges": self.history.edge_count,
            "top_peers": [{"peer_id": pid, "weight": w} for pid, w in top],
        }
