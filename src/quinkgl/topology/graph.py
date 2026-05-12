"""Graph and reliability-aware peer selection strategies."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, List, Sequence

from quinkgl.fingerprint.fingerprint import AffinityWeights
from quinkgl.topology.base import (
    PeerInfo,
    SelectionContext,
    TopologyStrategy,
    is_version_compatible,
)


def _compatible(context: SelectionContext) -> List[PeerInfo]:
    return context.get_compatible_peers(exclude_self=True)


def _ring_order(context: SelectionContext, peers: Sequence[PeerInfo]) -> List[str]:
    ids = {peer.peer_id for peer in peers}
    ids.add(context.my_peer_id)
    return sorted(ids)


def _stable_score(seed: int | None, round_number: int, peer_id: str) -> int:
    data = f"{seed or 0}:{round_number}:{peer_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def _accepts_compatible(context: SelectionContext, peer_info: PeerInfo) -> bool:
    if context.my_manifest_id is not None and peer_info.manifest_id is not None:
        return peer_info.manifest_id == context.my_manifest_id
    return (
        peer_info.domain == context.my_domain
        and peer_info.data_schema_hash == context.my_data_schema_hash
        and is_version_compatible(context.my_model_version, peer_info.model_version)
    )


def _metadata_float(metadata: dict[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return default


class Ring(TopologyStrategy):
    """Select nearest compatible peers in a stable logical ring."""

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3,
    ) -> List[str]:
        peers = _compatible(context)
        if not peers or count <= 0:
            return []

        peer_by_id = {peer.peer_id: peer for peer in peers}
        order = _ring_order(context, peers)
        own_index = order.index(context.my_peer_id)
        targets: List[str] = []

        for distance in range(1, len(order)):
            for direction in (1, -1):
                candidate_id = order[(own_index + direction * distance) % len(order)]
                if candidate_id in peer_by_id and candidate_id not in targets:
                    targets.append(candidate_id)
                    if len(targets) >= count:
                        return targets

        return targets

    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo,
    ) -> bool:
        return _accepts_compatible(context, peer_info)


class RandomRegular(TopologyStrategy):
    """Approximate a fixed-degree random regular overlay by stable hashing."""

    def __init__(self, degree: int | None = None, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.seed = seed

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3,
    ) -> List[str]:
        peers = _compatible(context)
        if not peers or count <= 0:
            return []

        limit = count
        if self.degree is not None:
            limit = min(limit, max(0, self.degree))
        if limit <= 0:
            return []

        ranked = sorted(
            peers,
            key=lambda peer: (
                _stable_score(self.seed, context.current_round, peer.peer_id),
                peer.peer_id,
            ),
        )
        return [peer.peer_id for peer in ranked[:limit]]

    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo,
    ) -> bool:
        return _accepts_compatible(context, peer_info)


class SmallWorld(TopologyStrategy):
    """Combine ring-local peers with long-range random shortcuts."""

    def __init__(
        self,
        local_ratio: float = 0.67,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_ratio = min(1.0, max(0.0, local_ratio))
        self._rng = random.Random(seed)

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3,
    ) -> List[str]:
        peers = _compatible(context)
        if not peers or count <= 0:
            return []

        local_count = min(count, max(1, int(round(count * self.local_ratio))))
        ring_targets = await Ring().select_targets(context, local_count)
        remaining = [
            peer.peer_id for peer in peers
            if peer.peer_id not in ring_targets
        ]
        shortcut_count = min(count - len(ring_targets), len(remaining))
        shortcuts = (
            self._rng.sample(remaining, shortcut_count)
            if shortcut_count > 0 else []
        )
        return (ring_targets + shortcuts)[:count]

    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo,
    ) -> bool:
        return _accepts_compatible(context, peer_info)


class ReliabilityAware(TopologyStrategy):
    """Prefer peers with successful, low-latency transfer history."""

    def __init__(
        self,
        success_weight: float = 0.7,
        latency_weight: float = 0.2,
        penalty_weight: float = 0.1,
        max_latency_ms: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.success_weight = success_weight
        self.latency_weight = latency_weight
        self.penalty_weight = penalty_weight
        self.max_latency_ms = max(1.0, max_latency_ms)

    def reliability_score(self, peer: PeerInfo) -> float:
        metadata = peer.metadata or {}
        explicit = _metadata_float(
            metadata,
            ("reliability_score", "reliability", "trust_score"),
            -1.0,
        )
        if explicit >= 0.0:
            return min(1.0, max(0.0, explicit))

        success = _metadata_float(
            metadata,
            ("transfer_success_rate", "success_rate", "chunk_success_rate"),
            0.5,
        )
        latency_ms = _metadata_float(
            metadata,
            ("last_latency_ms", "latency_ms", "rtt_ms"),
            self.max_latency_ms,
        )
        latency_score = 1.0 - min(1.0, max(0.0, latency_ms / self.max_latency_ms))
        failures = _metadata_float(
            metadata,
            ("failed_transfers", "timeouts", "stale_transfers", "nack_count"),
            0.0,
        )
        penalty = min(1.0, max(0.0, failures / 10.0))
        score = (
            self.success_weight * min(1.0, max(0.0, success))
            + self.latency_weight * latency_score
            - self.penalty_weight * penalty
        )
        return min(1.0, max(0.0, score))

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3,
    ) -> List[str]:
        peers = _compatible(context)
        if not peers or count <= 0:
            return []

        ranked = sorted(
            peers,
            key=lambda peer: (-self.reliability_score(peer), peer.peer_id),
        )
        return [peer.peer_id for peer in ranked[:count]]

    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo,
    ) -> bool:
        return _accepts_compatible(context, peer_info)


class HybridAffinityReliability(ReliabilityAware):
    """Blend data affinity with transfer reliability."""

    def __init__(
        self,
        affinity_weight: float = 0.6,
        reliability_weight: float = 0.4,
        min_affinity: float = 0.0,
        affinity_weights: AffinityWeights | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        total = affinity_weight + reliability_weight
        if total <= 0:
            affinity_weight, reliability_weight, total = 0.0, 1.0, 1.0
        self.affinity_weight = affinity_weight / total
        self.reliability_weight = reliability_weight / total
        self.min_affinity = min_affinity
        self.affinity_weights = affinity_weights or AffinityWeights()

    def _affinity_score(self, context: SelectionContext, peer: PeerInfo) -> float:
        if context.my_fingerprint is None or peer.data_fingerprint is None:
            return 0.5
        return context.my_fingerprint.affinity_score(
            peer.data_fingerprint,
            self.affinity_weights,
        )

    def hybrid_score(self, context: SelectionContext, peer: PeerInfo) -> float:
        affinity = self._affinity_score(context, peer)
        if affinity < self.min_affinity:
            return -1.0
        return (
            self.affinity_weight * affinity
            + self.reliability_weight * self.reliability_score(peer)
        )

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3,
    ) -> List[str]:
        peers = _compatible(context)
        if not peers or count <= 0:
            return []

        ranked = sorted(
            peers,
            key=lambda peer: (-self.hybrid_score(context, peer), peer.peer_id),
        )
        return [
            peer.peer_id for peer in ranked
            if self.hybrid_score(context, peer) >= 0.0
        ][:count]
