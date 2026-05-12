import pytest

from quinkgl.topology import (
    HybridAffinityReliability,
    RandomRegular,
    ReliabilityAware,
    Ring,
    SmallWorld,
)
from quinkgl.topology.base import PeerInfo, SelectionContext


def _peer(peer_id, metadata=None):
    return PeerInfo(
        peer_id=peer_id,
        domain="demo",
        data_schema_hash="schema",
        model_version="1.0.0",
        metadata=metadata or {},
    )


def _context(peers):
    return SelectionContext(
        my_peer_id="peer-0",
        my_domain="demo",
        my_data_schema_hash="schema",
        my_model_version="1.0.0",
        known_peers=peers,
    )


@pytest.mark.asyncio
async def test_ring_selects_nearest_ring_neighbors():
    topology = Ring()
    ctx = _context([_peer(f"peer-{i}") for i in range(5)])

    targets = await topology.select_targets(ctx, count=3)

    assert targets == ["peer-1", "peer-4", "peer-2"]


@pytest.mark.asyncio
async def test_random_regular_is_seeded_and_count_bounded():
    ctx = _context([_peer(f"peer-{i}") for i in range(8)])
    first = RandomRegular(seed=7, degree=4)
    second = RandomRegular(seed=7, degree=4)

    targets = await first.select_targets(ctx, count=6)
    repeated = await second.select_targets(ctx, count=6)

    assert targets == repeated
    assert len(targets) == 4
    assert "peer-0" not in targets


@pytest.mark.asyncio
async def test_small_world_mixes_ring_neighbors_with_long_range_peer():
    topology = SmallWorld(seed=3, local_ratio=0.5)
    ctx = _context([_peer(f"peer-{i}") for i in range(8)])

    targets = await topology.select_targets(ctx, count=4)

    assert targets[:2] == ["peer-1", "peer-7"]
    assert len(targets) == 4
    assert len(set(targets)) == 4


@pytest.mark.asyncio
async def test_reliability_aware_prefers_successful_low_latency_peers():
    topology = ReliabilityAware()
    ctx = _context(
        [
            _peer("slow", {"transfer_success_rate": 0.9, "last_latency_ms": 900}),
            _peer("flaky", {"transfer_success_rate": 0.2, "last_latency_ms": 20}),
            _peer("stable", {"transfer_success_rate": 0.95, "last_latency_ms": 40}),
        ]
    )

    targets = await topology.select_targets(ctx, count=2)

    assert targets == ["stable", "slow"]


@pytest.mark.asyncio
async def test_hybrid_affinity_reliability_uses_reliability_without_fingerprints():
    topology = HybridAffinityReliability(reliability_weight=1.0, affinity_weight=0.0)
    ctx = _context(
        [
            _peer("bad", {"transfer_success_rate": 0.1}),
            _peer("good", {"transfer_success_rate": 0.99}),
        ]
    )

    targets = await topology.select_targets(ctx, count=1)

    assert targets == ["good"]
