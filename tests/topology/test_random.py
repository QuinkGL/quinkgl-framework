"""Tests for RandomTopology."""

import pytest

from quinkgl.topology.random import RandomTopology
from quinkgl.topology.base import PeerInfo, SelectionContext  # noqa: F401 — also used in manifest tests


def _peer(pid, domain="d", schema="s", version="1.0.0"):
    return PeerInfo(peer_id=pid, domain=domain, data_schema_hash=schema, model_version=version)


def _ctx(my_id="me", peers=None, domain="d", schema="s"):
    return SelectionContext(
        my_peer_id=my_id,
        my_domain=domain,
        my_data_schema_hash=schema,
        known_peers=peers or [],
        my_model_version="1.0.0",
    )


class TestRandomTopologySelectTargets:
    @pytest.mark.asyncio
    async def test_returns_up_to_count(self):
        topo = RandomTopology(seed=0)
        peers = [_peer(f"p{i}") for i in range(10)]
        ctx = _ctx(peers=peers)
        targets = await topo.select_targets(ctx, count=3)
        assert len(targets) == 3

    @pytest.mark.asyncio
    async def test_count_larger_than_peers_returns_all(self):
        topo = RandomTopology(seed=0)
        peers = [_peer("p1"), _peer("p2")]
        ctx = _ctx(peers=peers)
        targets = await topo.select_targets(ctx, count=10)
        assert set(targets) == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_empty_peers_returns_empty(self):
        topo = RandomTopology(seed=0)
        ctx = _ctx(peers=[])
        targets = await topo.select_targets(ctx, count=3)
        assert targets == []

    @pytest.mark.asyncio
    async def test_self_is_excluded(self):
        topo = RandomTopology(seed=0)
        peers = [_peer("me"), _peer("p1"), _peer("p2")]
        ctx = _ctx(my_id="me", peers=peers)
        for _ in range(20):
            targets = await topo.select_targets(ctx, count=2)
            assert "me" not in targets

    @pytest.mark.asyncio
    async def test_incompatible_domain_excluded(self):
        topo = RandomTopology(seed=0)
        peers = [_peer("p1", domain="other"), _peer("p2", domain="d")]
        ctx = _ctx(peers=peers, domain="d")
        targets = await topo.select_targets(ctx, count=5)
        assert "p1" not in targets
        assert "p2" in targets

    @pytest.mark.asyncio
    async def test_incompatible_schema_excluded(self):
        topo = RandomTopology(seed=0)
        peers = [_peer("p1", schema="wrong"), _peer("p2", schema="s")]
        ctx = _ctx(peers=peers, schema="s")
        targets = await topo.select_targets(ctx, count=5)
        assert "p1" not in targets

    @pytest.mark.asyncio
    async def test_seeded_rng_is_deterministic(self):
        peers = [_peer(f"p{i}") for i in range(10)]
        ctx = _ctx(peers=peers)
        topo1 = RandomTopology(seed=42, cache_duration=0)
        topo2 = RandomTopology(seed=42, cache_duration=0)
        r1 = await topo1.select_targets(ctx, count=3)
        r2 = await topo2.select_targets(ctx, count=3)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_targets_are_strings(self):
        topo = RandomTopology(seed=0)
        peers = [_peer("p1"), _peer("p2")]
        ctx = _ctx(peers=peers)
        targets = await topo.select_targets(ctx, count=2)
        for t in targets:
            assert isinstance(t, str)

    @pytest.mark.asyncio
    async def test_no_duplicate_targets(self):
        topo = RandomTopology(seed=0)
        peers = [_peer(f"p{i}") for i in range(5)]
        ctx = _ctx(peers=peers)
        targets = await topo.select_targets(ctx, count=5)
        assert len(targets) == len(set(targets))


class TestRandomTopologyShouldAccept:
    @pytest.mark.asyncio
    async def test_accept_compatible_peer(self):
        topo = RandomTopology()
        ctx = _ctx()
        peer = _peer("p1")
        assert await topo.should_accept_connection(ctx, peer) is True

    @pytest.mark.asyncio
    async def test_reject_domain_mismatch(self):
        topo = RandomTopology()
        ctx = _ctx(domain="d")
        peer = _peer("p1", domain="other")
        assert await topo.should_accept_connection(ctx, peer) is False

    @pytest.mark.asyncio
    async def test_reject_schema_mismatch(self):
        topo = RandomTopology()
        ctx = _ctx(schema="s")
        peer = _peer("p1", schema="wrong")
        assert await topo.should_accept_connection(ctx, peer) is False

    @pytest.mark.asyncio
    async def test_accept_manifest_id_match(self):
        mid = b"\xaa" * 30
        topo = RandomTopology()
        ctx = SelectionContext(
            my_peer_id="me", my_domain="d", my_data_schema_hash="s",
            known_peers=[], my_model_version="1.0.0", my_manifest_id=mid,
        )
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s",
                        model_version="1.0.0", manifest_id=mid)
        assert await topo.should_accept_connection(ctx, peer) is True

    @pytest.mark.asyncio
    async def test_reject_manifest_id_mismatch(self):
        topo = RandomTopology()
        ctx = SelectionContext(
            my_peer_id="me", my_domain="d", my_data_schema_hash="s",
            known_peers=[], my_model_version="1.0.0", my_manifest_id=b"\xaa" * 30,
        )
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s",
                        model_version="1.0.0", manifest_id=b"\xbb" * 30)
        assert await topo.should_accept_connection(ctx, peer) is False

    @pytest.mark.asyncio
    async def test_fallback_to_domain_when_manifest_absent(self):
        topo = RandomTopology()
        ctx = SelectionContext(
            my_peer_id="me", my_domain="d", my_data_schema_hash="s",
            known_peers=[], my_model_version="1.0.0", my_manifest_id=None,
        )
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s",
                        model_version="1.0.0", manifest_id=None)
        assert await topo.should_accept_connection(ctx, peer) is True


# ── D: RandomTopology — partition scenario (0, 1 peer) ───────────────────

class TestRandomPartition:
    @pytest.mark.asyncio
    async def test_zero_compatible_peers_returns_empty(self):
        topo = RandomTopology(seed=0)
        # All peers have wrong domain
        peers = [_peer("p1", domain="other"), _peer("p2", domain="other")]
        ctx = _ctx(peers=peers, domain="d")
        targets = await topo.select_targets(ctx, count=3)
        assert targets == []

    @pytest.mark.asyncio
    async def test_single_compatible_peer_returns_it(self):
        topo = RandomTopology(seed=0)
        ctx = _ctx(peers=[_peer("p1")])
        targets = await topo.select_targets(ctx, count=3)
        assert targets == ["p1"]


# ── D: RandomTopology — no peer-id ordering bias ─────────────────────────

class TestRandomNoPeerIdBias:
    @pytest.mark.asyncio
    async def test_all_peers_reachable_over_many_draws(self):
        """rng.sample is uniformly random — every peer should eventually be picked."""
        peers = [_peer(f"p{i:03d}") for i in range(5)]
        ctx = _ctx(peers=peers)
        seen = set()
        for seed in range(50):
            topo = RandomTopology(seed=seed, cache_duration=0)
            targets = await topo.select_targets(ctx, count=1)
            seen.update(targets)
        assert seen == {p.peer_id for p in peers}
