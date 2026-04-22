"""Tests for CyclonTopology."""

import asyncio
import pytest

from quinkgl.topology.cyclon import CyclonTopology
from quinkgl.topology.base import PeerInfo, SelectionContext


def _peer(pid, domain="d", schema="s", manifest_id=None):
    return PeerInfo(
        peer_id=pid,
        domain=domain,
        data_schema_hash=schema,
        model_version="1.0.0",
        manifest_id=manifest_id,
    )


def _ctx(my_id="me", peers=None, domain="d", schema="s", manifest_id=None):
    return SelectionContext(
        my_peer_id=my_id,
        my_domain=domain,
        my_data_schema_hash=schema,
        known_peers=peers or [],
        my_model_version="1.0.0",
        my_manifest_id=manifest_id,
    )


class TestCyclonSelectTargets:
    @pytest.mark.asyncio
    async def test_empty_view_bootstraps_from_known_peers(self):
        topo = CyclonTopology(view_size=10, seed=0)
        peers = [_peer("p1"), _peer("p2"), _peer("p3")]
        ctx = _ctx(peers=peers)
        targets = await topo.select_targets(ctx, count=2)
        assert len(targets) <= 2
        assert all(t in {"p1", "p2", "p3"} for t in targets)

    @pytest.mark.asyncio
    async def test_select_from_populated_view(self):
        topo = CyclonTopology(seed=0)
        for i in range(5):
            await topo.sampler.add_peer(_peer(f"p{i}"))
        ctx = _ctx()
        targets = await topo.select_targets(ctx, count=3)
        assert 0 < len(targets) <= 3

    @pytest.mark.asyncio
    async def test_empty_view_no_known_peers_returns_empty(self):
        topo = CyclonTopology(seed=0)
        ctx = _ctx(peers=[])
        targets = await topo.select_targets(ctx, count=3)
        assert targets == []

    @pytest.mark.asyncio
    async def test_no_duplicate_targets(self):
        topo = CyclonTopology(seed=42)
        for i in range(10):
            await topo.sampler.add_peer(_peer(f"p{i}"))
        ctx = _ctx()
        targets = await topo.select_targets(ctx, count=5)
        assert len(targets) == len(set(targets))

    @pytest.mark.asyncio
    async def test_count_larger_than_view_returns_all(self):
        topo = CyclonTopology(seed=0)
        await topo.sampler.add_peer(_peer("p1"))
        await topo.sampler.add_peer(_peer("p2"))
        ctx = _ctx()
        targets = await topo.select_targets(ctx, count=100)
        assert set(targets) == {"p1", "p2"}


class TestCyclonShouldAccept:
    @pytest.mark.asyncio
    async def test_accept_compatible_peer(self):
        topo = CyclonTopology()
        ctx = _ctx()
        assert await topo.should_accept_connection(ctx, _peer("p1")) is True

    @pytest.mark.asyncio
    async def test_reject_domain_mismatch(self):
        topo = CyclonTopology()
        ctx = _ctx(domain="d")
        assert await topo.should_accept_connection(ctx, _peer("p1", domain="other")) is False

    @pytest.mark.asyncio
    async def test_reject_schema_mismatch(self):
        topo = CyclonTopology()
        ctx = _ctx(schema="s")
        assert await topo.should_accept_connection(ctx, _peer("p1", schema="wrong")) is False

    @pytest.mark.asyncio
    async def test_accept_manifest_id_match(self):
        """L-4 fix: manifest_id checked when both sides have one."""
        mid = b"\xaa" * 30
        topo = CyclonTopology()
        ctx = _ctx(manifest_id=mid)
        assert await topo.should_accept_connection(ctx, _peer("p1", manifest_id=mid)) is True

    @pytest.mark.asyncio
    async def test_reject_manifest_id_mismatch(self):
        """L-4 fix: peers with mismatched manifest IDs are rejected."""
        topo = CyclonTopology()
        ctx = _ctx(manifest_id=b"\xaa" * 30)
        peer = _peer("p1", manifest_id=b"\xbb" * 30)
        assert await topo.should_accept_connection(ctx, peer) is False

    @pytest.mark.asyncio
    async def test_fallback_to_domain_when_manifest_absent(self):
        """If either side has no manifest_id, fall back to domain+schema."""
        topo = CyclonTopology()
        ctx = _ctx(manifest_id=None)
        peer = _peer("p1", manifest_id=None)
        assert await topo.should_accept_connection(ctx, peer) is True


class TestCyclonPeerLifecycle:
    @pytest.mark.asyncio
    async def test_on_new_peer_discovered_adds_to_view(self):
        topo = CyclonTopology()
        assert topo.sampler.get_view_size() == 0
        await topo.on_new_peer_discovered(_peer("p1"))
        assert topo.sampler.get_view_size() == 1

    @pytest.mark.asyncio
    async def test_on_peer_disconnected_removes_from_view(self):
        topo = CyclonTopology()
        await topo.on_new_peer_discovered(_peer("p1"))
        await topo.on_new_peer_discovered(_peer("p2"))
        await topo.on_peer_disconnected("p1")
        view_ids = {p.peer_id for p in topo.get_active_view()}
        assert "p1" not in view_ids
        assert "p2" in view_ids

    @pytest.mark.asyncio
    async def test_get_active_view_returns_peers(self):
        topo = CyclonTopology()
        await topo.on_new_peer_discovered(_peer("p1"))
        await topo.on_new_peer_discovered(_peer("p2"))
        view = topo.get_active_view()
        assert {p.peer_id for p in view} == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_adding_same_peer_twice_no_duplicates(self):
        topo = CyclonTopology()
        await topo.on_new_peer_discovered(_peer("p1"))
        await topo.on_new_peer_discovered(_peer("p1"))
        assert topo.sampler.get_view_size() == 1


class TestCyclonShuffle:
    @pytest.mark.asyncio
    async def test_handle_incoming_shuffle_merges_peers(self):
        topo = CyclonTopology(shuffle_length=5)
        await topo.on_new_peer_discovered(_peer("local1"))
        remote_peers = [_peer("remote1"), _peer("remote2")]
        response = await topo.handle_incoming_shuffle("sender", remote_peers)
        # remote peers should now be in view
        view_ids = {p.peer_id for p in topo.get_active_view()}
        assert "remote1" in view_ids or "remote2" in view_ids

    @pytest.mark.asyncio
    async def test_handle_incoming_shuffle_excludes_sender(self):
        topo = CyclonTopology(shuffle_length=10)
        await topo.on_new_peer_discovered(_peer("p1"))
        await topo.on_new_peer_discovered(_peer("p2"))
        await topo.on_new_peer_discovered(_peer("sender"))
        response = await topo.handle_incoming_shuffle("sender", [])
        response_ids = [p.peer_id for p in response]
        assert "sender" not in response_ids

    @pytest.mark.asyncio
    async def test_periodic_maintenance_recovery_empty_view(self):
        """Recovery: empty view + known_peers → view refilled."""
        topo = CyclonTopology()
        peers = [_peer("p1"), _peer("p2")]
        ctx = _ctx(peers=peers)
        # View starts empty; maintenance should bootstrap it
        await topo.periodic_maintenance(ctx)
        assert topo.sampler.get_view_size() > 0


class TestCyclonStartStop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        topo = CyclonTopology(shuffle_interval=60.0)
        ctx = _ctx()
        await topo.start(ctx)
        assert topo._running is True
        await topo.stop()
        assert topo._running is False

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self):
        topo = CyclonTopology(shuffle_interval=60.0)
        ctx = _ctx()
        await topo.start(ctx)
        task1 = topo._shuffle_task
        await topo.start(ctx)  # second start should be no-op
        assert topo._shuffle_task is task1
        await topo.stop()


# ── D: CyclonTopology — partition scenario (0, 1 peer) ───────────────────

class TestCyclonPartition:
    @pytest.mark.asyncio
    async def test_zero_compatible_peers_no_known_peers(self):
        topo = CyclonTopology(seed=0)
        ctx = _ctx(peers=[])
        targets = await topo.select_targets(ctx, count=3)
        assert targets == []

    @pytest.mark.asyncio
    async def test_single_peer_in_view_selectable(self):
        topo = CyclonTopology(seed=0)
        await topo.on_new_peer_discovered(_peer("only"))
        ctx = _ctx()
        targets = await topo.select_targets(ctx, count=3)
        assert targets == ["only"]


# ── D: CyclonTopology — no peer-id ordering bias ─────────────────────────

class TestCyclonNoPeerIdBias:
    @pytest.mark.asyncio
    async def test_all_peers_reachable_across_different_seeds(self):
        """Different seeds must produce different selections — no ordering bias."""
        peers = [_peer(f"p{i:03d}") for i in range(5)]
        seen = set()
        for seed in range(20):
            topo = CyclonTopology(seed=seed)
            for p in peers:
                await topo.on_new_peer_discovered(p)
            ctx = _ctx()
            targets = await topo.select_targets(ctx, count=2)
            seen.update(targets)
        # With 20 different seeds selecting 2 of 5 peers each time,
        # all peers should appear at least once.
        assert len(seen) == 5, f"Only {seen} ever selected — ordering bias detected"
