"""Tests for AffinityTopology, CollaborationEdge, CollaborationHistory."""

import asyncio

import pytest

from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    AffinityWeights,
)
from quinkgl.topology.affinity import (
    AffinityTopology,
    CollaborationEdge,
    CollaborationHistory,
)
from quinkgl.topology.base import PeerInfo, SelectionContext


def _make_fp(label_buckets=None, moments=None):
    return DataFingerprint(
        label_buckets=label_buckets or {"a": "high"},
        noised_moments=moments or {"l": (1.0, 0.5)},
        sample_bucket="1k-10k",
        num_classes=1,
    )


def _make_ctx(my_fp=None, peers=None, manifest_id=None):
    return SelectionContext(
        my_peer_id="me",
        my_domain="d",
        my_data_schema_hash="s",
        known_peers=peers or [],
        my_model_version="1.0.0",
        my_manifest_id=manifest_id,
        my_fingerprint=my_fp,
    )


# ── CollaborationEdge ──────────────────────────────────────────────────

class TestCollaborationEdge:
    def test_initial_state(self):
        e = CollaborationEdge(peer_id="p1")
        assert e.affinity == 0.0
        assert e.weight == 0.5
        assert e.rounds_since_update == 0
        assert e.success_rate == 0.0

    def test_success_rate(self):
        e = CollaborationEdge(peer_id="p1", successful_rounds=7, total_rounds=10)
        assert e.success_rate == pytest.approx(0.7)

    def test_success_rate_zero_division(self):
        e = CollaborationEdge(peer_id="p1")
        assert e.success_rate == 0.0

    def test_ema_update(self):
        e = CollaborationEdge(peer_id="p1", weight=0.5)
        e.update_affinity(0.9, ema_alpha=0.2)
        assert e.affinity == 0.9
        assert e.weight == pytest.approx(0.8 * 0.5 + 0.2 * 0.9)
        assert e.rounds_since_update == 0

    def test_ema_update_alpha_1(self):
        e = CollaborationEdge(peer_id="p1", weight=0.5)
        e.update_affinity(0.9, ema_alpha=1.0)
        assert e.weight == pytest.approx(0.9)

    def test_decay(self):
        e = CollaborationEdge(peer_id="p1", weight=1.0)
        e.decay(decay_factor=0.9)
        assert e.rounds_since_update == 1
        assert e.weight == pytest.approx(0.9)

    def test_decay_accumulates_staleness(self):
        e = CollaborationEdge(peer_id="p1", weight=1.0)
        e.decay(0.9)
        e.decay(0.9)
        assert e.rounds_since_update == 2
        assert e.weight < 0.9

    def test_record_collaboration_success(self):
        e = CollaborationEdge(peer_id="p1")
        e.record_collaboration(True)
        assert e.successful_rounds == 1
        assert e.total_rounds == 1

    def test_record_collaboration_failure(self):
        e = CollaborationEdge(peer_id="p1")
        e.record_collaboration(False)
        assert e.successful_rounds == 0
        assert e.total_rounds == 1


# ── CollaborationHistory ────────────────────────────────────────────────

class TestCollaborationHistory:
    def test_create_edge(self):
        h = CollaborationHistory()
        e = h.get_or_create_edge("p1")
        assert e.peer_id == "p1"
        assert e.weight == 0.5

    def test_get_edge_returns_none(self):
        h = CollaborationHistory()
        assert h.get_edge("p1") is None

    def test_history_score_insufficient_data(self):
        h = CollaborationHistory()
        e = h.get_or_create_edge("p1")
        e.record_collaboration(True)
        e.record_collaboration(True)
        assert h.get_history_score("p1") == 0.0

    def test_history_score_sufficient_data(self):
        h = CollaborationHistory()
        e = h.get_or_create_edge("p1")
        for _ in range(5):
            e.record_collaboration(True)
        assert h.get_history_score("p1") == pytest.approx(1.0)

    def test_history_score_mixed(self):
        h = CollaborationHistory()
        e = h.get_or_create_edge("p1")
        for _ in range(3):
            e.record_collaboration(True)
        for _ in range(2):
            e.record_collaboration(False)
        assert h.get_history_score("p1") == pytest.approx(0.6)

    def test_update_peer(self):
        h = CollaborationHistory()
        h.update_peer("p1", 0.8, ema_alpha=0.2)
        e = h.get_edge("p1")
        assert e is not None
        assert e.affinity == 0.8

    def test_decay_all(self):
        h = CollaborationHistory()
        h.update_peer("p1", 0.9)
        h.decay_all(0.95)
        e = h.get_edge("p1")
        assert e.weight < 0.9

    def test_evict_dead_edges(self):
        h = CollaborationHistory()
        h.update_peer("p1", 0.01)
        e = h.get_edge("p1")
        e.weight = 0.01
        h.evict_dead_edges(min_weight=0.05)
        assert h.get_edge("p1") is None

    def test_evict_keeps_strong_edges(self):
        h = CollaborationHistory()
        h.update_peer("p1", 0.9)
        h.evict_dead_edges(min_weight=0.05)
        assert h.get_edge("p1") is not None

    def test_max_peers_eviction(self):
        h = CollaborationHistory(max_peers=2)
        h.update_peer("p1", 0.5)
        h.update_peer("p2", 0.8)
        h.update_peer("p3", 0.9)
        assert h.edge_count == 2
        assert h.get_edge("p3") is not None

    def test_get_top_peers(self):
        h = CollaborationHistory()
        h.update_peer("p1", 0.5)
        h.get_edge("p1").weight = 0.5
        h.update_peer("p2", 0.9)
        h.get_edge("p2").weight = 0.9
        top = h.get_top_peers(5)
        assert top[0][0] == "p2"
        assert top[1][0] == "p1"


# ── AffinityTopology: select_targets ───────────────────────────────────

class TestAffinityTopologySelectTargets:
    @pytest.mark.asyncio
    async def test_no_compatible_peers(self):
        topo = AffinityTopology()
        ctx = _make_ctx(my_fp=_make_fp(), peers=[])
        targets = await topo.select_targets(ctx, count=3)
        assert targets == []

    @pytest.mark.asyncio
    async def test_no_fingerprint_falls_back_to_random(self):
        topo = AffinityTopology()
        peers = [
            PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", model_version="1.0.0"),
            PeerInfo(peer_id="p2", domain="d", data_schema_hash="s", model_version="1.0.0"),
        ]
        ctx = _make_ctx(my_fp=None, peers=peers)
        targets = await topo.select_targets(ctx, count=2)
        assert len(targets) == 2
        assert set(targets) == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_affinity_driven_selection(self):
        topo = AffinityTopology(min_affinity=0.3, exploration_initial=0.0, exploration_min=0.0)
        fp_high = _make_fp(label_buckets={"a": "high"}, moments={"l": (1.0, 0.5)})
        fp_low = _make_fp(label_buckets={"a": "low"}, moments={"l": (0.0, 1.0)})

        peers = [
            PeerInfo(peer_id="p_high", domain="d", data_schema_hash="s",
                     model_version="1.0.0", data_fingerprint=fp_high),
            PeerInfo(peer_id="p_low", domain="d", data_schema_hash="s",
                     model_version="1.0.0", data_fingerprint=fp_low),
        ]
        my_fp = _make_fp(label_buckets={"a": "high"}, moments={"l": (1.0, 0.5)})
        ctx = _make_ctx(my_fp=my_fp, peers=peers)

        targets = await topo.select_targets(ctx, count=1)
        assert "p_high" in targets

    @pytest.mark.asyncio
    async def test_cold_start_high_exploration(self):
        topo = AffinityTopology(min_affinity=0.3, exploration_initial=0.8, exploration_decay=0.95, exploration_min=0.1)
        fp = _make_fp()
        fp_diff = _make_fp(label_buckets={"a": "low"}, moments={"l": (0.0, 1.0)})
        peers = [
            PeerInfo(peer_id="p_similar", domain="d", data_schema_hash="s",
                     model_version="1.0.0", data_fingerprint=fp),
            PeerInfo(peer_id="p_diff", domain="d", data_schema_hash="s",
                     model_version="1.0.0", data_fingerprint=fp_diff),
            PeerInfo(peer_id="p_no_fp", domain="d", data_schema_hash="s",
                     model_version="1.0.0"),
        ]
        ctx = _make_ctx(my_fp=fp, peers=peers)

        targets = await topo.select_targets(ctx, count=3)
        assert "p_similar" in targets
        assert len(targets) >= 2

    @pytest.mark.asyncio
    async def test_exploration_ratio_decays(self):
        topo = AffinityTopology(exploration_initial=0.8, exploration_decay=0.5, exploration_min=0.1)
        fp = _make_fp()
        ctx = _make_ctx(my_fp=fp, peers=[
            PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", model_version="1.0.0", data_fingerprint=fp),
        ])
        initial = topo.get_exploration_ratio()
        for _ in range(5):
            await topo.select_targets(ctx, count=1)
        final = topo.get_exploration_ratio()
        assert final < initial
        assert final >= 0.1

    @pytest.mark.asyncio
    async def test_exploration_never_goes_below_min(self):
        topo = AffinityTopology(exploration_initial=0.2, exploration_decay=0.5, exploration_min=0.15)
        fp = _make_fp()
        ctx = _make_ctx(my_fp=fp, peers=[
            PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", model_version="1.0.0", data_fingerprint=fp),
        ])
        for _ in range(50):
            await topo.select_targets(ctx, count=1)
        assert topo.get_exploration_ratio() >= 0.15

    @pytest.mark.asyncio
    async def test_peers_without_fingerprint_go_to_explore_pool(self):
        topo = AffinityTopology(min_affinity=0.3, exploration_initial=0.5, exploration_min=0.3)
        fp = _make_fp()
        peers = [
            PeerInfo(peer_id="p_fp", domain="d", data_schema_hash="s",
                     model_version="1.0.0", data_fingerprint=fp),
            PeerInfo(peer_id="p_no_fp", domain="d", data_schema_hash="s",
                     model_version="1.0.0"),
        ]
        ctx = _make_ctx(my_fp=fp, peers=peers)

        targets = await topo.select_targets(ctx, count=2)
        assert len(targets) == 2


# ── AffinityTopology: should_accept_connection ─────────────────────────

class TestAffinityTopologyAcceptConnection:
    @pytest.mark.asyncio
    async def test_accept_manifest_id_match(self):
        topo = AffinityTopology()
        mid = b"\xaa" * 30
        ctx = _make_ctx(manifest_id=mid)
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", manifest_id=mid)
        assert await topo.should_accept_connection(ctx, peer) is True

    @pytest.mark.asyncio
    async def test_reject_manifest_id_mismatch(self):
        topo = AffinityTopology()
        ctx = _make_ctx(manifest_id=b"\xaa" * 30)
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", manifest_id=b"\xbb" * 30)
        assert await topo.should_accept_connection(ctx, peer) is False

    @pytest.mark.asyncio
    async def test_fallback_to_domain_schema(self):
        topo = AffinityTopology()
        ctx = _make_ctx()
        peer = PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", model_version="1.0.0")
        assert await topo.should_accept_connection(ctx, peer) is True

    @pytest.mark.asyncio
    async def test_reject_domain_mismatch(self):
        topo = AffinityTopology()
        ctx = _make_ctx()
        peer = PeerInfo(peer_id="p1", domain="other", data_schema_hash="s", model_version="1.0.0")
        assert await topo.should_accept_connection(ctx, peer) is False


# ── AffinityTopology: cold-start phases ────────────────────────────────

class TestAffinityTopologyColdStart:
    def test_initial_phase_blind(self):
        topo = AffinityTopology(cold_start_rounds=3)
        assert topo.cold_start_phase == "blind"

    def test_phase_learning(self):
        topo = AffinityTopology(cold_start_rounds=3)
        topo._round_count = 5
        assert topo.cold_start_phase == "learning"

    def test_phase_exploiting(self):
        topo = AffinityTopology(cold_start_rounds=3)
        topo._round_count = 10
        assert topo.cold_start_phase == "exploiting"

    def test_custom_cold_start_rounds(self):
        topo = AffinityTopology(cold_start_rounds=5)
        topo._round_count = 5
        assert topo.cold_start_phase == "blind"
        topo._round_count = 6
        assert topo.cold_start_phase == "learning"
        topo._round_count = 16
        assert topo.cold_start_phase == "exploiting"


# ── AffinityTopology: record_round_result + history ──────────────────────

class TestAffinityTopologyRoundResult:
    @pytest.mark.asyncio
    async def test_record_round_result(self):
        topo = AffinityTopology()
        topo.record_round_result("p1", True)
        topo.record_round_result("p1", True)
        topo.record_round_result("p1", False)
        score = topo.history.get_history_score("p1")
        assert score == pytest.approx(2 / 3)


# ── AffinityTopology: collaboration graph summary ──────────────────────

class TestAffinityTopologySummary:
    @pytest.mark.asyncio
    async def test_summary_structure(self):
        topo = AffinityTopology()
        fp = _make_fp()
        ctx = _make_ctx(my_fp=fp, peers=[
            PeerInfo(peer_id="p1", domain="d", data_schema_hash="s", model_version="1.0.0", data_fingerprint=fp),
        ])
        await topo.select_targets(ctx, count=1)
        summary = topo.get_collaboration_graph_summary()
        assert "round" in summary
        assert "phase" in summary
        assert "exploration_ratio" in summary
        assert "active_edges" in summary
        assert "top_peers" in summary
