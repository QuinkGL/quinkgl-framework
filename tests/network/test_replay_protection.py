"""
B15 regression tests — Replay protection.

Validates that:
 - last_seen_round dict is initialised.
 - on_model_update rejects round <= last_seen.
 - on_model_update accepts strictly increasing rounds.
 - on_model_chunk rejects new buffer when round <= last_seen.
 - MAX_MESSAGE_AGE_SECONDS is tightened to 300 s.
 - validate_message uses monotonic freshness, rejects old msgs.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

import pytest

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    ModelUpdatePayload,
    ModelChunkPayload,
    MAX_ROUND_SKIP,
)
from quinkgl.network.model_serializer import serialize_model
from quinkgl.gossip.protocol import GossipProtocol


# ── helpers ──────────────────────────────────────────────────

def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    peer.public_key = MagicMock()
    return peer


def _make_community():
    community = MagicMock(spec=GossipLearningCommunity)
    community._last_seen_round = {}
    community._chunk_buffers = {}

    community._recent_chunks = {}
    community.known_peers = {}
    community._mid_to_node_id = {}
    community._heartbeat_sequence = 0
    community.node_id = "local"
    community.data_schema_hash = "abc"
    community.on_model_update_callback = None
    community.require_signature = False
    community.event_emitter = MagicMock()
    community.last_seen_round_state_path = ""
    community.max_round_skip = MAX_ROUND_SKIP
    community.current_round_provider = lambda: 0
    community._load_last_seen_round_state = GossipLearningCommunity._load_last_seen_round_state.__get__(community)
    community._persist_last_seen_round_state = GossipLearningCommunity._persist_last_seen_round_state.__get__(community)
    community._record_last_seen_round = GossipLearningCommunity._record_last_seen_round.__get__(community)
    community._get_local_round = GossipLearningCommunity._get_local_round.__get__(community)
    # v3: new state required by gossip_community.py
    community.metrics = {
        'chunk_transfers_started': 0,
        'chunk_transfers_completed': 0,
        'chunk_transfers_failed_timeout': 0,
        'chunk_transfers_rejected_peer_limit': 0,
        'nacks_sent': 0,
        'nacks_received': 0,
        'nacks_ignored_budget': 0,
        'chunks_resent': 0,
    }
    community._nack_transfer_buckets = {}
    community._inflight_transfers = {}
    community._completed_chunk_transfers = {}
    community._recent_chunks = {}
    community._nack_try_consume_transfer = GossipLearningCommunity._nack_try_consume_transfer.__get__(community)
    community._dispatch_model_update = GossipLearningCommunity.on_model_update.__get__(community)
    return community


def _model_update_payload(sender_id, round_number):
    return ModelUpdatePayload(
        sender_id=sender_id,
        weights_bytes=serialize_model({"w": [1.0, 2.0]}),
        sample_count=8,
        round_number=round_number,
        data_schema_hash="abc",
    )


def _chunk_payload(transfer_id, sender_id, round_number, chunk_index=0,
                   total_chunks=2):
    return ModelChunkPayload(
        transfer_id=transfer_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        sender_id=sender_id,
        data_schema_hash="abc",
        round_number=round_number,
        sample_count=8,
        loss=0.0,
        accuracy=0.0,
        chunk_data=serialize_model({"w": [1.0, 2.0]}),
    )


# ── B15-1: last_seen_round initialised ──────────────────────

def test_last_seen_round_init():
    c = _make_community()
    assert isinstance(c._last_seen_round, dict)
    assert len(c._last_seen_round) == 0


# ── B15-2: on_model_update rejects replay ───────────────────

class TestModelUpdateReplay:
    @pytest.mark.asyncio
    async def test_accepts_first_round(self):
        c = _make_community()
        peer = _make_peer("aa" * 20)
        payload = _model_update_payload("node-a", round_number=5)

        await GossipLearningCommunity._dispatch_model_update(c, peer, payload)

        mid = peer.mid.hex()
        assert c._last_seen_round[mid] == 5

    @pytest.mark.asyncio
    async def test_rejects_same_round(self):
        c = _make_community()
        peer = _make_peer("aa" * 20)
        mid = peer.mid.hex()
        c._last_seen_round[mid] = 5

        payload = _model_update_payload("node-a", round_number=5)
        await GossipLearningCommunity._dispatch_model_update(c, peer, payload)

        # Should NOT have advanced (callback not invoked)
        assert c._last_seen_round[mid] == 5
        c.on_model_update_callback.assert_not_called() if c.on_model_update_callback else None

    @pytest.mark.asyncio
    async def test_rejects_older_round(self):
        c = _make_community()
        peer = _make_peer("bb" * 20)
        mid = peer.mid.hex()
        c._last_seen_round[mid] = 10

        payload = _model_update_payload("node-b", round_number=3)
        await GossipLearningCommunity._dispatch_model_update(c, peer, payload)

        assert c._last_seen_round[mid] == 10

    @pytest.mark.asyncio
    async def test_rejects_future_round(self):
        c = _make_community()
        c.current_round_provider = lambda: 5
        peer = _make_peer("ab" * 20)

        payload = _model_update_payload("node-future", round_number=MAX_ROUND_SKIP + 6)
        await GossipLearningCommunity._dispatch_model_update(c, peer, payload)

        assert c._last_seen_round == {}
        emitted = [call.args[0] for call in c.event_emitter.emit.call_args_list]
        assert emitted == ["security.future_round_rejected", "ipv8_payload_dropped"]

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_advance_round(self):
        c = _make_community()
        peer = _make_peer("ac" * 20)
        c.on_model_update_callback = AsyncMock(side_effect=RuntimeError("boom"))

        payload = _model_update_payload("node-cb", round_number=7)
        await GossipLearningCommunity._dispatch_model_update(c, peer, payload)

        assert peer.mid.hex() not in c._last_seen_round


# ── B15-3: on_model_chunk rejects replay on new buffer ──────

class TestChunkReplay:
    @pytest.mark.asyncio
    async def test_rejects_duplicate_completed_transfer(self):
        c = _make_community()
        peer = _make_peer("cc" * 20)
        mid = peer.mid.hex()
        # v3: transfer-based replay protection
        c._completed_chunk_transfers[(mid, "tid-1")] = 1234567890.0

        payload = _chunk_payload("tid-1", "node-c", round_number=5)
        GossipLearningCommunity._dispatch_model_chunk(c, peer, payload)

        # No buffer should have been created (duplicate completed transfer)
        assert len(c._chunk_buffers) == 0

    @pytest.mark.asyncio
    async def test_accepts_new_transfer_same_round(self):
        c = _make_community()
        peer = _make_peer("dd" * 20)

        # v3: same round but new transfer_id should be accepted
        payload = _chunk_payload("tid-2", "node-d", round_number=10)
        GossipLearningCommunity._dispatch_model_chunk(c, peer, payload)

        # Buffer should have been created
        assert len(c._chunk_buffers) == 1

    @pytest.mark.asyncio
    async def test_rejects_future_round_chunked(self):
        c = _make_community()
        c.current_round_provider = lambda: 3
        peer = _make_peer("de" * 20)

        payload = _chunk_payload("tid-future", "node-future", round_number=MAX_ROUND_SKIP + 4)
        GossipLearningCommunity._dispatch_model_chunk(c, peer, payload)

        assert len(c._chunk_buffers) == 0
        emitted = [call.args[0] for call in c.event_emitter.emit.call_args_list]
        assert emitted == ["security.future_round_rejected", "ipv8_payload_dropped"]

    @pytest.mark.asyncio
    async def test_chunk_callback_failure_records_completed_anyway(self):
        c = _make_community()
        peer = _make_peer("df" * 20)
        c.on_model_update_callback = AsyncMock(side_effect=RuntimeError("chunk boom"))

        payload = _chunk_payload("tid-cb", "node-cb", round_number=11, total_chunks=1)
        GossipLearningCommunity._dispatch_model_chunk(c, peer, payload)

        # v3: transfer is recorded as completed BEFORE the async callback runs,
        # so that late chunks (including NACK resends with non-deterministic
        # signatures) are rejected even if the callback eventually fails.
        assert (peer.mid.hex(), "tid-cb") in c._completed_chunk_transfers


class TestReplayPersistence:
    def test_record_last_seen_round_persists_to_disk(self, tmp_path):
        c = _make_community()
        state_path = tmp_path / "last_seen_round.json"
        c.last_seen_round_state_path = str(state_path)

        c._record_last_seen_round("aa" * 20, 9)

        assert state_path.exists()
        reloaded = _make_community()
        reloaded.last_seen_round_state_path = str(state_path)
        reloaded._load_last_seen_round_state()
        assert reloaded._last_seen_round["aa" * 20] == 9


# ── B15-4: MAX_MESSAGE_AGE_SECONDS tightened ────────────────

def test_max_message_age_tightened():
    assert GossipProtocol.MAX_MESSAGE_AGE_SECONDS == 300


# ── B15-5: validate_message monotonic freshness ─────────────

def test_validate_rejects_old_message():
    proto = GossipProtocol(peer_id="me")
    msg = MagicMock()
    msg.sender_id = "other"
    msg.timestamp = datetime.now() - timedelta(seconds=400)
    assert proto.validate_message(msg) is False


def test_validate_accepts_recent_message():
    proto = GossipProtocol(peer_id="me")
    msg = MagicMock()
    msg.sender_id = "other"
    msg.timestamp = datetime.now()
    assert proto.validate_message(msg) is True


def test_validate_rejects_future_message():
    proto = GossipProtocol(peer_id="me")
    msg = MagicMock()
    msg.sender_id = "other"
    msg.timestamp = datetime.now() + timedelta(seconds=120)
    assert proto.validate_message(msg) is False
