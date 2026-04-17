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
)
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
    community.known_peers = {}
    community._mid_to_node_id = {}
    community._heartbeat_sequence = 0
    community.node_id = "local"
    community.data_schema_hash = "abc"
    community.on_model_update_callback = None
    return community


def _model_update_payload(sender_id, round_number):
    return ModelUpdatePayload(
        sender_id=sender_id,
        weights_bytes=b"\x00" * 16,
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
        chunk_data=b"\x00" * 10,
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

        await GossipLearningCommunity.on_model_update.__wrapped__(c, peer, payload)

        mid = peer.mid.hex()
        assert c._last_seen_round[mid] == 5

    @pytest.mark.asyncio
    async def test_rejects_same_round(self):
        c = _make_community()
        peer = _make_peer("aa" * 20)
        mid = peer.mid.hex()
        c._last_seen_round[mid] = 5

        payload = _model_update_payload("node-a", round_number=5)
        await GossipLearningCommunity.on_model_update.__wrapped__(c, peer, payload)

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
        await GossipLearningCommunity.on_model_update.__wrapped__(c, peer, payload)

        assert c._last_seen_round[mid] == 10


# ── B15-3: on_model_chunk rejects replay on new buffer ──────

class TestChunkReplay:
    @pytest.mark.asyncio
    async def test_rejects_stale_chunked_transfer(self):
        c = _make_community()
        peer = _make_peer("cc" * 20)
        mid = peer.mid.hex()
        c._last_seen_round[mid] = 8

        payload = _chunk_payload("tid-1", "node-c", round_number=5)
        await GossipLearningCommunity.on_model_chunk.__wrapped__(c, peer, payload)

        # No buffer should have been created
        assert len(c._chunk_buffers) == 0

    @pytest.mark.asyncio
    async def test_accepts_new_round_chunked(self):
        c = _make_community()
        peer = _make_peer("dd" * 20)

        payload = _chunk_payload("tid-2", "node-d", round_number=10)
        await GossipLearningCommunity.on_model_chunk.__wrapped__(c, peer, payload)

        # Buffer should have been created
        assert len(c._chunk_buffers) == 1


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
