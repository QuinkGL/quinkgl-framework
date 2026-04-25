"""Regression tests: early NACK on gap detection.

Covers:
- Incomplete buffer older than threshold triggers NACK.
- Fresh buffer is not NACKed.
- Complete buffer is not NACKed.
- B5 rate-limit applies to early-NACK path.
"""

import time
import pytest
from unittest.mock import MagicMock

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    ChunkBuffer,
    EARLY_NACK_AGE_THRESHOLD,
    NACK_BUCKET_MAX_TOKENS,
)


def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    return peer


def _make_community():
    community = MagicMock(spec=GossipLearningCommunity)
    community._chunk_buffers = {}
    community._nack_buckets = {}
    community._nack_transfer_buckets = {}
    community.known_peers = {}
    community.node_id = "local"
    # Bind real methods
    community._nack_try_consume = GossipLearningCommunity._nack_try_consume.__get__(community)
    community._nack_try_consume_transfer = GossipLearningCommunity._nack_try_consume_transfer.__get__(community)
    community._nack_incomplete_buffers = GossipLearningCommunity._nack_incomplete_buffers.__get__(community)
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
    return community


def _add_peer(community, node_id, mid_hex):
    peer = _make_peer(mid_hex)
    peer_info = MagicMock()
    peer_info.peer = peer
    peer_info.node_id = node_id
    community.known_peers[node_id] = peer_info
    return peer


class TestEarlyNACK:
    @pytest.mark.asyncio
    async def test_old_incomplete_buffer_triggers_nack(self):
        """Buffer older than EARLY_NACK_AGE_THRESHOLD with missing chunks → NACK."""
        community = _make_community()
        mid = "aa" * 20
        peer = _add_peer(community, "node-a", mid)

        buf = ChunkBuffer(
            transfer_id="t1", sender_id="node-a", total_chunks=5,
            data_schema_hash="abc", round_number=1,
            sample_count=4, loss=0.1, accuracy=0.9,
        )
        buf.created_at = time.time() - EARLY_NACK_AGE_THRESHOLD - 1
        buf.chunks = {0: b"x", 1: b"x", 3: b"x"}  # missing 2, 4
        community._chunk_buffers[(mid, "t1")] = buf

        await community._nack_incomplete_buffers()

        community.ez_send.assert_called_once()
        sent_payload = community.ez_send.call_args[0][1]
        assert sent_payload.transfer_id == "t1"

    @pytest.mark.asyncio
    async def test_fresh_buffer_not_nacked(self):
        """Buffer younger than threshold should NOT trigger NACK."""
        community = _make_community()
        mid = "bb" * 20
        _add_peer(community, "node-b", mid)

        buf = ChunkBuffer(
            transfer_id="t2", sender_id="node-b", total_chunks=5,
            data_schema_hash="abc", round_number=1,
            sample_count=4, loss=0.1, accuracy=0.9,
        )
        buf.created_at = time.time()  # just created
        buf.chunks = {0: b"x"}
        community._chunk_buffers[(mid, "t2")] = buf

        await community._nack_incomplete_buffers()

        community.ez_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_buffer_not_nacked(self):
        """Complete buffer should not trigger NACK regardless of age."""
        community = _make_community()
        mid = "cc" * 20
        _add_peer(community, "node-c", mid)

        buf = ChunkBuffer(
            transfer_id="t3", sender_id="node-c", total_chunks=2,
            data_schema_hash="abc", round_number=1,
            sample_count=4, loss=0.1, accuracy=0.9,
        )
        buf.created_at = time.time() - 100
        buf.chunks = {0: b"x", 1: b"x"}  # complete
        community._chunk_buffers[(mid, "t3")] = buf

        await community._nack_incomplete_buffers()

        community.ez_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_applies_to_early_nack(self):
        """After exhausting the token bucket, early-NACKs are suppressed."""
        community = _make_community()
        mid = "dd" * 20
        _add_peer(community, "node-d", mid)

        # Exhaust bucket
        for _ in range(NACK_BUCKET_MAX_TOKENS):
            community._nack_try_consume(mid)

        buf = ChunkBuffer(
            transfer_id="t4", sender_id="node-d", total_chunks=3,
            data_schema_hash="abc", round_number=1,
            sample_count=4, loss=0.1, accuracy=0.9,
        )
        buf.created_at = time.time() - EARLY_NACK_AGE_THRESHOLD - 1
        buf.chunks = {0: b"x"}
        community._chunk_buffers[(mid, "t4")] = buf

        await community._nack_incomplete_buffers()

        community.ez_send.assert_not_called()
