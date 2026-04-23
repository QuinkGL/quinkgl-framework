"""Regression tests: chunk-buffer hijack prevention.

Covers:
- Two chunk streams with same transfer_id from different peer.mid remain isolated.
- Buffer keyed by (peer_mid, transfer_id), not just transfer_id.
- Cleanup uses tuple keys correctly.
"""

import time
from unittest.mock import MagicMock

import pytest

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    ChunkBuffer,
    ModelChunkPayload,
    CHUNK_TRANSFER_TIMEOUT,
)


def _make_peer(mid_hex: str):
    """Create a mock IPv8 Peer with a given mid (hex string → bytes)."""
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    return peer


def _make_community():
    """Create a minimal community stub with the fields on_model_chunk needs."""
    community = MagicMock(spec=GossipLearningCommunity)
    community._chunk_buffers = {}
    community.known_peers = {}
    community._heartbeat_sequence = 0
    community.node_id = "local"
    community.on_model_update_callback = None
    community._last_seen_round = {}
    community._mid_to_node_id = {}
    community.require_signature = False
    return community


def _make_chunk_payload(transfer_id, sender_id, chunk_index, total_chunks,
                        chunk_data=b"\x00"):
    return ModelChunkPayload(
        transfer_id=transfer_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        sender_id=sender_id,
        data_schema_hash="abc",
        round_number=5,
        sample_count=8,
        loss=0.1,
        accuracy=0.9,
        chunk_data=chunk_data,
    )


# ── Tests ────────────────────────────────────────────────────────────

class TestChunkBufferIsolation:
    @pytest.mark.asyncio
    async def test_same_transfer_id_different_peers_isolated(self):
        """Two peers sending chunks with the SAME transfer_id must get
        separate buffers (no hijack)."""
        community = _make_community()
        tid = "shared-transfer-id-12345678"

        peer_a = _make_peer("aa" * 20)
        peer_b = _make_peer("bb" * 20)

        payload_a = _make_chunk_payload(tid, "attacker", 0, 2, b"A-data-0")
        payload_b = _make_chunk_payload(tid, "victim", 0, 2, b"B-data-0")

        # Invoke the real handler logic
        await GossipLearningCommunity._dispatch_model_chunk(
            community, peer_a, payload_a
        )
        await GossipLearningCommunity._dispatch_model_chunk(
            community, peer_b, payload_b
        )

        # Must have TWO separate buffers
        assert len(community._chunk_buffers) == 2

        mid_a = peer_a.mid.hex()
        mid_b = peer_b.mid.hex()
        buf_a = community._chunk_buffers[(mid_a, tid)]
        buf_b = community._chunk_buffers[(mid_b, tid)]

        assert buf_a.sender_id == "attacker"
        assert buf_b.sender_id == "victim"
        assert buf_a.chunks[0] == b"A-data-0"
        assert buf_b.chunks[0] == b"B-data-0"

    @pytest.mark.asyncio
    async def test_same_peer_same_transfer_id_shared(self):
        """Same peer + same transfer_id should go to the same buffer."""
        community = _make_community()
        tid = "transfer-xyz"
        peer = _make_peer("cc" * 20)

        p0 = _make_chunk_payload(tid, "node-1", 0, 3, b"chunk-0")
        p1 = _make_chunk_payload(tid, "node-1", 1, 3, b"chunk-1")

        await GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p0
        )
        await GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p1
        )

        assert len(community._chunk_buffers) == 1
        mid = peer.mid.hex()
        buf = community._chunk_buffers[(mid, tid)]
        assert len(buf.chunks) == 2


class TestCleanupTupleKeys:
    @pytest.mark.asyncio
    async def test_cleanup_stale_transfers_with_tuple_keys(self):
        """_cleanup_stale_transfers must work with (peer_mid, transfer_id) keys."""
        community = _make_community()

        # Add an expired buffer
        buf = ChunkBuffer(
            transfer_id="t1",
            sender_id="p1",
            total_chunks=10,
            data_schema_hash="abc",
            round_number=5,
            sample_count=8,
            loss=0.1,
            accuracy=0.9,
        )
        buf.created_at = time.monotonic() - CHUNK_TRANSFER_TIMEOUT - 10
        community._chunk_buffers[("mid-a", "t1")] = buf

        # Add a fresh buffer
        community._chunk_buffers[("mid-b", "t2")] = ChunkBuffer(
            transfer_id="t2",
            sender_id="p2",
            total_chunks=5,
            data_schema_hash="abc",
            round_number=5,
            sample_count=4,
            loss=0.2,
            accuracy=0.8,
        )

        await GossipLearningCommunity._cleanup_stale_transfers(community)

        # Only the fresh one should remain
        assert ("mid-a", "t1") not in community._chunk_buffers
        assert ("mid-b", "t2") in community._chunk_buffers
