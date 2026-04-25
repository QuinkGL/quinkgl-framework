"""Regression tests: chunk-buffer memory caps.

Covers:
- Reject transfer when total_chunks exceeds MAX_CHUNKS_PER_TRANSFER.
- Reject new buffer when per-peer transfer limit reached.
- Reject new buffer when global transfer limit reached.
- Reject new buffer when per-peer byte budget exceeded.
"""

import pytest
from unittest.mock import MagicMock

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    ChunkBuffer,
    ModelChunkPayload,
    MAX_CONCURRENT_TRANSFERS_PER_PEER,
    MAX_TOTAL_TRANSFERS,
    MAX_BUFFERED_BYTES_PER_PEER,
    MAX_CHUNKS_PER_TRANSFER,
)


def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    return peer


def _make_community():
    community = MagicMock(spec=GossipLearningCommunity)
    community._chunk_buffers = {}
    community._recent_chunks = {}
    community._completed_chunk_transfers = {}
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
    community.known_peers = {}
    community._heartbeat_sequence = 0
    community.node_id = "local"
    community.on_model_update_callback = None
    community._last_seen_round = {}
    community._mid_to_node_id = {}
    community.require_signature = False
    community.max_round_skip = 1000
    community.current_round_provider = lambda: 0
    return community


def _payload(transfer_id, sender_id, chunk_index=0, total_chunks=10,
             chunk_data=b"\x00"):
    return ModelChunkPayload(
        transfer_id=transfer_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        sender_id=sender_id,
        data_schema_hash="abc",
        round_number=1,
        sample_count=8,
        loss=0.1,
        accuracy=0.9,
        chunk_data=chunk_data,
    )


class TestMaxChunksPerTransfer:
    @pytest.mark.asyncio
    async def test_reject_when_total_chunks_exceeds_limit(self):
        community = _make_community()
        peer = _make_peer("aa" * 20)
        p = _payload("t1", "p1", total_chunks=MAX_CHUNKS_PER_TRANSFER + 1)

        GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p
        )

        assert len(community._chunk_buffers) == 0

    @pytest.mark.asyncio
    async def test_accept_when_total_chunks_at_limit(self):
        community = _make_community()
        peer = _make_peer("aa" * 20)
        p = _payload("t1", "p1", total_chunks=MAX_CHUNKS_PER_TRANSFER)

        GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p
        )

        assert len(community._chunk_buffers) == 1


class TestPerPeerTransferLimit:
    @pytest.mark.asyncio
    async def test_reject_exceeding_per_peer_limit(self):
        community = _make_community()
        peer = _make_peer("bb" * 20)
        mid = peer.mid.hex()

        # Fill up per-peer limit
        for i in range(MAX_CONCURRENT_TRANSFERS_PER_PEER):
            tid = f"transfer-{i}"
            community._chunk_buffers[(mid, tid)] = ChunkBuffer(
                transfer_id=tid, sender_id="p1", total_chunks=5,
                data_schema_hash="abc", round_number=1,
                sample_count=4, loss=0.1, accuracy=0.9,
            )

        # This one should be rejected
        p = _payload("transfer-overflow", "p1")
        GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p
        )

        assert (mid, "transfer-overflow") not in community._chunk_buffers
        assert len(community._chunk_buffers) == MAX_CONCURRENT_TRANSFERS_PER_PEER


class TestGlobalTransferLimit:
    @pytest.mark.asyncio
    async def test_reject_exceeding_global_limit(self):
        community = _make_community()

        # Fill up global limit with different peers
        for i in range(MAX_TOTAL_TRANSFERS):
            fake_mid = f"{i:040x}"
            tid = f"t-{i}"
            community._chunk_buffers[(fake_mid, tid)] = ChunkBuffer(
                transfer_id=tid, sender_id=f"node-{i}", total_chunks=5,
                data_schema_hash="abc", round_number=1,
                sample_count=4, loss=0.1, accuracy=0.9,
            )

        peer = _make_peer("cc" * 20)
        p = _payload("overflow", "new-node")
        GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p
        )

        assert (peer.mid.hex(), "overflow") not in community._chunk_buffers
        assert len(community._chunk_buffers) == MAX_TOTAL_TRANSFERS


class TestPerPeerByteBudget:
    @pytest.mark.asyncio
    async def test_reject_exceeding_byte_budget(self):
        community = _make_community()
        peer = _make_peer("dd" * 20)
        mid = peer.mid.hex()

        # Create a buffer that consumes the entire byte budget
        buf = ChunkBuffer(
            transfer_id="big-t", sender_id="p1", total_chunks=2,
            data_schema_hash="abc", round_number=1,
            sample_count=4, loss=0.1, accuracy=0.9,
        )
        buf.chunks[0] = b"\x00" * MAX_BUFFERED_BYTES_PER_PEER
        community._chunk_buffers[(mid, "big-t")] = buf

        # New transfer should be rejected
        p = _payload("new-t", "p1")
        GossipLearningCommunity._dispatch_model_chunk(
            community, peer, p
        )

        assert (mid, "new-t") not in community._chunk_buffers
