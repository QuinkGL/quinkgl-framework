import pytest
from unittest.mock import AsyncMock, MagicMock

import quinkgl.network.gossip_community as gc
from quinkgl.network.gossip_community import (
    CHUNK_ACK_EVERY,
    CHUNK_SIZE,
    ChunkAckPayload,
    GossipLearningCommunity,
    OutgoingChunkTransfer,
    _pack_chunk_ranges,
    _unpack_chunk_ranges,
)


def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    peer.public_key = MagicMock()
    peer.address = ("127.0.0.1", 7001)
    return peer


def _make_receiver():
    community = MagicMock(spec=GossipLearningCommunity)
    community._chunk_buffers = {}
    community._recent_chunks = {}
    community._last_seen_round = {}
    community._completed_chunk_transfers = {}
    community._mid_to_node_id = {}
    community.known_peers = {}
    community._heartbeat_sequence = 0
    community.node_id = "receiver"
    community.data_schema_hash = "abc"
    community.require_signature = False
    community.event_emitter = MagicMock()
    community.max_round_skip = 1000
    community.current_round_provider = lambda: 0
    community.on_model_update_callback = AsyncMock()
    community.my_peer = MagicMock()
    community.my_peer.key.signature = MagicMock(return_value=b"\x00" * 64)
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
    community._send_chunk_ack_ranges = GossipLearningCommunity._send_chunk_ack_ranges.__get__(community)
    community._send_chunk_ack = GossipLearningCommunity._send_chunk_ack.__get__(community)
    community._maybe_send_chunk_ack = GossipLearningCommunity._maybe_send_chunk_ack.__get__(community)
    community._process_completed_model = GossipLearningCommunity._process_completed_model.__get__(community)
    community.on_model_chunk = GossipLearningCommunity.on_model_chunk.__wrapped__.__get__(
        community, GossipLearningCommunity
    )
    return community


def _chunk_payload(transfer_id: str, index: int, total: int):
    return gc.ModelChunkPayload(
        transfer_id=transfer_id,
        chunk_index=index,
        total_chunks=total,
        sender_id="sender",
        data_schema_hash="abc",
        round_number=1,
        sample_count=8,
        loss=0.0,
        accuracy=0.0,
        chunk_data=b"x",
    )


class TestChunkRangePacking:
    def test_pack_unpack_empty(self):
        assert _pack_chunk_ranges([]) == b""
        assert _unpack_chunk_ranges(b"") == set()

    def test_pack_unpack_sparse_ranges(self):
        packed = _pack_chunk_ranges([0, 1, 2, 5, 7, 8])
        assert _unpack_chunk_ranges(packed, total_chunks=10) == {0, 1, 2, 5, 7, 8}

    def test_unpack_rejects_out_of_bounds(self):
        packed = _pack_chunk_ranges([0, 4])
        with pytest.raises(ValueError):
            _unpack_chunk_ranges(packed, total_chunks=4)


class TestReceiverChunkAck:
    def test_receiver_sends_ack_after_threshold(self):
        receiver = _make_receiver()
        peer = _make_peer("aa" * 20)
        total = CHUNK_ACK_EVERY + 10

        for index in range(CHUNK_ACK_EVERY):
            receiver.on_model_chunk(peer, _chunk_payload("tid", index, total))

        receiver.ez_send.assert_called()
        ack = receiver.ez_send.call_args[0][1]
        assert isinstance(ack, ChunkAckPayload)
        assert ack.transfer_id == "tid"
        assert _unpack_chunk_ranges(ack.sack_ranges, total) == set(range(CHUNK_ACK_EVERY))

    def test_receiver_sends_final_ack_on_completion(self):
        receiver = _make_receiver()
        peer = _make_peer("bb" * 20)

        for index in range(2):
            receiver.on_model_chunk(peer, _chunk_payload("tid-final", index, 2))

        sent_payloads = [call.args[1] for call in receiver.ez_send.call_args_list]
        assert any(isinstance(payload, ChunkAckPayload) for payload in sent_payloads)

    def test_duplicate_completed_transfer_sends_completion_ack(self):
        receiver = _make_receiver()
        peer = _make_peer("ee" * 20)
        receiver._completed_chunk_transfers[(peer.mid.hex(), "tid-complete")] = 1.0

        receiver.on_model_chunk(peer, _chunk_payload("tid-complete", 1, 3))

        ack = receiver.ez_send.call_args[0][1]
        assert isinstance(ack, ChunkAckPayload)
        assert _unpack_chunk_ranges(ack.sack_ranges, 3) == {0, 1, 2}


class TestChunkAckHandler:
    @pytest.mark.asyncio
    async def test_ack_marks_chunks_and_completes_transfer(self):
        community = MagicMock(spec=GossipLearningCommunity)
        peer = _make_peer("cc" * 20)
        transfer = OutgoingChunkTransfer(
            transfer_id="tid-ack",
            target_node_id="receiver",
            recipient_mid=peer.mid.hex(),
            peer=peer,
            weights_bytes=b"x" * (CHUNK_SIZE * 3),
            sample_count=8,
            round_number=1,
            loss=0.0,
            accuracy=0.0,
            timestamp=0,
            total_chunks=3,
            inflight_key=("receiver", 1, "hash"),
        )
        community._active_outgoing_transfers = {"tid-ack": transfer}

        payload = ChunkAckPayload(
            transfer_id="tid-ack",
            sender_id="receiver",
            data_schema_hash="abc",
            round_number=1,
            total_chunks=3,
            sack_ranges=_pack_chunk_ranges([0, 1, 2]),
        )

        await GossipLearningCommunity._dispatch_chunk_ack(community, peer, payload)

        assert transfer.acked == {0, 1, 2}
        assert transfer.completed.is_set()


class TestChunkWindow:
    @pytest.mark.asyncio
    async def test_initial_window_bounds_unacked_sends(self, monkeypatch):
        monkeypatch.setattr(gc, "CHUNK_SEND_INTERVAL", 0)
        monkeypatch.setattr(gc, "CHUNK_WINDOW_POLL_INTERVAL", 0.001)
        monkeypatch.setattr(gc, "CHUNK_TRANSFER_TIMEOUT", 0.01)

        sender = MagicMock(spec=GossipLearningCommunity)
        sender.node_id = "sender"
        sender.data_schema_hash = "abc"
        sender._outgoing_transfers = {}
        sender._active_outgoing_transfers = {}
        sender._inflight_transfers = {}
        sender.metrics = {
            'chunk_transfers_started': 0,
            'chunk_transfers_completed': 0,
            'chunk_transfers_failed_timeout': 0,
            'chunk_transfers_rejected_peer_limit': 0,
            'nacks_sent': 0,
            'nacks_received': 0,
            'nacks_ignored_budget': 0,
            'chunks_resent': 0,
        }
        sender.my_peer = MagicMock()
        sender.my_peer.key.signature = MagicMock(return_value=b"\x00" * 64)
        sender._send_model_chunk = GossipLearningCommunity._send_model_chunk.__get__(sender)
        sender._send_chunked_model_update = GossipLearningCommunity._send_chunked_model_update.__get__(
            sender, GossipLearningCommunity
        )

        peer = _make_peer("dd" * 20)
        peer_info = MagicMock()
        peer_info.peer = peer

        result = await sender._send_chunked_model_update(
            "receiver",
            peer_info,
            b"x" * (CHUNK_SIZE * (gc.CHUNK_WINDOW_INITIAL + 8)),
            8,
            1,
            0.0,
            0.0,
        )

        assert result is False
        assert sender.ez_send.call_count <= gc.CHUNK_WINDOW_INITIAL
