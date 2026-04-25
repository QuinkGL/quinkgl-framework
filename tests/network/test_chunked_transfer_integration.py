"""Integration test: chunked transfer with artificial packet loss and NACK recovery.

Simulates a sender streaming 10 chunks to a receiver, dropping 3 chunks,
and verifying that NACK-driven retransmission completes the transfer and
fires the aggregation callback.
"""

import asyncio
import hashlib
import time
import pytest
from unittest.mock import MagicMock, AsyncMock

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    ModelChunkPayload,
    RequestChunksPayload,
    ChunkBuffer,
    CHUNK_SIZE,
)
from quinkgl.network.model_serializer import serialize_model


def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    peer.public_key = MagicMock()
    return peer


def _make_receiver():
    """Create a mock receiver community with all required state."""
    c = MagicMock(spec=GossipLearningCommunity)
    c._chunk_buffers = {}

    c._recent_chunks = {}
    c._last_seen_round = {}
    c._nack_buckets = {}
    c._nack_transfer_buckets = {}
    c._completed_chunk_transfers = {}
    c._recent_chunks = {}
    c.known_peers = {}
    c._mid_to_node_id = {}
    c._heartbeat_sequence = 0
    c.node_id = "receiver"
    c.data_schema_hash = "abc"
    c.on_model_update_callback = AsyncMock()
    c.require_signature = False
    c.event_emitter = MagicMock()
    c.max_round_skip = 1000
    c.current_round_provider = lambda: 0
    c.metrics = {
        'chunk_transfers_started': 0,
        'chunk_transfers_completed': 0,
        'chunk_transfers_failed_timeout': 0,
        'chunk_transfers_rejected_peer_limit': 0,
        'nacks_sent': 0,
        'nacks_received': 0,
        'nacks_ignored_budget': 0,
        'chunks_resent': 0,
    }
    # Bind real methods (bypass @lazy_wrapper to avoid serializer dependencies)
    c._nack_try_consume = GossipLearningCommunity._nack_try_consume.__get__(c)
    c._nack_try_consume_transfer = GossipLearningCommunity._nack_try_consume_transfer.__get__(c)
    c.on_model_chunk = GossipLearningCommunity.on_model_chunk.__wrapped__.__get__(c, GossipLearningCommunity)
    c.on_request_chunks = GossipLearningCommunity.on_request_chunks.__wrapped__.__get__(c, GossipLearningCommunity)
    c._nack_incomplete_buffers = GossipLearningCommunity._nack_incomplete_buffers.__get__(c)
    c._process_completed_model = GossipLearningCommunity._process_completed_model.__get__(c, GossipLearningCommunity)
    return c


def _chunk_payload(transfer_id, sender_id, round_number, chunk_index,
                   total_chunks, chunk_data):
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
        chunk_data=chunk_data,
    )


class TestChunkedTransferNACKRecovery:
    @pytest.mark.asyncio
    async def test_nack_recover_missing_chunks_and_fire_callback(self):
        """Drop 3 of 10 chunks; NACK should recover them and invoke callback."""
        receiver = _make_receiver()
        sender_peer = _make_peer("aa" * 20)
        sender_mid = sender_peer.mid.hex()
        transfer_id = "t-recovery-01"
        round_number = 3
        total_chunks = 10

        # Build a single model payload and split it into 10 chunks
        model = {"data": [float(i) for i in range(500)]}
        weights_bytes = serialize_model(model)
        chunk_size = (len(weights_bytes) + total_chunks - 1) // total_chunks
        chunks = []
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(weights_bytes))
            chunks.append(weights_bytes[start:end])

        # Simulate sending chunks 0-9 but drop indices 2, 5, 7
        dropped = {2, 5, 7}
        for i in range(total_chunks):
            if i in dropped:
                continue
            payload = _chunk_payload(
                transfer_id, "sender", round_number, i, total_chunks, chunks[i]
            )
            receiver.on_model_chunk(sender_peer, payload)

        # Buffer should exist but be incomplete
        buf_key = (sender_mid, transfer_id)
        assert buf_key in receiver._chunk_buffers
        assert not receiver._chunk_buffers[buf_key].is_complete()
        assert receiver.metrics['chunk_transfers_started'] == 1

        # Simulate early-NACK (proactive gap detection)
        # Make buffer old enough
        receiver._chunk_buffers[buf_key].created_at = (
            time.time() - 10.0  # older than EARLY_NACK_AGE_THRESHOLD
        )
        # Add sender to known_peers so NACK can be sent
        peer_info = MagicMock()
        peer_info.peer = sender_peer
        receiver.known_peers["sender"] = peer_info
        receiver._mid_to_node_id[sender_mid] = "sender"

        await receiver._nack_incomplete_buffers()

        # A NACK should have been sent for the missing chunks
        receiver.ez_send.assert_called()
        nack_payload = receiver.ez_send.call_args[0][1]
        assert isinstance(nack_payload, RequestChunksPayload)
        assert nack_payload.transfer_id == transfer_id

        # Decode missing indices
        import struct
        missing = list(struct.unpack(f'{len(nack_payload.missing_indices_bytes)//4}I',
                                     nack_payload.missing_indices_bytes))
        assert set(missing) == dropped

        # Simulate sender retransmitting the missing chunks
        for i in missing:
            payload = _chunk_payload(
                transfer_id, "sender", round_number, i, total_chunks, chunks[i]
            )
            receiver.on_model_chunk(sender_peer, payload)

        # Allow the event loop to run the async callback task
        await asyncio.sleep(0)

        # Now buffer should be complete and callback fired
        assert buf_key not in receiver._chunk_buffers  # removed after completion
        receiver.on_model_update_callback.assert_awaited_once()
        assert receiver.metrics['chunk_transfers_completed'] == 1

        # Verify callback args
        call_kwargs = receiver.on_model_update_callback.call_args.kwargs
        assert call_kwargs['sender_id'] == "sender"
        assert call_kwargs['round_number'] == round_number
        assert call_kwargs['sample_count'] == 8

    @pytest.mark.asyncio
    async def test_idempotency_prevents_duplicate_transfer(self):
        """Sender should not start a new transfer for the same (peer, round, model)."""
        sender = MagicMock(spec=GossipLearningCommunity)
        sender._inflight_transfers = {}
        sender._outgoing_transfers = {}
        sender.node_id = "sender"
        sender.data_schema_hash = "abc"
        sender.known_peers = {}
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
        sender.my_peer.key = MagicMock()
        sender.my_peer.key.signature = MagicMock(return_value=b"\x00" * 64)

        # Create a known peer
        peer = _make_peer("bb" * 20)
        peer_info = MagicMock()
        peer_info.peer = peer
        sender.known_peers["receiver"] = peer_info

        weights_bytes = serialize_model({"w": [1.0, 2.0]})
        model_hash = hashlib.sha256(weights_bytes).hexdigest()[:16]
        inflight_key = ("receiver", 5, model_hash)

        # First call creates transfer
        sender._inflight_transfers[inflight_key] = "existing-tid"
        sender._outgoing_transfers["existing-tid"] = {
            "weights": weights_bytes,
            "timestamp": time.time(),
        }

        # v3 idempotency check: second call should skip
        result = await GossipLearningCommunity.send_model_update(
            sender, "receiver", {"w": [1.0, 2.0]}, 8, 5
        )
        # Because weights dict is serialized, the hash may differ; just assert
        # that if an inflight transfer exists, the method returns True without
        # creating a new one.
        assert result is True
        assert len(sender._outgoing_transfers) == 1
