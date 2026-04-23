"""Regression tests: NACK rate-limiting & authorization.

Covers:
- Unauthorized peer (wrong peer.mid) is rejected.
- Per-transfer resend budget is enforced.
- Token-bucket rate limiting stops burst NACKs.
- Malformed missing_indices_bytes are rejected.
"""

import struct
import time
import pytest
from unittest.mock import MagicMock

from quinkgl.network.gossip_community import (
    GossipLearningCommunity,
    RequestChunksPayload,
    NACK_MAX_RESENDS_PER_TRANSFER,
    NACK_BUCKET_MAX_TOKENS,
    CHUNK_SIZE,
)


def _make_peer(mid_hex: str):
    peer = MagicMock()
    peer.mid = bytes.fromhex(mid_hex)
    return peer


def _make_community():
    community = MagicMock(spec=GossipLearningCommunity)
    community._outgoing_transfers = {}
    community._nack_resend_counts = {}
    community._nack_buckets = {}
    community.node_id = "local"
    community.data_schema_hash = "abc"
    community.event_emitter = MagicMock()
    # mock my_peer.key for chunk signing in NACK resend path
    community.my_peer = MagicMock()
    community.my_peer.key.signature = MagicMock(return_value=b"\x00" * 64)
    # Bind the real method
    community._nack_try_consume = GossipLearningCommunity._nack_try_consume.__get__(community)
    return community


def _seed_transfer(community, tid, weights_len=1024, recipient_mid="aa" * 20):
    community._outgoing_transfers[tid] = {
        "weights": b"\x00" * weights_len,
        "loss": 0.0,
        "accuracy": 0.0,
        "round": 1,
        "samples": 8,
        "timestamp": time.time(),
        "recipient_mid": recipient_mid,
    }


def _nack_payload(tid, sender_id, missing_indices):
    data = struct.pack(f'{len(missing_indices)}I', *missing_indices)
    return RequestChunksPayload(
        transfer_id=tid,
        sender_id=sender_id,
        missing_indices_bytes=data,
    )


class TestNACKAuthorization:
    @pytest.mark.asyncio
    async def test_wrong_peer_rejected(self):
        """A peer that is NOT the original recipient must be refused."""
        community = _make_community()
        recipient_mid = "aa" * 20
        _seed_transfer(community, "t1", recipient_mid=recipient_mid)

        attacker = _make_peer("bb" * 20)
        payload = _nack_payload("t1", "attacker", [0])

        await GossipLearningCommunity._dispatch_request_chunks(
            community, attacker, payload
        )

        # ez_send should NOT have been called
        community.ez_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_correct_peer_accepted(self):
        """The original recipient's NACK should be serviced."""
        community = _make_community()
        mid = "aa" * 20
        _seed_transfer(community, "t1", weights_len=CHUNK_SIZE * 3, recipient_mid=mid)

        peer = _make_peer(mid)
        payload = _nack_payload("t1", "node-a", [1])

        await GossipLearningCommunity._dispatch_request_chunks(
            community, peer, payload
        )

        community.ez_send.assert_called_once()


class TestResendBudget:
    @pytest.mark.asyncio
    async def test_budget_exhaustion(self):
        """After NACK_MAX_RESENDS_PER_TRANSFER resends, further NACKs are refused."""
        community = _make_community()
        mid = "cc" * 20
        _seed_transfer(community, "t2", weights_len=CHUNK_SIZE * 5, recipient_mid=mid)
        peer = _make_peer(mid)

        for i in range(NACK_MAX_RESENDS_PER_TRANSFER):
            payload = _nack_payload("t2", "node-c", [0])
            await GossipLearningCommunity._dispatch_request_chunks(
                community, peer, payload
            )

        # All budget used
        assert community._nack_resend_counts["t2"] == NACK_MAX_RESENDS_PER_TRANSFER

        # Next NACK should be refused
        community.ez_send.reset_mock()
        payload = _nack_payload("t2", "node-c", [0])
        await GossipLearningCommunity._dispatch_request_chunks(
            community, peer, payload
        )
        community.ez_send.assert_not_called()


class TestTokenBucketRateLimit:
    @pytest.mark.asyncio
    async def test_burst_exhaustion(self):
        """Rapid NACKs beyond the bucket burst size are rate-limited."""
        community = _make_community()
        mid = "dd" * 20

        # Exhaust the token bucket
        for _ in range(NACK_BUCKET_MAX_TOKENS):
            assert community._nack_try_consume(mid) is True

        # Next one must fail
        assert community._nack_try_consume(mid) is False

    @pytest.mark.asyncio
    async def test_rate_limited_nack_emits_security_event(self):
        community = _make_community()
        mid = "ab" * 20
        _seed_transfer(community, "t-rate", weights_len=CHUNK_SIZE * 2, recipient_mid=mid)
        peer = _make_peer(mid)

        for _ in range(NACK_BUCKET_MAX_TOKENS):
            assert community._nack_try_consume(mid) is True

        payload = _nack_payload("t-rate", "node-rate", [0])
        await GossipLearningCommunity._dispatch_request_chunks(community, peer, payload)

        emitted = [call.args[0] for call in community.event_emitter.emit.call_args_list]
        assert emitted == ["security.nack_rate_limited", "ipv8_payload_dropped"]


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_odd_length_rejected(self):
        """missing_indices_bytes with length not divisible by 4 is rejected."""
        community = _make_community()
        mid = "ee" * 20
        _seed_transfer(community, "t3", recipient_mid=mid)
        peer = _make_peer(mid)

        payload = RequestChunksPayload(
            transfer_id="t3",
            sender_id="node-e",
            missing_indices_bytes=b"\x00\x00\x00",  # 3 bytes — invalid
        )

        await GossipLearningCommunity._dispatch_request_chunks(
            community, peer, payload
        )

        community.ez_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_payload_rejected(self):
        """Empty missing_indices_bytes is rejected."""
        community = _make_community()
        mid = "ff" * 20
        _seed_transfer(community, "t4", recipient_mid=mid)
        peer = _make_peer(mid)

        payload = RequestChunksPayload(
            transfer_id="t4",
            sender_id="node-f",
            missing_indices_bytes=b"",
        )

        await GossipLearningCommunity._dispatch_request_chunks(
            community, peer, payload
        )

        community.ez_send.assert_not_called()
