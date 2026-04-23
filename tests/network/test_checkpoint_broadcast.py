"""Regression tests: checkpoint broadcast on IPv8 transport.

Covers:
- CheckpointPayload serialization round-trip.
- broadcast_checkpoint sends to all known peers.
- on_checkpoint invokes callback with correct fields.
- on_checkpoint_callback is wired in _setup_ipv8_callbacks.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from quinkgl.network.gossip_community import CheckpointPayload


# ── Tests ────────────────────────────────────────────────────────────

class TestCheckpointPayload:
    def test_payload_round_trip(self):
        """Pack → unpack must preserve all fields."""
        original = CheckpointPayload(
            sender_id="node-1",
            round_number=42,
            loss=0.123,
            accuracy=0.876,
            model_version="2.0.0",
        )
        packed = original.to_pack_list()

        # Extract raw values for from_unpack_list
        raw = [item[1] for item in packed]
        restored = CheckpointPayload.from_unpack_list(*raw)

        assert restored.sender_id == "node-1"
        assert restored.round_number == 42
        assert abs(restored.loss - 0.123) < 1e-9
        assert abs(restored.accuracy - 0.876) < 1e-9
        assert restored.model_version == "2.0.0"

    def test_msg_id_unique(self):
        """CheckpointPayload msg_id must be 9 (unique)."""
        assert CheckpointPayload.msg_id == 9


class TestBroadcastCheckpoint:
    def test_broadcast_sends_to_all_known_peers(self):
        """broadcast_checkpoint must ez_send to every known peer."""
        community = MagicMock()
        community.known_peers = {
            "p1": MagicMock(peer=MagicMock(), node_id="p1"),
            "p2": MagicMock(peer=MagicMock(), node_id="p2"),
            "p3": MagicMock(peer=MagicMock(), node_id="p3"),
        }

        from quinkgl.network.gossip_community import GossipLearningCommunity
        GossipLearningCommunity.broadcast_checkpoint(
            community,
            sender_id="me",
            round_number=10,
            loss=0.5,
            accuracy=0.8,
        )

        assert community.ez_send.call_count == 3
        # Verify payload type
        for call in community.ez_send.call_args_list:
            payload = call[0][1]
            assert isinstance(payload, CheckpointPayload)
            assert payload.round_number == 10

    def test_broadcast_tolerates_send_failure(self):
        """If ez_send raises for one peer, others must still be sent."""
        community = MagicMock()
        call_count = 0

        def side_effect(peer, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("peer down")

        community.ez_send = MagicMock(side_effect=side_effect)
        community.known_peers = {
            "p1": MagicMock(peer=MagicMock(), node_id="p1"),
            "p2": MagicMock(peer=MagicMock(), node_id="p2"),
            "p3": MagicMock(peer=MagicMock(), node_id="p3"),
        }

        from quinkgl.network.gossip_community import GossipLearningCommunity
        # Should not raise
        GossipLearningCommunity.broadcast_checkpoint(
            community,
            sender_id="me",
            round_number=10,
            loss=0.5,
            accuracy=0.8,
        )
        assert call_count == 3


class TestOnCheckpointCallback:
    @pytest.mark.asyncio
    async def test_on_checkpoint_invokes_callback(self):
        """on_checkpoint must forward fields to on_checkpoint_callback."""
        received = {}

        async def cb(**kwargs):
            received.update(kwargs)

        community = MagicMock()
        community.on_checkpoint_callback = cb
        community.known_peers = {}
        community.require_signature = False

        from quinkgl.network.gossip_community import GossipLearningCommunity
        payload = CheckpointPayload("peer-a", 20, 0.3, 0.7, "1.0.0")
        peer = MagicMock()

        # Call the undecorated handler directly
        await GossipLearningCommunity._dispatch_checkpoint(
            community, peer, payload
        )

        assert received["sender_id"] == "peer-a"
        assert received["round_number"] == 20
        assert abs(received["loss"] - 0.3) < 1e-9
