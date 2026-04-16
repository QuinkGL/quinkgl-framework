"""Regression tests: tunnel-mode receive path.

Covers:
- Valid tunnel MODEL_UPDATE reaches aggregator pending_updates.
- Missing required fields are rejected with observability event.
- Domain/schema mismatch is rejected.
- Oversized weights are rejected.
- Malformed hex weights are rejected.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from quinkgl.network.model_serializer import serialize_model


# ── Helpers ──────────────────────────────────────────────────────────

def _make_gossip_node(domain="test", schema="abc123"):
    """Build a minimal GossipNode-like object with the fields
    _setup_tunnel_callbacks needs, without real IPv8 / gRPC."""

    # Minimal LearningNode stub
    aggregator = MagicMock()
    aggregator.event_emitter = MagicMock()
    aggregator.running = True
    aggregator.current_round = 5
    aggregator.stale_round_tolerance = 2
    aggregator.pending_updates = []

    async def _handle_network_message(msg):
        from quinkgl.aggregation.base import ModelUpdate
        update = ModelUpdate(
            peer_id=msg.sender_id,
            weights=msg.weights,
            sample_count=msg.sample_count,
            round_number=msg.round_number,
        )
        aggregator.pending_updates.append(update)

    gl_node = SimpleNamespace(
        aggregator=aggregator,
        _handle_network_message=_handle_network_message,
    )

    node = SimpleNamespace(
        node_id="node-1",
        domain=domain,
        data_schema_hash=schema,
        gl_node=gl_node,
        _on_tunnel_model_update=None,
        _on_tunnel_peer_discovered=None,
        tunnel_client=SimpleNamespace(
            on_chat_message=None,
            on_peer_list=None,
        ),
        _tunnel_peers={},
    )
    return node


def _setup_callbacks(node):
    """Invoke the real _setup_tunnel_callbacks logic on our stub."""
    from quinkgl.network.gossip_node import GossipNode
    # Bind the real method to our stub
    GossipNode._setup_tunnel_callbacks(node)


def _valid_payload(domain="test", schema="abc123"):
    """Build a valid tunnel MODEL_UPDATE dict."""
    weights = {"w": np.array([1.0, 2.0, 3.0])}
    weights_bytes = serialize_model(weights)
    return {
        "type": "MODEL_UPDATE",
        "sender_id": "peer-a",
        "round_number": 5,
        "weights": weights_bytes.hex(),
        "domain": domain,
        "data_schema_hash": schema,
        "sample_count": 16,
        "loss": 0.1,
        "accuracy": 0.9,
    }


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_tunnel_update_reaches_handler():
    """A valid MODEL_UPDATE must invoke _on_tunnel_model_update and not be None."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    assert node._on_tunnel_model_update is not None

    data = _valid_payload()
    await node._on_tunnel_model_update(data)

    # The update must have reached the aggregator via _handle_network_message
    assert len(node.gl_node.aggregator.pending_updates) == 1
    update = node.gl_node.aggregator.pending_updates[0]
    assert update.peer_id == "peer-a"
    assert update.round_number == 5


@pytest.mark.asyncio
async def test_missing_fields_emits_dropped_event():
    """Missing sender_id must emit tunnel_payload_dropped."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = {"type": "MODEL_UPDATE", "round_number": 5, "weights": "aabb"}
    # Missing sender_id
    await node._on_tunnel_model_update(data)

    node.gl_node.aggregator.event_emitter.emit.assert_called()
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert call_args[0][0] == "tunnel_payload_dropped"
    assert "missing fields" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_domain_mismatch_rejected():
    """A payload with wrong domain must be rejected."""
    node = _make_gossip_node(domain="health", schema="abc123")
    _setup_callbacks(node)

    data = _valid_payload(domain="finance", schema="abc123")
    await node._on_tunnel_model_update(data)

    node.gl_node.aggregator.event_emitter.emit.assert_called()
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "domain/schema mismatch" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_oversized_weights_rejected():
    """Hex payload exceeding MAX_INCOMING_MESSAGE_SIZE must be rejected."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    # Create a huge hex string (> 150MB when decoded)
    # We'll just make it bigger than the constant
    from quinkgl.network.gossip_community import MAX_INCOMING_MESSAGE_SIZE
    data["weights"] = "aa" * (MAX_INCOMING_MESSAGE_SIZE + 1)

    await node._on_tunnel_model_update(data)

    node.gl_node.aggregator.event_emitter.emit.assert_called()
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "oversized" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_malformed_hex_rejected():
    """Invalid hex in weights must emit a deserialization error event."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    data["weights"] = "not_valid_hex"

    await node._on_tunnel_model_update(data)

    node.gl_node.aggregator.event_emitter.emit.assert_called()
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "deserialization error" in call_args[0][1]["reason"]
