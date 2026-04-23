"""Regression tests: tunnel-mode receive path.

Covers:
- Valid tunnel MODEL_UPDATE reaches aggregator pending_updates.
- Missing required fields are rejected with observability event.
- Domain/schema mismatch is rejected.
- Oversized weights are rejected.
- Malformed hex weights are rejected.
- Stream sender mismatch is rejected.
- Replay is rejected.
- Future round beyond MAX_ROUND_SKIP is rejected.
- Stream-binding: payload without _tunnel_sender_id is rejected.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# T10: Skip test if ipv8 is not installed
pytest.importorskip("ipv8")
from ipv8.keyvault.crypto import default_eccrypto

from quinkgl.network.model_serializer import serialize_model
from quinkgl.network.gossip_node import _tunnel_sign
from quinkgl.network.gossip_community import MAX_ROUND_SKIP


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
        _tunnel_last_seen_round={},
    )
    return node


def _setup_callbacks(node):
    """Invoke the real _setup_tunnel_callbacks logic on our stub."""
    from quinkgl.network.gossip_node import GossipNode
    # Bind the real method to our stub
    GossipNode._setup_tunnel_callbacks(node)


def _valid_payload(domain="test", schema="abc123", round_number=5):
    """Build a valid tunnel MODEL_UPDATE dict."""
    weights = {"w": np.array([1.0, 2.0, 3.0])}
    weights_bytes = serialize_model(weights)
    key = default_eccrypto.generate_key("medium")
    timestamp = 1_700_000_000
    signature = _tunnel_sign(key, "peer-a", domain, round_number, schema, 16, 0.1, 0.9, timestamp, weights_bytes)
    return {
        "type": "MODEL_UPDATE",
        "sender_id": "peer-a",
        "round_number": round_number,
        "weights": weights_bytes.hex(),
        "signature": signature.hex(),
        "signer_public_key": key.pub().key_to_bin().hex(),
        "domain": domain,
        "data_schema_hash": schema,
        "sample_count": 16,
        "loss": 0.1,
        "accuracy": 0.9,
        "timestamp": timestamp,
    }


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_tunnel_update_reaches_handler():
    """A valid MODEL_UPDATE must invoke _on_tunnel_model_update and not be None."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    assert node._on_tunnel_model_update is not None

    data = _valid_payload()
    data["_tunnel_sender_id"] = "peer-a"
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
    data["_tunnel_sender_id"] = "peer-a"
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
    data["_tunnel_sender_id"] = "peer-a"
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
    data["_tunnel_sender_id"] = "peer-a"
    data["weights"] = "not_valid_hex"

    await node._on_tunnel_model_update(data)

    node.gl_node.aggregator.event_emitter.emit.assert_called()
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "deserialization error" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_stream_sender_mismatch_rejected():
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    data["_tunnel_sender_id"] = "peer-b"
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 0
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "sender mismatch" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_replayed_round_rejected():
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    data["_tunnel_sender_id"] = "peer-a"
    await node._on_tunnel_model_update(data)
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 1
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "replayed round" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_tampered_sample_count_rejected_as_invalid_signature():
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    data["_tunnel_sender_id"] = "peer-a"
    data["sample_count"] = 999
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 0
    call_args = node.gl_node.aggregator.event_emitter.emit.call_args
    assert "invalid signature" in call_args[0][1]["reason"]


@pytest.mark.asyncio
async def test_future_round_rejected():
    """A round_number exceeding current_round + MAX_ROUND_SKIP must be rejected."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    future_round = node.gl_node.aggregator.current_round + MAX_ROUND_SKIP + 1
    data = _valid_payload(round_number=future_round)
    data["_tunnel_sender_id"] = "peer-a"
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 0
    calls = node.gl_node.aggregator.event_emitter.emit.call_args_list
    # emit_drop emits security_event as event type, then tunnel_payload_dropped
    security_calls = [c for c in calls if c[0][0] == "security.future_round_rejected"]
    assert len(security_calls) >= 1, f"Expected security.future_round_rejected event, got {[c[0][0] for c in calls]}"
    dropped_calls = [c for c in calls if c[0][0] == "tunnel_payload_dropped"]
    assert any("future round" in c[0][1]["reason"] for c in dropped_calls)


@pytest.mark.asyncio
async def test_round_at_max_skip_accepted():
    """A round_number exactly at current_round + MAX_ROUND_SKIP must be accepted."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    boundary_round = node.gl_node.aggregator.current_round + MAX_ROUND_SKIP
    data = _valid_payload(round_number=boundary_round)
    data["_tunnel_sender_id"] = "peer-a"
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 1
    assert node.gl_node.aggregator.pending_updates[0].round_number == boundary_round


@pytest.mark.asyncio
async def test_stream_binding_missing_sender_id_rejected():
    """A MODEL_UPDATE without _tunnel_sender_id must be rejected (stream-binding)."""
    node = _make_gossip_node()
    _setup_callbacks(node)

    data = _valid_payload()
    # No _tunnel_sender_id set — simulates a payload not bound to a gRPC stream
    await node._on_tunnel_model_update(data)

    assert len(node.gl_node.aggregator.pending_updates) == 0
    calls = node.gl_node.aggregator.event_emitter.emit.call_args_list
    dropped_calls = [c for c in calls if c[0][0] == "tunnel_payload_dropped"]
    assert any("stream sender missing" in c[0][1]["reason"] for c in dropped_calls)
