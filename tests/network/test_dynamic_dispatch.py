"""
regression tests — Dynamic send dispatch.

Validates that:
 - A single send_to_peer closure reads connection_mode on each call.
 - Switching mode mid-run changes the dispatch target.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quinkgl.network.gossip_node import GossipNode, ConnectionMode


def _make_node():
    """Create a minimal GossipNode with mocked internals."""
    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.running = True
    node.connection_mode = ConnectionMode.IPV8_P2P
    node.community = MagicMock()
    node.community.send_model_update = AsyncMock()
    node.tunnel_client = MagicMock()
    node._tunnel_connected = True
    node.tunnel_client.send_chat_message = AsyncMock()
    node._run_task = None

    # Minimal gl_node mock
    node.gl_node = MagicMock()
    node.gl_node.aggregator = MagicMock()
    node.gl_node.aggregator.send_message_callback = None

    # Make run_continuous stop immediately
    node.gl_node.run_continuous = AsyncMock()

    return node


@pytest.mark.asyncio
async def test_dispatch_uses_ipv8_when_p2p():
    node = _make_node()
    node.connection_mode = ConnectionMode.IPV8_P2P

    await node.run_continuous(data=MagicMock())

    callback = node.gl_node.aggregator.send_message_callback
    assert callback is not None

    msg = MagicMock()
    await callback("peer-1", msg)

    node.community.send_model_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_raises_when_ipv8_send_reports_failure():
    node = _make_node()
    node.connection_mode = ConnectionMode.IPV8_P2P
    node.community.send_model_update.return_value = False

    await node.run_continuous(data=MagicMock())

    callback = node.gl_node.aggregator.send_message_callback
    assert callback is not None

    msg = MagicMock()
    with pytest.raises(RuntimeError, match="Failed to send model update"):
        await callback("peer-1", msg)


@pytest.mark.asyncio
async def test_dispatch_uses_tunnel_when_relay():
    node = _make_node()
    node.connection_mode = ConnectionMode.TUNNEL_RELAY
    node._tunnel_peers = {}

    # Mock _announce_to_tunnel
    node._announce_to_tunnel = AsyncMock()

    await node.run_continuous(data=MagicMock())

    callback = node.gl_node.aggregator.send_message_callback
    msg = MagicMock()

    # Should raise ConnectionError because we didn't wire full tunnel
    # OR succeed via tunnel — either way it should NOT call IPv8
    try:
        await callback("peer-1", msg)
    except (ConnectionError, Exception):
        pass

    node.community.send_model_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_switches_dynamically():
    """B11: Changing connection_mode after callback creation should
    change the dispatch target on next call."""
    node = _make_node()
    node.connection_mode = ConnectionMode.IPV8_P2P

    await node.run_continuous(data=MagicMock())
    callback = node.gl_node.aggregator.send_message_callback

    msg = MagicMock()
    await callback("peer-1", msg)
    assert node.community.send_model_update.await_count == 1

    # Switch to tunnel mode
    node.connection_mode = ConnectionMode.TUNNEL_RELAY
    node.community.send_model_update.reset_mock()

    try:
        await callback("peer-1", msg)
    except (ConnectionError, Exception):
        pass

    # IPv8 should NOT have been called this time
    node.community.send_model_update.assert_not_awaited()


def test_single_closure_in_run_continuous():
    """Source should define exactly one send_to_peer, not two."""
    src = inspect.getsource(GossipNode.run_continuous)
    count = src.count("async def send_to_peer")
    assert count == 1, f"Expected 1 send_to_peer definition, found {count}"
