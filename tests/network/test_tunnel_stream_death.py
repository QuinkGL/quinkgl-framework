"""
regression tests — Detect tunnel stream death and surface it.

Validates that:
 - TunnelClient has an on_disconnected callback.
 - message_queue is bounded (maxsize).
 - on_disconnected is called when the stream dies.
 - _send_model_update_via_tunnel raises ConnectionError when tunnel is down.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# T12: Constant for tunnel server address
DEFAULT_TUNNEL_SERVER = "localhost:50051"


# ---------------------------------------------------------------------------
# B9-1: message_queue has maxsize
# ---------------------------------------------------------------------------

def test_message_queue_bounded():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server=DEFAULT_TUNNEL_SERVER, node_id="n1")
    assert client.message_queue.maxsize > 0, "message_queue must have a maxsize"
    assert client.message_queue.maxsize == 1024


# ---------------------------------------------------------------------------
# B9-2: on_disconnected callback attribute exists
# ---------------------------------------------------------------------------

def test_on_disconnected_attribute():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server=DEFAULT_TUNNEL_SERVER, node_id="n1")
    assert hasattr(client, "on_disconnected")
    assert client.on_disconnected is None  # default


# ---------------------------------------------------------------------------
# B9-3: on_disconnected fires when _tunnel_stream catches an exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_disconnected_fires_on_stream_error():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server=DEFAULT_TUNNEL_SERVER, node_id="n1")
    # Disable reconnect so the test doesn't spawn background tasks
    client._reconnect_enabled = False

    # Mock stub so RegisterTunnel raises immediately
    client.stub = MagicMock()
    client.stub.RegisterTunnel = MagicMock(side_effect=RuntimeError("stream died"))
    client.running = True

    disconnected_called = asyncio.Event()

    async def on_disconnect():
        disconnected_called.set()

    client.on_disconnected = on_disconnect

    # Put a dummy message so the generator can yield something
    await client.message_queue.put(MagicMock())

    # Run the stream — it should fail, call on_disconnected
    await client._tunnel_stream()

    assert client.running is False
    assert disconnected_called.is_set(), "on_disconnected should have been called"


# ---------------------------------------------------------------------------
# B9-4: _send_model_update_via_tunnel raises ConnectionError when disconnected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_via_tunnel_raises_when_disconnected():
    from quinkgl.network.gossip_node import GossipNode

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.tunnel_client = MagicMock()
    node._tunnel_connected = False

    with pytest.raises(ConnectionError, match="tunnel not connected"):
        await node._send_model_update_via_tunnel("peer-1", MagicMock())


# ---------------------------------------------------------------------------
# B9-5: _send_model_update_via_tunnel raises when tunnel_client is None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_via_tunnel_raises_when_no_client():
    from quinkgl.network.gossip_node import GossipNode

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.tunnel_client = None
    node._tunnel_connected = False

    with pytest.raises(ConnectionError):
        await node._send_model_update_via_tunnel("peer-1", MagicMock())
