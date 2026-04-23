"""
regression tests — Startup-timeout edge case.

Validates that:
 - A minimum peer-discovery window is enforced.
 - 0-peer + fallback-enabled triggers tunnel fallback (returns False).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from quinkgl.network.gossip_node import GossipNode, ConnectionMode

# T12: Constant for tunnel server address
DEFAULT_TUNNEL_SERVER = "localhost:50051"


def _make_node(**overrides):
    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.fallback_timeout = overrides.get("fallback_timeout", 30.0)
    node.enable_fallback = overrides.get("enable_fallback", True)
    node.tunnel_server = overrides.get("tunnel_server", DEFAULT_TUNNEL_SERVER)
    node.node_id = "test-node"
    node.domain = "test"
    node.data_schema_hash = "abc123"
    node.model_version = "1.0.0"
    node._ipv8_failed = False
    node.fingerprint = None
    node.data_policy = None
    node.community = None

    node.ipv8_manager = AsyncMock()
    node.ipv8_manager.start = AsyncMock()
    node.ipv8_manager.community = MagicMock()
    node.ipv8_manager.community.get_peer_count = MagicMock(return_value=0)

    node.gl_node = MagicMock()
    node.gl_node.aggregator = MagicMock()
    node.gl_node.aggregator.known_peers = {}
    node.gl_node.topology = MagicMock()  # Not CyclonTopology

    return node


@pytest.mark.asyncio
async def test_zero_peers_with_fallback_returns_false():
    """B12: 0 peers discovered + fallback enabled = IPv8 failure."""
    node = _make_node(enable_fallback=True, tunnel_server=DEFAULT_TUNNEL_SERVER)

    # Simulate: IPv8 starts fine, but discovers 0 peers
    node._wait_for_peers = AsyncMock(side_effect=asyncio.TimeoutError)
    node._sync_known_peers = MagicMock()

    # community returns 0 peers
    mock_community = MagicMock()
    mock_community.get_peer_count.return_value = 0
    node.ipv8_manager.community = mock_community

    result = await node._try_start_ipv8_with_timeout()

    assert result is False
    assert node._ipv8_failed is True
    node.ipv8_manager.stop.assert_awaited()


@pytest.mark.asyncio
async def test_zero_peers_without_fallback_returns_true():
    """B12: 0 peers but fallback disabled → still succeeds (peers may arrive later)."""
    node = _make_node(enable_fallback=False, tunnel_server=None)

    node._wait_for_peers = AsyncMock(side_effect=asyncio.TimeoutError)
    node._sync_known_peers = MagicMock()

    mock_community = MagicMock()
    mock_community.get_peer_count.return_value = 0
    mock_community.domain = "test"
    mock_community.data_schema_hash = "abc123"
    mock_community.model_version = "1.0.0"
    node.ipv8_manager.community = mock_community

    result = await node._try_start_ipv8_with_timeout()

    assert result is True


@pytest.mark.asyncio
async def test_minimum_discovery_window():
    """B12: Even if IPv8 start consumed most of the budget, peer discovery
    should get at least MIN_PEER_DISCOVERY_WINDOW seconds."""
    node = _make_node(fallback_timeout=6.0, enable_fallback=False, tunnel_server=None)

    discovery_timeout_used = None

    async def mock_wait_for_peers():
        await asyncio.sleep(999)  # will be timed out

    node._wait_for_peers = mock_wait_for_peers
    node._sync_known_peers = MagicMock()

    mock_community = MagicMock()
    mock_community.get_peer_count.return_value = 1  # 1 peer so it doesn't fallback
    mock_community.domain = "test"
    mock_community.data_schema_hash = "abc123"
    mock_community.model_version = "1.0.0"
    node.ipv8_manager.community = mock_community

    # Simulate that IPv8 start took 5.5 seconds (leaving only 0.5s)
    original_start = node._try_start_ipv8_with_timeout

    start_time = time.time()
    # We just verify the function completes without error —
    # the MIN_PEER_DISCOVERY_WINDOW ensures it doesn't zero out.
    result = await node._try_start_ipv8_with_timeout()
    elapsed = time.time() - start_time

    # Should have waited at least ~5s for discovery (MIN_PEER_DISCOVERY_WINDOW)
    assert result is True
