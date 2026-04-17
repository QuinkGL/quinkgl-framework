"""
regression tests — Clean up half-started IPv8.

Validates that:
 - IPv8Manager.stop() is idempotent (safe to call multiple times or before start).
 - _try_start_ipv8_with_timeout cleans up on timeout.
 - _try_start_ipv8_with_timeout cleans up on exception.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# B8-1: IPv8Manager.stop() is idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ipv8_manager_stop_idempotent():
    """Calling stop() twice must not raise."""
    from quinkgl.network.ipv8_manager import IPv8Manager

    mgr = IPv8Manager(node_id="test-node", port=0)
    # Never started — stop should be safe
    await mgr.stop()
    assert mgr.running is False
    assert mgr.community is None

    # Second call — still safe
    await mgr.stop()
    assert mgr.running is False


# ---------------------------------------------------------------------------
# B8-2: IPv8Manager.stop() clears ipv8 reference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ipv8_manager_stop_clears_ipv8():
    from quinkgl.network.ipv8_manager import IPv8Manager

    mgr = IPv8Manager(node_id="test-node", port=0)
    # Simulate a partially-started state
    mock_ipv8 = AsyncMock()
    mgr.ipv8 = mock_ipv8
    mgr.running = True
    mgr.community = MagicMock()

    await mgr.stop()

    mock_ipv8.stop.assert_awaited_once()
    assert mgr.ipv8 is None
    assert mgr.running is False
    assert mgr.community is None


# ---------------------------------------------------------------------------
# B8-3: IPv8Manager.stop() handles exception in ipv8.stop() gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ipv8_manager_stop_handles_exception():
    from quinkgl.network.ipv8_manager import IPv8Manager

    mgr = IPv8Manager(node_id="test-node", port=0)
    mock_ipv8 = AsyncMock()
    mock_ipv8.stop.side_effect = RuntimeError("kaboom")
    mgr.ipv8 = mock_ipv8
    mgr.running = True

    # Should not raise
    await mgr.stop()
    assert mgr.running is False


# ---------------------------------------------------------------------------
# B8-4: _try_start_ipv8_with_timeout calls stop on timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_start_cleans_up_on_timeout():
    """When IPv8 start times out, ipv8_manager.stop() should be called."""
    from quinkgl.network.gossip_node import GossipNode
    from quinkgl.models.base import ModelWrapper, TrainingConfig

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    # Minimal wiring
    node.fallback_timeout = 0.01  # very short timeout
    node.ipv8_manager = AsyncMock()
    # Simulate slow start that exceeds timeout
    node.ipv8_manager.start = AsyncMock(side_effect=lambda **kw: asyncio.sleep(10))

    result = await node._try_start_ipv8_with_timeout()

    assert result is False
    node.ipv8_manager.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# B8-5: _try_start_ipv8_with_timeout calls stop on generic exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_start_cleans_up_on_exception():
    from quinkgl.network.gossip_node import GossipNode

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.fallback_timeout = 5.0
    node.ipv8_manager = AsyncMock()
    node.ipv8_manager.start = AsyncMock(return_value=None)
    # community property will trigger an AttributeError → lands in except
    node.ipv8_manager.community = None

    # This will hit the except Exception branch because community is None
    # and node.community = None will try to access .domain on None
    node._ipv8_failed = False

    result = await node._try_start_ipv8_with_timeout()

    assert result is False
    node.ipv8_manager.stop.assert_awaited()
