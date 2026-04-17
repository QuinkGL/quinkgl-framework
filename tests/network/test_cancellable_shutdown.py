"""
regression tests — Cancellable shutdown.

Validates that:
 - _run_task is set during run_continuous.
 - shutdown() cancels and awaits the run task.
 - shutdown() is safe even when no task is running.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quinkgl.network.gossip_node import GossipNode, ConnectionMode


def _make_node():
    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.running = True
    node.connection_mode = ConnectionMode.IPV8_P2P
    node.community = MagicMock()
    node._run_task = None
    node._tunnel_connected = False
    node.tunnel_client = None
    node._telemetry_client = None
    node._start_time = 1.0
    node._ipv8_failed = False
    node.node_id = "test"

    node.gl_node = MagicMock()
    node.gl_node.aggregator = MagicMock()
    node.gl_node.aggregator.event_emitter = None
    node.gl_node.current_round = 0
    node.gl_node.topology = MagicMock()  # Not CyclonTopology

    node.ipv8_manager = AsyncMock()

    return node


@pytest.mark.asyncio
async def test_run_task_is_tracked():
    """B13: _run_task should be set to the current task during run_continuous."""
    node = _make_node()

    task_seen = None

    async def mock_run_continuous(**kw):
        nonlocal task_seen
        task_seen = node._run_task

    node.gl_node.run_continuous = mock_run_continuous
    node._sync_known_peers = MagicMock()

    await node.run_continuous(data=MagicMock())

    assert task_seen is not None
    # After completion, _run_task should be cleared
    assert node._run_task is None


@pytest.mark.asyncio
async def test_shutdown_cancels_run_task():
    """B13: shutdown() should cancel the gossip-loop task."""
    node = _make_node()

    cancelled = asyncio.Event()

    async def long_running(**kw):
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    node.gl_node.run_continuous = long_running
    node._sync_known_peers = MagicMock()

    # Start run_continuous in background
    run_task = asyncio.ensure_future(node.run_continuous(data=MagicMock()))

    # Give it a moment to start
    await asyncio.sleep(0.05)

    assert node._run_task is not None

    # Shutdown should cancel the run task
    await node.shutdown()

    assert cancelled.is_set()
    assert run_task.done()


@pytest.mark.asyncio
async def test_shutdown_safe_without_run_task():
    """B13: shutdown() should be safe when no run task exists."""
    node = _make_node()
    node._run_task = None

    # Should not raise
    await node.shutdown()
