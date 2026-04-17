"""
B10 regression tests — Tunnel reconnect with exponential backoff.

Validates that:
 - Reconnect constants are defined.
 - _reconnect_loop retries with backoff and stops after max attempts.
 - Successful reconnect stops the loop.
 - close() cancels pending reconnect task and disables future reconnects.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# B10-1: Reconnect constants exist
# ---------------------------------------------------------------------------

def test_reconnect_constants():
    from quinkgl.network.fallback.tunnel_client import (
        RECONNECT_INITIAL_DELAY,
        RECONNECT_MAX_DELAY,
        RECONNECT_BACKOFF_FACTOR,
        RECONNECT_MAX_ATTEMPTS,
    )
    assert RECONNECT_INITIAL_DELAY > 0
    assert RECONNECT_MAX_DELAY > RECONNECT_INITIAL_DELAY
    assert RECONNECT_BACKOFF_FACTOR > 1
    assert RECONNECT_MAX_ATTEMPTS >= 1


# ---------------------------------------------------------------------------
# B10-2: _reconnect_loop gives up after RECONNECT_MAX_ATTEMPTS failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconnect_gives_up_after_max_attempts():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server="localhost:50051", node_id="n1")

    attempt_count = 0

    async def failing_connect():
        nonlocal attempt_count
        attempt_count += 1
        raise ConnectionError("refused")

    client.connect = failing_connect

    # Patch sleep to skip delays
    with patch("quinkgl.network.fallback.tunnel_client.asyncio.sleep", new_callable=AsyncMock):
        # Patch module-level constants for fast test
        with patch("quinkgl.network.fallback.tunnel_client.RECONNECT_MAX_ATTEMPTS", 3), \
             patch("quinkgl.network.fallback.tunnel_client.RECONNECT_INITIAL_DELAY", 0.001):
            await client._reconnect_loop()

    assert attempt_count == 3


# ---------------------------------------------------------------------------
# B10-3: _reconnect_loop stops on successful reconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconnect_stops_on_success():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server="localhost:50051", node_id="n1")

    attempt_count = 0

    async def succeed_on_second():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ConnectionError("not yet")
        # Success on attempt 2
        client.running = True

    client.connect = succeed_on_second

    with patch("quinkgl.network.fallback.tunnel_client.asyncio.sleep", new_callable=AsyncMock):
        with patch("quinkgl.network.fallback.tunnel_client.RECONNECT_INITIAL_DELAY", 0.001):
            await client._reconnect_loop()

    assert attempt_count == 2
    assert client.running is True


# ---------------------------------------------------------------------------
# B10-4: close() disables reconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_disables_reconnect():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server="localhost:50051", node_id="n1")
    client.channel = AsyncMock()

    # Simulate a pending reconnect task
    async def dummy():
        await asyncio.sleep(100)

    client._reconnect_task = asyncio.ensure_future(dummy())

    await client.close()

    # Yield to the event loop so the task finishes cancellation
    await asyncio.sleep(0)

    assert client._reconnect_enabled is False
    assert client.running is False
    assert client._reconnect_task.done()


# ---------------------------------------------------------------------------
# B10-5: _reconnect_loop respects _reconnect_enabled=False (early exit)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reconnect_loop_exits_when_disabled():
    from quinkgl.network.fallback.tunnel_client import TunnelClient

    client = TunnelClient(tunnel_server="localhost:50051", node_id="n1")
    client._reconnect_enabled = False

    connect_called = False

    async def should_not_call():
        nonlocal connect_called
        connect_called = True

    client.connect = should_not_call

    with patch("quinkgl.network.fallback.tunnel_client.asyncio.sleep", new_callable=AsyncMock):
        await client._reconnect_loop()

    assert not connect_called
