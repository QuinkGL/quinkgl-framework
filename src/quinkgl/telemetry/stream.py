from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, Dict, Set


STREAM_CLOSE_CODE_QUEUE_FULL = 1013


class TelemetryStreamHub:
    """Simple queue-based fan-out for live telemetry updates."""

    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    async def publish(self, message: Dict[str, Any]) -> None:
        dead = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    while True:
                        queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    queue.put_nowait(
                        {
                            "type": "stream_closed",
                            "code": STREAM_CLOSE_CODE_QUEUE_FULL,
                            "reason": "Telemetry subscriber queue overflow",
                        }
                    )
                dead.append(queue)
        for queue in dead:
            self._subscribers.discard(queue)

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
        with suppress(asyncio.QueueEmpty):
            while True:
                queue.get_nowait()
