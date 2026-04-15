import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from quinkgl.observability.events import RuntimeEvent

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]
AsyncSink = Callable[[JsonDict], Awaitable[None]]


class TelemetryClient:
    """Forward runtime events and heartbeats to a central telemetry service."""

    def __init__(
        self,
        base_url: str,
        *,
        event_sink: Optional[AsyncSink] = None,
        heartbeat_sink: Optional[AsyncSink] = None,
        heartbeat_interval: float = 5.0,
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self._event_sink = event_sink
        self._heartbeat_sink = heartbeat_sink
        self._status_provider: Optional[Callable[[], JsonDict]] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    def _serialize_event(self, event: RuntimeEvent) -> JsonDict:
        return {
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "payload": dict(event.payload or {}),
        }

    def _serialize_heartbeat(self, snapshot: JsonDict) -> JsonDict:
        payload = dict(snapshot)
        payload["timestamp"] = datetime.now().isoformat()
        return payload

    async def _post_json(self, path: str, payload: JsonDict) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()

    async def send_event(self, event: RuntimeEvent) -> None:
        payload = self._serialize_event(event)
        try:
            if self._event_sink:
                await self._event_sink(payload)
            else:
                await self._post_json("/api/telemetry/events", payload)
        except Exception as exc:
            logger.debug("Telemetry event delivery failed: %s", exc)

    async def send_heartbeat(self, snapshot: JsonDict) -> None:
        payload = self._serialize_heartbeat(snapshot)
        try:
            if self._heartbeat_sink:
                await self._heartbeat_sink(payload)
            else:
                await self._post_json("/api/telemetry/heartbeats", payload)
        except Exception as exc:
            logger.debug("Telemetry heartbeat delivery failed: %s", exc)

    def handle(self, event: RuntimeEvent) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.send_event(event))

    def start(self, status_provider: Callable[[], JsonDict]) -> None:
        self._status_provider = status_provider
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())

    def pause(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    async def stop(self) -> None:
        self.pause()
        if self._heartbeat_task:
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        try:
            while self._status_provider is not None:
                await self.send_heartbeat(self._status_provider())
                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            raise
