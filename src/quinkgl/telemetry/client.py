import asyncio
from collections import deque
from copy import deepcopy
import logging
import os
from datetime import datetime
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple

import httpx

from quinkgl.observability.events import RuntimeEvent
from quinkgl.telemetry.api import (
    DEFAULT_TELEMETRY_AUTH_HEADER,
    TELEMETRY_AUTH_HEADER_ENV,
    TELEMETRY_AUTH_SECRET_ENV,
)

logger = logging.getLogger(__name__)

# T-OBS-10: Module-level shared httpx.AsyncClient for connection pooling
_module_http_client: Optional[httpx.AsyncClient] = None
_module_http_client_lock = asyncio.Lock()


async def get_module_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create the module-level shared httpx.AsyncClient."""
    global _module_http_client
    async with _module_http_client_lock:
        if _module_http_client is None or _module_http_client.is_closed:
            _module_http_client = httpx.AsyncClient(timeout=timeout)
        return _module_http_client


async def close_module_http_client() -> None:
    """Close the module-level shared httpx.AsyncClient."""
    global _module_http_client
    async with _module_http_client_lock:
        if _module_http_client is not None and not _module_http_client.is_closed:
            await _module_http_client.aclose()
        _module_http_client = None


JsonDict = Dict[str, Any]
AsyncSink = Callable[[JsonDict], Awaitable[None]]
RuntimeEventSink = Callable[[str, JsonDict], None]


DEFAULT_TELEMETRY_MAX_PENDING_ITEMS = 256
DEFAULT_TELEMETRY_MAX_DELIVERY_ATTEMPTS = 3
DEFAULT_TELEMETRY_RETRY_INITIAL_DELAY = 0.5
DEFAULT_TELEMETRY_RETRY_MAX_DELAY = 5.0


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
        auth_secret: Optional[str] = None,
        auth_header_name: Optional[str] = None,
        max_pending_items: int = DEFAULT_TELEMETRY_MAX_PENDING_ITEMS,
        max_delivery_attempts: int = DEFAULT_TELEMETRY_MAX_DELIVERY_ATTEMPTS,
        retry_initial_delay: float = DEFAULT_TELEMETRY_RETRY_INITIAL_DELAY,
        retry_max_delay: float = DEFAULT_TELEMETRY_RETRY_MAX_DELAY,
        runtime_event_sink: Optional[RuntimeEventSink] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self._event_sink = event_sink
        self._heartbeat_sink = heartbeat_sink
        self._status_provider: Optional[Callable[[], JsonDict]] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._delivery_lock = asyncio.Lock()
        self._pending_deliveries: Deque[Tuple[str, JsonDict]] = deque()
        self.max_pending_items = max_pending_items
        self.max_delivery_attempts = max_delivery_attempts
        self.retry_initial_delay = retry_initial_delay
        self.retry_max_delay = retry_max_delay
        self._runtime_event_sink = runtime_event_sink
        self._delivery_failed = False
        self._consecutive_failures = 0
        self._failure_window_started_at: Optional[datetime] = None
        self.auth_secret = auth_secret if auth_secret is not None else os.getenv(TELEMETRY_AUTH_SECRET_ENV)
        self.auth_header_name = (
            auth_header_name
            or os.getenv(TELEMETRY_AUTH_HEADER_ENV)
            or DEFAULT_TELEMETRY_AUTH_HEADER
        )

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

    def _request_headers(self) -> JsonDict:
        if not self.auth_secret:
            return {}
        return {self.auth_header_name: self.auth_secret}

    def bind_runtime_event_sink(self, sink: RuntimeEventSink) -> None:
        self._runtime_event_sink = sink

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _spawn_task(self, coro) -> Optional[asyncio.Task]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return self._track_task(loop.create_task(coro))

    async def _get_http_client(self) -> httpx.AsyncClient:
        # T-OBS-10: Use module-level shared httpx.AsyncClient
        return await get_module_http_client(self.timeout)

    async def _post_json(self, path: str, payload: JsonDict) -> None:
        client = await self._get_http_client()
        response = await client.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._request_headers(),
        )
        response.raise_for_status()

    async def _deliver_serialized(self, kind: str, payload: JsonDict) -> None:
        if kind == "event":
            if self._event_sink:
                await self._event_sink(deepcopy(payload))
            else:
                await self._post_json("/api/telemetry/events", payload)
            return

        if self._heartbeat_sink:
            await self._heartbeat_sink(deepcopy(payload))
        else:
            await self._post_json("/api/telemetry/heartbeats", payload)

    def _emit_runtime_event(self, event_type: str, payload: JsonDict) -> None:
        if not self._runtime_event_sink:
            return
        try:
            self._runtime_event_sink(event_type, payload)
        except Exception as exc:
            logger.debug("Telemetry runtime event emission failed: %s", exc)

    def _record_delivery_success(self) -> None:
        self._delivery_failed = False
        self._consecutive_failures = 0
        self._failure_window_started_at = None

    def _record_delivery_failure(self, kind: str, exc: Exception) -> None:
        if self._failure_window_started_at is None:
            self._failure_window_started_at = datetime.now()
        self._consecutive_failures += 1
        if self._delivery_failed:
            self._maybe_emit_delivery_failed(kind, exc)
            return
        self._delivery_failed = True
        self._emit_runtime_event(
            "telemetry.disconnected",
            {
                "base_url": self.base_url,
                "kind": kind,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "consecutive_failures": self._consecutive_failures,
                "pending_queue_size": len(self._pending_deliveries),
            },
        )
        self._maybe_emit_delivery_failed(kind, exc)

    def _maybe_emit_delivery_failed(self, kind: str, exc: Exception) -> None:
        if self._consecutive_failures < 2:
            return
        if self._consecutive_failures != 2 and self._consecutive_failures % 5 != 0:
            return
        window_started_at = self._failure_window_started_at or datetime.now()
        window_seconds = max((datetime.now() - window_started_at).total_seconds(), 1e-6)
        failure_rate_per_minute = self._consecutive_failures * 60.0 / window_seconds
        logger.warning(
            "Telemetry %s delivery repeatedly failing for %s: %s (%s failures, %.2f/min)",
            kind,
            self.base_url,
            exc,
            self._consecutive_failures,
            failure_rate_per_minute,
        )
        self._emit_runtime_event(
            "telemetry.delivery_failed",
            {
                "base_url": self.base_url,
                "kind": kind,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "consecutive_failures": self._consecutive_failures,
                "pending_queue_size": len(self._pending_deliveries),
                "failure_window_seconds": round(window_seconds, 6),
                "failure_rate_per_minute": round(failure_rate_per_minute, 4),
            },
        )

    def _enqueue_delivery(self, kind: str, payload: JsonDict) -> None:
        if self.max_pending_items <= 0:
            return
        if len(self._pending_deliveries) >= self.max_pending_items:
            dropped = self._pending_deliveries.popleft()
            # T-OBS-09: Emit telemetry.events_dropped when queue overflows
            self._emit_runtime_event(
                "telemetry.events_dropped",
                {
                    "base_url": self.base_url,
                    "dropped_kind": dropped[0],
                    "queue_size": len(self._pending_deliveries),
                    "max_pending_items": self.max_pending_items,
                },
            )
        self._pending_deliveries.append((kind, deepcopy(payload)))

    async def _deliver_with_retry(self, kind: str, payload: JsonDict) -> tuple[bool, Optional[Exception]]:
        delay = max(0.0, self.retry_initial_delay)
        last_exc: Optional[Exception] = None
        attempts = max(1, self.max_delivery_attempts)

        for attempt in range(attempts):
            try:
                await self._deliver_serialized(kind, payload)
                self._record_delivery_success()
                return True, None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(delay)
                if self.retry_max_delay > 0:
                    delay = min(max(delay * 2, 0.0), self.retry_max_delay)

        return False, last_exc

    async def _flush_pending_deliveries_locked(self) -> bool:
        while self._pending_deliveries:
            kind, payload = self._pending_deliveries[0]
            success, exc = await self._deliver_with_retry(kind, payload)
            if not success:
                if exc is not None:
                    self._record_delivery_failure(kind, exc)
                return False
            self._pending_deliveries.popleft()
        return True

    async def _send_serialized(self, kind: str, payload: JsonDict) -> None:
        async with self._delivery_lock:
            flushed = await self._flush_pending_deliveries_locked()
            if not flushed:
                self._enqueue_delivery(kind, payload)
                return
            success, exc = await self._deliver_with_retry(kind, payload)
            if success:
                return
            self._enqueue_delivery(kind, payload)
            if exc is not None:
                self._record_delivery_failure(kind, exc)

    async def send_event(self, event: RuntimeEvent) -> None:
        payload = self._serialize_event(event)
        try:
            await self._send_serialized("event", payload)
        except Exception as exc:
            logger.debug("Telemetry event delivery failed: %s", exc)

    async def send_heartbeat(self, snapshot: JsonDict) -> None:
        payload = self._serialize_heartbeat(snapshot)
        try:
            await self._send_serialized("heartbeat", payload)
        except Exception as exc:
            logger.debug("Telemetry heartbeat delivery failed: %s", exc)

    def handle(self, event: RuntimeEvent) -> None:
        if event.event_type in {
            "telemetry.disconnected",
            "telemetry.delivery_failed",
            "telemetry.status_provider_warning",
        }:
            return
        self._spawn_task(self.send_event(event))

    def start(self, status_provider: Callable[[], JsonDict]) -> None:
        self._status_provider = status_provider
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_task = self._spawn_task(self._heartbeat_loop())

    def pause(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    async def stop(self) -> None:
        self._status_provider = None
        self.pause()
        current_task = asyncio.current_task()
        pending_tasks = [
            task for task in list(self._background_tasks)
            if task != current_task and not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._heartbeat_task = None

        # T-OBS-10: Close module-level httpx client
        await close_module_http_client()

    async def _heartbeat_loop(self) -> None:
        try:
            while self._status_provider is not None:
                try:
                    snapshot = self._status_provider()
                except Exception as exc:
                    logger.exception("Failed to get status snapshot in heartbeat loop")
                    # Continue with empty snapshot to avoid breaking the heartbeat
                    snapshot = {}
                    logger.warning("Telemetry status provider failed: %s", exc)
                    self._emit_runtime_event(
                        "telemetry.status_provider_warning",
                        {
                            "base_url": self.base_url,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    await asyncio.sleep(self.heartbeat_interval)
                    continue
                await self.send_heartbeat(snapshot)
                await asyncio.sleep(self.heartbeat_interval)
        except asyncio.CancelledError:
            raise
