from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import json
import logging
import os
from time import monotonic

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from quinkgl.telemetry.api import (
    DEFAULT_TELEMETRY_AUTH_HEADER,
    TELEMETRY_AUTH_HEADER_ENV,
    TELEMETRY_CORS_ALLOW_ORIGINS_ENV,
    TELEMETRY_AUTH_SECRET_ENV,
    TelemetryConnectionStatusIngest,
    TelemetryEventIngest,
    TelemetryHeartbeatIngest,
)
from quinkgl.telemetry.store import TelemetryStore
from quinkgl.telemetry.stream import STREAM_CLOSE_CODE_QUEUE_FULL, TelemetryStreamHub


DEFAULT_TELEMETRY_MAX_REQUEST_BYTES = 64 * 1024
DEFAULT_TELEMETRY_RATE_LIMIT_MAX_REQUESTS = 120
DEFAULT_TELEMETRY_RATE_LIMIT_WINDOW_SECONDS = 60.0
TELEMETRY_MAX_REQUEST_BYTES_ENV = "QUINKGL_TELEMETRY_MAX_REQUEST_BYTES"
TELEMETRY_RATE_LIMIT_MAX_REQUESTS_ENV = "QUINKGL_TELEMETRY_RATE_LIMIT_MAX_REQUESTS"
TELEMETRY_RATE_LIMIT_WINDOW_SECONDS_ENV = "QUINKGL_TELEMETRY_RATE_LIMIT_WINDOW_SECONDS"
INGEST_PATHS = {"/api/telemetry/events", "/api/telemetry/heartbeats", "/api/telemetry/connection-status"}


logger = logging.getLogger(__name__)


def _resolved_cors_allow_origins(explicit: list[str] | None) -> list[str]:
    """Browser dashboards need CORS; an empty env would otherwise deny every origin."""
    if explicit is not None:
        return list(explicit)
    raw = os.getenv(TELEMETRY_CORS_ALLOW_ORIGINS_ENV, "")
    parsed = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if parsed:
        return parsed
    # Hosted SPAs (e.g. Vercel) and local Vite without extra server configuration.
    return ["*"]


def _rate_limit_bucket_key(client_host: str | None, forwarded_for: str | None) -> str:
    """When uvicorn sits behind nginx on loopback, rate-limit per real client."""
    host = (client_host or "").strip() or "unknown"
    if host in ("127.0.0.1", "::1") and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return host


def create_telemetry_app(
    store: TelemetryStore | None = None,
    stream_hub: TelemetryStreamHub | None = None,
    *,
    auth_secret: str | None = None,
    auth_header_name: str | None = None,
    cors_allow_origins: list[str] | None = None,
    max_request_bytes: int | None = None,
    rate_limit_max_requests: int | None = None,
    rate_limit_window_seconds: float | None = None,
) -> FastAPI:
    store = store or TelemetryStore()
    stream_hub = stream_hub or TelemetryStreamHub()
    auth_secret = auth_secret if auth_secret is not None else os.getenv(TELEMETRY_AUTH_SECRET_ENV)
    auth_header_name = auth_header_name or os.getenv(TELEMETRY_AUTH_HEADER_ENV) or DEFAULT_TELEMETRY_AUTH_HEADER
    cors_allow_origins = _resolved_cors_allow_origins(cors_allow_origins)
    max_request_bytes = int(
        max_request_bytes
        if max_request_bytes is not None
        else os.getenv(TELEMETRY_MAX_REQUEST_BYTES_ENV, DEFAULT_TELEMETRY_MAX_REQUEST_BYTES)
    )
    rate_limit_max_requests = int(
        rate_limit_max_requests
        if rate_limit_max_requests is not None
        else os.getenv(TELEMETRY_RATE_LIMIT_MAX_REQUESTS_ENV, DEFAULT_TELEMETRY_RATE_LIMIT_MAX_REQUESTS)
    )
    rate_limit_window_seconds = float(
        rate_limit_window_seconds
        if rate_limit_window_seconds is not None
        else os.getenv(TELEMETRY_RATE_LIMIT_WINDOW_SECONDS_ENV, DEFAULT_TELEMETRY_RATE_LIMIT_WINDOW_SECONDS)
    )
    rate_limit_state = defaultdict(deque)
    app = FastAPI(title="QuinkGL Telemetry")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.telemetry_store = store
    app.state.telemetry_stream_hub = stream_hub
    app.state.telemetry_auth_secret = auth_secret
    app.state.telemetry_auth_header_name = auth_header_name
    app.state.telemetry_cors_allow_origins = list(cors_allow_origins)
    app.state.telemetry_max_request_bytes = max_request_bytes
    app.state.telemetry_rate_limit_max_requests = rate_limit_max_requests
    app.state.telemetry_rate_limit_window_seconds = rate_limit_window_seconds

    def require_ingest_auth(header_value: str | None) -> None:
        if not auth_secret:
            return
        if header_value != auth_secret:
            raise HTTPException(status_code=401, detail="Invalid telemetry auth secret")

    def check_rate_limit(client_host: str | None) -> None:
        if rate_limit_max_requests <= 0:
            return
        host = client_host or "unknown"
        now = monotonic()
        bucket = rate_limit_state[host]
        cutoff = now - rate_limit_window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= rate_limit_max_requests:
            raise HTTPException(status_code=429, detail="Telemetry rate limit exceeded")
        bucket.append(now)

    @app.middleware("http")
    async def protect_ingest_surface(request: Request, call_next):
        if request.url.path not in INGEST_PATHS:
            return await call_next(request)
        body = await request.body()
        if len(body) > max_request_bytes:
            return JSONResponse(status_code=413, content={"detail": "Telemetry request too large"})
        try:
            check_rate_limit(
                _rate_limit_bucket_key(
                    request.client.host if request.client else None,
                    request.headers.get("x-forwarded-for"),
                )
            )
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        return await call_next(Request(request.scope, receive))

    @app.get("/api/session")
    async def get_session():
        return store.get_session()

    @app.get("/api/nodes")
    async def get_nodes():
        return store.get_nodes()

    @app.get("/api/nodes/{node_id}")
    async def get_node(node_id: str):
        node = store.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return node

    @app.get("/api/nodes/{node_id}/events")
    async def get_node_events(node_id: str):
        return store.get_node_events(node_id)

    @app.get("/api/events")
    async def get_events():
        return store.get_events()

    @app.get("/api/nodes/{node_id}/rounds")
    async def get_node_rounds(node_id: str):
        return store.get_node_rounds(node_id)

    @app.get("/api/rounds")
    async def get_rounds():
        return store.get_rounds()

    @app.get("/api/network/graph")
    async def get_network_graph():
        return store.get_network_graph()

    @app.get("/api/network/stats")
    async def get_network_stats():
        return store.get_network_stats()

    @app.get("/api/swarms")
    async def get_swarms():
        return store.get_swarms()

    @app.get("/api/swarms/{swarm_id}/manifest")
    async def get_swarm_manifest(swarm_id: str):
        manifest = store.get_manifest(swarm_id)
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found for this swarm")
        filename = f"{swarm_id[:16]}.qgl"
        return Response(
            content=json.dumps(manifest, indent=2, sort_keys=True),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    @app.post("/api/telemetry/events", status_code=202)
    async def ingest_event(request: Request, event: TelemetryEventIngest):
        require_ingest_auth(request.headers.get(auth_header_name))
        try:
            broadcasts = store.ingest_event(event.event_type, event.payload, event.timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for message in broadcasts:
            await stream_hub.publish(message)
        return {"accepted": True}

    @app.post("/api/telemetry/heartbeats", status_code=202)
    async def ingest_heartbeat(request: Request, heartbeat: TelemetryHeartbeatIngest):
        require_ingest_auth(request.headers.get(auth_header_name))
        payload = heartbeat.model_dump(exclude_none=True)
        try:
            broadcasts = store.ingest_heartbeat(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for message in broadcasts:
            await stream_hub.publish(message)
        return {"accepted": True}

    @app.post("/api/telemetry/connection-status", status_code=202)
    async def ingest_connection_status(request: Request, status: TelemetryConnectionStatusIngest):
        require_ingest_auth(request.headers.get(auth_header_name))
        store.set_connection_status(
            status.status,
            detail=status.detail,
            mode=status.mode,
            url=status.url,
            last_error=status.last_error,
        )
        await stream_hub.publish(
            {
                "type": "connection_status_updated",
                "payload": store.get_dashboard_snapshot()["connection"],
            }
        )
        return {"accepted": True}

    @app.websocket("/api/stream")
    @app.websocket("/api/ws")
    async def telemetry_stream(websocket: WebSocket):
        await websocket.accept()
        queue = await stream_hub.subscribe()
        queue_task: asyncio.Task | None = None
        disconnect_task: asyncio.Task | None = None

        def consume_task_result(task: asyncio.Task | None, name: str):
            if task is None or not task.done():
                return None
            try:
                return task.result()
            except (asyncio.CancelledError, WebSocketDisconnect):
                return None
            except Exception:
                logger.exception("Telemetry %s task failed", name)
                return None

        try:
            while True:
                queue_task = asyncio.create_task(queue.get())
                disconnect_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait(
                    {queue_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if queue_task in done:
                    disconnect_task.cancel()
                    message = consume_task_result(queue_task, "queue")
                    if message is None:
                        break
                    if message.get("type") == "stream_closed":
                        await websocket.close(
                            code=message.get("code", STREAM_CLOSE_CODE_QUEUE_FULL),
                            reason=message.get("reason", "Telemetry subscriber queue overflow"),
                        )
                        break
                    await websocket.send_json(message)
                    continue

                queue_task.cancel()
                consume_task_result(disconnect_task, "disconnect")
                break
        except WebSocketDisconnect:
            pass
        finally:
            consume_task_result(queue_task, "queue")
            consume_task_result(disconnect_task, "disconnect")
            await stream_hub.unsubscribe(queue)

    return app
