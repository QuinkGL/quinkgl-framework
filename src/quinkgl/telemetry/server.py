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
from quinkgl.telemetry.tokens import TelemetryTokenRegistry
from quinkgl.telemetry.viewer import DashboardAccessRegistry, DashboardViewerScope


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
    token_registry: TelemetryTokenRegistry | None = None,
    dashboard_access_registry: DashboardAccessRegistry | None = None,
    cors_allow_origins: list[str] | None = None,
    max_request_bytes: int | None = None,
    rate_limit_max_requests: int | None = None,
    rate_limit_window_seconds: float | None = None,
) -> FastAPI:
    store = store or TelemetryStore()
    stream_hub = stream_hub or TelemetryStreamHub()
    if dashboard_access_registry is None and token_registry is not None:
        dashboard_access_registry = DashboardAccessRegistry()
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
    app.state.telemetry_token_registry = token_registry
    app.state.dashboard_access_registry = dashboard_access_registry
    app.state.telemetry_cors_allow_origins = list(cors_allow_origins)
    app.state.telemetry_max_request_bytes = max_request_bytes
    app.state.telemetry_rate_limit_max_requests = rate_limit_max_requests
    app.state.telemetry_rate_limit_window_seconds = rate_limit_window_seconds

    def require_ingest_auth(header_value: str | None, payload: dict | None = None) -> None:
        if token_registry is not None:
            record = token_registry.resolve(header_value)
            if record is None:
                raise HTTPException(status_code=401, detail="Invalid telemetry token")
            data = payload or {}
            payload_swarm_id = data.get("swarm_id") or data.get("manifest_hash")
            if payload_swarm_id and payload_swarm_id != record.swarm_id:
                raise HTTPException(
                    status_code=403,
                    detail="Telemetry token is not valid for this swarm",
                )
            data["swarm_id"] = record.swarm_id
            return
        if not auth_secret:
            return
        if header_value != auth_secret:
            raise HTTPException(status_code=401, detail="Invalid telemetry auth secret")

    def resolve_ingest_swarm(header_value: str | None) -> str:
        if token_registry is None:
            raise HTTPException(status_code=503, detail="Swarm-scoped dashboard access requires token registry")
        record = token_registry.resolve(header_value)
        if record is None:
            raise HTTPException(status_code=401, detail="Invalid telemetry token")
        return record.swarm_id

    def _bearer_token(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return None
        return header[7:].strip()

    def require_viewer_scope(request: Request) -> DashboardViewerScope | None:
        if dashboard_access_registry is None:
            return None
        scope = dashboard_access_registry.resolve_viewer_token(_bearer_token(request))
        if scope is None:
            raise HTTPException(status_code=401, detail="Invalid dashboard viewer token")
        return scope

    def _node_ids_for_scope(scope: DashboardViewerScope | None) -> set[str]:
        if scope is None:
            return {
                node.get("node_id")
                for node in store.get_nodes()
                if node.get("node_id")
            }
        return {
            node.get("node_id")
            for node in store.get_nodes()
            if node.get("swarm_id") == scope.swarm_id and node.get("node_id")
        }

    def _filter_nodes(scope: DashboardViewerScope | None) -> list[dict]:
        nodes = store.get_nodes()
        if scope is None:
            return nodes
        return [node for node in nodes if node.get("swarm_id") == scope.swarm_id]

    def _filter_events(scope: DashboardViewerScope | None) -> list[dict]:
        events = store.get_events()
        if scope is None:
            return events
        allowed_node_ids = _node_ids_for_scope(scope)
        return [
            event for event in events
            if event.get("node_id") in allowed_node_ids
            or event.get("payload", {}).get("swarm_id") == scope.swarm_id
        ]

    def _filter_rounds(scope: DashboardViewerScope | None) -> list[dict]:
        rounds = store.get_rounds()
        if scope is None:
            return rounds
        allowed_node_ids = _node_ids_for_scope(scope)
        return [round_item for round_item in rounds if round_item.get("node_id") in allowed_node_ids]

    def _filter_swarms(scope: DashboardViewerScope | None) -> list[dict]:
        swarms = store.get_swarms()
        if scope is None:
            return swarms
        return [swarm for swarm in swarms if swarm.get("swarm_id") == scope.swarm_id]

    def _filtered_network_graph(scope: DashboardViewerScope | None) -> dict:
        graph = store.get_network_graph()
        if scope is None:
            return graph
        allowed_node_ids = _node_ids_for_scope(scope)
        return {
            "nodes": [node for node in graph.get("nodes", []) if node.get("node_id") in allowed_node_ids],
            "edges": [
                edge for edge in graph.get("edges", [])
                if edge.get("source_node_id", edge.get("source")) in allowed_node_ids
                and edge.get("target_node_id", edge.get("target")) in allowed_node_ids
            ],
        }

    def _filtered_network_stats(scope: DashboardViewerScope | None) -> dict:
        if scope is None:
            return store.get_network_stats()
        graph = _filtered_network_graph(scope)
        return {
            "total_nodes": len(graph["nodes"]),
            "active_edge_count": len(graph["edges"]),
            "isolated_nodes": [
                node["node_id"]
                for node in graph["nodes"]
                if int(node.get("known_peer_count") or 0) == 0
            ],
            "message_volume": sum(int(edge.get("exchange_count") or 0) for edge in graph["edges"]),
        }

    def _filtered_session(scope: DashboardViewerScope | None) -> dict:
        if scope is None:
            return store.get_session()
        base = store.get_session()
        nodes = _filter_nodes(scope)
        graph = _filtered_network_graph(scope)
        running_nodes = [node for node in nodes if node.get("running")]
        base.update(
            {
                "active_node_count": len(running_nodes),
                "total_edge_count": len(graph["edges"]),
                "recent_exchange_count": sum(int(edge.get("exchange_count") or 0) for edge in graph["edges"]),
                "recent_aggregation_count": sum(int(node.get("aggregations_completed") or 0) for node in nodes),
                "active_domains": sorted({node.get("domain") for node in nodes if node.get("domain")}),
                "selected_node_id": running_nodes[0]["node_id"] if running_nodes else (nodes[0]["node_id"] if nodes else None),
            }
        )
        return base

    def _filtered_dashboard_snapshot(scope: DashboardViewerScope | None) -> dict:
        graph = _filtered_network_graph(scope)
        return {
            "connection": store.get_dashboard_snapshot()["connection"],
            "session": _filtered_session(scope),
            "nodes": _filter_nodes(scope),
            "events": _filter_events(scope),
            "rounds": _filter_rounds(scope),
            "network": {
                **graph,
                "stats": _filtered_network_stats(scope),
                "swarms": _filter_swarms(scope),
            },
            "swarms": _filter_swarms(scope),
        }

    def _message_visible_to_scope(message: dict, scope: DashboardViewerScope | None) -> bool:
        if scope is None:
            return True
        payload = message.get("payload") or {}
        payload_swarm_id = payload.get("swarm_id") or payload.get("manifest_hash")
        if payload_swarm_id == scope.swarm_id:
            return True
        node_id = payload.get("node_id")
        return bool(node_id and node_id in _node_ids_for_scope(scope))

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

    @app.post("/api/telemetry/enroll", status_code=201)
    async def enroll_swarm(request: Request):
        if token_registry is None:
            raise HTTPException(status_code=503, detail="Telemetry enrollment is not configured")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Enrollment payload must be an object")
        swarm_id = payload.get("swarm_id")
        if not isinstance(swarm_id, str) or not swarm_id.strip():
            raise HTTPException(status_code=422, detail="swarm_id is required")
        dashboard_url = payload.get("dashboard_url") or ""
        if dashboard_url and not isinstance(dashboard_url, str):
            raise HTTPException(status_code=422, detail="dashboard_url must be a string")
        display_name = payload.get("display_name") or swarm_id
        token = token_registry.create_token(swarm_id=swarm_id, name=str(display_name))
        manifest = payload.get("manifest")
        if isinstance(manifest, dict):
            store._manifests[swarm_id] = manifest
        return {
            "swarm_id": swarm_id,
            "dashboard_url": dashboard_url,
            "ingest_token": token,
            "qglkey": {
                "schema_version": 1,
                "swarm_id": swarm_id,
                "dashboard_url": dashboard_url,
                "ingest_token": token,
            },
        }

    @app.post("/api/dashboard/codes", status_code=201)
    async def create_dashboard_code(request: Request):
        if dashboard_access_registry is None:
            raise HTTPException(status_code=503, detail="Dashboard login is not configured")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Dashboard code payload must be an object")
        swarm_id = payload.get("swarm_id")
        if not isinstance(swarm_id, str) or not swarm_id.strip():
            raise HTTPException(status_code=422, detail="swarm_id is required")
        authorized_swarm_id = resolve_ingest_swarm(request.headers.get(auth_header_name))
        if swarm_id != authorized_swarm_id:
            raise HTTPException(status_code=403, detail="Telemetry token is not valid for this swarm")
        node_id = payload.get("node_id")
        if node_id is not None and not isinstance(node_id, str):
            raise HTTPException(status_code=422, detail="node_id must be a string")
        code, scope = dashboard_access_registry.create_code(
            swarm_id=swarm_id,
            issued_from_node_id=node_id,
        )
        return {"code": code, "scope": scope.to_dict()}

    @app.post("/api/dashboard/login")
    async def login_dashboard(request: Request):
        if dashboard_access_registry is None:
            raise HTTPException(status_code=503, detail="Dashboard login is not configured")
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Dashboard login payload must be an object")
        redeemed = dashboard_access_registry.redeem_code(str(payload.get("code") or ""))
        if redeemed is None:
            raise HTTPException(status_code=401, detail="Invalid dashboard code")
        viewer_token, scope = redeemed
        return {"viewer_token": viewer_token, "scope": scope.to_dict()}

    @app.get("/api/session")
    async def get_session(request: Request):
        return _filtered_session(require_viewer_scope(request))

    @app.get("/api/nodes")
    async def get_nodes(request: Request):
        return _filter_nodes(require_viewer_scope(request))

    @app.get("/api/nodes/{node_id}")
    async def get_node(request: Request, node_id: str):
        scope = require_viewer_scope(request)
        node = store.get_node(node_id)
        if node is None or (scope is not None and node.get("swarm_id") != scope.swarm_id):
            raise HTTPException(status_code=404, detail="Node not found")
        return node

    @app.get("/api/nodes/{node_id}/events")
    async def get_node_events(request: Request, node_id: str):
        scope = require_viewer_scope(request)
        if scope is not None and node_id not in _node_ids_for_scope(scope):
            raise HTTPException(status_code=404, detail="Node not found")
        return store.get_node_events(node_id)

    @app.get("/api/events")
    async def get_events(request: Request):
        return _filter_events(require_viewer_scope(request))

    @app.get("/api/nodes/{node_id}/rounds")
    async def get_node_rounds(request: Request, node_id: str):
        scope = require_viewer_scope(request)
        if scope is not None and node_id not in _node_ids_for_scope(scope):
            raise HTTPException(status_code=404, detail="Node not found")
        return store.get_node_rounds(node_id)

    @app.get("/api/rounds")
    async def get_rounds(request: Request):
        return _filter_rounds(require_viewer_scope(request))

    @app.get("/api/network/graph")
    async def get_network_graph(request: Request):
        return _filtered_network_graph(require_viewer_scope(request))

    @app.get("/api/network/stats")
    async def get_network_stats(request: Request):
        return _filtered_network_stats(require_viewer_scope(request))

    @app.get("/api/swarms")
    async def get_swarms(request: Request):
        return _filter_swarms(require_viewer_scope(request))

    @app.get("/api/swarms/{swarm_id}/manifest")
    async def get_swarm_manifest(request: Request, swarm_id: str):
        scope = require_viewer_scope(request)
        if scope is not None and swarm_id != scope.swarm_id:
            raise HTTPException(status_code=404, detail="Manifest not found for this swarm")
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
        require_ingest_auth(request.headers.get(auth_header_name), event.payload)
        try:
            broadcasts = store.ingest_event(event.event_type, event.payload, event.timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for message in broadcasts:
            await stream_hub.publish(message)
        return {"accepted": True}

    @app.post("/api/telemetry/heartbeats", status_code=202)
    async def ingest_heartbeat(request: Request, heartbeat: TelemetryHeartbeatIngest):
        payload = heartbeat.model_dump(exclude_none=True)
        require_ingest_auth(request.headers.get(auth_header_name), payload)
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
        scope = None
        if dashboard_access_registry is not None:
            scope = dashboard_access_registry.resolve_viewer_token(
                websocket.query_params.get("viewer_token")
            )
            if scope is None:
                await websocket.close(code=1008, reason="Invalid dashboard viewer token")
                return
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
                    if message.get("type") == "session_stats_updated":
                        message = {
                            "type": "session_stats_updated",
                            "payload": _filtered_session(scope),
                        }
                    elif not _message_visible_to_scope(message, scope):
                        continue
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
