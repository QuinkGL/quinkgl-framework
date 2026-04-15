from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from quinkgl.telemetry.api import TelemetryEventIngest, TelemetryHeartbeatIngest
from quinkgl.telemetry.store import TelemetryStore
from quinkgl.telemetry.stream import TelemetryStreamHub


def create_telemetry_app(
    store: TelemetryStore | None = None,
    stream_hub: TelemetryStreamHub | None = None,
) -> FastAPI:
    store = store or TelemetryStore()
    stream_hub = stream_hub or TelemetryStreamHub()
    app = FastAPI(title="QuinkGL Telemetry")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.telemetry_store = store
    app.state.telemetry_stream_hub = stream_hub

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

    @app.post("/api/telemetry/events", status_code=202)
    async def ingest_event(event: TelemetryEventIngest):
        if not event.payload.get("node_id"):
            raise HTTPException(status_code=422, detail="payload.node_id is required")
        timestamp = datetime.fromisoformat(event.timestamp) if event.timestamp else None
        broadcasts = store.ingest_event(event.event_type, event.payload, timestamp)
        for message in broadcasts:
            await stream_hub.publish(message)
        return {"accepted": True}

    @app.post("/api/telemetry/heartbeats", status_code=202)
    async def ingest_heartbeat(heartbeat: TelemetryHeartbeatIngest):
        payload = heartbeat.model_dump(exclude_none=True)
        broadcasts = store.ingest_heartbeat(payload)
        for message in broadcasts:
            await stream_hub.publish(message)
        return {"accepted": True}

    @app.websocket("/api/stream")
    @app.websocket("/api/ws")
    async def telemetry_stream(websocket: WebSocket):
        await websocket.accept()
        queue = await stream_hub.subscribe()
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
                    await websocket.send_json(queue_task.result())
                    continue

                queue_task.cancel()
                disconnect_task.result()
        except WebSocketDisconnect:
            pass
        finally:
            await stream_hub.unsubscribe(queue)

    return app
