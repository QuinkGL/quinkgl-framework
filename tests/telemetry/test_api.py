from fastapi.testclient import TestClient

from quinkgl.telemetry.server import create_telemetry_app
from quinkgl.telemetry.store import TelemetryStore


def test_api_returns_empty_state_payloads():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    assert client.get("/api/session").status_code == 200
    assert client.get("/api/nodes").json() == []
    assert client.get("/api/network/graph").json()["edges"] == []


def test_api_ingest_updates_node_endpoints():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    heartbeat_response = client.post(
        "/api/telemetry/heartbeats",
        json={
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 2,
            "known_peers": ["node-b"],
        },
    )
    event_response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "training_completed",
            "payload": {
                "node_id": "node-a",
                "round": 2,
                "loss": 0.25,
                "accuracy": 0.75,
                "samples_trained": 32,
            },
        },
    )

    node = client.get("/api/nodes/node-a").json()
    events = client.get("/api/nodes/node-a/events").json()
    rounds = client.get("/api/nodes/node-a/rounds").json()

    assert heartbeat_response.status_code == 202
    assert event_response.status_code == 202
    assert node["current_round"] == 2
    assert node["last_loss"] == 0.25
    assert len(events) == 1
    assert rounds[0]["round_number"] == 2


def test_api_rejects_missing_payload_node_id():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/events",
        json={"event_type": "training_completed", "payload": {"round": 1}},
    )

    assert response.status_code == 422


def test_api_exposes_cors_headers_for_browser_clients():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.options(
        "/api/session",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
