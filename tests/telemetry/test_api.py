from fastapi.testclient import TestClient

from quinkgl.telemetry.api import DEFAULT_TELEMETRY_AUTH_HEADER
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
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        cors_allow_origins=["http://localhost:5173"],
    )
    client = TestClient(app)

    response = client.options(
        "/api/session",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_api_defaults_to_same_origin_without_cors_headers_for_foreign_origin():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.options(
        "/api/session",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_api_requires_auth_for_ingest_when_secret_configured():
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        auth_secret="secret-123",
    )
    client = TestClient(app)

    unauthorized = client.post(
        "/api/telemetry/heartbeats",
        json={
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
        },
    )
    authorized = client.post(
        "/api/telemetry/heartbeats",
        headers={DEFAULT_TELEMETRY_AUTH_HEADER: "secret-123"},
        json={
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
        },
    )
    read_response = client.get("/api/session")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 202
    assert read_response.status_code == 200


def test_api_accepts_custom_auth_header_name_for_ingest():
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        auth_secret="secret-456",
        auth_header_name="X-Test-Telemetry-Secret",
    )
    client = TestClient(app)

    wrong_header = client.post(
        "/api/telemetry/events",
        headers={DEFAULT_TELEMETRY_AUTH_HEADER: "secret-456"},
        json={
            "event_type": "training_completed",
            "payload": {"node_id": "node-a", "round": 1},
        },
    )
    correct_header = client.post(
        "/api/telemetry/events",
        headers={"X-Test-Telemetry-Secret": "secret-456"},
        json={
            "event_type": "training_completed",
            "payload": {"node_id": "node-a", "round": 1},
        },
    )

    assert wrong_header.status_code == 401
    assert correct_header.status_code == 202


def test_api_rejects_oversized_ingest_requests():
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        max_request_bytes=32,
    )
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/events",
        content='{"event_type":"training_completed","payload":{"node_id":"node-a","summary":"abcdefghijklmnopqrstuvwxyz"}}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Telemetry request too large"


def test_api_rate_limits_ingest_requests_per_client_ip():
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        rate_limit_max_requests=1,
        rate_limit_window_seconds=60.0,
    )
    client = TestClient(app)

    first = client.post(
        "/api/telemetry/heartbeats",
        json={
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
        },
    )
    second = client.post(
        "/api/telemetry/heartbeats",
        json={
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
        },
    )

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.json()["detail"] == "Telemetry rate limit exceeded"


def test_api_rejects_invalid_event_timestamp():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "training_completed",
            "timestamp": "not-a-datetime",
            "payload": {"node_id": "node-a", "round": 1},
        },
    )

    assert response.status_code == 422


def test_api_rejects_invalid_event_round_type():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "training_completed",
            "payload": {"node_id": "node-a", "round": "oops"},
        },
    )

    assert response.status_code == 422
    assert "payload.round must be an integer" in response.text


def test_api_rejects_unknown_event_types():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "custom_debug",
            "payload": {"node_id": "node-a", "round": 1},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown telemetry event type: custom_debug"


def test_api_accepts_connection_status_updates_and_exposes_snapshot():
    store = TelemetryStore(session_id="session-1")
    app = create_telemetry_app(store=store)
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/connection-status",
        json={
            "status": "error",
            "detail": "Socket error",
            "mode": "relay",
            "url": "http://telemetry.local",
            "last_error": "Socket error",
        },
    )

    assert response.status_code == 202
    snapshot = store.get_dashboard_snapshot()["connection"]
    assert snapshot["status"] == "error"
    assert snapshot["detail"] == "Socket error"
    assert snapshot["mode"] == "relay"
    assert snapshot["url"] == "http://telemetry.local"
