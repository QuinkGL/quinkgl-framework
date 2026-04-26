from fastapi.testclient import TestClient

from quinkgl.telemetry.api import DEFAULT_TELEMETRY_AUTH_HEADER
from quinkgl.telemetry.server import _rate_limit_bucket_key, create_telemetry_app
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


def test_api_accepts_runtime_node_state_and_round_events():
    app = create_telemetry_app(store=TelemetryStore(session_id="session-1"))
    client = TestClient(app)

    state_response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "node.state.training",
            "payload": {
                "node_id": "node-a",
                "from": "peers_discovered",
                "to": "training",
                "rounds": 3,
            },
        },
    )
    round_started_response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "round_started",
            "payload": {"node_id": "node-a", "round": 1},
        },
    )
    round_completed_response = client.post(
        "/api/telemetry/events",
        json={
            "event_type": "round_completed",
            "payload": {"node_id": "node-a", "round": 1, "duration": 0.5},
        },
    )

    events = client.get("/api/nodes/node-a/events").json()

    assert state_response.status_code == 202
    assert round_started_response.status_code == 202
    assert round_completed_response.status_code == 202
    assert [event["event_type"] for event in events] == [
        "node.state.training",
        "round_started",
        "round_completed",
    ]


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


def test_api_defaults_to_permissive_cors_for_cross_origin_dashboards():
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
    assert response.headers.get("access-control-allow-origin") == "*"


def test_api_accepts_explicit_empty_cors_via_kwarg():
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        cors_allow_origins=[],
    )
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


def test_rate_limit_bucket_prefers_x_forwarded_for_from_loopback():
    assert _rate_limit_bucket_key("127.0.0.1", "203.0.113.9, 127.0.0.1") == "203.0.113.9"
    assert _rate_limit_bucket_key("192.168.1.1", "203.0.113.9") == "192.168.1.1"
    assert _rate_limit_bucket_key("127.0.0.1", "") == "127.0.0.1"


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


def test_api_accepts_swarm_scoped_token_for_matching_swarm():
    from quinkgl.telemetry.tokens import TelemetryTokenRegistry

    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        token_registry=TelemetryTokenRegistry.from_plain_tokens(
            [{"swarm_id": "swarm-1", "token": "qgl_live_swarm_1"}]
        ),
    )
    client = TestClient(app)

    accepted = client.post(
        "/api/telemetry/events",
        headers={DEFAULT_TELEMETRY_AUTH_HEADER: "qgl_live_swarm_1"},
        json={
            "event_type": "node.started",
            "payload": {"node_id": "node-a", "swarm_id": "swarm-1"},
        },
    )
    rejected = client.post(
        "/api/telemetry/events",
        headers={DEFAULT_TELEMETRY_AUTH_HEADER: "qgl_live_swarm_1"},
        json={
            "event_type": "node.started",
            "payload": {"node_id": "node-b", "swarm_id": "swarm-2"},
        },
    )

    assert accepted.status_code == 202
    assert rejected.status_code == 403


def test_api_open_enrollment_persists_swarm_token_and_returns_qglkey(tmp_path):
    from quinkgl.telemetry.tokens import TelemetryTokenRegistry

    token_file = tmp_path / "tokens.json"
    registry = TelemetryTokenRegistry.from_file(token_file, missing_ok=True)
    app = create_telemetry_app(
        store=TelemetryStore(session_id="session-1"),
        token_registry=registry,
    )
    client = TestClient(app)

    response = client.post(
        "/api/telemetry/enroll",
        json={
            "swarm_id": "swarm-enroll-1",
            "dashboard_url": "https://dash.example.com",
            "display_name": "Enroll Test",
            "manifest": {"name": "Enroll Test"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["swarm_id"] == "swarm-enroll-1"
    assert body["qglkey"]["swarm_id"] == "swarm-enroll-1"
    assert body["qglkey"]["dashboard_url"] == "https://dash.example.com"
    assert body["qglkey"]["ingest_token"].startswith("qgl_live_")
    assert "token_hash" in token_file.read_text(encoding="utf-8")

    accepted = client.post(
        "/api/telemetry/events",
        headers={DEFAULT_TELEMETRY_AUTH_HEADER: body["qglkey"]["ingest_token"]},
        json={
            "event_type": "node.started",
            "payload": {"node_id": "node-a", "swarm_id": "swarm-enroll-1"},
        },
    )
    assert accepted.status_code == 202


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
