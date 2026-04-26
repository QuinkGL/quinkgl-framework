import sys
import json
from types import SimpleNamespace

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import SUCCESS
from quinkgl.manifest import ModelSpec, SwarmManifest, TaskSpec


def test_telemetry_serve_defaults_to_loopback_8765(monkeypatch):
    calls = {}

    def fake_uvicorn_run(app, *, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_uvicorn_run))

    assert main(["telemetry", "serve"]) == SUCCESS

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8765
    assert calls["app"].state.telemetry_store.get_session()["active_node_count"] == 0


def test_telemetry_serve_passes_security_options_to_app(monkeypatch):
    calls = {}

    def fake_uvicorn_run(app, *, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_uvicorn_run))

    assert (
        main(
            [
                "telemetry",
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--auth-secret",
                "secret-123",
                "--cors-origin",
                "https://dash.example.com",
                "--cors-origin",
                "http://localhost:4173",
                "--max-request-bytes",
                "4096",
                "--rate-limit-max-requests",
                "10",
                "--rate-limit-window-seconds",
                "2.5",
            ]
        )
        == SUCCESS
    )

    app = calls["app"]
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9001
    assert app.state.telemetry_auth_secret == "secret-123"
    assert app.state.telemetry_cors_allow_origins == [
        "https://dash.example.com",
        "http://localhost:4173",
    ]
    assert app.state.telemetry_max_request_bytes == 4096
    assert app.state.telemetry_rate_limit_max_requests == 10
    assert app.state.telemetry_rate_limit_window_seconds == 2.5


def test_telemetry_serve_accepts_swarm_token_file(tmp_path, monkeypatch):
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "swarm_id": "swarm-1",
                        "token": "qgl_live_swarm_1",
                        "name": "swarm one",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_uvicorn_run(app, *, host, port):
        calls["app"] = app

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_uvicorn_run))

    assert main(["telemetry", "serve", "--token-file", str(token_file)]) == SUCCESS

    registry = calls["app"].state.telemetry_token_registry
    assert registry.resolve("qgl_live_swarm_1").swarm_id == "swarm-1"


def test_telemetry_enroll_writes_adjacent_qglkey(tmp_path, monkeypatch):
    manifest = SwarmManifest(
        name="enrolled-swarm",
        model_arch_fingerprint="sha256:" + "1" * 64,
        data_schema_hash="sha256:" + "2" * 64,
        task=TaskSpec(
            type="classification",
            input_shape=[1],
            output_shape=[1],
            label_type="integer",
        ),
        model=ModelSpec(framework="pytorch", arch_hash="sha256:" + "1" * 64),
    )
    manifest_path = tmp_path / "my-swarm.qgl"
    manifest.to_file(manifest_path, pretty=True)
    captured = {}

    def fake_post_json(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {
            "qglkey": {
                "schema_version": 1,
                "swarm_id": payload["swarm_id"],
                "dashboard_url": "https://dash.example.com",
                "ingest_token": "qgl_live_from_enroll",
            }
        }

    from quinkgl.cli import telemetry_cmd

    monkeypatch.setattr(telemetry_cmd, "_post_json", fake_post_json)

    assert (
        main(
            [
                "telemetry",
                "enroll",
                str(manifest_path),
                "--dashboard-url",
                "https://dash.example.com",
            ]
        )
        == SUCCESS
    )

    qglkey = json.loads((tmp_path / "my-swarm.telemetry.qglkey").read_text(encoding="utf-8"))
    assert captured["url"] == "https://dash.example.com/api/telemetry/enroll"
    assert captured["payload"]["swarm_id"] == manifest.manifest_hash()
    assert qglkey["ingest_token"] == "qgl_live_from_enroll"
