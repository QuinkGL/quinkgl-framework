import json
from argparse import Namespace

import pytest

from quinkgl.cli import run_cmd
from quinkgl.manifest.schema import SwarmManifest, TelemetryConfig
from quinkgl.telemetry.qglkey import load_qglkey
from quinkgl.telemetry.tokens import TelemetryTokenRegistry


def test_manifest_round_trips_secret_free_telemetry_metadata():
    manifest = SwarmManifest(
        name="telemetry-swarm",
        telemetry=TelemetryConfig(
            dashboard_url="https://dash.example.com",
            enrollment="invite-required",
        ),
    )

    parsed = SwarmManifest.from_dict(manifest.to_dict())

    assert parsed.telemetry.dashboard_url == "https://dash.example.com"
    assert parsed.telemetry.enrollment == "invite-required"
    manifest_json = json.dumps(parsed.to_dict())
    assert "invite_code" not in manifest_json
    assert "ingest_token" not in manifest_json


def test_load_qglkey_requires_matching_swarm_id(tmp_path):
    key_path = tmp_path / "my-swarm.telemetry.qglkey"
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "swarm_id": "swarm-1",
                "dashboard_url": "https://dash.example.com",
                "ingest_token": "qgl_live_secret",
            }
        ),
        encoding="utf-8",
    )

    key = load_qglkey(key_path, expected_swarm_id="swarm-1")

    assert key.ingest_token == "qgl_live_secret"
    assert key.dashboard_url == "https://dash.example.com"
    with pytest.raises(ValueError, match="swarm_id"):
        load_qglkey(key_path, expected_swarm_id="other-swarm")


def test_run_prefers_adjacent_qglkey_over_global_secret(tmp_path, monkeypatch):
    manifest = SwarmManifest(name="keyed-swarm")
    manifest_path = tmp_path / "my-swarm.qgl"
    manifest.to_file(manifest_path, pretty=True)
    swarm_id = manifest.manifest_hash()
    key_path = tmp_path / "my-swarm.telemetry.qglkey"
    key_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "swarm_id": swarm_id,
                "dashboard_url": "https://from-key.example.com",
                "ingest_token": "qgl_live_from_key",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUINKGL_TELEMETRY_SECRET", "global-secret")

    auth = run_cmd._resolve_telemetry_auth(
        Namespace(telemetry_secret=None),
        manifest=manifest,
        manifest_path=manifest_path,
    )

    assert auth.base_url == "https://from-key.example.com"
    assert auth.secret == "qgl_live_from_key"


def test_run_falls_back_to_global_secret_when_qglkey_missing(tmp_path, monkeypatch):
    manifest = SwarmManifest(name="legacy-swarm")
    manifest_path = tmp_path / "legacy.qgl"
    manifest.to_file(manifest_path, pretty=True)
    monkeypatch.setenv("QUINKGL_TELEMETRY_SECRET", "global-secret")

    auth = run_cmd._resolve_telemetry_auth(
        Namespace(telemetry_secret=None),
        manifest=manifest,
        manifest_path=manifest_path,
    )

    assert auth.secret == "global-secret"


def test_token_registry_accepts_empty_touched_file(tmp_path):
    token_file = tmp_path / "tokens.json"
    token_file.write_text("", encoding="utf-8")

    registry = TelemetryTokenRegistry.from_file(token_file)

    assert registry.resolve("qgl_live_missing") is None
