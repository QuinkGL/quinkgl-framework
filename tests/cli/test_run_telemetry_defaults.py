# Copyright 2026 Ali Seyhan, Baki Turhan
"""Default telemetry URL resolution for ``quinkgl run``."""

from argparse import Namespace
import json

from quinkgl.cli import run_cmd
from quinkgl.manifest import ModelSpec, SwarmManifest, TaskSpec
from quinkgl.telemetry.api import DEFAULT_TELEMETRY_BASE_URL


def test_resolve_effective_telemetry_url_defaults_to_local(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_URL", raising=False)
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    args = Namespace(no_telemetry=False)
    assert run_cmd._resolve_effective_telemetry_url(args) == DEFAULT_TELEMETRY_BASE_URL


def test_resolve_effective_telemetry_url_no_telemetry_flag():
    args = Namespace(no_telemetry=True)
    assert run_cmd._resolve_effective_telemetry_url(args) is None


def test_resolve_effective_telemetry_url_disable_env(monkeypatch):
    monkeypatch.setenv("QUINKGL_TELEMETRY_DISABLE", "1")
    args = Namespace(no_telemetry=False)
    assert run_cmd._resolve_effective_telemetry_url(args) is None


def test_resolve_effective_telemetry_url_env_when_no_cli(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    monkeypatch.setenv("QUINKGL_TELEMETRY_URL", "http://from-env:1")
    args = Namespace(no_telemetry=False)
    assert run_cmd._resolve_effective_telemetry_url(args) == "http://from-env:1"


def test_run_help_does_not_offer_telemetry_url(capsys):
    from quinkgl.cli.__main__ import main

    try:
        main(["run", "--help"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "--telemetry-url" not in captured.out


def test_maybe_print_dashboard_code_uses_manifest_qglkey(tmp_path, monkeypatch, capsys):
    manifest = SwarmManifest(
        name="run-dashboard-code",
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
    manifest_path = tmp_path / "swarm.qgl"
    manifest.to_file(manifest_path, pretty=True)
    (tmp_path / "swarm.telemetry.qglkey").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "swarm_id": manifest.manifest_hash(),
                "dashboard_url": "https://dash.example.com",
                "ingest_token": "qgl_live_from_key",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_request_dashboard_code(*, manifest_path, node_id=None, dashboard_url=None):
        captured["manifest_path"] = manifest_path
        captured["node_id"] = node_id
        captured["dashboard_url"] = dashboard_url
        return "QGL-RUN1-CODE", {"scope": {"expires_at": "2026-04-27T12:00:00"}}

    from quinkgl.cli import telemetry_cmd

    monkeypatch.setattr(telemetry_cmd, "_request_dashboard_code", fake_request_dashboard_code)

    run_cmd._maybe_print_dashboard_code(
        Namespace(node_id="peer-1", no_telemetry=False),
        manifest_path=manifest_path,
    )

    output = capsys.readouterr().out
    assert captured["manifest_path"] == manifest_path
    assert captured["node_id"] == "peer-1"
    assert "Dashboard code: QGL-RUN1-CODE" in output
