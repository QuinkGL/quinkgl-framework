# Copyright 2026 Ali Seyhan, Baki Turhan
"""Default telemetry URL resolution for ``quinkgl run``."""

from argparse import Namespace

from quinkgl.cli import run_cmd
from quinkgl.telemetry.api import DEFAULT_TELEMETRY_BASE_URL


def test_resolve_effective_telemetry_url_defaults_to_local(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_URL", raising=False)
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    args = Namespace(no_telemetry=False, telemetry_url=None)
    assert run_cmd._resolve_effective_telemetry_url(args) == DEFAULT_TELEMETRY_BASE_URL


def test_resolve_effective_telemetry_url_no_telemetry_flag():
    args = Namespace(no_telemetry=True, telemetry_url="http://example.invalid:9")
    assert run_cmd._resolve_effective_telemetry_url(args) is None


def test_resolve_effective_telemetry_url_disable_env(monkeypatch):
    monkeypatch.setenv("QUINKGL_TELEMETRY_DISABLE", "1")
    args = Namespace(no_telemetry=False, telemetry_url=None)
    assert run_cmd._resolve_effective_telemetry_url(args) is None


def test_resolve_effective_telemetry_url_cli_overrides_env(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    monkeypatch.setenv("QUINKGL_TELEMETRY_URL", "http://from-env:1")
    args = Namespace(no_telemetry=False, telemetry_url="http://from-cli:2")
    assert run_cmd._resolve_effective_telemetry_url(args) == "http://from-cli:2"


def test_resolve_effective_telemetry_url_env_when_no_cli(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    monkeypatch.setenv("QUINKGL_TELEMETRY_URL", "http://from-env:1")
    args = Namespace(no_telemetry=False, telemetry_url=None)
    assert run_cmd._resolve_effective_telemetry_url(args) == "http://from-env:1"


def test_resolve_effective_telemetry_url_blank_cli_falls_through_to_env(monkeypatch):
    monkeypatch.delenv("QUINKGL_TELEMETRY_DISABLE", raising=False)
    monkeypatch.setenv("QUINKGL_TELEMETRY_URL", "http://from-env:1")
    args = Namespace(no_telemetry=False, telemetry_url="  ")
    assert run_cmd._resolve_effective_telemetry_url(args) == "http://from-env:1"
