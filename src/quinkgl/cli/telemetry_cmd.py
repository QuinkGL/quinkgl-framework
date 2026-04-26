# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Telemetry server CLI commands."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from quinkgl.cli.exit_codes import SUCCESS, VALIDATION_ERROR
from quinkgl.manifest import SwarmManifest
from quinkgl.telemetry.api import DEFAULT_TELEMETRY_BASE_URL
from quinkgl.telemetry.api import TELEMETRY_AUTH_SECRET_ENV
from quinkgl.telemetry.qglkey import default_qglkey_path
from quinkgl.telemetry.server import (
    DEFAULT_TELEMETRY_MAX_REQUEST_BYTES,
    DEFAULT_TELEMETRY_RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_TELEMETRY_RATE_LIMIT_WINDOW_SECONDS,
    create_telemetry_app,
)


DEFAULT_TELEMETRY_HOST = "127.0.0.1"
DEFAULT_TELEMETRY_PORT = 8765


def build_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("telemetry", help="Run telemetry backend utilities")
    nested = parser.add_subparsers(dest="telemetry_command")

    serve = nested.add_parser(
        "serve",
        help="Run the FastAPI telemetry backend for the dashboard",
    )
    serve.add_argument(
        "--host",
        default=DEFAULT_TELEMETRY_HOST,
        help="Host interface for uvicorn. Use 127.0.0.1 behind Caddy.",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_TELEMETRY_PORT,
        help="Port for uvicorn telemetry API.",
    )
    serve.add_argument(
        "--auth-secret",
        default=None,
        help=f"Ingest auth secret. Defaults to ${TELEMETRY_AUTH_SECRET_ENV}.",
    )
    serve.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="Allowed dashboard origin. Repeat for multiple origins.",
    )
    serve.add_argument(
        "--token-file",
        default=None,
        help="JSON file with swarm-scoped telemetry ingest tokens.",
    )
    serve.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_TELEMETRY_MAX_REQUEST_BYTES,
        help="Maximum ingest request size.",
    )
    serve.add_argument(
        "--rate-limit-max-requests",
        type=int,
        default=DEFAULT_TELEMETRY_RATE_LIMIT_MAX_REQUESTS,
        help="Maximum ingest requests per rate-limit window.",
    )
    serve.add_argument(
        "--rate-limit-window-seconds",
        type=float,
        default=DEFAULT_TELEMETRY_RATE_LIMIT_WINDOW_SECONDS,
        help="Rate-limit window length in seconds.",
    )

    enroll = nested.add_parser(
        "enroll",
        help="Enroll a manifest with the telemetry backend and write a .qglkey",
    )
    enroll.add_argument("manifest", help="Path to the .qgl manifest")
    enroll.add_argument(
        "--dashboard-url",
        default=None,
        help="Telemetry dashboard origin. Defaults to manifest metadata or hosted default.",
    )
    enroll.add_argument(
        "--output",
        default=None,
        help="Output .qglkey path. Defaults to <manifest>.telemetry.qglkey.",
    )
    enroll.add_argument("--overwrite", action="store_true", help="Replace an existing .qglkey file.")


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"enrollment failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"enrollment request failed: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("enrollment response must be a JSON object")
    return parsed


def _run_enroll(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        manifest = SwarmManifest.from_file(manifest_path)
    except Exception as exc:
        print(f"Error: failed to load manifest: {exc}")
        return VALIDATION_ERROR

    manifest_telemetry = getattr(manifest, "telemetry", None)
    dashboard_url = (
        args.dashboard_url
        or getattr(manifest_telemetry, "dashboard_url", "")
        or DEFAULT_TELEMETRY_BASE_URL
    ).rstrip("/")
    output_path = Path(args.output) if args.output else default_qglkey_path(manifest_path)
    if output_path.exists() and not args.overwrite:
        print(f"Error: {output_path} already exists; use --overwrite to replace it")
        return VALIDATION_ERROR

    swarm_id = manifest.manifest_hash()
    payload = {
        "swarm_id": swarm_id,
        "dashboard_url": dashboard_url,
        "display_name": manifest.name,
        "manifest": manifest.to_dict(),
    }
    try:
        response = _post_json(f"{dashboard_url}/api/telemetry/enroll", payload)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return VALIDATION_ERROR

    qglkey = response.get("qglkey")
    if not isinstance(qglkey, dict):
        print("Error: enrollment response did not include qglkey")
        return VALIDATION_ERROR
    if qglkey.get("swarm_id") != swarm_id:
        print("Error: enrollment response swarm_id does not match manifest")
        return VALIDATION_ERROR
    output_path.write_text(json.dumps(qglkey, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")
    return SUCCESS


def run(args: argparse.Namespace) -> int:
    if getattr(args, "telemetry_command", None) == "enroll":
        return _run_enroll(args)
    if getattr(args, "telemetry_command", None) != "serve":
        return VALIDATION_ERROR

    import uvicorn
    from quinkgl.telemetry.tokens import TelemetryTokenRegistry

    token_registry = (
        TelemetryTokenRegistry.from_file(args.token_file)
        if args.token_file
        else None
    )

    app = create_telemetry_app(
        auth_secret=args.auth_secret,
        token_registry=token_registry,
        cors_allow_origins=args.cors_origin,
        max_request_bytes=args.max_request_bytes,
        rate_limit_max_requests=args.rate_limit_max_requests,
        rate_limit_window_seconds=args.rate_limit_window_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return SUCCESS
