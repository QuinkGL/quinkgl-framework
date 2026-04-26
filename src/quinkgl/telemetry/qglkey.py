from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryQglKey:
    swarm_id: str
    ingest_token: str
    dashboard_url: str = ""
    schema_version: int = 1


def default_qglkey_path(manifest_path: str | Path) -> Path:
    return Path(manifest_path).with_suffix(".telemetry.qglkey")


def load_qglkey(path: str | Path, *, expected_swarm_id: str | None = None) -> TelemetryQglKey:
    p = Path(path)
    data: Any = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("telemetry qglkey must contain a JSON object")
    schema_version = data.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"unsupported telemetry qglkey schema_version: {schema_version}")
    swarm_id = data.get("swarm_id")
    token = data.get("ingest_token")
    if not isinstance(swarm_id, str) or not swarm_id:
        raise ValueError("telemetry qglkey swarm_id must be a non-empty string")
    if expected_swarm_id is not None and swarm_id != expected_swarm_id:
        raise ValueError(
            f"telemetry qglkey swarm_id {swarm_id!r} does not match manifest {expected_swarm_id!r}"
        )
    if not isinstance(token, str) or not token:
        raise ValueError("telemetry qglkey ingest_token must be a non-empty string")
    dashboard_url = data.get("dashboard_url") or ""
    if not isinstance(dashboard_url, str):
        raise ValueError("telemetry qglkey dashboard_url must be a string")
    return TelemetryQglKey(
        schema_version=schema_version,
        swarm_id=swarm_id,
        ingest_token=token,
        dashboard_url=dashboard_url.rstrip("/"),
    )
