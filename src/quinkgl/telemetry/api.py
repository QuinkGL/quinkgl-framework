from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


DEFAULT_TELEMETRY_AUTH_HEADER = "X-QuinkGL-Telemetry-Secret"
TELEMETRY_AUTH_SECRET_ENV = "QUINKGL_TELEMETRY_SECRET"
TELEMETRY_AUTH_HEADER_ENV = "QUINKGL_TELEMETRY_SECRET_HEADER"
TELEMETRY_CORS_ALLOW_ORIGINS_ENV = "QUINKGL_TELEMETRY_CORS_ALLOW_ORIGINS"
# Default telemetry API origin (no ``/api`` suffix). Override with
# :envvar:`TELEMETRY_URL_ENV`. Disable with :envvar:`TELEMETRY_DISABLE_ENV`
# or ``quinkgl run --no-telemetry``.
DEFAULT_TELEMETRY_BASE_URL = "https://141-147-36-24.sslip.io"
TELEMETRY_URL_ENV = "QUINKGL_TELEMETRY_URL"
TELEMETRY_DISABLE_ENV = "QUINKGL_TELEMETRY_DISABLE"


class TelemetryEventIngest(BaseModel):
    event_type: str = Field(min_length=1)
    timestamp: Optional[datetime] = None
    payload: Dict[str, Any]

    @model_validator(mode="after")
    def validate_payload(self) -> "TelemetryEventIngest":
        node_id = self.payload.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("payload.node_id is required")

        round_number = self.payload.get("round")
        if round_number is not None:
            try:
                payload = dict(self.payload)
                payload["round"] = int(round_number)
                self.payload = payload
            except (TypeError, ValueError) as exc:
                raise ValueError("payload.round must be an integer") from exc
        return self


class TelemetryConnectionStatusIngest(BaseModel):
    status: str = Field(min_length=1)
    detail: Optional[str] = None
    mode: Optional[str] = None
    url: Optional[str] = None
    last_error: Optional[str] = None


class TelemetryHeartbeatIngest(BaseModel):
    node_id: str = Field(min_length=1)
    domain: Optional[str] = None
    connection_mode: Optional[str] = None
    running: Optional[bool] = None
    current_round: Optional[int] = None
    known_peers: Optional[list[str]] = None
    peer_ids: Optional[list[str]] = None
    known_peer_count: Optional[int] = None
    ipv8_peers: Optional[int] = None
    tunnel_peers: Optional[int] = None
    timestamp: Optional[datetime] = None
