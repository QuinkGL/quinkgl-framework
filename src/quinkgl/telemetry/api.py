from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TelemetryEventIngest(BaseModel):
    event_type: str = Field(min_length=1)
    timestamp: Optional[str] = None
    payload: Dict[str, Any]


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
    timestamp: Optional[str] = None
