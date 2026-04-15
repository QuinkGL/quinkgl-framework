from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _iso(value: datetime | None) -> Optional[str]:
    return value.isoformat() if value else None


@dataclass
class NodeEvent:
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.payload.get("node_id"),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }


@dataclass
class NodeRoundSummary:
    node_id: str
    round_number: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    samples_trained: Optional[int] = None
    selected_targets: List[str] = field(default_factory=list)
    sent_peer_ids: List[str] = field(default_factory=list)
    received_peer_ids: List[str] = field(default_factory=list)
    aggregated_peer_ids: List[str] = field(default_factory=list)
    aggregation_total_samples: Optional[int] = None
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "round_number": self.round_number,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "samples_trained": self.samples_trained,
            "selected_targets": list(self.selected_targets),
            "sent_peer_ids": list(self.sent_peer_ids),
            "received_peer_ids": list(self.received_peer_ids),
            "aggregated_peer_ids": list(self.aggregated_peer_ids),
            "aggregation_total_samples": self.aggregation_total_samples,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class NetworkEdge:
    source_node_id: str
    target_node_id: str
    edge_type: str = "peer_link"
    last_active_at: datetime = field(default_factory=datetime.now)
    exchange_count: int = 0
    discovery_count: int = 0
    last_round: Optional[int] = None
    last_event_type: Optional[str] = None
    last_weight_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": f"{self.source_node_id}::{self.target_node_id}::{self.edge_type}",
            "source": self.source_node_id,
            "target": self.target_node_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "last_active_at": self.last_active_at.isoformat(),
            "exchange_count": self.exchange_count,
            "discovery_count": self.discovery_count,
            "last_round": self.last_round,
            "last_event_type": self.last_event_type,
            "last_weight_summary": dict(self.last_weight_summary),
        }


@dataclass
class NodeSnapshot:
    node_id: str
    domain: str = "unknown"
    connection_mode: Optional[str] = None
    status: str = "unknown"
    running: bool = False
    current_round: int = 0
    session_started_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    known_peer_count: int = 0
    peer_ids: List[str] = field(default_factory=list)
    last_loss: Optional[float] = None
    last_accuracy: Optional[float] = None
    last_samples_trained: Optional[int] = None
    training_rounds_completed: int = 0
    models_sent: int = 0
    models_received: int = 0
    last_sent_peer_ids: List[str] = field(default_factory=list)
    last_received_peer_id: Optional[str] = None
    aggregations_completed: int = 0
    last_aggregation_peer_count: Optional[int] = None
    last_aggregation_total_samples: Optional[int] = None
    aggregation_strategy: Optional[str] = None
    chunked_transfers_sent: int = 0
    chunked_transfers_received: int = 0
    resend_requests: int = 0
    send_failures: int = 0
    stale_updates: int = 0
    duplicate_updates: int = 0
    last_training_at: Optional[datetime] = None
    last_send_at: Optional[datetime] = None
    last_receive_at: Optional[datetime] = None
    last_aggregation_at: Optional[datetime] = None
    last_selected_peer_ids: List[str] = field(default_factory=list)
    last_weight_summary: Dict[str, Any] = field(default_factory=dict)

    def refresh_uptime(self, now: datetime) -> None:
        self.last_seen_at = now
        self.uptime_seconds = max(
            0.0, (now - self.session_started_at).total_seconds()
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "session_started_at",
            "last_seen_at",
            "last_training_at",
            "last_send_at",
            "last_receive_at",
            "last_aggregation_at",
        ):
            result[key] = _iso(getattr(self, key))
        return result


@dataclass
class SessionSnapshot:
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    last_updated_at: datetime = field(default_factory=datetime.now)
    active_node_count: int = 0
    total_edge_count: int = 0
    recent_exchange_count: int = 0
    recent_aggregation_count: int = 0
    active_domains: List[str] = field(default_factory=list)
    selected_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "active_node_count": self.active_node_count,
            "total_edge_count": self.total_edge_count,
            "recent_exchange_count": self.recent_exchange_count,
            "recent_aggregation_count": self.recent_aggregation_count,
            "active_domains": list(self.active_domains),
            "selected_node_id": self.selected_node_id,
        }
