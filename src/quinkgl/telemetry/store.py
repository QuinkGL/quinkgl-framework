from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from collections import defaultdict

from quinkgl.telemetry.models import (
    NetworkEdge,
    NodeEvent,
    NodeRoundSummary,
    NodeSnapshot,
    SessionSnapshot,
    SwarmSnapshot,
)


DEFAULT_TELEMETRY_MAX_NODES = 256
DEFAULT_TELEMETRY_MAX_EVENTS_PER_NODE = 500
DEFAULT_TELEMETRY_MAX_ROUNDS_PER_NODE = 128
DEFAULT_TELEMETRY_MAX_EDGES = 2048
DEFAULT_TELEMETRY_MAX_PEER_IDS_PER_NODE = 256
SUPPORTED_TELEMETRY_EVENT_TYPES = {
    "aggregation_completed",
    "aggregation_failed",
    "consensus_reached",
    "early_stopping",
    "ipv8_payload_dropped",
    "model_received",
    "model_rejected_backpressure",
    "model_rejected_duplicate",
    "model_rejected_stale",
    "model_send_failed",
    "model_send_started",
    "model_sent",
    "models_converged",
    "node.config",
    "node.started",
    "node.stopped",
    "peer_disconnected",
    "peer_discovered",
    "post_aggregation_eval",
    "subscriber.error",
    "targets_selected",
    "telemetry.connected",
    "telemetry.delivery_failed",
    "telemetry.disconnected",
    "telemetry.events_dropped",
    "telemetry.status_provider_warning",
    "training_completed",
    "training_started",
    "tunnel_payload_dropped",
}

# T-OBS-14: Allowlist of payload keys per event type.
# Events not listed here accept any payload keys (open schema).
PAYLOAD_KEY_ALLOWLIST: Dict[str, set] = {
    "training_completed": {"node_id", "round", "loss", "accuracy", "samples_trained", "domain"},
    "model_sent": {"node_id", "round", "peer_ids", "weight_summary", "domain"},
    "model_received": {"node_id", "round", "peer_id", "weight_summary", "domain"},
    "aggregation_completed": {"node_id", "round", "peer_ids", "sample_count", "weight_summary", "domain"},
    "peer_discovered": {"node_id", "round", "peer_id", "domain"},
    "peer_disconnected": {"node_id", "peer_id", "domain"},
    "targets_selected": {"node_id", "round", "selected_targets", "domain"},
    "ipv8_payload_dropped": {"node_id", "peer_id", "transport", "reason", "security_event", "domain"},
    "tunnel_payload_dropped": {"node_id", "peer_id", "reason", "security_event", "domain"},
    "telemetry.events_dropped": {"base_url", "dropped_kind", "queue_size", "max_pending_items"},
    "telemetry.delivery_failed": {"base_url", "kind", "error", "error_type", "consecutive_failures", "pending_queue_size"},
    "telemetry.disconnected": {"base_url", "kind", "error", "error_type", "consecutive_failures", "pending_queue_size"},
}


class TelemetryStore:
    """In-memory session telemetry store for the dashboard backend."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        max_nodes: int = DEFAULT_TELEMETRY_MAX_NODES,
        max_events_per_node: int = DEFAULT_TELEMETRY_MAX_EVENTS_PER_NODE,
        max_rounds_per_node: int = DEFAULT_TELEMETRY_MAX_ROUNDS_PER_NODE,
        max_edges: int = DEFAULT_TELEMETRY_MAX_EDGES,
        max_peer_ids_per_node: int = DEFAULT_TELEMETRY_MAX_PEER_IDS_PER_NODE,
    ):
        self.session = SessionSnapshot(session_id=session_id or uuid4().hex)
        self.max_nodes = max_nodes
        self.max_events_per_node = max_events_per_node
        self.max_rounds_per_node = max_rounds_per_node
        self.max_edges = max_edges
        self.max_peer_ids_per_node = max_peer_ids_per_node
        self.connection: Dict[str, Any] = {
            "status": "idle",
            "detail": "Waiting for telemetry bootstrap",
            "mode": "unknown",
            "url": None,
            "attempts": 0,
            "last_error": None,
            "last_connected_at": None,
        }
        self._nodes: Dict[str, NodeSnapshot] = {}
        self._events: Dict[str, List[NodeEvent]] = {}
        self._rounds: Dict[str, Dict[int, NodeRoundSummary]] = {}
        self._edges: Dict[Tuple[str, str, str], NetworkEdge] = {}
        self._manifests: Dict[str, dict] = {}
        self._swarm_nodes: Dict[str, set] = defaultdict(set)

    def ingest_heartbeat(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        timestamp = _parse_datetime(snapshot.get("timestamp")) or datetime.now()
        node_id = self._require_non_empty_node_id(snapshot.get("node_id"))
        node = self._get_or_create_node(node_id, snapshot.get("domain"), timestamp)
        node.domain = snapshot.get("domain", node.domain)
        node.connection_mode = snapshot.get("connection_mode", node.connection_mode)
        node.running = bool(snapshot.get("running", node.running))
        node.status = "running" if node.running else "idle"
        node.current_round = self._coerce_int(
            snapshot.get("current_round"),
            default=node.current_round or 0,
            field_name="current_round",
        )
        node.peer_ids = self._cap_sequence(
            list(snapshot.get("known_peers", snapshot.get("peer_ids", node.peer_ids))),
            self.max_peer_ids_per_node,
        )
        node.known_peer_count = self._coerce_int(
            snapshot.get("known_peer_count", snapshot.get("ipv8_peers", 0) + snapshot.get("tunnel_peers", 0))
            or len(node.peer_ids),
            default=len(node.peer_ids),
            field_name="known_peer_count",
        )
        self._update_node_swarm_state(node, snapshot)
        node.refresh_uptime(timestamp)
        return self._broadcasts_for(node, None, None)

    def set_connection_status(
        self,
        status: str,
        *,
        detail: Optional[str] = None,
        mode: Optional[str] = None,
        url: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        self.connection["status"] = status
        if detail is not None:
            self.connection["detail"] = detail
        if mode is not None:
            self.connection["mode"] = mode
        if url is not None:
            self.connection["url"] = url
        self.connection["last_error"] = last_error
        if status == "open":
            self.connection["last_connected_at"] = datetime.now().isoformat()
        elif status in {"connecting", "error"}:
            self.connection["attempts"] += 1

    def ingest_event(self, event_type: str, payload: Dict[str, Any], timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        timestamp = timestamp or datetime.now()
        if not self._is_supported_event_type(event_type):
            raise ValueError(f"Unknown telemetry event type: {event_type}")

        # T-OBS-14: Filter payload keys against per-type allowlist
        allowed_keys = PAYLOAD_KEY_ALLOWLIST.get(event_type)
        if allowed_keys is not None:
            payload = {k: v for k, v in payload.items() if k in allowed_keys}

        node_id = self._require_non_empty_node_id(payload.get("node_id"))
        node = self._get_or_create_node(node_id, payload.get("domain"), timestamp)
        if event_type == "node.started":
            self._update_node_swarm_state(node, payload)
        event = NodeEvent(event_type=event_type, timestamp=timestamp, payload=dict(payload))
        node_events = self._events.setdefault(node_id, [])
        node_events.append(event)
        if self.max_events_per_node > 0 and len(node_events) > self.max_events_per_node:
            self._events[node_id] = node_events[-self.max_events_per_node:]

        edge: Optional[NetworkEdge] = None
        round_summary = self._round_summary_for(node_id, payload.get("round"))
        node.refresh_uptime(timestamp)

        if event_type == "training_completed":
            node.last_loss = payload.get("loss")
            node.last_accuracy = payload.get("accuracy")
            node.last_samples_trained = payload.get("samples_trained")
            node.last_training_at = timestamp
            node.current_round = self._coerce_int(payload.get("round"), default=node.current_round, field_name="round")
            node.training_rounds_completed = max(
                node.training_rounds_completed,
                node.current_round,
            )
            if round_summary:
                round_summary.loss = payload.get("loss")
                round_summary.accuracy = payload.get("accuracy")
                round_summary.samples_trained = payload.get("samples_trained")
                round_summary.updated_at = timestamp
        elif event_type == "targets_selected":
            targets = list(payload.get("selected_targets") or [])
            node.last_selected_peer_ids = self._cap_sequence(targets, self.max_peer_ids_per_node)
            if round_summary:
                round_summary.selected_targets = self._cap_sequence(targets, self.max_peer_ids_per_node)
                round_summary.updated_at = timestamp
        elif event_type == "model_sent":
            peer_ids = list(payload.get("peer_ids") or [])
            round_number = self._coerce_int(payload.get("round"), default=None, field_name="round")
            weight_summary = dict(payload.get("weight_summary") or {})
            node.models_sent += len(peer_ids)
            node.last_send_at = timestamp
            node.last_sent_peer_ids = self._cap_sequence(peer_ids, self.max_peer_ids_per_node)
            node.last_weight_summary = weight_summary
            if round_summary:
                round_summary.sent_peer_ids = self._cap_sequence(peer_ids, self.max_peer_ids_per_node)
                round_summary.updated_at = timestamp
            for peer_id in peer_ids:
                edge = self._touch_edge(
                    node_id,
                    peer_id,
                    timestamp,
                    event_type,
                    edge_type="model_transfer",
                    exchange_delta=1,
                    round_number=round_number,
                    weight_summary=weight_summary,
                )
        elif event_type == "model_received":
            peer_id = payload.get("peer_id")
            round_number = self._coerce_int(payload.get("round"), default=None, field_name="round")
            weight_summary = dict(payload.get("weight_summary") or {})
            node.models_received += 1
            node.last_receive_at = timestamp
            node.last_received_peer_id = peer_id
            node.last_weight_summary = weight_summary
            if round_summary and peer_id and peer_id not in round_summary.received_peer_ids:
                round_summary.received_peer_ids.append(peer_id)
                round_summary.received_peer_ids = self._cap_sequence(
                    round_summary.received_peer_ids,
                    self.max_peer_ids_per_node,
                )
                round_summary.updated_at = timestamp
            if peer_id:
                edge = self._touch_edge(
                    peer_id,
                    node_id,
                    timestamp,
                    event_type,
                    edge_type="model_transfer",
                    exchange_delta=1,
                    round_number=round_number,
                    weight_summary=weight_summary,
                )
        elif event_type == "aggregation_completed":
            peer_ids = list(payload.get("peer_ids") or [])
            node.aggregations_completed += 1
            node.last_aggregation_at = timestamp
            node.last_aggregation_peer_count = len(peer_ids)
            node.last_aggregation_total_samples = payload.get("sample_count")
            node.last_weight_summary = dict(payload.get("weight_summary") or {})
            if round_summary:
                round_summary.aggregated_peer_ids = self._cap_sequence(peer_ids, self.max_peer_ids_per_node)
                round_summary.aggregation_total_samples = payload.get("sample_count")
                round_summary.updated_at = timestamp
        elif event_type == "peer_discovered":
            peer_id = payload.get("peer_id")
            round_number = self._coerce_int(payload.get("round"), default=None, field_name="round")
            if peer_id:
                if peer_id not in node.peer_ids:
                    node.peer_ids.append(peer_id)
                    node.peer_ids = self._cap_sequence(node.peer_ids, self.max_peer_ids_per_node)
                node.known_peer_count = len(node.peer_ids)
                edge = self._touch_edge(
                    node_id,
                    peer_id,
                    timestamp,
                    event_type,
                    discovery_delta=1,
                    round_number=round_number,
                )
        elif event_type == "peer_disconnected":
            peer_id = payload.get("peer_id")
            if peer_id and peer_id in node.peer_ids:
                node.peer_ids.remove(peer_id)
                node.known_peer_count = len(node.peer_ids)
                edge = self._touch_edge(node_id, peer_id, timestamp, event_type)

        return self._broadcasts_for(node, event, edge)

    def get_session(self) -> Dict[str, Any]:
        self._refresh_session()
        return self.session.to_dict()

    def get_nodes(self) -> List[Dict[str, Any]]:
        self._refresh_session()
        return [
            self._nodes[node_id].to_dict()
            for node_id in sorted(self._nodes)
        ]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        node = self._nodes.get(node_id)
        return node.to_dict() if node else None

    def get_node_events(self, node_id: str) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._events.get(node_id, [])]

    def get_events(self) -> List[Dict[str, Any]]:
        return [self._normalize_event(event) for event in self._sorted_events()]

    def get_node_rounds(self, node_id: str) -> List[Dict[str, Any]]:
        rounds = self._rounds.get(node_id, {})
        return [
            rounds[round_number].to_dict()
            for round_number in sorted(rounds)
        ]

    def get_rounds(self) -> List[Dict[str, Any]]:
        rounds = [
            round_summary
            for node_rounds in self._rounds.values()
            for round_summary in node_rounds.values()
        ]
        rounds.sort(key=lambda round_summary: round_summary.updated_at, reverse=True)
        return [round_summary.to_dict() for round_summary in rounds]

    def get_network_graph(self) -> Dict[str, Any]:
        return {
            "nodes": self.get_nodes(),
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }

    def get_network_stats(self) -> Dict[str, Any]:
        self._refresh_session()
        isolated_nodes = [
            node.node_id
            for node in self._nodes.values()
            if node.known_peer_count == 0
        ]
        return {
            "total_nodes": len(self._nodes),
            "active_edge_count": len(self._edges),
            "isolated_nodes": isolated_nodes,
            "message_volume": sum(edge.exchange_count for edge in self._edges.values()),
        }

    def get_swarms(self) -> List[Dict[str, Any]]:
        swarms: Dict[str, SwarmSnapshot] = {}
        domain_sets: Dict[str, set] = defaultdict(set)
        for node_id, node in self._nodes.items():
            sid = node.swarm_id
            if not sid:
                continue
            manifest = self._manifests.get(sid, {})
            if sid not in swarms:
                swarms[sid] = SwarmSnapshot(
                    swarm_id=sid,
                    swarm_name=manifest.get("name") or node.swarm_name or "Unnamed swarm",
                    manifest_hash=node.manifest_hash or sid,
                    description=manifest.get("description", ""),
                    peer_count=0,
                    aggregation_name=manifest.get("aggregation", {}).get("name") or node.aggregation_name or "",
                    topology_name=manifest.get("topology", {}).get("name") or node.topology_name or "",
                    task_type=manifest.get("task", {}).get("type", ""),
                    input_shape=manifest.get("task", {}).get("input_shape", []),
                    output_shape=manifest.get("task", {}).get("output_shape", []),
                    label_type=manifest.get("task", {}).get("label_type", ""),
                    round_limit=manifest.get("round_limit"),
                    created_at=manifest.get("created_at"),
                    domains=[],
                )
            domain_sets[sid].add(node.domain)
        for sid in self._swarm_nodes:
            if sid not in swarms:
                manifest = self._manifests.get(sid, {})
                swarms[sid] = SwarmSnapshot(
                    swarm_id=sid,
                    swarm_name=manifest.get("name") or "Unnamed swarm",
                    manifest_hash=manifest.get("manifest_hash") or sid,
                    description=manifest.get("description", ""),
                    peer_count=0,
                    aggregation_name=manifest.get("aggregation", {}).get("name", ""),
                    topology_name=manifest.get("topology", {}).get("name", ""),
                    task_type=manifest.get("task", {}).get("type", ""),
                    input_shape=manifest.get("task", {}).get("input_shape", []),
                    output_shape=manifest.get("task", {}).get("output_shape", []),
                    label_type=manifest.get("task", {}).get("label_type", ""),
                    round_limit=manifest.get("round_limit"),
                    created_at=manifest.get("created_at"),
                    domains=[],
                )
        for sid in swarms:
            swarms[sid].peer_count = len(self._swarm_nodes.get(sid, set()))
            swarms[sid].domains = list(domain_sets[sid])
        return [s.to_dict() for s in swarms.values()]

    def get_manifest(self, swarm_id: str) -> Optional[dict]:
        return self._manifests.get(swarm_id)

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        self._refresh_session()
        return {
            "connection": self._connection_snapshot(),
            "session": self.session.to_dict(),
            "nodes": self.get_nodes(),
            "events": [self._normalize_event(event) for event in self._sorted_events()],
            "rounds": self.get_rounds(),
            "network": {
                "nodes": self.get_nodes(),
                "edges": [edge.to_dict() for edge in self._edges.values()],
                "stats": self.get_network_stats(),
                "swarms": self.get_swarms(),
            },
        }

    def _get_or_create_node(self, node_id: str, domain: Optional[str], timestamp: datetime) -> NodeSnapshot:
        node = self._nodes.get(node_id)
        if node is None:
            self._prune_nodes_if_needed(incoming_node_id=node_id)
            node = NodeSnapshot(
                node_id=node_id,
                domain=domain or "unknown",
                session_started_at=timestamp,
                last_seen_at=timestamp,
            )
            self._nodes[node_id] = node
        return node

    def _update_node_swarm_state(self, node: NodeSnapshot, data: Dict[str, Any]) -> None:
        old_swarm_id = node.swarm_id
        new_swarm_id = data.get("swarm_id") or data.get("manifest_hash")
        if old_swarm_id and old_swarm_id != new_swarm_id:
            self._swarm_nodes[old_swarm_id].discard(node.node_id)
        if new_swarm_id:
            node.swarm_id = new_swarm_id
        node.swarm_name = data.get("swarm_name") or node.swarm_name
        node.manifest_hash = data.get("manifest_hash") or node.manifest_hash
        node.aggregation_name = data.get("aggregation_name") or node.aggregation_name
        node.topology_name = data.get("topology_name") or node.topology_name
        if node.swarm_id:
            self._swarm_nodes[node.swarm_id].add(node.node_id)
        if "manifest" in data and node.swarm_id:
            self._manifests[node.swarm_id] = data["manifest"]

    def _round_summary_for(self, node_id: str, round_number: Any) -> Optional[NodeRoundSummary]:
        if round_number is None:
            return None
        round_number = self._coerce_int(round_number, field_name="round")
        rounds = self._rounds.get(node_id)
        if rounds is None:
            rounds = {}
            self._rounds[node_id] = rounds
        if round_number not in rounds:
            self._prune_rounds_if_needed(node_id, incoming_round_number=round_number)
            rounds[round_number] = NodeRoundSummary(node_id=node_id, round_number=round_number)
        return rounds[round_number]

    @staticmethod
    def _is_supported_event_type(event_type: str) -> bool:
        return event_type in SUPPORTED_TELEMETRY_EVENT_TYPES or event_type.startswith("security.")

    @staticmethod
    def _require_non_empty_node_id(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("payload.node_id is required")
        return value

    @staticmethod
    def _coerce_int(value: Any, default: Any = 0, field_name: str = "round") -> Any:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer") from exc

    def _touch_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        timestamp: datetime,
        event_type: str,
        *,
        edge_type: str = "peer_link",
        exchange_delta: int = 0,
        discovery_delta: int = 0,
        round_number: Optional[int] = None,
        weight_summary: Optional[Dict[str, Any]] = None,
    ) -> NetworkEdge:
        key = (source_node_id, target_node_id, edge_type)
        edge = self._edges.get(key)
        if edge is None:
            self._prune_edges_if_needed(incoming_key=key)
            edge = NetworkEdge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
                last_active_at=timestamp,
            )
            self._edges[key] = edge
        edge.last_active_at = timestamp
        edge.last_event_type = event_type
        edge.exchange_count += exchange_delta
        edge.discovery_count += discovery_delta
        edge.last_round = round_number
        edge.last_weight_summary = dict(weight_summary or edge.last_weight_summary)
        return edge

    def _prune_nodes_if_needed(self, incoming_node_id: Optional[str] = None) -> None:
        if self.max_nodes <= 0:
            return
        while len(self._nodes) >= self.max_nodes:
            candidates = [node for node in self._nodes.values() if node.node_id != incoming_node_id]
            if not candidates:
                break
            oldest = min(candidates, key=lambda node: (node.last_seen_at or datetime.min, node.node_id))
            self._drop_node(oldest.node_id)

    def _drop_node(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node is not None and node.swarm_id:
            self._swarm_nodes[node.swarm_id].discard(node_id)
        self._nodes.pop(node_id, None)
        self._events.pop(node_id, None)
        self._rounds.pop(node_id, None)
        doomed_edges = [key for key, edge in self._edges.items() if edge.source_node_id == node_id or edge.target_node_id == node_id]
        for key in doomed_edges:
            self._edges.pop(key, None)

    def _prune_rounds_if_needed(self, node_id: str, incoming_round_number: Optional[int] = None) -> None:
        if self.max_rounds_per_node <= 0:
            return
        rounds = self._rounds.get(node_id, {})
        while len(rounds) >= self.max_rounds_per_node:
            candidates = [rn for rn in rounds if rn != incoming_round_number]
            if not candidates:
                break
            oldest_round = min(candidates)
            rounds.pop(oldest_round, None)

    def _prune_edges_if_needed(self, incoming_key: Optional[Tuple[str, str, str]] = None) -> None:
        if self.max_edges <= 0:
            return
        while len(self._edges) >= self.max_edges:
            candidates = [(key, edge) for key, edge in self._edges.items() if key != incoming_key]
            if not candidates:
                break
            oldest_key, _ = min(candidates, key=lambda item: (item[1].last_active_at or datetime.min, item[0]))
            self._edges.pop(oldest_key, None)

    @staticmethod
    def _cap_sequence(values: List[Any], limit: int) -> List[Any]:
        if limit <= 0:
            return []
        if len(values) <= limit:
            return list(values)
        return list(values[-limit:])

    def _sorted_events(self) -> List[NodeEvent]:
        events = [event for node_events in self._events.values() for event in node_events]
        events.sort(key=lambda event: event.timestamp, reverse=True)
        return events

    def _connection_snapshot(self) -> Dict[str, Any]:
        return {
            "status": self.connection["status"],
            "detail": self.connection["detail"],
            "mode": self.connection["mode"],
            "url": self.connection["url"],
            "attempts": self.connection["attempts"],
            "last_error": self.connection["last_error"],
            "last_connected_at": self.connection["last_connected_at"],
        }

    def _normalize_event(self, event: NodeEvent) -> Dict[str, Any]:
        payload = dict(event.payload)
        peer_ids = list(payload.get("peer_ids") or [])
        related_node_id = payload.get("peer_id") or (peer_ids[0] if peer_ids else None)
        round_number = payload.get("round")
        node_id = payload.get("node_id")

        if event.event_type == "model_received":
            title = "Model received"
            summary = f"{related_node_id} -> {node_id}"
            if round_number is not None:
                summary = f"{summary} · round {round_number}"
        elif event.event_type == "model_sent":
            title = "Model sent"
            summary = f"{node_id} -> {related_node_id}"
            if round_number is not None:
                summary = f"{summary} · round {round_number}"
        else:
            title = event.event_type.replace("_", " ").title()
            summary = payload.get("summary") or node_id
            if round_number is not None and payload.get("summary") is None:
                summary = f"{summary} · round {round_number}"

        return {
            "id": f"{event.timestamp.isoformat()}::{node_id}::{event.event_type}",
            "timestamp": event.timestamp.isoformat(),
            "node_id": node_id,
            "event_type": event.event_type,
            "title": title,
            "summary": summary,
            "related_node_id": related_node_id,
            "round": round_number,
            "severity": "info",
            "payload": payload,
        }

    def _refresh_session(self) -> None:
        active_nodes = [node for node in self._nodes.values() if node.running]
        self.session.last_updated_at = datetime.now()
        self.session.active_node_count = len(active_nodes)
        self.session.total_edge_count = len(self._edges)
        self.session.recent_exchange_count = sum(
            edge.exchange_count for edge in self._edges.values()
        )
        self.session.recent_aggregation_count = sum(
            node.aggregations_completed for node in self._nodes.values()
        )
        self.session.active_domains = sorted(
            {node.domain for node in self._nodes.values() if node.domain}
        )
        self.session.selected_node_id = self._select_default_node_id()

    def _select_default_node_id(self) -> Optional[str]:
        running_nodes = sorted(
            (node for node in self._nodes.values() if node.running),
            key=lambda node: (node.last_seen_at or datetime.min, node.node_id),
            reverse=True,
        )
        if running_nodes:
            return running_nodes[0].node_id
        return sorted(self._nodes)[0] if self._nodes else None

    def _broadcasts_for(
        self,
        node: NodeSnapshot,
        event: Optional[NodeEvent],
        edge: Optional[NetworkEdge],
    ) -> List[Dict[str, Any]]:
        self._refresh_session()
        broadcasts = [
            {
                "type": "node_snapshot_updated",
                "payload": node.to_dict(),
            },
            {
                "type": "session_stats_updated",
                "payload": self.session.to_dict(),
            },
        ]
        if event is not None:
            broadcasts.append(
                {
                    "type": "node_event_received",
                    "payload": self._normalize_event(event),
                }
            )
        if edge is not None:
            broadcasts.append(
                {
                    "type": "network_edge_updated",
                    "payload": edge.to_dict(),
                }
            )
        return broadcasts


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
