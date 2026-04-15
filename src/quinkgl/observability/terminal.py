from typing import Any, Callable, Iterable, List

from quinkgl.observability.events import RuntimeEvent


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _format_peer_list(peer_ids: Any) -> str:
    if peer_ids is None:
        return ""
    if isinstance(peer_ids, (list, tuple)):
        values = [str(item) for item in peer_ids if item is not None]
        return ", ".join(values)
    if isinstance(peer_ids, set):
        values = sorted((str(item) for item in peer_ids if item is not None))
        return ", ".join(values)
    return str(peer_ids)


def _format_weight_summary(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""

    parts: List[str] = []
    kind = summary.get("kind")
    if kind:
        parts.append(f"weights={kind}")

    if "layer_count" in summary:
        parts.append(f"layers={summary['layer_count']}")
    elif "field_count" in summary:
        parts.append(f"fields={summary['field_count']}")
    elif "item_count" in summary:
        parts.append(f"items={summary['item_count']}")

    if "total_elements" in summary:
        parts.append(f"total_elements={summary['total_elements']}")

    return " ".join(parts)


def _format_safe_value(value: Any) -> str:
    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in sorted(value, key=lambda item: repr(item)))
        return f"dict[{keys}]"
    if isinstance(value, (list, tuple, set)):
        return _format_peer_list(value)
    if isinstance(value, (int, float)):
        return _format_number(value)
    return str(value)


def _prefix(event: RuntimeEvent) -> str:
    payload = event.payload or {}
    node_id = payload.get("node_id", "?")
    prefix = [f"[NODE {node_id}]"]
    if payload.get("round") is not None:
        prefix.append(f"[ROUND {payload['round']}]")
    return "".join(prefix)


def _join_parts(parts: Iterable[str]) -> str:
    filtered = [part for part in parts if part]
    return " ".join(filtered)


def _format_training_event(event: RuntimeEvent, action: str) -> str:
    payload = event.payload
    parts = [_prefix(event), action]
    if payload.get("loss") is not None:
        parts.append(f"loss={_format_number(payload['loss'])}")
    if payload.get("accuracy") is not None:
        parts.append(f"acc={_format_number(payload['accuracy'])}")
    if payload.get("samples_trained") is not None:
        parts.append(f"samples={payload['samples_trained']}")
    return _join_parts(parts)


def _format_selected_targets(event: RuntimeEvent) -> str:
    payload = event.payload
    targets = _format_peer_list(payload.get("selected_targets"))
    suffix = f"selected peers -> {targets}" if targets else "selected peers -> none"
    return _join_parts([_prefix(event), suffix])


def _format_send_event(event: RuntimeEvent, action: str, peer_key: str) -> str:
    payload = event.payload
    parts = [_prefix(event), action]
    peers = _format_peer_list(payload.get(peer_key))
    if peers:
        arrow = "->" if peer_key == "peer_ids" else "<-"
        parts.append(f"{arrow} {peers}")
    weight_summary = _format_weight_summary(payload.get("weight_summary"))
    if weight_summary:
        parts.append(weight_summary)
    return _join_parts(parts)


def _format_aggregation(event: RuntimeEvent) -> str:
    payload = event.payload
    parts = [_prefix(event), "aggregated models"]
    peer_ids = payload.get("peer_ids") or []
    if peer_ids:
        peer_count = len([peer for peer in peer_ids if peer is not None])
        parts.append(f"peers={peer_count}")
    if payload.get("sample_count") is not None:
        parts.append(f"total_samples={payload['sample_count']}")
    return _join_parts(parts)


def _format_node_config(event: RuntimeEvent) -> str:
    """Render the startup configuration banner."""
    p = event.payload
    sep = "=" * 72
    lines = [sep, f"QuinkGL v{p.get('version', '?')}", sep]

    fields = [
        ("Node ID", p.get("node_id")),
        ("Domain", p.get("domain")),
        ("Connection", f"{p.get('connection_mode', '?')} (port {p.get('port', '?')})"),
        ("Topology", p.get("topology")),
        ("Aggregation", p.get("aggregation")),
        ("Model", p.get("model")),
        ("Gossip Interval", f"{p.get('gossip_interval', '?')}s"),
    ]

    dp = p.get("data_policy")
    if dp:
        fields.append(("Fingerprint", str(dp.get("fingerprint_enabled", False))))
        if dp.get("min_affinity") is not None:
            fields.append(("Min Affinity", str(dp["min_affinity"])))
        if dp.get("privacy_level") is not None:
            fields.append(("Privacy Level", str(dp["privacy_level"])))

    fp = p.get("fingerprint_summary")
    if fp:
        fields.append(("Fingerprint", f"{fp.get('label_buckets', 0)} label buckets, sample_bucket={fp.get('sample_bucket', '?')}"))

    max_label = max((len(label) for label, _ in fields), default=0)
    for label, value in fields:
        lines.append(f"{label:<{max_label}} : {value}")

    lines.append(sep)
    return "\n".join(lines)


def _format_node_started(event: RuntimeEvent) -> str:
    p = event.payload
    return f"[NODE {p.get('node_id', '?')}] started ({p.get('connection_mode', '?')})"


def _format_node_stopped(event: RuntimeEvent) -> str:
    p = event.payload
    parts = [f"[NODE {p.get('node_id', '?')}]", "stopped"]
    if p.get("total_rounds") is not None:
        parts.append(f"rounds={p['total_rounds']}")
    if p.get("uptime_seconds") is not None:
        parts.append(f"uptime={p['uptime_seconds']}s")
    return " ".join(parts)


def _format_telemetry_connected(event: RuntimeEvent) -> str:
    p = event.payload
    return f"[TELEMETRY] connected to {p.get('base_url', '?')} (heartbeat every {p.get('heartbeat_interval', '?')}s)"


def _format_fallback_payload(payload: dict) -> str:
    pieces: List[str] = []
    for key in sorted(payload):
        if key in {"node_id", "round"}:
            continue
        value = payload[key]
        if value is None:
            continue
        if key == "weight_summary":
            formatted = _format_weight_summary(value)
            if formatted:
                pieces.append(formatted)
            continue
        if key in {"selected_targets", "peer_ids"}:
            formatted = _format_peer_list(value)
            if formatted:
                pieces.append(f"{key}={formatted}")
            continue
        pieces.append(f"{key}={_format_safe_value(value)}")
    return " ".join(pieces)


def format_runtime_event(event: RuntimeEvent) -> str:
    if event.event_type == "training_started":
        return _format_training_event(event, "training started")
    if event.event_type == "training_completed":
        return _format_training_event(event, "training completed")
    if event.event_type == "targets_selected":
        return _format_selected_targets(event)
    if event.event_type == "model_send_started":
        return _format_send_event(event, "sending model", "peer_ids")
    if event.event_type == "model_sent":
        return _format_send_event(event, "sent model", "peer_ids")
    if event.event_type == "model_received":
        return _format_send_event(event, "received model", "peer_id")
    if event.event_type == "aggregation_completed":
        return _format_aggregation(event)
    if event.event_type == "node.config":
        return _format_node_config(event)
    if event.event_type == "node.started":
        return _format_node_started(event)
    if event.event_type == "node.stopped":
        return _format_node_stopped(event)
    if event.event_type == "telemetry.connected":
        return _format_telemetry_connected(event)

    fallback = _format_fallback_payload(event.payload or {})
    if fallback:
        return _join_parts([_prefix(event), event.event_type, fallback])
    return _join_parts([_prefix(event), event.event_type])


class TerminalObserver:
    def __init__(self, printer: Callable[[str], None] = print):
        self.printer = printer

    def handle(self, event: RuntimeEvent) -> None:
        self.printer(format_runtime_event(event))
