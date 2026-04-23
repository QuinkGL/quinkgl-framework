from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import enum
import logging
from typing import Any, Callable, Dict, List, Set


logger = logging.getLogger(__name__)


# T-OBS-02: Enumerated event types for schema stability.
# New types must be added here AND in telemetry/store.py:SUPPORTED_TELEMETRY_EVENT_TYPES.
class RuntimeEventType(str, enum.Enum):
    """Canonical event type identifiers for the QuinkGL runtime."""

    AGGREGATION_COMPLETED = "aggregation_completed"
    AGGREGATION_FAILED = "aggregation_failed"
    CONSENSUS_REACHED = "consensus_reached"
    EARLY_STOPPING = "early_stopping"
    IPV8_PAYLOAD_DROPPED = "ipv8_payload_dropped"
    MODEL_RECEIVED = "model_received"
    MODEL_REJECTED_BACKPRESSURE = "model_rejected_backpressure"
    MODEL_REJECTED_DUPLICATE = "model_rejected_duplicate"
    MODEL_REJECTED_STALE = "model_rejected_stale"
    MODEL_SEND_FAILED = "model_send_failed"
    MODEL_SEND_STARTED = "model_send_started"
    MODEL_SENT = "model_sent"
    MODELS_CONVERGED = "models_converged"
    NODE_CONFIG = "node.config"
    NODE_STARTED = "node.started"
    NODE_STOPPED = "node.stopped"
    PEER_DISCONNECTED = "peer_disconnected"
    PEER_DISCOVERED = "peer_discovered"
    POST_AGGREGATION_EVAL = "post_aggregation_eval"
    ROUND_COMPLETED = "round_completed"
    ROUND_STARTED = "round_started"
    SUBSCRIBER_ERROR = "subscriber.error"
    TARGETS_SELECTED = "targets_selected"
    TELEMETRY_CONNECTED = "telemetry.connected"
    TELEMETRY_DELIVERY_FAILED = "telemetry.delivery_failed"
    TELEMETRY_DISCONNECTED = "telemetry.disconnected"
    TELEMETRY_EVENTS_DROPPED = "telemetry.events_dropped"
    TELEMETRY_STATUS_PROVIDER_WARNING = "telemetry.status_provider_warning"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_STARTED = "training_started"
    TUNNEL_PAYLOAD_DROPPED = "tunnel_payload_dropped"


# T-OBS-02: Schema version for forward-compatible event deserialization.
EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeEvent:
    """Frozen event envelope carrying a safe per-delivery payload snapshot.

    T-OBS-02: ``schema_version`` enables future consumers to detect schema
    changes and migrate accordingly.  ``event_type`` is accepted as a plain
    string for backward compatibility; callers should prefer the
    ``RuntimeEventType`` enum.
    """

    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    schema_version: int = EVENT_SCHEMA_VERSION


def _copy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Each delivery gets its own deep-copied snapshot so subscribers can
    # inspect ordinary dict/list payloads without mutating shared state.
    return _snapshot_value(payload, set())


class _UnreprSentinel:
    """Sentinel for values that cannot be copied or represented safely."""
    def __repr__(self):
        return "<unrepresentable_value>"


_UNREPR_SENTINEL = _UnreprSentinel()


def _snapshot_value(value: Any, active: Set[int]) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        object_id = id(value)
        if object_id in active:
            return f"<recursive {type(value).__name__}>"

        active.add(object_id)
        try:
            if isinstance(value, dict):
                return {
                    _snapshot_value(key, active): _snapshot_value(inner, active)
                    for key, inner in value.items()
                }
            if isinstance(value, list):
                return [_snapshot_value(item, active) for item in value]
            if isinstance(value, tuple):
                return tuple(_snapshot_value(item, active) for item in value)
            return {_snapshot_value(item, active) for item in value}
        finally:
            active.remove(object_id)

    try:
        return deepcopy(value)
    except Exception:
        return _UNREPR_SENTINEL


class EventEmitter:
    """Minimal runtime event fan-out with isolated subscriber deliveries."""

    def __init__(self):
        self._subscribers: List[Callable[[RuntimeEvent], None]] = []

    def subscribe(self, callback: Callable[[RuntimeEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, event_type: str, payload: Dict[str, Any]) -> RuntimeEvent:
        """Emit a frozen event whose payload is copied for each subscriber."""
        timestamp = datetime.now()
        snapshot = _copy_payload(payload)
        event = RuntimeEvent(
            event_type=event_type,
            payload=snapshot,
            timestamp=timestamp,
        )
        for callback in list(self._subscribers):
            try:
                callback_payload = _copy_payload(snapshot) if getattr(callback, "needs_isolated_payload", False) else snapshot
                callback(
                    RuntimeEvent(
                        event_type=event_type,
                        payload=callback_payload,
                        timestamp=timestamp,
                    )
                )
            except Exception as exc:
                logger.exception("Event subscriber failed while handling %s", event_type)
                if event_type != "subscriber.error":
                    self.emit(
                        "subscriber.error",
                        {
                            "source_event_type": event_type,
                            "subscriber": getattr(callback, "__name__", repr(callback)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                continue
        return event
