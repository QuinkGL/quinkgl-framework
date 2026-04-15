from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Set


@dataclass(frozen=True)
class RuntimeEvent:
    """Frozen event envelope carrying a safe per-delivery payload snapshot."""

    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


def _copy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Each delivery gets its own deep-copied snapshot so subscribers can
    # inspect ordinary dict/list payloads without mutating shared state.
    return _snapshot_value(payload, set())


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
        return repr(value)


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
                callback(
                    RuntimeEvent(
                        event_type=event_type,
                        payload=_copy_payload(snapshot),
                        timestamp=timestamp,
                    )
                )
            except Exception:
                continue
        return event
