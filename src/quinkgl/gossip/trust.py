# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Trust policy enum + training metrics dataclass (spec §10.6.1, §15).

These were carved out of the CURRENT public surface in spec v2.0.0 and
re-categorised as PLANNED (Phase 2).  This module is the Phase 2 landing
for both symbols.

Design notes
------------
* :class:`TrustPolicy` is a plain ``str`` enum so existing call sites
  that pass ``trust_policy="open"`` continue to work unchanged.  New
  callers get a proper typed enum + IDE autocomplete.  The value strings
  are lowercase to match the spec (§15 table) and the CLI flag choices
  (``--trust-policy open|tofu|pinned``).
* :class:`TrainingMetrics` is intentionally separate from the existing
  ``quinkgl.models.TrainingResult`` — the latter describes the outcome
  of an *epoch loop inside a single peer*, while ``TrainingMetrics`` is
  the per-round telemetry snapshot a gossip community emits to observers
  and the telemetry server (§10.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["TrustPolicy", "TrainingMetrics"]


class TrustPolicy(str, Enum):
    """Peer-level trust posture (spec §15).

    Uses ``str`` mixin so the enum values compare equal to the lowercase
    policy names that ``GossipNode`` has always accepted, and so
    serialisation (JSON, logs, telemetry events) falls through to the
    bare string without special handling::

        >>> TrustPolicy.OPEN == "open"
        True
        >>> json.dumps({"policy": TrustPolicy.TOFU})
        '{"policy": "tofu"}'
    """

    OPEN = "open"
    TOFU = "tofu"
    PINNED = "pinned"

    @classmethod
    def coerce(cls, value: Any) -> "TrustPolicy":
        """Normalise a user-provided value into a :class:`TrustPolicy`.

        Accepts either a :class:`TrustPolicy` member or one of the
        lowercase string names.  Raises ``ValueError`` with the allowed
        set on anything else so the error propagates cleanly through the
        existing ``GossipNode.__init__`` validation path.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        allowed = ", ".join(repr(p.value) for p in cls)
        raise ValueError(
            f"invalid trust_policy {value!r}; expected one of {{{allowed}}}"
        )


@dataclass
class TrainingMetrics:
    """Per-round training telemetry snapshot.

    This dataclass is the shape carried by observability events
    (``training.round_completed`` etc. — §10.8.3) and by the
    ``on_round_end`` hook the CLI layer dispatches after every gossip
    round.  All numeric fields are optional because different
    frameworks (and different training loops) can only compute a
    subset: a regression task has no ``accuracy``, a one-sample debug
    run has no meaningful ``samples_trained``, and so on.
    """

    round_number: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    samples_trained: Optional[int] = None
    duration_s: Optional[float] = None
    peer_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dict.

        Omits ``None`` values so downstream observability sinks can tell
        "not reported" from "reported as zero".  ``extra`` is flattened
        into the top level to mirror how the telemetry server indexes
        ad-hoc metric keys today.
        """
        out: Dict[str, Any] = {"round_number": int(self.round_number)}
        for key in ("loss", "accuracy", "samples_trained", "duration_s", "peer_count"):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        for key, val in (self.extra or {}).items():
            if key in out:
                # Never let extra overwrite a canonical key.
                continue
            out[key] = val
        return out
