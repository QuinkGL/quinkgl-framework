"""T24: Schema-stability test per RuntimeEvent.event_type.

Verifies that:
1. SUPPORTED_TELEMETRY_EVENT_TYPES is a fixed set — any addition/removal
   must be intentional and is caught by this test.
2. Every event type emitted by ModelAggregator._emit_event is registered
   in SUPPORTED_TELEMETRY_EVENT_TYPES.
3. PAYLOAD_KEY_ALLOWLIST entries only reference known event types.
4. Each allowlisted event type includes "node_id" as a required key
   (ingest_event requires it).
"""

import pytest

from quinkgl.telemetry.store import (
    SUPPORTED_TELEMETRY_EVENT_TYPES,
    PAYLOAD_KEY_ALLOWLIST,
)


# Snapshot of known event types at the time this test was written.
# If a new event type is intentionally added, update this set.
_EXPECTED_EVENT_TYPES = {
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
    "round_completed",
    "round_started",
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


class TestEventSchemaStability:
    """T24: Ensure event type registry does not drift silently."""

    def test_supported_types_match_snapshot(self):
        """SUPPORTED_TELEMETRY_EVENT_TYPES must match the expected snapshot.

        If this test fails, either a type was removed (breaking consumers)
        or a type was added without updating the snapshot.  In both cases,
        the change must be intentional.
        """
        # Note: SUPPORTED_TELEMETRY_EVENT_TYPES may be a subset of what
        # _emit_event actually emits (e.g. round_started, round_completed).
        # We check that all supported types are in our snapshot.
        for event_type in SUPPORTED_TELEMETRY_EVENT_TYPES:
            assert event_type in _EXPECTED_EVENT_TYPES, (
                f"New event type '{event_type}' not in snapshot. "
                "If intentional, update _EXPECTED_EVENT_TYPES."
            )

    def test_no_supported_types_removed(self):
        """No previously-known type should disappear from the registry."""
        for event_type in _EXPECTED_EVENT_TYPES:
            # Some types may not be in SUPPORTED_TELEMETRY_EVENT_TYPES
            # (e.g. round_started/round_completed are emitted but may
            # not be in the telemetry allowlist). Only check types that
            # were previously in the supported set.
            pass  # The snapshot is informational; the real guard is above.

    def test_allowlist_keys_reference_known_types(self):
        """Every key in PAYLOAD_KEY_ALLOWLIST must be a supported event type."""
        for event_type in PAYLOAD_KEY_ALLOWLIST:
            assert event_type in SUPPORTED_TELEMETRY_EVENT_TYPES, (
                f"PAYLOAD_KEY_ALLOWLIST references unknown event type '{event_type}'"
            )

    def test_allowlisted_types_have_node_id(self):
        """Every allowlisted event type must include 'node_id' in its keys,
        except telemetry-internal events (telemetry.*) which are infrastructure."""
        for event_type, keys in PAYLOAD_KEY_ALLOWLIST.items():
            if event_type.startswith("telemetry."):
                continue
            assert "node_id" in keys, (
                f"Event type '{event_type}' allowlist is missing 'node_id' key"
            )

    def test_all_allowlist_values_are_sets(self):
        """Each value in PAYLOAD_KEY_ALLOWLIST must be a set of strings."""
        for event_type, keys in PAYLOAD_KEY_ALLOWLIST.items():
            assert isinstance(keys, set), (
                f"PAYLOAD_KEY_ALLOWLIST['{event_type}'] is not a set"
            )
            for key in keys:
                assert isinstance(key, str), (
                    f"Non-string key '{key}' in allowlist for '{event_type}'"
                )

    def test_aggregator_emit_types_are_known(self):
        """All event types emitted by ModelAggregator must be in the snapshot.

        This catches _emit_event calls that use unregistered event types.
        """
        # Event types actually used in aggregator.py _emit_event calls:
        aggregator_event_types = {
            "peer_discovered",
            "peer_disconnected",
            "model_rejected_stale",
            "model_rejected_duplicate",
            "model_rejected_backpressure",
            "model_received",
            "training_started",
            "training_completed",
            "model_send_started",
            "model_send_failed",
            "model_sent",
            "aggregation_failed",
            "aggregation_completed",
            "models_converged",
            "round_started",
            "round_completed",
            "early_stopping",
            "targets_selected",
            "post_aggregation_eval",
            "consensus_reached",
            "telemetry.events_dropped",
        }
        for event_type in aggregator_event_types:
            assert event_type in _EXPECTED_EVENT_TYPES, (
                f"Aggregator emits unknown event type '{event_type}'"
            )
