from quinkgl.observability.events import RuntimeEvent
from quinkgl.observability.terminal import TerminalObserver, format_runtime_event


def test_format_runtime_event_renders_selected_targets_concisely():
    event = RuntimeEvent(
        event_type="targets_selected",
        payload={
            "node_id": "alice",
            "round": 4,
            "selected_targets": ["bob", "carol"],
            "candidate_count": 5,
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == "[NODE alice][ROUND 4] selected peers -> bob, carol"


def test_format_runtime_event_renders_set_selected_targets_deterministically():
    event = RuntimeEvent(
        event_type="targets_selected",
        payload={
            "node_id": "alice",
            "round": 4,
            "selected_targets": {"carol", "bob"},
            "candidate_count": 5,
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == "[NODE alice][ROUND 4] selected peers -> bob, carol"


def test_format_runtime_event_renders_model_sent_with_peer_ids_and_weight_summary():
    event = RuntimeEvent(
        event_type="model_sent",
        payload={
            "node_id": "alice",
            "round": 4,
            "peer_ids": ["bob", "carol"],
            "sample_count": 128,
            "weight_summary": {
                "kind": "dict",
                "field_count": 2,
                "layer_count": 8,
                "total_elements": 124586,
            },
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == (
        "[NODE alice][ROUND 4] sent model -> bob, carol "
        "weights=dict layers=8 total_elements=124586"
    )


def test_format_runtime_event_renders_model_received_with_peer_id_and_weight_summary():
    event = RuntimeEvent(
        event_type="model_received",
        payload={
            "node_id": "alice",
            "round": 4,
            "peer_id": "carol",
            "sample_count": 64,
            "weight_summary": {
                "kind": "dict",
                "field_count": 2,
                "layer_count": 8,
                "total_elements": 124586,
            },
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == (
        "[NODE alice][ROUND 4] received model <- carol "
        "weights=dict layers=8 total_elements=124586"
    )


def test_format_runtime_event_renders_aggregation_with_peer_count():
    event = RuntimeEvent(
        event_type="aggregation_completed",
        payload={
            "node_id": "alice",
            "round": 4,
            "peer_ids": ["alice", "bob", "carol"],
            "sample_count": 384,
            "weight_summary": {
                "kind": "dict",
                "field_count": 2,
                "layer_count": 8,
                "total_elements": 124586,
            },
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == "[NODE alice][ROUND 4] aggregated models peers=3 total_samples=384"


def test_format_runtime_event_handles_unknown_events_sensibly():
    event = RuntimeEvent(
        event_type="custom_debug",
        payload={
            "node_id": "alice",
            "round": 4,
            "message": "hello",
            "peer_ids": ["bob"],
        },
    )

    rendered = format_runtime_event(event)

    assert rendered == "[NODE alice][ROUND 4] custom_debug message=hello peer_ids=bob"


def test_terminal_observer_delegates_to_printer_callable():
    seen = []
    observer = TerminalObserver(printer=seen.append)

    observer.handle(
        RuntimeEvent(
            event_type="training_completed",
            payload={
                "node_id": "alice",
                "round": 4,
                "loss": 0.25,
                "accuracy": 0.75,
                "samples_trained": 128,
            },
        )
    )

    assert seen == [
        "[NODE alice][ROUND 4] training completed loss=0.25 acc=0.75 samples=128"
    ]
