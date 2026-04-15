from quinkgl.telemetry.store import TelemetryStore


def test_store_initializes_empty_session():
    store = TelemetryStore(session_id="session-1")

    session = store.get_session()

    assert session["session_id"] == "session-1"
    assert session["active_node_count"] == 0
    assert session["total_edge_count"] == 0


def test_store_heartbeat_creates_node_snapshot():
    store = TelemetryStore(session_id="session-1")

    broadcasts = store.ingest_heartbeat(
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 2,
            "connection_mode": "ipv8_p2p",
            "known_peers": ["node-b"],
        }
    )

    node = store.get_node("node-a")

    assert node["domain"] == "demo"
    assert node["running"] is True
    assert node["current_round"] == 2
    assert node["known_peer_count"] == 1
    assert broadcasts[0]["type"] == "node_snapshot_updated"


def test_store_event_updates_training_exchange_aggregation_and_graph():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})

    store.ingest_event(
        "training_completed",
        {
            "node_id": "node-a",
            "round": 3,
            "loss": 0.25,
            "accuracy": 0.75,
            "samples_trained": 32,
        },
    )
    store.ingest_event(
        "targets_selected",
        {
            "node_id": "node-a",
            "round": 3,
            "selected_targets": ["node-b"],
        },
    )
    store.ingest_event(
        "model_sent",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_ids": ["node-b"],
            "sample_count": 32,
            "weight_summary": {"kind": "dict", "layer_count": 2, "total_elements": 8},
        },
    )
    store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_id": "node-b",
            "sample_count": 16,
            "weight_summary": {"kind": "dict", "layer_count": 2, "total_elements": 8},
        },
    )
    store.ingest_event(
        "aggregation_completed",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_ids": ["node-a", "node-b"],
            "sample_count": 48,
            "weight_summary": {"kind": "dict", "layer_count": 2, "total_elements": 8},
        },
    )
    store.ingest_event(
        "peer_discovered",
        {
            "node_id": "node-a",
            "peer_id": "node-b",
            "round": 3,
        },
    )

    node = store.get_node("node-a")
    graph = store.get_network_graph()
    rounds = store.get_node_rounds("node-a")

    assert node["last_loss"] == 0.25
    assert node["models_sent"] == 1
    assert node["models_received"] == 1
    assert node["aggregations_completed"] == 1
    assert node["known_peer_count"] == 1
    assert graph["edges"]
    assert rounds[0]["round_number"] == 3
    assert rounds[0]["selected_targets"] == ["node-b"]
    assert rounds[0]["aggregated_peer_ids"] == ["node-a", "node-b"]


def test_store_builds_directed_transfer_edges_and_selected_node():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat(
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 3,
            "known_peers": ["node-b"],
        }
    )
    store.ingest_heartbeat(
        {
            "node_id": "node-b",
            "domain": "demo",
            "running": True,
            "current_round": 3,
            "known_peers": ["node-a"],
        }
    )
    store.ingest_event(
        "model_sent",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_ids": ["node-b"],
            "weight_summary": {
                "layer_count": 2,
                "total_elements": 8,
                "payload_size_bytes": 512,
            },
        },
    )

    session = store.get_session()
    graph = store.get_network_graph()

    assert session["selected_node_id"] == "node-a"
    transfer_edges = [
        edge for edge in graph["edges"] if edge["edge_type"] == "model_transfer"
    ]
    assert len(transfer_edges) == 1
    assert {
        "id": transfer_edges[0]["id"],
        "source_node_id": transfer_edges[0]["source_node_id"],
        "target_node_id": transfer_edges[0]["target_node_id"],
        "edge_type": transfer_edges[0]["edge_type"],
        "exchange_count": transfer_edges[0]["exchange_count"],
        "last_round": transfer_edges[0]["last_round"],
        "last_weight_summary": transfer_edges[0]["last_weight_summary"],
    } == {
        "id": "node-a::node-b::model_transfer",
        "source_node_id": "node-a",
        "target_node_id": "node-b",
        "edge_type": "model_transfer",
        "exchange_count": 1,
        "last_round": 3,
        "last_weight_summary": {
            "layer_count": 2,
            "total_elements": 8,
            "payload_size_bytes": 512,
        },
    }
    assert transfer_edges[0]["last_active_at"]


def test_store_reselects_session_node_when_runtime_state_changes():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat(
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 1,
            "timestamp": "2026-04-08T08:00:00",
        }
    )

    assert store.get_session()["selected_node_id"] == "node-a"

    store.ingest_heartbeat(
        {
            "node_id": "node-b",
            "domain": "demo",
            "running": True,
            "current_round": 2,
            "timestamp": "2026-04-08T09:00:00",
        }
    )

    assert store.get_session()["selected_node_id"] == "node-b"

    store.ingest_heartbeat(
        {
            "node_id": "node-b",
            "domain": "demo",
            "running": False,
            "current_round": 2,
            "timestamp": "2026-04-08T09:01:00",
        }
    )

    assert store.get_session()["selected_node_id"] == "node-a"


def test_store_touches_discovery_edges_for_known_peers():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat(
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "known_peers": ["node-b"],
        }
    )

    store.ingest_event(
        "peer_discovered",
        {
            "node_id": "node-a",
            "peer_id": "node-b",
            "round": 1,
        },
    )
    store.ingest_event(
        "peer_discovered",
        {
            "node_id": "node-a",
            "peer_id": "node-b",
            "round": 2,
        },
    )

    graph = store.get_network_graph()
    discovery_edges = [
        edge
        for edge in graph["edges"]
        if edge["source_node_id"] == "node-a"
        and edge["target_node_id"] == "node-b"
        and edge["edge_type"] == "peer_link"
    ]

    assert len(discovery_edges) == 1
    assert discovery_edges[0]["discovery_count"] == 2
    assert discovery_edges[0]["last_event_type"] == "peer_discovered"
    assert discovery_edges[0]["last_round"] == 2


def test_store_exposes_normalized_activity_and_connection_snapshot():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})
    store.set_connection_status(
        "error",
        detail="Socket error",
        last_error="Socket error",
    )
    store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "peer_id": "node-b",
            "round": 7,
            "weight_summary": {"payload_size_bytes": 1024},
        },
    )

    snapshot = store.get_dashboard_snapshot()
    events = store.get_events()

    assert snapshot["connection"]["status"] == "error"
    assert snapshot["connection"]["detail"] == "Socket error"
    assert snapshot["session"]["active_node_count"] == 1
    assert snapshot["events"][0]["title"] == "Model received"
    assert snapshot["events"][0]["summary"] == "node-b -> node-a · round 7"
    assert snapshot["events"][0]["related_node_id"] == "node-b"
    assert events[0]["title"] == "Model received"
    assert events[0]["summary"] == "node-b -> node-a · round 7"
    assert events[0]["related_node_id"] == "node-b"

    snapshot["connection"]["status"] = "mutated"

    assert store.get_dashboard_snapshot()["connection"]["status"] == "error"


def test_store_broadcasts_normalized_event_payloads():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})

    broadcasts = store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "peer_id": "node-b",
            "round": 4,
        },
    )

    event_broadcast = next(
        message for message in broadcasts if message["type"] == "node_event_received"
    )

    assert event_broadcast["payload"]["title"] == "Model received"
    assert event_broadcast["payload"]["summary"] == "node-b -> node-a · round 4"
    assert event_broadcast["payload"]["related_node_id"] == "node-b"


def test_store_accumulates_unique_received_peers_per_round():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})

    store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_id": "node-b",
            "weight_summary": {"layer_count": 1},
        },
    )
    store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_id": "node-c",
            "weight_summary": {"layer_count": 1},
        },
    )
    store.ingest_event(
        "model_received",
        {
            "node_id": "node-a",
            "round": 3,
            "peer_id": "node-b",
            "weight_summary": {"layer_count": 1},
        },
    )

    round_summary = store.get_node_rounds("node-a")[0]

    assert round_summary["received_peer_ids"] == ["node-b", "node-c"]
