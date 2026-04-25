from datetime import datetime

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


def test_store_caps_node_count_and_prunes_oldest_node_state():
    store = TelemetryStore(session_id="session-1", max_nodes=2)

    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "timestamp": "2026-04-08T08:00:00",
    })
    store.ingest_event("training_completed", {
        "node_id": "node-a",
        "round": 1,
        "loss": 0.1,
        "accuracy": 0.9,
        "samples_trained": 8,
    }, timestamp=datetime.fromisoformat("2026-04-08T08:10:00"))
    store.ingest_heartbeat({
        "node_id": "node-b",
        "domain": "demo",
        "running": True,
        "timestamp": "2026-04-08T09:00:00",
    })
    store.ingest_event("peer_discovered", {
        "node_id": "node-a",
        "peer_id": "node-b",
        "round": 1,
    }, timestamp=datetime.fromisoformat("2026-04-08T08:30:00"))
    store.ingest_heartbeat({
        "node_id": "node-c",
        "domain": "demo",
        "running": True,
        "timestamp": "2026-04-08T10:00:00",
    })

    node_ids = [node["node_id"] for node in store.get_nodes()]
    graph = store.get_network_graph()

    assert node_ids == ["node-b", "node-c"]
    assert store.get_node("node-a") is None
    assert store.get_node_events("node-a") == []
    assert store.get_node_rounds("node-a") == []
    assert graph["edges"] == []


def test_store_caps_rounds_per_node_to_most_recent_rounds():
    store = TelemetryStore(session_id="session-1", max_rounds_per_node=2)
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})

    for round_number in (1, 2, 3):
        store.ingest_event("training_completed", {
            "node_id": "node-a",
            "round": round_number,
            "loss": 0.1 * round_number,
            "accuracy": 0.5,
            "samples_trained": 8,
        })

    rounds = store.get_node_rounds("node-a")

    assert [item["round_number"] for item in rounds] == [2, 3]


def test_store_caps_edge_count_to_most_recent_edges():
    store = TelemetryStore(session_id="session-1", max_edges=2)
    store.ingest_heartbeat({"node_id": "node-a", "domain": "demo", "running": True})

    for peer_id in ("node-b", "node-c", "node-d"):
        store.ingest_event("peer_discovered", {
            "node_id": "node-a",
            "peer_id": peer_id,
            "round": 1,
        })

    edges = store.get_network_graph()["edges"]
    edge_ids = sorted(edge["id"] for edge in edges)

    assert edge_ids == [
        "node-a::node-c::peer_link",
        "node-a::node-d::peer_link",
    ]


def test_store_caps_peer_id_lists_to_latest_entries():
    store = TelemetryStore(session_id="session-1", max_peer_ids_per_node=2)
    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "known_peers": ["node-b", "node-c", "node-d"],
    })

    node = store.get_node("node-a")

    assert node["peer_ids"] == ["node-c", "node-d"]
    assert node["known_peer_count"] == 2


def test_ingest_heartbeat_sets_swarm_metadata():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
        "swarm_name": "Alpha Swarm",
        "manifest_hash": "hash-abc",
        "aggregation_name": "FedAvg",
        "topology_name": "Ring",
    })

    node = store.get_node("node-a")

    assert node["swarm_id"] == "swarm-1"
    assert node["swarm_name"] == "Alpha Swarm"
    assert node["manifest_hash"] == "hash-abc"
    assert node["aggregation_name"] == "FedAvg"
    assert node["topology_name"] == "Ring"


def test_ingest_event_node_started_caches_manifest_and_sets_swarm_fields():
    store = TelemetryStore(session_id="session-1")
    store.ingest_event(
        "node.started",
        {
            "node_id": "node-a",
            "swarm_id": "swarm-1",
            "swarm_name": "Beta Swarm",
            "manifest_hash": "hash-def",
            "aggregation_name": "FedProx",
            "topology_name": "Star",
            "manifest": {"name": "Beta", "round_limit": 10},
        },
    )

    node = store.get_node("node-a")
    assert node["swarm_id"] == "swarm-1"
    assert node["swarm_name"] == "Beta Swarm"
    assert node["manifest_hash"] == "hash-def"
    assert node["aggregation_name"] == "FedProx"
    assert node["topology_name"] == "Star"

    swarms = store.get_swarms()
    assert len(swarms) == 1
    assert swarms[0]["swarm_id"] == "swarm-1"
    assert swarms[0]["swarm_name"] == "Beta Swarm"
    assert swarms[0]["peer_count"] == 1
    assert store.get_manifest("swarm-1") == {"name": "Beta", "round_limit": 10}


def test_get_swarms_returns_correct_listings_and_peer_counts():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
        "swarm_name": "Swarm One",
    })
    store.ingest_heartbeat({
        "node_id": "node-b",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
    })
    store.ingest_heartbeat({
        "node_id": "node-c",
        "domain": "test",
        "running": True,
        "swarm_id": "swarm-2",
        "swarm_name": "Swarm Two",
    })

    swarms = store.get_swarms()
    by_id = {s["swarm_id"]: s for s in swarms}

    assert len(swarms) == 2
    assert by_id["swarm-1"]["peer_count"] == 2
    assert by_id["swarm-1"]["domains"] == ["demo"]
    assert by_id["swarm-2"]["peer_count"] == 1
    assert set(by_id["swarm-2"]["domains"]) == {"test"}


def test_drop_node_cleans_up_swarm_nodes():
    store = TelemetryStore(session_id="session-1", max_nodes=2)
    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
        "timestamp": "2026-04-08T08:00:00",
    })
    store.ingest_heartbeat({
        "node_id": "node-b",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
        "timestamp": "2026-04-08T09:00:00",
    })
    store.ingest_heartbeat({
        "node_id": "node-c",
        "domain": "demo",
        "running": True,
        "timestamp": "2026-04-08T10:00:00",
    })

    assert store.get_node("node-a") is None
    swarms = store.get_swarms()
    assert len(swarms) == 1
    assert swarms[0]["peer_count"] == 1


def test_ingest_event_node_started_without_manifest_sets_swarm_fields_and_swarm_nodes():
    store = TelemetryStore(session_id="session-1")
    store.ingest_event(
        "node.started",
        {
            "node_id": "node-a",
            "swarm_id": "swarm-1",
            "swarm_name": "Beta Swarm",
            "manifest_hash": "hash-def",
            "aggregation_name": "FedProx",
            "topology_name": "Star",
        },
    )

    node = store.get_node("node-a")
    assert node["swarm_id"] == "swarm-1"
    assert node["swarm_name"] == "Beta Swarm"
    assert node["manifest_hash"] == "hash-def"
    assert node["aggregation_name"] == "FedProx"
    assert node["topology_name"] == "Star"

    swarms = store.get_swarms()
    assert len(swarms) == 1
    assert swarms[0]["swarm_id"] == "swarm-1"
    assert swarms[0]["swarm_name"] == "Beta Swarm"
    assert swarms[0]["peer_count"] == 1
    assert store.get_manifest("swarm-1") is None


def test_ingest_heartbeat_caches_manifest_when_present():
    store = TelemetryStore(session_id="session-1")
    store.ingest_heartbeat({
        "node_id": "node-a",
        "domain": "demo",
        "running": True,
        "swarm_id": "swarm-1",
        "manifest": {"name": "Cached", "description": "from heartbeat"},
    })

    assert store.get_manifest("swarm-1") == {"name": "Cached", "description": "from heartbeat"}
