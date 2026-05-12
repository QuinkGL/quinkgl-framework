import asyncio
import logging
import time

import numpy as np
import pytest

from quinkgl import GossipNode, TerminalObserver, TrainingConfig
from quinkgl.gossip.protocol import ModelUpdateMessage
from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy
from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.models.base import ModelWrapper, TrainingResult
from quinkgl.observability.events import EventEmitter, RuntimeEvent
from quinkgl.topology.base import PeerInfo, SelectionContext, TopologyStrategy


class DummyModel(ModelWrapper):
    def __init__(self):
        super().__init__(model={})

    def get_weights(self):
        return {"w": np.array([1.0, 2.0])}

    def set_weights(self, weights):
        self.model = weights

    async def train(self, data, config=None):
        return TrainingResult(
            epochs_completed=1,
            final_loss=0.25,
            final_accuracy=0.75,
            samples_trained=8,
        )

    def evaluate(self, data, loss_fn=None):
        return {"loss": 0.25, "accuracy": 0.75}


class DummyTopology(TopologyStrategy):
    def __init__(self, on_select=None, targets=None):
        super().__init__()
        self.on_select = on_select
        self.targets = targets or ["peer-a", "peer-b"]

    async def select_targets(self, context: SelectionContext, count: int = 3):
        if self.on_select:
            self.on_select(context, count)
        return list(self.targets)

    async def should_accept_connection(self, context: SelectionContext, peer_info: PeerInfo):
        return True


class DummyAggregator(AggregationStrategy):
    async def aggregate(self, updates):
        return AggregatedModel(
            weights=updates[0].weights,
            contributing_peers=[update.peer_id for update in updates],
            total_samples=sum(update.sample_count for update in updates),
            updates=updates,
        )


def test_event_emitter_returns_runtime_event_with_isolated_payload_snapshot():
    payload = {"round": 3, "metrics": {"loss": 0.25}, "tags": ["alpha"]}
    seen = []
    emitter = EventEmitter()

    def subscriber(event: RuntimeEvent):
        event.payload["metrics"]["loss"] = 0.99
        event.payload["tags"].append("beta")

    subscriber.needs_isolated_payload = True

    def later_subscriber(event: RuntimeEvent):
        seen.append(
            (
                event.event_type,
                event.payload["metrics"]["loss"],
                list(event.payload["tags"]),
            )
        )

    emitter.subscribe(subscriber)
    emitter.subscribe(later_subscriber)
    event = emitter.emit("training_completed", payload)
    payload["metrics"]["loss"] = 0.5
    payload["tags"].append("gamma")

    assert isinstance(event, RuntimeEvent)
    assert event.event_type == "training_completed"
    assert isinstance(event.payload, dict)
    assert isinstance(event.payload["metrics"], dict)
    assert isinstance(event.payload["tags"], list)
    assert event.payload == {
        "round": 3,
        "metrics": {"loss": 0.25},
        "tags": ["alpha"],
    }
    assert seen == [("training_completed", 0.25, ["alpha"])]


def test_event_emitter_isolates_subscriber_exceptions():
    seen = []
    emitter = EventEmitter()

    def raising_subscriber(event: RuntimeEvent):
        raise RuntimeError("subscriber failure")

    def healthy_subscriber(event: RuntimeEvent):
        seen.append((event.event_type, event.payload["round"]))

    emitter.subscribe(raising_subscriber)
    emitter.subscribe(healthy_subscriber)

    event = emitter.emit("training_completed", {"round": 3})

    assert isinstance(event, RuntimeEvent)
    assert seen == [("training_completed", 3)]


def test_event_emitter_logs_subscriber_failures_and_emits_subscriber_error(caplog):
    seen = []
    emitter = EventEmitter()

    def raising_subscriber(event: RuntimeEvent):
        if event.event_type == "training_completed":
            raise RuntimeError("subscriber failure")

    def healthy_subscriber(event: RuntimeEvent):
        seen.append((event.event_type, dict(event.payload)))

    emitter.subscribe(raising_subscriber)
    emitter.subscribe(healthy_subscriber)

    with caplog.at_level(logging.ERROR, logger="quinkgl.observability.events"):
        emitter.emit("training_completed", {"round": 3})

    assert ("training_completed", {"round": 3}) in seen
    subscriber_error_events = [payload for event_type, payload in seen if event_type == "subscriber.error"]
    assert len(subscriber_error_events) == 1
    assert subscriber_error_events[0]["source_event_type"] == "training_completed"
    assert subscriber_error_events[0]["error_type"] == "RuntimeError"
    assert subscriber_error_events[0]["error"] == "subscriber failure"
    assert "Event subscriber failed while handling training_completed" in caplog.text


def test_event_emitter_preserves_returned_payload_shape():
    payload = {"round": 3, "metrics": {"loss": 0.25}, "tags": ["alpha"]}
    emitter = EventEmitter()

    event = emitter.emit("training_completed", payload)
    payload["metrics"]["loss"] = 0.99
    payload["tags"].append("beta")

    assert event.payload["metrics"]["loss"] == 0.25
    assert event.payload["tags"] == ["alpha"]


def test_subscriber_payload_mutation_does_not_affect_later_subscribers_or_returned_event():
    seen = []
    emitter = EventEmitter()

    def mutating_subscriber(event: RuntimeEvent):
        event.payload["metrics"]["loss"] = 0.99
        event.payload["tags"].append("beta")

    mutating_subscriber.needs_isolated_payload = True

    def later_subscriber(event: RuntimeEvent):
        seen.append(
            (
                event.event_type,
                event.payload["metrics"]["loss"],
                list(event.payload["tags"]),
            )
        )

    emitter.subscribe(mutating_subscriber)
    emitter.subscribe(later_subscriber)

    event = emitter.emit(
        "training_completed",
        {"round": 3, "metrics": {"loss": 0.25}, "tags": ["alpha"]},
    )

    assert event.payload == {
        "round": 3,
        "metrics": {"loss": 0.25},
        "tags": ["alpha"],
    }
    assert seen == [("training_completed", 0.25, ["alpha"])]


def test_emit_degrades_unsafe_payload_values_without_raising():
    class BadCopy:
        def __deepcopy__(self, memo):
            raise TypeError("cannot copy")

        def __repr__(self):
            return "<bad-copy>"

    seen = []
    emitter = EventEmitter()

    def subscriber(event: RuntimeEvent):
        seen.append((event.event_type, event.payload["bad"], event.payload["round"]))

    emitter.subscribe(subscriber)

    event = emitter.emit("training_completed", {"round": 3, "bad": BadCopy()})

    assert event.payload == {"round": 3, "bad": "<bad-copy>"}
    assert seen == [("training_completed", "<bad-copy>", 3)]


def test_emit_handles_cyclic_payload_without_raising():
    payload = {"round": 3}
    payload["self"] = payload
    seen = []
    emitter = EventEmitter()

    def subscriber(event: RuntimeEvent):
        seen.append((event.event_type, event.payload["self"], event.payload["round"]))

    emitter.subscribe(subscriber)

    event = emitter.emit("training_completed", payload)

    assert event.payload == {"round": 3, "self": "<recursive dict>"}
    assert seen == [("training_completed", "<recursive dict>", 3)]


def test_weight_summary_handles_mixed_dict_keys_without_raising():
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=DummyAggregator(),
        training_config=TrainingConfig(),
    )

    summary = aggregator._weight_summary({1: np.array([1.0]), "layer": np.array([2.0, 3.0])})

    assert summary["kind"] == "dict"
    assert summary["field_count"] == 2
    assert summary["total_elements"] == 3


@pytest.mark.asyncio
async def test_aggregator_emits_training_send_receive_and_aggregation_events():
    emitter = EventEmitter()
    seen = []

    def subscriber(event: RuntimeEvent):
        seen.append((event.event_type, dict(event.payload)))

    emitter.subscribe(subscriber)

    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=DummyAggregator(),
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter
    aggregator.current_round = 7
    aggregator.running = True

    async def send_message_callback(peer_id, message):
        return None

    aggregator.send_message_callback = send_message_callback

    loss, acc, samples = await aggregator._train_local(data=[1])
    await aggregator._send_model(["peer-a"], loss=loss, accuracy=acc, samples_trained=samples)
    await aggregator._handle_model_update(
        ModelUpdateMessage.create(
            sender_id="peer-b",
            weights={"w": np.array([2.0, 3.0])},
            sample_count=4,
            loss=0.5,
            accuracy=0.6,
            round_number=7,
        )
    )
    await aggregator._aggregate_models()


    await asyncio.sleep(0)

    event_types = [event_type for event_type, _ in seen]
    assert event_types == [
        "training_started",
        "training_completed",
        "model_send_started",
        "model_sent",
        "model_received",
        "models_converged",
        "aggregation_completed",
    ]

    training_started = seen[0][1]
    training_completed = seen[1][1]
    model_send_started = seen[2][1]
    model_sent = seen[3][1]
    model_received = seen[4][1]
    models_converged = seen[5][1]
    aggregation_completed = seen[6][1]

    assert training_started["node_id"] == "n1"
    assert training_started["round"] == 7
    assert training_started["loss"] is None
    assert training_started["accuracy"] is None
    assert training_started["samples_trained"] == 0
    assert training_completed["loss"] == 0.25
    assert training_completed["accuracy"] == 0.75
    assert training_completed["samples_trained"] == 8
    assert model_send_started["peer_ids"] == ["peer-a"]
    assert model_send_started["sample_count"] == 8
    assert model_send_started["weight_summary"]["kind"] == "dict"
    assert model_send_started["weight_summary"]["total_elements"] == 2
    assert model_sent["peer_ids"] == ["peer-a"]
    assert model_sent["sample_count"] == 8
    assert model_sent["weight_summary"]["kind"] == "dict"
    assert model_sent["weight_summary"]["total_elements"] == 2
    assert model_received["peer_id"] == "peer-b"
    assert model_received["sample_count"] == 4
    assert model_received["weight_summary"]["kind"] == "dict"
    assert model_received["weight_summary"]["total_elements"] == 2
    assert aggregation_completed["peer_ids"] == ["n1", "peer-b"]
    assert aggregation_completed["sample_count"] == 12
    assert aggregation_completed["weight_summary"]["kind"] == "dict"
    assert aggregation_completed["weight_summary"]["total_elements"] == 2


@pytest.mark.asyncio
async def test_aggregator_emits_peer_discovery_and_disconnect_events():
    emitter = EventEmitter()
    seen = []

    def subscriber(event: RuntimeEvent):
        seen.append((event.event_type, dict(event.payload)))

    emitter.subscribe(subscriber)

    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=DummyAggregator(),
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter

    peer = PeerInfo(
        peer_id="peer-a",
        domain="demo",
        data_schema_hash="abc",
        model_version="1.0.0",
    )
    aggregator.add_peer(peer)
    await asyncio.sleep(0)
    await aggregator.remove_peer("peer-a")
    await asyncio.sleep(0)

    event_types = [event_type for event_type, _payload in seen]

    assert event_types == ["peer_discovered", "peer_disconnected"]
    assert seen[0][1]["peer_id"] == "peer-a"
    assert seen[1][1]["node_id"] == "n1"


@pytest.mark.asyncio
async def test_aggregator_emits_aggregation_failed_event_and_restores_batch():
    emitter = EventEmitter()
    seen = []

    def subscriber(event: RuntimeEvent):
        seen.append((event.event_type, dict(event.payload)))

    emitter.subscribe(subscriber)

    class FailingAggregator(DummyAggregator):
        async def aggregate(self, updates):
            raise RuntimeError("aggregate boom")

    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=FailingAggregator(),
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter
    aggregator.current_round = 7
    aggregator.running = True

    await aggregator._handle_model_update(
        ModelUpdateMessage.create(
            sender_id="peer-b",
            weights={"w": np.array([2.0, 3.0])},
            sample_count=4,
            loss=0.5,
            accuracy=0.6,
            round_number=7,
        )
    )

    with pytest.raises(RuntimeError, match="aggregate boom"):
        await aggregator._aggregate_models()

    await asyncio.sleep(0)

    failed_events = [payload for event_type, payload in seen if event_type == "aggregation_failed"]
    assert len(failed_events) == 1
    failed = failed_events[0]
    assert failed["node_id"] == "n1"
    assert failed["round"] == 7
    assert failed["pending_updates_restored"] == 1
    assert failed["error_type"] == "RuntimeError"
    assert failed["error"] == "aggregate boom"
    assert failed["peer_ids"] == ["n1", "peer-b"]
    assert len(aggregator.pending_updates) == 1
    assert aggregator.pending_updates[0].peer_id == "peer-b"


def test_gossip_node_can_attach_terminal_observer_and_receive_emitted_events():
    node = GossipNode(
        node_id="n1",
        domain="demo",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregation=DummyAggregator(),
        training_config=TrainingConfig(),
        enable_fallback=False,
    )

    seen = []
    observer = node.attach_terminal_observer()

    assert isinstance(observer, TerminalObserver)

    observer.printer = seen.append

    node.gl_node.aggregator.event_emitter.emit(
        "training_completed",
        {
            "node_id": "n1",
            "round": 2,
            "loss": 0.125,
            "accuracy": 0.875,
            "samples_trained": 16,
        },
    )

    assert seen == [
        "[NODE n1][ROUND 2] training completed loss=0.125 acc=0.875 samples=16"
    ]


@pytest.mark.asyncio
async def test_gossip_node_ipv8_send_callback_awaits_community_send():
    node = GossipNode(
        node_id="n1",
        domain="demo",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregation=DummyAggregator(),
        training_config=TrainingConfig(),
        enable_fallback=False,
    )

    class DummyCommunity:
        def __init__(self):
            self.called = False
            self.payloads = []

        def get_compatible_peers(self):
            return []

        def get_peer_count(self):
            return 0

        async def send_model_update(self, **kwargs):
            self.called = True
            self.payloads.append(kwargs)
            return True

    community = DummyCommunity()
    node.community = community
    node.running = True

    async def fake_gl_run_continuous(data_provider=None):
        await node.gl_node.aggregator.send_message_callback(
            "peer-a",
            ModelUpdateMessage.create(
                sender_id="n1",
                weights={"w": np.array([1.0, 2.0])},
                sample_count=8,
                round_number=1,
                loss=0.25,
                accuracy=0.75,
            ),
        )

    node.gl_node.run_continuous = fake_gl_run_continuous

    await node.run_continuous(data=[1])

    assert community.called is True
    assert len(community.payloads) == 1
    payload = community.payloads[0]
    assert payload["target_node_id"] == "peer-a"
    assert payload["sample_count"] == 8
    assert payload["round_number"] == 1
    assert payload["loss"] == 0.25
    assert payload["accuracy"] == 0.75
    np.testing.assert_array_equal(payload["weights"]["w"], np.array([1.0, 2.0]))


@pytest.mark.asyncio
async def test_aggregator_event_delivery_is_scheduled_off_hot_path():
    seen = []
    slow_started = False

    def slow_subscriber(event: RuntimeEvent):
        nonlocal slow_started
        slow_started = True
        seen.append(event.event_type)
        time.sleep(0.05)

    emitter = EventEmitter()
    emitter.subscribe(slow_subscriber)

    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=DummyAggregator(),
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter

    async def send_message_callback(peer_id, message):
        return None

    aggregator.send_message_callback = send_message_callback

    await aggregator._train_local(data=[1])

    # _train_local has no internal awaits that yield to the event loop,
    # so delivery tasks are still pending at this point.
    assert slow_started is False
    assert seen == []

    # _send_model uses asyncio.gather which yields control, allowing
    # previously scheduled event-delivery tasks to run.
    await aggregator._send_model(["peer-a"], loss=0.25, accuracy=0.75, samples_trained=8)
    await asyncio.sleep(0)

    assert slow_started is True
    assert "training_started" in seen
    assert "model_send_started" in seen
    assert "model_sent" in seen


@pytest.mark.asyncio
async def test_aggregator_does_not_emit_model_sent_without_send_callback():
    seen = []

    emitter = EventEmitter()
    emitter.subscribe(lambda event: seen.append((event.event_type, dict(event.payload))))

    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregator=DummyAggregator(),
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter

    await aggregator._send_model(["peer-a"], loss=0.25, accuracy=0.75, samples_trained=8)

    await asyncio.sleep(0)

    event_types = [event_type for event_type, _ in seen]
    assert "model_send_started" in event_types
    assert "model_sent" not in event_types


@pytest.mark.asyncio
async def test_run_continuous_emits_targets_selected():
    seen = []

    emitter = EventEmitter()
    emitter.subscribe(lambda event: seen.append((event.event_type, dict(event.payload))))

    topology = DummyTopology(targets=["peer-a"])
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=topology,
        aggregator=DummyAggregator(),
        gossip_interval=0.0,
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter
    aggregator.known_peers["peer-a"] = PeerInfo(peer_id="peer-a", domain="demo", data_schema_hash="abc", model_version="1.0.0")
    aggregator.known_peers["peer-b"] = PeerInfo(peer_id="peer-b", domain="demo", data_schema_hash="abc", model_version="1.0.0")

    async def send_message_callback(peer_id, message):
        return None

    aggregator.send_message_callback = send_message_callback
    aggregator.running = True

    topology.on_select = lambda context, count: setattr(aggregator, "running", False)

    await aggregator.run_continuous(data_provider=None)

    targets_selected = next(payload for event_type, payload in seen if event_type == "targets_selected")
    assert targets_selected["node_id"] == "n1"
    assert targets_selected["round"] == 1
    assert targets_selected["candidate_count"] == 2
    assert targets_selected["selected_targets"] == ["peer-a"]


@pytest.mark.asyncio
async def test_run_continuous_uses_adaptive_fanout_for_large_swarms():
    seen = []
    selected_counts = []

    emitter = EventEmitter()
    emitter.subscribe(lambda event: seen.append((event.event_type, dict(event.payload))))

    topology = DummyTopology(targets=[])
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=topology,
        aggregator=DummyAggregator(),
        gossip_interval=0.0,
        training_config=TrainingConfig(),
    )
    aggregator.event_emitter = emitter

    for index in range(251):
        peer_id = f"peer-{index}"
        aggregator.known_peers[peer_id] = PeerInfo(
            peer_id=peer_id,
            domain="demo",
            data_schema_hash="abc",
            model_version="1.0.0",
        )

    def on_select(context, count):
        selected_counts.append(count)
        aggregator.running = False

    topology.on_select = on_select
    aggregator.running = True

    await aggregator.run_continuous(data_provider=None)

    assert selected_counts == [7]
    targets_selected = next(payload for event_type, payload in seen if event_type == "targets_selected")
    assert targets_selected["candidate_count"] == 251
    assert targets_selected["fanout"] == 7


# ---------------------------------------------------------------------------
# Lifecycle event formatting tests
# ---------------------------------------------------------------------------

class TestLifecycleEventFormatting:
    """Tests for node.config, node.started, and node.stopped banner rendering."""

    def test_node_config_banner_contains_version(self):
        from quinkgl.observability.terminal import format_runtime_event
        event = RuntimeEvent(
            event_type="node.config",
            payload={
                "node_id": "alice",
                "version": "0.3.4",
                "domain": "health",
                "port": 7000,
                "topology": "AffinityTopology",
                "aggregation": "FedAvg",
                "connection_mode": "IPv8 P2P",
                "model": "PyTorchPersonalizedModel",
                "gossip_interval": 10.0,
                "data_policy": None,
                "fingerprint_summary": None,
            },
        )
        output = format_runtime_event(event)
        assert "QuinkGL v0.3.4" in output
        assert "alice" in output
        assert "health" in output
        assert "AffinityTopology" in output
        assert "FedAvg" in output
        assert "IPv8 P2P" in output

    def test_node_config_banner_with_data_policy(self):
        from quinkgl.observability.terminal import format_runtime_event
        event = RuntimeEvent(
            event_type="node.config",
            payload={
                "node_id": "bob",
                "version": "0.3.4",
                "domain": "demo",
                "port": 8000,
                "topology": "RandomTopology",
                "aggregation": "TrimmedMean",
                "connection_mode": "Tunnel Relay",
                "model": "PyTorchModel",
                "gossip_interval": 30.0,
                "data_policy": {
                    "fingerprint_enabled": True,
                    "min_affinity": 0.3,
                    "privacy_level": "standard",
                },
                "fingerprint_summary": {
                    "label_buckets": 4,
                    "sample_bucket": "1k-10k",
                },
            },
        )
        output = format_runtime_event(event)
        assert "0.3" in output
        assert "standard" in output
        assert "4 label buckets" in output
        assert "1k-10k" in output

    def test_node_started_format(self):
        from quinkgl.observability.terminal import format_runtime_event
        event = RuntimeEvent(
            event_type="node.started",
            payload={"node_id": "alice", "connection_mode": "IPv8 P2P"},
        )
        output = format_runtime_event(event)
        assert "[NODE alice]" in output
        assert "started" in output
        assert "IPv8 P2P" in output

    def test_node_stopped_format(self):
        from quinkgl.observability.terminal import format_runtime_event
        event = RuntimeEvent(
            event_type="node.stopped",
            payload={"node_id": "alice", "total_rounds": 10, "uptime_seconds": 45.3},
        )
        output = format_runtime_event(event)
        assert "[NODE alice]" in output
        assert "stopped" in output
        assert "rounds=10" in output
        assert "uptime=45.3s" in output

    def test_node_stopped_without_rounds(self):
        from quinkgl.observability.terminal import format_runtime_event
        event = RuntimeEvent(
            event_type="node.stopped",
            payload={"node_id": "x"},
        )
        output = format_runtime_event(event)
        assert "[NODE x]" in output
        assert "stopped" in output
        assert "rounds" not in output

    def test_terminal_observer_prints_banner(self):
        lines = []
        observer = TerminalObserver(printer=lines.append)
        event = RuntimeEvent(
            event_type="node.config",
            payload={
                "node_id": "n1",
                "version": "0.3.4",
                "domain": "test",
                "port": 0,
                "topology": "RandomTopology",
                "aggregation": "FedAvg",
                "connection_mode": "IPv8 P2P",
                "model": "PyTorchModel",
                "gossip_interval": 60.0,
                "data_policy": None,
                "fingerprint_summary": None,
            },
        )
        observer.handle(event)
        assert len(lines) == 1
        assert "QuinkGL v0.3.4" in lines[0]

    def test_auto_observer_via_emitter(self):
        """Verify that emitting node.config through EventEmitter reaches TerminalObserver."""
        captured = []
        emitter = EventEmitter()
        observer = TerminalObserver(printer=captured.append)
        emitter.subscribe(observer.handle)

        emitter.emit("node.started", {"node_id": "z", "connection_mode": "IPv8 P2P"})
        assert len(captured) == 1
        assert "[NODE z]" in captured[0]
