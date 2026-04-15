import asyncio

import numpy as np
import pytest

from quinkgl import GossipNode, TrainingConfig
from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy
from quinkgl.models.base import ModelWrapper, TrainingResult
from quinkgl.observability.events import RuntimeEvent
from quinkgl.topology.base import PeerInfo, SelectionContext, TopologyStrategy
from quinkgl.telemetry.client import TelemetryClient


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
    async def select_targets(self, context: SelectionContext, count: int = 3):
        return []

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


@pytest.mark.asyncio
async def test_telemetry_client_serializes_and_forwards_runtime_events():
    seen_events = []

    async def event_sink(payload):
        seen_events.append(payload)

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=event_sink,
        heartbeat_sink=None,
        heartbeat_interval=60.0,
    )

    client.handle(
        RuntimeEvent(
            event_type="training_completed",
            payload={
                "node_id": "node-a",
                "round": 3,
                "loss": 0.25,
                "accuracy": 0.75,
                "samples_trained": 32,
            },
        )
    )

    await asyncio.sleep(0)

    assert len(seen_events) == 1
    assert seen_events[0]["event_type"] == "training_completed"
    assert seen_events[0]["payload"]["node_id"] == "node-a"
    assert seen_events[0]["payload"]["round"] == 3
    assert "timestamp" in seen_events[0]


@pytest.mark.asyncio
async def test_telemetry_client_heartbeat_loop_reports_node_status():
    seen_heartbeats = []

    async def heartbeat_sink(payload):
        seen_heartbeats.append(payload)

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=None,
        heartbeat_sink=heartbeat_sink,
        heartbeat_interval=0.01,
    )

    snapshots = [
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 1,
            "connection_mode": "ipv8_p2p",
        },
        {
            "node_id": "node-a",
            "domain": "demo",
            "running": True,
            "current_round": 2,
            "connection_mode": "ipv8_p2p",
        },
    ]

    def status_provider():
        return snapshots[min(len(seen_heartbeats), len(snapshots) - 1)]

    client.start(status_provider)
    await asyncio.sleep(0.03)
    await client.stop()

    assert len(seen_heartbeats) >= 2
    assert seen_heartbeats[0]["node_id"] == "node-a"
    assert seen_heartbeats[-1]["current_round"] == 2
    assert "timestamp" in seen_heartbeats[0]


@pytest.mark.asyncio
async def test_gossip_node_can_attach_telemetry_client_without_affecting_runtime():
    seen_events = []

    async def event_sink(payload):
        seen_events.append(payload)

    node = GossipNode(
        node_id="node-a",
        domain="demo",
        model=DummyModel(),
        topology=DummyTopology(),
        aggregation=DummyAggregator(),
        training_config=TrainingConfig(),
        enable_fallback=False,
    )

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=event_sink,
        heartbeat_sink=None,
        heartbeat_interval=60.0,
    )
    attached = node.attach_telemetry_client(client)

    assert attached is client

    node.gl_node.aggregator.event_emitter.emit(
        "training_completed",
        {
            "node_id": "node-a",
            "round": 1,
            "loss": 0.5,
            "accuracy": 0.8,
            "samples_trained": 16,
        },
    )

    await asyncio.sleep(0)

    # First event is the telemetry.connected lifecycle event
    assert seen_events[0]["event_type"] == "telemetry.connected"
    assert seen_events[0]["payload"]["base_url"] == "http://telemetry.local"

    # Second event is the manually emitted training_completed
    assert seen_events[1]["event_type"] == "training_completed"
    assert seen_events[1]["payload"]["samples_trained"] == 16


@pytest.mark.asyncio
async def test_telemetry_client_swallowing_sink_failures_does_not_raise():
    async def failing_event_sink(payload):
        raise RuntimeError("service unavailable")

    async def failing_heartbeat_sink(payload):
        raise RuntimeError("service unavailable")

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=failing_event_sink,
        heartbeat_sink=failing_heartbeat_sink,
        heartbeat_interval=0.01,
    )

    client.handle(
        RuntimeEvent(
            event_type="targets_selected",
            payload={"node_id": "node-a", "round": 1, "selected_targets": ["node-b"]},
        )
    )
    client.start(lambda: {"node_id": "node-a", "running": True})

    await asyncio.sleep(0.03)
    await client.stop()

