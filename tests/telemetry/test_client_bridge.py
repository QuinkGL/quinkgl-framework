import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from quinkgl import GossipNode, TrainingConfig
from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy
from quinkgl.models.base import ModelWrapper, TrainingResult
from quinkgl.observability.events import RuntimeEvent
from quinkgl.telemetry.api import DEFAULT_TELEMETRY_AUTH_HEADER
from quinkgl.telemetry.client import TelemetryClient
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
async def test_telemetry_client_stop_cancels_and_awaits_pending_event_tasks():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_event_sink(payload):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=blocking_event_sink,
        heartbeat_interval=60.0,
    )

    client.handle(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 1},
        )
    )
    await started.wait()

    assert len(client._background_tasks) >= 1

    await client.stop()

    assert cancelled.is_set()
    assert client._heartbeat_task is None
    assert len(client._background_tasks) == 0


@pytest.mark.asyncio
async def test_telemetry_client_stop_cleans_up_heartbeat_task():
    snapshots = [{"node_id": "node-a", "running": True}]

    async def blocking_heartbeat_sink(payload):
        await asyncio.Event().wait()

    client = TelemetryClient(
        base_url="http://telemetry.local",
        heartbeat_sink=blocking_heartbeat_sink,
        heartbeat_interval=0.01,
    )

    client.start(lambda: snapshots[0])
    await asyncio.sleep(0)

    assert client._heartbeat_task is not None
    assert len(client._background_tasks) >= 1

    await client.stop()

    assert client._heartbeat_task is None
    assert len(client._background_tasks) == 0


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


@pytest.mark.asyncio
async def test_telemetry_client_posts_default_auth_header():
    response = MagicMock()
    response.raise_for_status.return_value = None
    post = AsyncMock(return_value=response)
    http_client = AsyncMock()
    http_client.is_closed = False
    http_client.post = post

    client = TelemetryClient(
        base_url="http://telemetry.local",
        auth_secret="secret-123",
    )

    with patch("quinkgl.telemetry.client.httpx.AsyncClient", return_value=http_client):
        await client.send_heartbeat({"node_id": "node-a", "running": True})

    post.assert_awaited_once()
    assert post.await_args.kwargs["headers"] == {DEFAULT_TELEMETRY_AUTH_HEADER: "secret-123"}


@pytest.mark.asyncio
async def test_telemetry_client_posts_custom_auth_header():
    response = MagicMock()
    response.raise_for_status.return_value = None
    post = AsyncMock(return_value=response)
    http_client = AsyncMock()
    http_client.is_closed = False
    http_client.post = post

    client = TelemetryClient(
        base_url="http://telemetry.local",
        auth_secret="secret-456",
        auth_header_name="X-Test-Telemetry-Secret",
    )

    with patch("quinkgl.telemetry.client.httpx.AsyncClient", return_value=http_client):
        await client.send_event(
            RuntimeEvent(
                event_type="training_completed",
                payload={"node_id": "node-a", "round": 1},
            )
        )

    post.assert_awaited_once()
    assert post.await_args.kwargs["headers"] == {"X-Test-Telemetry-Secret": "secret-456"}


@pytest.mark.asyncio
async def test_telemetry_client_retries_and_flushes_bounded_queue_after_recovery():
    delivered_events = []
    outage = {"active": True}

    async def flaky_event_sink(payload):
        if outage["active"]:
            raise RuntimeError("temporary outage")
        delivered_events.append(payload)

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=flaky_event_sink,
        max_pending_items=2,
        max_delivery_attempts=1,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
    )

    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 1},
        )
    )
    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 2},
        )
    )
    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 3},
        )
    )

    assert len(client._pending_deliveries) == 2

    outage["active"] = False

    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 4},
        )
    )

    delivered_rounds = [payload["payload"]["round"] for payload in delivered_events]
    assert delivered_rounds == [2, 3, 4]
    assert len(client._pending_deliveries) == 0


@pytest.mark.asyncio
async def test_telemetry_client_emits_disconnected_runtime_event_on_sustained_failure():
    runtime_events = []

    async def failing_event_sink(payload):
        raise RuntimeError("service unavailable")

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=failing_event_sink,
        max_delivery_attempts=1,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
        runtime_event_sink=lambda event_type, payload: runtime_events.append((event_type, payload)),
    )

    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 1},
        )
    )
    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 2},
        )
    )

    disconnected = [payload for event_type, payload in runtime_events if event_type == "telemetry.disconnected"]
    assert len(disconnected) == 1
    assert disconnected[0]["kind"] == "event"
    assert disconnected[0]["error_type"] == "RuntimeError"
    assert disconnected[0]["error"] == "service unavailable"


@pytest.mark.asyncio
async def test_telemetry_client_reuses_shared_http_client_and_closes_on_stop():
    response = MagicMock()
    response.raise_for_status.return_value = None
    http_client = AsyncMock()
    http_client.is_closed = False
    http_client.post = AsyncMock(return_value=response)
    http_client.aclose = AsyncMock()

    client = TelemetryClient(base_url="http://telemetry.local")

    with patch("quinkgl.telemetry.client.httpx.AsyncClient", return_value=http_client) as client_factory:
        await client.send_heartbeat({"node_id": "node-a", "running": True})
        await client.send_heartbeat({"node_id": "node-a", "running": True})
        await client.stop()

    assert client_factory.call_count == 1
    assert http_client.post.await_count == 2
    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_telemetry_client_status_provider_exceptions_emit_warning_and_loop_continues():
    runtime_events = []
    seen_heartbeats = []
    calls = {"count": 0}

    async def heartbeat_sink(payload):
        seen_heartbeats.append(payload)

    def flaky_status_provider():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("provider failed")
        return {"node_id": "node-a", "running": True, "current_round": calls["count"]}

    client = TelemetryClient(
        base_url="http://telemetry.local",
        heartbeat_sink=heartbeat_sink,
        heartbeat_interval=0.01,
        runtime_event_sink=lambda event_type, payload: runtime_events.append((event_type, payload)),
    )

    client.start(flaky_status_provider)
    await asyncio.sleep(0.05)
    await client.stop()

    warning_events = [payload for event_type, payload in runtime_events if event_type == "telemetry.status_provider_warning"]
    assert len(warning_events) == 1
    assert warning_events[0]["error_type"] == "RuntimeError"
    assert warning_events[0]["error"] == "provider failed"
    assert len(seen_heartbeats) >= 1


@pytest.mark.asyncio
async def test_telemetry_client_emits_delivery_failed_summary_on_repeated_failures():
    runtime_events = []

    async def failing_event_sink(payload):
        raise RuntimeError("still down")

    client = TelemetryClient(
        base_url="http://telemetry.local",
        event_sink=failing_event_sink,
        max_delivery_attempts=1,
        retry_initial_delay=0.0,
        retry_max_delay=0.0,
        runtime_event_sink=lambda event_type, payload: runtime_events.append((event_type, payload)),
    )

    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 1},
        )
    )
    await client.send_event(
        RuntimeEvent(
            event_type="training_completed",
            payload={"node_id": "node-a", "round": 2},
        )
    )

    delivery_failed = [payload for event_type, payload in runtime_events if event_type == "telemetry.delivery_failed"]
    assert len(delivery_failed) == 1
    assert delivery_failed[0]["kind"] == "event"
    assert delivery_failed[0]["error_type"] == "RuntimeError"
    assert delivery_failed[0]["failure_rate_per_minute"] > 0

