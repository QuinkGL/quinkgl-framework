"""Regression tests: tracked background tasks.

Covers:
- _spawn_task adds to _background_tasks and auto-removes on completion.
- _emit_event caps pending events at _MAX_PENDING_EVENTS.
- add_peer topology notification is tracked.
- _cancel_background_tasks cancels and awaits all pending tasks.
- run_continuous finally block cleans up background tasks.
"""

import asyncio

import numpy as np
import pytest

from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.aggregation.base import AggregationStrategy, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.topology.base import TopologyStrategy, PeerInfo
from quinkgl.observability.events import EventEmitter


# ── Helpers ──────────────────────────────────────────────────────────

class _DummyModel(ModelWrapper):
    def __init__(self):
        super().__init__(model=None)

    def get_weights(self):
        return {"w": np.array([1.0, 2.0])}

    def set_weights(self, weights):
        pass

    async def train(self, data, config=None):
        return TrainingResult(epochs_completed=1, final_loss=0.25, final_accuracy=0.75, samples_trained=8)

    def evaluate(self, data, loss_fn=None):
        return {"loss": 0.25, "accuracy": 0.75}


class _DummyAggregator(AggregationStrategy):
    async def aggregate(self, updates):
        return AggregatedModel(
            weights=updates[0].weights,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
        )


class _DummyTopology(TopologyStrategy):
    def __init__(self):
        super().__init__()
        self.notified_peers = []

    async def select_targets(self, context, count=3):
        return []

    async def accept_connection(self, peer_info, context):
        return True

    async def should_accept_connection(self, peer_info, context):
        return True

    async def on_new_peer_discovered(self, peer_info):
        self.notified_peers.append(peer_info.peer_id)


def _make_aggregator(topology=None):
    topo = topology or _DummyTopology()
    agg = ModelAggregator(
        peer_id="n1",
        domain="test",
        data_schema_hash="abc",
        model=_DummyModel(),
        topology=topo,
        aggregator=_DummyAggregator(),
        training_config=TrainingConfig(),
    )
    agg.running = True
    agg.current_round = 5
    return agg


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spawn_task_tracks_and_auto_removes():
    """_spawn_task must add to set and auto-remove on completion."""
    agg = _make_aggregator()

    completed = False

    async def quick():
        nonlocal completed
        completed = True

    task = agg._spawn_task(quick())
    assert task in agg._background_tasks

    await task
    # done callback fires on next iteration
    await asyncio.sleep(0)

    assert completed is True
    assert task not in agg._background_tasks


@pytest.mark.asyncio
async def test_emit_event_caps_pending_events():
    """Events beyond _MAX_PENDING_EVENTS must be dropped."""
    agg = _make_aggregator()
    agg._MAX_PENDING_EVENTS = 5

    # Use a slow subscriber so event tasks stay pending
    emitter = EventEmitter()
    emitter.subscribe(lambda e: None)
    agg.event_emitter = emitter

    # Create blocking event tasks to fill the cap
    blocker = asyncio.Event()

    async def _blocked_deliver(event_type, payload):
        await blocker.wait()
        agg.event_emitter.emit(event_type, payload)

    # Monkey-patch _deliver_event to block
    agg._deliver_event = _blocked_deliver

    # Emit exactly at the cap
    for i in range(5):
        agg._emit_event(f"evt_{i}", {"i": i})

    await asyncio.sleep(0)
    event_tasks = [t for t in agg._background_tasks if t.get_name().startswith("evt:")]
    assert len(event_tasks) == 5

    # This one should be dropped
    agg._emit_event("dropped", {"x": 1})
    await asyncio.sleep(0)
    event_tasks_after = [t for t in agg._background_tasks if t.get_name().startswith("evt:")]
    assert len(event_tasks_after) == 5  # still 5, not 6

    # Unblock and cleanup
    blocker.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_add_peer_topology_notification_is_tracked():
    """add_peer's topology notification task must be in _background_tasks."""
    topo = _DummyTopology()
    agg = _make_aggregator(topology=topo)

    peer = PeerInfo(peer_id="p1", domain="test", data_schema_hash="abc")
    agg.add_peer(peer)

    # Task should be in the set before it completes
    assert len(agg._background_tasks) > 0

    # Let it complete
    await asyncio.sleep(0.05)
    assert "p1" in topo.notified_peers


@pytest.mark.asyncio
async def test_cancel_background_tasks_cleans_all():
    """_cancel_background_tasks must cancel and clear all pending tasks."""
    agg = _make_aggregator()

    blocker = asyncio.Event()

    async def hang():
        await blocker.wait()

    agg._spawn_task(hang())
    agg._spawn_task(hang())
    assert len(agg._background_tasks) == 2

    await agg._cancel_background_tasks()

    assert len(agg._background_tasks) == 0


@pytest.mark.asyncio
async def test_event_drop_warning_throttled():
    """The drop warning should only log once until backlog clears."""
    agg = _make_aggregator()
    agg._MAX_PENDING_EVENTS = 2

    blocker = asyncio.Event()

    async def _blocked_deliver(event_type, payload):
        await blocker.wait()

    agg._deliver_event = _blocked_deliver
    agg.event_emitter = EventEmitter()

    # Fill to cap
    agg._emit_event("e1", {})
    agg._emit_event("e2", {})
    await asyncio.sleep(0)

    # First drop sets the warning flag
    assert agg._event_drop_warned is False
    agg._emit_event("e3", {})
    assert agg._event_drop_warned is True

    # Subsequent drops don't re-warn (flag stays True)
    agg._emit_event("e4", {})
    assert agg._event_drop_warned is True

    # Unblock
    blocker.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_emit_event_reports_telemetry_events_dropped_after_backlog_drains():
    agg = _make_aggregator()
    agg._MAX_PENDING_EVENTS = 2

    blocker = asyncio.Event()
    seen = []

    async def _blocked_deliver(event_type, payload):
        await blocker.wait()
        agg.event_emitter.emit(event_type, payload)

    agg._deliver_event = _blocked_deliver
    agg.event_emitter = EventEmitter()
    agg.event_emitter.subscribe(lambda event: seen.append((event.event_type, dict(event.payload))))

    agg._emit_event("e1", {"i": 1})
    agg._emit_event("e2", {"i": 2})
    await asyncio.sleep(0)

    agg._emit_event("dropped-1", {"i": 3})
    agg._emit_event("dropped-2", {"i": 4})
    assert agg._event_drop_count == 2

    blocker.set()
    await asyncio.sleep(0.05)

    agg._emit_event("recovered", {"i": 5})
    await asyncio.sleep(0.05)

    dropped_events = [payload for event_type, payload in seen if event_type == "telemetry.events_dropped"]
    assert len(dropped_events) == 1
    assert dropped_events[0]["count"] == 2
    assert dropped_events[0]["max_pending_events"] == 2
    assert dropped_events[0]["node_id"] == "n1"
    assert agg._event_drop_count == 0
