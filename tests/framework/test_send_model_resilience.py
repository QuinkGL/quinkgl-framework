"""Regression tests: resilient & concurrent _send_model.

Covers:
- One flaky peer does not prevent delivery to remaining targets.
- Failed sends are logged in comm_log with error details.
- model_send_failed event is emitted for failed peers.
- model_sent event only lists successfully sent peers.
- All targets are sent concurrently (not sequentially).
"""

import asyncio
import time

import numpy as np
import pytest

from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.aggregation.base import AggregationStrategy, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.topology.base import TopologyStrategy
from quinkgl.observability.events import EventEmitter, RuntimeEvent


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
    async def select_targets(self, context, count=3):
        return []

    async def accept_connection(self, peer_info, context):
        return True

    async def should_accept_connection(self, peer_info, context):
        return True


def _make_aggregator():
    agg = ModelAggregator(
        peer_id="n1",
        domain="test",
        data_schema_hash="abc",
        model=_DummyModel(),
        topology=_DummyTopology(),
        aggregator=_DummyAggregator(),
        training_config=TrainingConfig(),
    )
    agg.running = True
    agg.current_round = 5
    return agg


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_one_flaky_peer_does_not_block_others():
    """If peer-b raises, peer-a and peer-c should still receive the model."""
    agg = _make_aggregator()
    delivered = []

    async def callback(peer_id, message):
        if peer_id == "peer-b":
            raise ConnectionError("peer-b is unreachable")
        delivered.append(peer_id)

    agg.send_message_callback = callback

    await agg._send_model(["peer-a", "peer-b", "peer-c"], loss=0.1, accuracy=0.9, samples_trained=16)

    assert sorted(delivered) == ["peer-a", "peer-c"]


@pytest.mark.asyncio
async def test_failed_peer_logged_in_comm_log():
    """Failed sends must appear in comm_log with an error field."""
    agg = _make_aggregator()

    async def callback(peer_id, message):
        if peer_id == "bad-peer":
            raise TimeoutError("timed out")

    agg.send_message_callback = callback

    await agg._send_model(["good-peer", "bad-peer"], loss=0.1, accuracy=0.9, samples_trained=16)

    error_entries = [e for e in agg.comm_log if "error" in e]
    assert len(error_entries) == 1
    assert error_entries[0]["target"] == "bad-peer"
    assert "timed out" in error_entries[0]["error"]

    success_entries = [e for e in agg.comm_log if "error" not in e]
    assert len(success_entries) == 1
    assert success_entries[0]["target"] == "good-peer"


@pytest.mark.asyncio
async def test_model_send_failed_event_emitted():
    """A model_send_failed event must be emitted listing the failed peers."""
    agg = _make_aggregator()
    events = []

    emitter = EventEmitter()
    emitter.subscribe(lambda e: events.append((e.event_type, dict(e.payload))))
    agg.event_emitter = emitter

    async def callback(peer_id, message):
        if peer_id == "flaky":
            raise RuntimeError("network down")

    agg.send_message_callback = callback

    await agg._send_model(["ok-peer", "flaky"], loss=0.1, accuracy=0.9, samples_trained=16)
    await asyncio.sleep(0)  # let event tasks run

    failed_events = [p for t, p in events if t == "model_send_failed"]
    assert len(failed_events) == 1
    assert "flaky" in failed_events[0]["failed_peers"]


@pytest.mark.asyncio
async def test_model_sent_event_only_lists_successful_peers():
    """model_sent event must only contain peers that were actually delivered to."""
    agg = _make_aggregator()
    events = []

    emitter = EventEmitter()
    emitter.subscribe(lambda e: events.append((e.event_type, dict(e.payload))))
    agg.event_emitter = emitter

    async def callback(peer_id, message):
        if peer_id == "dead":
            raise OSError("refused")

    agg.send_message_callback = callback

    await agg._send_model(["alive", "dead"], loss=0.1, accuracy=0.9, samples_trained=16)
    await asyncio.sleep(0)

    sent_events = [p for t, p in events if t == "model_sent"]
    assert len(sent_events) == 1
    assert sent_events[0]["peer_ids"] == ["alive"]


@pytest.mark.asyncio
async def test_false_callback_result_is_reported_as_failed_send():
    """An explicit False result from the send callback is a failed delivery."""
    agg = _make_aggregator()
    events = []

    emitter = EventEmitter()
    emitter.subscribe(lambda e: events.append((e.event_type, dict(e.payload))))
    agg.event_emitter = emitter

    async def callback(peer_id, message):
        if peer_id == "undelivered":
            return False
        return True

    agg.send_message_callback = callback

    await agg._send_model(
        ["delivered", "undelivered"],
        loss=0.1,
        accuracy=0.9,
        samples_trained=16,
    )
    await asyncio.sleep(0)

    failed_events = [p for t, p in events if t == "model_send_failed"]
    sent_events = [p for t, p in events if t == "model_sent"]

    assert failed_events[0]["failed_peers"] == ["undelivered"]
    assert sent_events[0]["peer_ids"] == ["delivered"]


@pytest.mark.asyncio
async def test_concurrent_send_is_faster_than_sequential():
    """asyncio.gather should run sends concurrently, not sequentially."""
    agg = _make_aggregator()
    delay = 0.1

    async def slow_callback(peer_id, message):
        await asyncio.sleep(delay)

    agg.send_message_callback = slow_callback

    targets = ["peer-a", "peer-b", "peer-c"]
    start = time.monotonic()
    await agg._send_model(targets, loss=0.1, accuracy=0.9, samples_trained=16)
    elapsed = time.monotonic() - start

    # Sequential would take ~0.3 s; concurrent should be ~0.1 s
    assert elapsed < delay * len(targets) * 0.8


@pytest.mark.asyncio
async def test_all_peers_fail_does_not_raise():
    """Even if every peer fails, _send_model must not raise."""
    agg = _make_aggregator()

    async def always_fail(peer_id, message):
        raise ConnectionError("all down")

    agg.send_message_callback = always_fail

    # Should not raise
    await agg._send_model(["p1", "p2", "p3"], loss=0.1, accuracy=0.9, samples_trained=16)

    assert len(agg.comm_log) == 3
    assert all("error" in e for e in agg.comm_log)


@pytest.mark.asyncio
async def test_no_callback_does_not_raise():
    """If send_message_callback is None, _send_model returns silently."""
    agg = _make_aggregator()
    agg.send_message_callback = None

    await agg._send_model(["peer-a"], loss=0.1, accuracy=0.9, samples_trained=16)
    assert len(agg.comm_log) == 0
