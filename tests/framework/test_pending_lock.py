"""Regression tests: pending_updates lock, round-gating, and drain.

Covers:
- TOCTOU: concurrent _handle_model_update during _aggregate_models loses zero updates.
- Round-gating: stale (round N-5) and future (round N+5) updates are rejected.
- Running guard: updates after stop() are refused.
- Re-entrancy guard: overlapping _aggregate_models calls are short-circuited.
- Drain semantics: pending_updates is empty after aggregation, new arrivals survive.
"""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.gossip.protocol import ModelUpdateMessage
from quinkgl.aggregation.base import AggregationStrategy, ModelUpdate, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo


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


class _BlockingModel(_DummyModel):
    def __init__(self, gate: asyncio.Event):
        super().__init__()
        self._gate = gate

    async def train(self, data, config=None):
        await self._gate.wait()
        return await super().train(data, config=config)


class _SlowAggregator(AggregationStrategy):
    """Aggregator that sleeps to simulate a slow merge, allowing concurrent appends."""

    def __init__(self, delay: float = 0.1):
        super().__init__()
        self.delay = delay

    async def aggregate(self, updates):
        await asyncio.sleep(self.delay)
        return AggregatedModel(
            weights=updates[0].weights,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
        )


class _InstantAggregator(AggregationStrategy):
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


def _make_aggregator(aggregator=None, stale_round_tolerance=2):
    agg = ModelAggregator(
        peer_id="n1",
        domain="test",
        data_schema_hash="abc",
        model=_DummyModel(),
        topology=_DummyTopology(),
        aggregator=aggregator or _InstantAggregator(),
        training_config=TrainingConfig(),
        stale_round_tolerance=stale_round_tolerance,
    )
    agg.running = True
    agg.current_round = 10
    return agg


def _make_aggregator_with_model(model, aggregator=None, stale_round_tolerance=2):
    agg = ModelAggregator(
        peer_id="n1",
        domain="test",
        data_schema_hash="abc",
        model=model,
        topology=_DummyTopology(),
        aggregator=aggregator or _InstantAggregator(),
        training_config=TrainingConfig(),
        stale_round_tolerance=stale_round_tolerance,
    )
    agg.running = True
    agg.current_round = 10
    return agg


def _make_update_msg(sender_id="peer-a", round_number=10):
    return ModelUpdateMessage.create(
        sender_id=sender_id,
        weights={"w": np.array([3.0, 4.0])},
        sample_count=16,
        loss=0.1,
        accuracy=0.9,
        round_number=round_number,
    )


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_updates_during_slow_aggregation_are_not_lost():
    """Updates arriving while _aggregate_models is running must survive."""
    agg = _make_aggregator(aggregator=_SlowAggregator(delay=0.15))

    # Seed one update so aggregation can start
    await agg._handle_model_update(_make_update_msg("peer-a", 10))
    assert len(agg.pending_updates) == 1

    # Start aggregation (drains immediately, then sleeps 0.15 s)
    agg_task = asyncio.create_task(agg._aggregate_models())

    # While aggregation is in flight, push a second update
    await asyncio.sleep(0.05)
    await agg._handle_model_update(_make_update_msg("peer-b", 10))

    await agg_task  # wait for aggregation to finish

    # The second update must have survived in pending_updates
    assert len(agg.pending_updates) == 1
    assert agg.pending_updates[0].peer_id == "peer-b"


@pytest.mark.asyncio
async def test_stale_round_update_is_rejected():
    """An update from 5 rounds ago must be rejected (tolerance=2)."""
    agg = _make_aggregator(stale_round_tolerance=2)
    agg.current_round = 10

    await agg._handle_model_update(_make_update_msg("stale-peer", round_number=5))

    assert len(agg.pending_updates) == 0


@pytest.mark.asyncio
async def test_future_round_update_is_rejected():
    """An update from 5 rounds in the future must be rejected (tolerance=2)."""
    agg = _make_aggregator(stale_round_tolerance=2)
    agg.current_round = 10

    await agg._handle_model_update(_make_update_msg("future-peer", round_number=15))

    assert len(agg.pending_updates) == 0


@pytest.mark.asyncio
async def test_within_tolerance_update_is_accepted():
    """Updates within tolerance (round 8, 9, 10, 11, 12 at round 10) are accepted."""
    agg = _make_aggregator(stale_round_tolerance=2)
    agg.current_round = 10

    for rnd in [8, 9, 10, 11, 12]:
        await agg._handle_model_update(_make_update_msg(f"peer-{rnd}", round_number=rnd))

    assert len(agg.pending_updates) == 5


@pytest.mark.asyncio
async def test_updates_refused_after_stop():
    """Once running=False, no updates should be appended."""
    agg = _make_aggregator()
    agg.running = False

    await agg._handle_model_update(_make_update_msg("late-peer", 10))

    assert len(agg.pending_updates) == 0


@pytest.mark.asyncio
async def test_reentrant_aggregation_is_skipped():
    """A second concurrent _aggregate_models call must short-circuit."""
    agg = _make_aggregator(aggregator=_SlowAggregator(delay=0.2))

    await agg._handle_model_update(_make_update_msg("peer-a", 10))

    task1 = asyncio.create_task(agg._aggregate_models())
    await asyncio.sleep(0.05)  # let task1 acquire _aggregating

    # Force another update + second aggregation attempt
    await agg._handle_model_update(_make_update_msg("peer-b", 10))
    result = await agg._aggregate_models()  # should return None (re-entrant)

    assert result is None
    await task1


@pytest.mark.asyncio
async def test_pending_update_queue_applies_backpressure_when_full():
    agg = _make_aggregator()
    agg.max_pending_updates = 1

    await agg._handle_model_update(_make_update_msg("peer-a", 10))
    await agg._handle_model_update(_make_update_msg("peer-b", 10))

    assert len(agg.pending_updates) == 1
    assert agg.pending_updates[0].peer_id == "peer-a"


@pytest.mark.asyncio
async def test_model_snapshot_waits_for_inflight_training_under_model_lock():
    gate = asyncio.Event()
    agg = _make_aggregator_with_model(_BlockingModel(gate))

    train_task = asyncio.create_task(agg._train_local(data=[1]))
    await asyncio.sleep(0)

    snapshot_task = asyncio.create_task(agg.get_model_weights_snapshot())
    await asyncio.sleep(0.05)
    assert snapshot_task.done() is False

    gate.set()
    await train_task
    snapshot = await snapshot_task

    assert snapshot is not None
    assert "w" in snapshot


@pytest.mark.asyncio
async def test_drain_clears_pending_and_aggregation_succeeds():
    """After aggregation, pending_updates must be empty."""
    agg = _make_aggregator()

    await agg._handle_model_update(_make_update_msg("peer-a", 10))
    await agg._handle_model_update(_make_update_msg("peer-b", 10))
    assert len(agg.pending_updates) == 2

    result = await agg._aggregate_models()

    assert result is not None
    assert len(agg.pending_updates) == 0
    assert "n1" in result.contributing_peers
    assert "peer-a" in result.contributing_peers
    assert "peer-b" in result.contributing_peers


@pytest.mark.asyncio
async def test_belt_and_braces_round_filter_in_drain():
    """Updates that become stale between append and drain are filtered out."""
    agg = _make_aggregator(stale_round_tolerance=2)
    agg.current_round = 10

    # Manually insert an update that was valid at round 10
    await agg._handle_model_update(_make_update_msg("peer-a", 10))
    assert len(agg.pending_updates) == 1

    # Simulate round advancing before aggregation fires
    agg.current_round = 15

    result = await agg._aggregate_models()

    # peer-a's update (round 10) is now |15-10|=5 > tolerance=2, filtered
    assert result is None
