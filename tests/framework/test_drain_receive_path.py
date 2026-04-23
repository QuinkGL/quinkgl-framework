"""Regression tests: drain receive path on early stop.

Covers:
- Updates after loop stop are refused.
- pending_updates is cleared on loop exit.
- Repeated post-stop updates keep pending_updates bounded at zero.
"""

import asyncio

import numpy as np
import pytest

from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.gossip.protocol import ModelUpdateMessage
from quinkgl.aggregation.base import AggregationStrategy, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.topology.base import TopologyStrategy


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
    return ModelAggregator(
        peer_id="n1",
        domain="test",
        data_schema_hash="abc",
        model=_DummyModel(),
        topology=_DummyTopology(),
        aggregator=_DummyAggregator(),
        training_config=TrainingConfig(),
    )


def _make_msg(sender_id="peer-a", round_number=5):
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
async def test_updates_refused_when_not_running():
    """Updates must be dropped when running=False."""
    agg = _make_aggregator()
    # Default running=False

    await agg._handle_model_update(_make_msg())
    assert len(agg.pending_updates) == 0


@pytest.mark.asyncio
async def test_pending_cleared_after_loop_exit():
    """pending_updates must be empty after run_continuous exits."""
    agg = _make_aggregator()

    # Seed a pending update by setting running temporarily
    agg.running = True
    agg.current_round = 5
    await agg._handle_model_update(_make_msg(round_number=5))
    assert len(agg.pending_updates) == 1

    # run_continuous should clear on exit
    async def _quick_loop():
        agg.running = True
        # Stop immediately after one iteration
        stop_task = asyncio.create_task(stop_soon())
        try:
            await agg.run_continuous(data_provider=None)
        finally:
            # T14: Cancel stop task on teardown
            if not stop_task.done():
                stop_task.cancel()

    await _quick_loop()
    assert len(agg.pending_updates) == 0


@pytest.mark.asyncio
async def test_repeated_post_stop_updates_stay_bounded():
    """Sending 100 updates after stop must not grow pending_updates."""
    agg = _make_aggregator()
    agg.running = False

    for i in range(100):
        await agg._handle_model_update(_make_msg(f"peer-{i}", round_number=5))

    assert len(agg.pending_updates) == 0
