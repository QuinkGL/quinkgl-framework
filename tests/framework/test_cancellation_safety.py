"""
T18 & T19: Cancellation-safety tests for run_in_executor and ModelWrapper.train.

Verifies that:
 - _evaluate_model releases _model_lock when the outer task is cancelled
 - _train_local releases _model_lock when the outer task is cancelled
 - CancelledError is properly re-raised after cleanup
 - The model lock is not left in a held state after cancellation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from quinkgl.aggregation.base import ModelUpdate
from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult


class _SlowModel(ModelWrapper):
    """Model wrapper whose train/evaluate methods block for a controlled duration."""

    def __init__(self, train_delay: float = 0.0, eval_delay: float = 0.0):
        super().__init__(model=MagicMock(), model_version="1.0.0")
        self._train_delay = train_delay
        self._eval_delay = eval_delay
        self._train_call_count = 0
        self._eval_call_count = 0

    def get_weights(self):
        return {"layer": np.array([1.0, 2.0])}

    def set_weights(self, weights):
        pass

    async def train(self, data, config=None):
        self._train_call_count += 1
        # Simulate synchronous CPU work by sleeping in the event loop
        await asyncio.sleep(self._train_delay)
        return TrainingResult(
            epochs_completed=1,
            final_loss=0.5,
            final_accuracy=0.8,
            samples_trained=100,
        )

    def evaluate(self, data, loss_fn=None):
        import time
        self._eval_call_count += 1
        # Simulate blocking CPU/GPU work (cannot be interrupted)
        time.sleep(self._eval_delay)
        return {"loss": 0.3, "accuracy": 0.9}


def _make_aggregator(model=None, **kwargs):
    """Create a ModelAggregator with minimal stubs."""
    from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
    from quinkgl.aggregation import FedAvg

    class _StubTopology(TopologyStrategy):
        async def select_targets(self, context, count=3):
            return []

        async def should_accept_connection(self, context, peer_info):
            return True

    if model is None:
        model = _SlowModel()

    defaults = dict(
        peer_id="test_node",
        domain="test",
        data_schema_hash="abc",
        model=model,
        topology=_StubTopology(),
        aggregator=FedAvg(),
        gossip_interval=0.01,
        min_peers_before_aggregate=1,
    )
    defaults.update(kwargs)
    return ModelAggregator(**defaults)


# ------------------------------------------------------------------ #
# T18: Cancellation-safety of _evaluate_model (run_in_executor)
# ------------------------------------------------------------------ #


class TestEvaluateModelCancellation:
    @pytest.mark.asyncio
    async def test_evaluate_model_releases_lock_on_cancel(self):
        """T18: _model_lock must be released when the outer task is cancelled
        while _evaluate_model is awaiting run_in_executor."""
        model = _SlowModel(eval_delay=0.5)
        agg = _make_aggregator(model=model)

        eval_data = MagicMock()

        # Start evaluation in a task, then cancel it
        task = asyncio.ensure_future(agg._evaluate_model(eval_data))
        # Give the executor a moment to start
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The model lock must be available now
        assert not agg._model_lock.locked(), (
            "_model_lock is still held after _evaluate_model was cancelled"
        )

        # We should be able to acquire the lock immediately
        async with agg._model_lock:
            pass  # lock acquired and released successfully

    @pytest.mark.asyncio
    async def test_evaluate_model_normal_completion(self):
        """T18: Normal (non-cancelled) evaluation should still work."""
        model = _SlowModel(eval_delay=0.0)
        agg = _make_aggregator(model=model)

        result = await agg._evaluate_model(MagicMock())
        assert "loss" in result
        assert "accuracy" in result
        assert model._eval_call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_model_cancel_then_normal(self):
        """T18: After cancellation, a subsequent normal evaluation should work."""
        model = _SlowModel(eval_delay=0.3)
        agg = _make_aggregator(model=model)

        # Cancel the first evaluation
        task = asyncio.ensure_future(agg._evaluate_model(MagicMock()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # A subsequent evaluation should work fine
        result = await agg._evaluate_model(MagicMock())
        assert "loss" in result


# ------------------------------------------------------------------ #
# T19: Cancellation-safety of _train_local (ModelWrapper.train)
# ------------------------------------------------------------------ #


class TestTrainLocalCancellation:
    @pytest.mark.asyncio
    async def test_train_local_releases_lock_on_cancel(self):
        """T19: _model_lock must be released when the outer task is cancelled
        while _train_local is awaiting model.train()."""
        model = _SlowModel(train_delay=0.5)
        agg = _make_aggregator(model=model)

        train_data = MagicMock()

        # Start training in a task, then cancel it
        task = asyncio.ensure_future(agg._train_local(train_data))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The model lock must be available now
        assert not agg._model_lock.locked(), (
            "_model_lock is still held after _train_local was cancelled"
        )

        # We should be able to acquire the lock immediately
        async with agg._model_lock:
            pass

    @pytest.mark.asyncio
    async def test_train_local_normal_completion(self):
        """T19: Normal (non-cancelled) training should still work."""
        model = _SlowModel(train_delay=0.0)
        agg = _make_aggregator(model=model)

        loss, acc, samples = await agg._train_local(MagicMock())
        assert loss == 0.5
        assert acc == 0.8
        assert samples == 100
        assert model._train_call_count == 1

    @pytest.mark.asyncio
    async def test_train_local_cancel_then_normal(self):
        """T19: After cancellation, a subsequent normal training should work."""
        model = _SlowModel(train_delay=0.3)
        agg = _make_aggregator(model=model)

        # Cancel the first training
        task = asyncio.ensure_future(agg._train_local(MagicMock()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # A subsequent training should work fine
        loss, acc, samples = await agg._train_local(MagicMock())
        assert loss == 0.5
        assert acc == 0.8

    @pytest.mark.asyncio
    async def test_run_continuous_handles_cancelled_training(self):
        """T19: run_continuous should handle CancelledError from training gracefully."""
        model = _SlowModel(train_delay=10.0)  # Long training
        agg = _make_aggregator(model=model, gossip_interval=0.01)

        # Start run_continuous with a data provider
        task = asyncio.ensure_future(
            agg.run_continuous(data_provider=MagicMock())
        )

        # Let it start the first round and enter training
        await asyncio.sleep(0.1)

        # Cancel the run
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The model lock should not be held
        assert not agg._model_lock.locked()
