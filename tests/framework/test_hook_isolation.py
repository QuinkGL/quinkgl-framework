"""Task 2b: verify that a raising hook does not abort the gossip round.

A single failing lifecycle hook should be logged and skipped, leaving all
subsequent hooks and the rest of the pipeline unaffected.
"""

import asyncio
import pytest

from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
from quinkgl.aggregation.base import AggregationStrategy, ModelUpdate, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _NoopTopology(TopologyStrategy):
    async def select_targets(self, context, count=3):
        return []

    async def should_accept_connection(self, context, peer_info):
        return True


class _NoopAggregator(AggregationStrategy):
    async def aggregate(self, updates):
        w = updates[0].weights
        return AggregatedModel(
            weights=w,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
        )


class _ConstantModel(ModelWrapper):
    def __init__(self):
        import numpy as np
        self._w = {"layer": np.zeros(4, dtype="float32")}

    def get_weights(self):
        return self._w

    def set_weights(self, w):
        self._w = w

    async def train(self, data, config):
        return TrainingResult(epochs_completed=1, final_loss=0.1, final_accuracy=0.9, samples_trained=32)

    async def evaluate(self, data, config=None):
        return {"loss": 0.1, "accuracy": 0.9}

    def get_model_version(self):
        return "1.0.0"

    def get_data_schema_hash(self):
        return "abc123"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHookIsolation:
    def _make_aggregator(self):
        return ModelAggregator(
            peer_id="test-node",
            domain="test",
            data_schema_hash="abc",
            model=_ConstantModel(),
            topology=_NoopTopology(),
            aggregator=_NoopAggregator(),
            gossip_interval=0.01,
        )

    async def test_raising_hook_does_not_raise_from_execute_hooks(self):
        """_execute_hooks must swallow hook exceptions and continue."""
        agg = self._make_aggregator()
        called_after = []

        def bad_hook(*args, **kwargs):
            raise RuntimeError("hook intentionally broken")

        def good_hook(*args, **kwargs):
            called_after.append(True)

        agg.register_hook("before_train", bad_hook)
        agg.register_hook("before_train", good_hook)

        # Must not raise
        await agg._execute_hooks("before_train")

        # The good hook after the bad one must still have been called
        assert called_after == [True], "Hook registered after a failing hook was not called"

    async def test_raising_async_hook_does_not_raise_from_execute_hooks(self):
        """Async hook exceptions are also swallowed."""
        agg = self._make_aggregator()
        reached = []

        async def bad_async_hook(*args, **kwargs):
            raise ValueError("async hook broken")

        async def good_async_hook(*args, **kwargs):
            reached.append(1)

        agg.register_hook("after_train", bad_async_hook)
        agg.register_hook("after_train", good_async_hook)

        await agg._execute_hooks("after_train", object())

        assert reached == [1]

    async def test_raising_hook_does_not_abort_training_round(self):
        """A raising before_train hook must not prevent _train_local from running."""
        agg = self._make_aggregator()
        agg.current_round = 1

        def crash_hook(*args, **kwargs):
            raise RuntimeError("crash before train")

        agg.register_hook("before_train", crash_hook)

        import numpy as np
        loss, acc, samples = await agg._train_local(np.zeros((10, 1)))
        assert loss == pytest.approx(0.1)
        assert samples == 32
