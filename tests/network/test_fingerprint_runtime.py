import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quinkgl.aggregation.base import AggregatedModel, AggregationStrategy
from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.network.gossip_node import GossipNode
from quinkgl.fingerprint.fingerprint import DataFingerprint
from quinkgl.topology.base import PeerInfo, SelectionContext, TopologyStrategy


class DummyModel(ModelWrapper):
    def __init__(self):
        super().__init__(model={})

    def get_weights(self):
        return {"w": np.array([1.0])}

    def set_weights(self, weights):
        self.model = weights

    async def train(self, data, config=None):
        return TrainingResult(
            epochs_completed=1,
            final_loss=0.0,
            final_accuracy=0.0,
            samples_trained=0,
        )

    def evaluate(self, data, loss_fn=None):
        return {"loss": 0.0, "accuracy": 0.0}


class DummyTopology(TopologyStrategy):
    def __init__(self):
        super().__init__()
        self.seen_contexts = []

    async def select_targets(self, context: SelectionContext, count: int = 3):
        self.seen_contexts.append(context)
        return []

    async def should_accept_connection(self, context: SelectionContext, peer_info: PeerInfo):
        return True


class DummyAggregation(AggregationStrategy):
    async def aggregate(self, updates):
        return AggregatedModel(
            weights=updates[0].weights,
            contributing_peers=[u.peer_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            updates=updates,
        )


def _make_fp(round_nonce=None):
    return DataFingerprint(
        label_buckets={"a": "high"},
        noised_moments={"l": (1.0, 0.5)},
        sample_bucket="0-100",
        num_classes=0,
        class_count_bucket="small",
        round_nonce=round_nonce,
    )


@pytest.mark.asyncio
async def test_model_aggregator_refreshes_local_fingerprint_each_round():
    topology = DummyTopology()
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=topology,
        aggregator=DummyAggregation(),
        gossip_interval=0.0,
        training_config=TrainingConfig(),
    )

    seen_rounds = []

    def provider(round_number):
        seen_rounds.append(round_number)
        return _make_fp(round_nonce=f"nonce-{round_number}")

    aggregator._local_fingerprint_provider = provider
    aggregator.running = True

    async def stop_soon():
        await asyncio.sleep(0)
        aggregator.stop()

    asyncio.create_task(stop_soon())
    await aggregator.run_continuous(data_provider=None)

    assert seen_rounds == [1]
    assert aggregator._local_fingerprint is not None
    assert aggregator._local_fingerprint.round_nonce == "nonce-1"
    assert topology.seen_contexts[0].my_fingerprint.round_nonce == "nonce-1"


def test_gossip_node_static_fingerprint_is_round_bound_and_propagated():
    topology = DummyTopology()
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=topology,
        aggregator=DummyAggregation(),
        gossip_interval=0.0,
        training_config=TrainingConfig(),
    )
    aggregator.current_round = 3

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    node.gl_node = MagicMock()
    node.gl_node.aggregator = aggregator
    node.community = MagicMock()
    node.community.local_fingerprint = None
    node._fingerprint_source = _make_fp(round_nonce=None)
    node.fingerprint = None

    node._configure_local_fingerprint_runtime()

    assert node.fingerprint is not None
    assert node.fingerprint.round_nonce is not None
    assert aggregator._local_fingerprint.round_nonce == node.fingerprint.round_nonce
    assert node.community.local_fingerprint.round_nonce == node.fingerprint.round_nonce

    first_nonce = node.fingerprint.round_nonce
    aggregator.current_round = 4
    aggregator._refresh_local_fingerprint()
    assert node.fingerprint.round_nonce != first_nonce
    assert node.community.local_fingerprint.round_nonce == node.fingerprint.round_nonce


def test_gossip_node_callable_fingerprint_source_receives_round_number():
    topology = DummyTopology()
    aggregator = ModelAggregator(
        peer_id="n1",
        domain="demo",
        data_schema_hash="abc",
        model=DummyModel(),
        topology=topology,
        aggregator=DummyAggregation(),
        gossip_interval=0.0,
        training_config=TrainingConfig(),
    )

    with patch.object(GossipNode, '__init__', lambda self, **kw: None):
        node = object.__new__(GossipNode)

    seen_rounds = []

    def source(round_number):
        seen_rounds.append(round_number)
        return _make_fp(round_nonce=None)

    node.gl_node = MagicMock()
    node.gl_node.aggregator = aggregator
    node.community = MagicMock()
    node.community.local_fingerprint = None
    node._fingerprint_source = source
    node.fingerprint = None

    aggregator.current_round = 6
    node._configure_local_fingerprint_runtime()

    assert seen_rounds == [6]
    assert node.fingerprint is not None
    assert node.fingerprint.round_nonce is not None
