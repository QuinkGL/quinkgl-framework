"""Tests for StalenessWeightedFedAvg and FedProx."""

import pytest
import numpy as np

from quinkgl.aggregation.base import ModelUpdate
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult
from quinkgl.topology.base import TopologyStrategy


class _CaptureModel(ModelWrapper):
    def __init__(self):
        super().__init__(model=None)
        self.seen_config = None

    def get_weights(self):
        return {"w": np.array([1.0])}

    def set_weights(self, weights):
        pass

    async def train(self, data, config=None):
        self.seen_config = config
        return TrainingResult(epochs_completed=1, final_loss=0.25, final_accuracy=0.75, samples_trained=8)

    def evaluate(self, data, loss_fn=None):
        return {"loss": 0.25, "accuracy": 0.75}


class _NoopTopology(TopologyStrategy):
    async def select_targets(self, context, count=3):
        return []

    async def accept_connection(self, peer_info, context):
        return True

    async def should_accept_connection(self, peer_info, context):
        return True


class TestStalenessWeightedFedAvg:
    def test_stale_update_gets_less_weight(self):
        sw = StalenessWeightedFedAvg(staleness_coefficient=0.1)
        stale = ModelUpdate(peer_id="p1", weights=np.array([1.0]), round_number=5)
        fresh = ModelUpdate(peer_id="p2", weights=np.array([1.0]), round_number=10)
        w_stale = sw.compute_staleness_weight(stale, current_round=10)
        w_fresh = sw.compute_staleness_weight(fresh, current_round=10)
        assert w_stale < w_fresh

    def test_zero_staleness_no_penalty(self):
        sw = StalenessWeightedFedAvg(staleness_coefficient=0.1)
        update = ModelUpdate(peer_id="p1", weights=np.array([1.0]), round_number=10)
        weight = sw.compute_staleness_weight(update, current_round=10)
        assert weight == 1.0

    @pytest.mark.asyncio
    async def test_aggregation_with_staleness(self):
        sw = StalenessWeightedFedAvg(staleness_coefficient=0.5)
        updates = [
            ModelUpdate(peer_id="p1", weights=np.array([2.0]), sample_count=100, round_number=5),
            ModelUpdate(peer_id="p2", weights=np.array([4.0]), sample_count=100, round_number=10),
        ]
        result = await sw.aggregate(updates, current_round=10)
        assert result.metadata["aggregation_method"] == "staleness_weighted_fedavg"
        assert "staleness_info" in result.metadata

    def test_higher_coefficient_more_penalty(self):
        sw_low = StalenessWeightedFedAvg(staleness_coefficient=0.01)
        sw_high = StalenessWeightedFedAvg(staleness_coefficient=1.0)
        update = ModelUpdate(peer_id="p1", weights=np.array([1.0]), round_number=5)
        w_low = sw_low.compute_staleness_weight(update, current_round=10)
        w_high = sw_high.compute_staleness_weight(update, current_round=10)
        assert w_high < w_low


class TestFedProxModes:
    def test_training_time_mode_default(self):
        fp = FedProx(mu=0.01)
        assert fp.mode == "training_time"

    def test_legacy_mode_available(self):
        fp = FedProx(mu=0.01, mode="weight_interpolation")
        assert fp.mode == "weight_interpolation"

    def test_get_training_config_overrides_empty_initially(self):
        fp = FedProx(mu=0.01, mode="training_time")
        overrides = fp.get_training_config_overrides()
        assert overrides == {}

    @pytest.mark.asyncio
    async def test_training_time_mode_stores_global_weights(self):
        fp = FedProx(mu=0.01, mode="training_time")
        updates = [
            ModelUpdate(peer_id="p1", weights=np.array([1.0, 2.0]), sample_count=10),
            ModelUpdate(peer_id="p2", weights=np.array([3.0, 4.0]), sample_count=10),
        ]
        result = await fp.aggregate(updates)
        assert fp.global_weights is not None
        overrides = fp.get_training_config_overrides()
        assert "proximal_coefficient" in overrides
        assert overrides["proximal_coefficient"] == 0.01
        assert "global_weights" in overrides

    @pytest.mark.asyncio
    async def test_weight_interpolation_mode_applies_correction(self):
        fp = FedProx(mu=0.5, mode="weight_interpolation")
        updates = [
            ModelUpdate(peer_id="p1", weights=np.array([0.0]), sample_count=10),
        ]
        result1 = await fp.aggregate(updates)
        updates2 = [
            ModelUpdate(peer_id="p1", weights=np.array([10.0]), sample_count=10),
        ]
        result2 = await fp.aggregate(updates2)
        assert result2.metadata["fedprox_mode"] == "weight_interpolation"

    @pytest.mark.asyncio
    async def test_metadata_includes_mu(self):
        fp = FedProx(mu=0.05)
        updates = [
            ModelUpdate(peer_id="p1", weights=np.array([1.0]), sample_count=10),
        ]
        result = await fp.aggregate(updates)
        # First round returns FedAvg (no global weights yet), subsequent rounds include mu
        updates2 = [
            ModelUpdate(peer_id="p1", weights=np.array([2.0]), sample_count=10),
        ]
        result2 = await fp.aggregate(updates2)
        assert result2.metadata["mu"] == 0.05

    @pytest.mark.asyncio
    async def test_training_time_mode_overrides_reach_model_train_via_aggregator(self):
        fp = FedProx(mu=0.25, mode="training_time")
        await fp.aggregate([
            ModelUpdate(peer_id="p1", weights={"w": np.array([1.0])}, sample_count=10),
            ModelUpdate(peer_id="p2", weights={"w": np.array([3.0])}, sample_count=10),
        ])

        model = _CaptureModel()
        aggregator = ModelAggregator(
            peer_id="n1",
            domain="demo",
            data_schema_hash="abc",
            model=model,
            topology=_NoopTopology(),
            aggregator=fp,
            training_config=TrainingConfig(epochs=3, batch_size=7, learning_rate=0.05, grad_clip_norm=1.5),
        )

        await aggregator._train_local(data=[1])

        assert model.seen_config is not None
        assert model.seen_config is not aggregator.training_config
        assert model.seen_config.proximal_coefficient == 0.25
        assert np.array_equal(model.seen_config.global_weights["w"], fp.global_weights["w"])
        assert model.seen_config.learning_rate == 0.05
        assert model.seen_config.grad_clip_norm == 1.5


class TestFedAvgM:
    @pytest.mark.asyncio
    async def test_first_round_matches_plain_average(self):
        agg = FedAvgM(server_momentum=0.9)
        updates = [
            ModelUpdate(peer_id="p1", weights=np.array([1.0]), sample_count=10),
            ModelUpdate(peer_id="p2", weights=np.array([3.0]), sample_count=10),
        ]

        result = await agg.aggregate(updates)

        assert np.allclose(result.weights, np.array([2.0]))
        assert np.allclose(agg.momentum_buffer, np.array([0.0]))

    @pytest.mark.asyncio
    async def test_subsequent_rounds_apply_server_momentum_to_deltas(self):
        agg = FedAvgM(server_momentum=0.5)

        await agg.aggregate([
            ModelUpdate(peer_id="p1", weights=np.array([1.0]), sample_count=10),
            ModelUpdate(peer_id="p2", weights=np.array([3.0]), sample_count=10),
        ])
        result2 = await agg.aggregate([
            ModelUpdate(peer_id="p1", weights=np.array([5.0]), sample_count=10),
            ModelUpdate(peer_id="p2", weights=np.array([7.0]), sample_count=10),
        ])
        result3 = await agg.aggregate([
            ModelUpdate(peer_id="p1", weights=np.array([5.0]), sample_count=10),
            ModelUpdate(peer_id="p2", weights=np.array([7.0]), sample_count=10),
        ])

        assert np.allclose(result2.weights, np.array([6.0]))
        assert np.allclose(result3.weights, np.array([8.0]))
        assert np.allclose(agg.momentum_buffer, np.array([-2.0]))

    @pytest.mark.asyncio
    async def test_dict_weights_follow_server_momentum_update(self):
        agg = FedAvgM(server_momentum=0.5)

        await agg.aggregate([
            ModelUpdate(peer_id="p1", weights={"w": np.array([1.0])}, sample_count=10),
            ModelUpdate(peer_id="p2", weights={"w": np.array([3.0])}, sample_count=10),
        ])
        result = await agg.aggregate([
            ModelUpdate(peer_id="p1", weights={"w": np.array([5.0])}, sample_count=10),
            ModelUpdate(peer_id="p2", weights={"w": np.array([7.0])}, sample_count=10),
        ])

        assert np.allclose(result.weights["w"], np.array([6.0]))
        assert np.allclose(agg.global_weights["w"], np.array([6.0]))
