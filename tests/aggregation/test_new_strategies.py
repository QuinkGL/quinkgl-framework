"""Tests for StalenessWeightedFedAvg and FedProx."""

import pytest
import numpy as np

from quinkgl.aggregation.base import ModelUpdate
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.fedprox import FedProx


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
