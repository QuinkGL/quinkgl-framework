"""Tests for FedAvgM (Federated Averaging with Momentum)."""

import pytest
import numpy as np
from copy import deepcopy

from quinkgl.aggregation.base import ModelUpdate
from quinkgl.aggregation.fedavgm import FedAvgM


def _updates(*values, sample_count=1):
    return [ModelUpdate(peer_id=f"p{i}", weights=np.array([v], dtype=np.float32),
                        sample_count=sample_count)
            for i, v in enumerate(values)]


class TestFedAvgMInit:
    def test_valid_momentum(self):
        agg = FedAvgM(server_momentum=0.9)
        assert agg.server_momentum == 0.9

    def test_zero_momentum_valid(self):
        agg = FedAvgM(server_momentum=0.0)
        assert agg.server_momentum == 0.0

    def test_momentum_too_large_raises(self):
        with pytest.raises(ValueError, match="server_momentum"):
            FedAvgM(server_momentum=1.0)

    def test_momentum_negative_raises(self):
        with pytest.raises(ValueError, match="server_momentum"):
            FedAvgM(server_momentum=-0.1)

    def test_buffer_starts_none(self):
        agg = FedAvgM()
        assert agg.momentum_buffer is None


class TestFedAvgMFirstRound:
    @pytest.mark.asyncio
    async def test_first_round_no_momentum(self):
        """First round: buffer = averaged (no blending)."""
        agg = FedAvgM(server_momentum=0.9)
        updates = _updates(2.0, 4.0)
        result = await agg.aggregate(updates)
        # uniform weight → avg = 3.0; no momentum on first round
        assert np.allclose(result.weights, [3.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_first_round_sets_buffer(self):
        agg = FedAvgM(server_momentum=0.9)
        await agg.aggregate(_updates(4.0, 6.0))
        assert agg.momentum_buffer is not None

    @pytest.mark.asyncio
    async def test_metadata_aggregation_method(self):
        agg = FedAvgM()
        result = await agg.aggregate(_updates(1.0, 2.0))
        assert result.metadata["aggregation_method"] == "fedavgm"
        assert result.metadata["momentum"] == agg.server_momentum


class TestFedAvgMMomentum:
    @pytest.mark.asyncio
    async def test_second_round_applies_momentum(self):
        """buffer_new = 0.9*buffer_old + 0.1*avg_new"""
        agg = FedAvgM(server_momentum=0.9)
        await agg.aggregate(_updates(0.0, 0.0))   # round 1: buffer = 0.0
        result = await agg.aggregate(_updates(10.0, 10.0))  # round 2: avg=10.0
        # expected: 0.9*0.0 + 0.1*10.0 = 1.0
        assert np.allclose(result.weights, [1.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_momentum_zero_replaces_buffer(self):
        """server_momentum=0 means buffer is fully replaced each round."""
        agg = FedAvgM(server_momentum=0.0)
        await agg.aggregate(_updates(5.0, 5.0))   # buffer = 5.0
        result = await agg.aggregate(_updates(2.0, 2.0))
        assert np.allclose(result.weights, [2.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_output_not_mutating_input(self):
        agg = FedAvgM(server_momentum=0.9)
        original = np.array([3.0], dtype=np.float32)
        updates = [ModelUpdate("a", original.copy())]
        await agg.aggregate(updates)
        assert np.allclose(original, [3.0])


class TestFedAvgMEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_updates_raises(self):
        agg = FedAvgM()
        with pytest.raises(ValueError, match="empty"):
            await agg.aggregate([])

    @pytest.mark.asyncio
    async def test_single_update(self):
        agg = FedAvgM(server_momentum=0.9)
        result = await agg.aggregate(_updates(7.0))
        assert np.allclose(result.weights, [7.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_nan_in_weights_raises(self):
        agg = FedAvgM()
        updates = [ModelUpdate("a", np.array([float("nan")]))]
        with pytest.raises(ValueError, match="NaN"):
            await agg.aggregate(updates)

    @pytest.mark.asyncio
    async def test_inf_in_weights_raises(self):
        agg = FedAvgM()
        updates = [ModelUpdate("a", np.array([float("inf")]))]
        with pytest.raises(ValueError, match="Inf"):
            await agg.aggregate(updates)

    @pytest.mark.asyncio
    async def test_dict_weights(self):
        agg = FedAvgM(server_momentum=0.0)
        updates = [
            ModelUpdate("a", {"w": np.array([1.0])}, sample_count=1),
            ModelUpdate("b", {"w": np.array([3.0])}, sample_count=1),
        ]
        result = await agg.aggregate(updates)
        assert np.allclose(result.weights["w"], [2.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_weighted_by_sample_count(self):
        agg = FedAvgM(server_momentum=0.0)
        updates = [
            ModelUpdate("a", np.array([0.0]), sample_count=1),
            ModelUpdate("b", np.array([4.0]), sample_count=3),
        ]
        result = await agg.aggregate(updates)
        # weighted avg: (0*1 + 4*3) / 4 = 3.0
        assert np.allclose(result.weights, [3.0], atol=1e-5)

    @pytest.mark.asyncio
    async def test_contributing_peers_recorded(self):
        agg = FedAvgM()
        updates = _updates(1.0, 2.0)
        result = await agg.aggregate(updates)
        assert set(result.contributing_peers) == {"p0", "p1"}
