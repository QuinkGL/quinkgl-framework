"""Tests for MultiKrum (Byzantine-resilient averaging)."""

import pytest
import numpy as np

from quinkgl.aggregation.base import ModelUpdate
from quinkgl.aggregation.krum import MultiKrum


def _u(peer_id, value, sample_count=1):
    return ModelUpdate(peer_id=peer_id, weights=np.array([value], dtype=np.float64),
                       sample_count=sample_count)


class TestMultiKrumPrecondition:
    @pytest.mark.asyncio
    async def test_n_less_than_2f_plus_3_raises(self):
        """n < 2f+3 must raise (H-1 fix: was n <= 2f)."""
        mk = MultiKrum(num_byzantines=1)
        # n=2 < 2*1+3=5 → raise
        with pytest.raises(ValueError, match="2f\\+3"):
            await mk.aggregate([_u("a", 1.0), _u("b", 2.0)])

    @pytest.mark.asyncio
    async def test_n_equals_2f_plus_2_raises(self):
        mk = MultiKrum(num_byzantines=1)
        # n=4 < 5 → raise
        with pytest.raises(ValueError, match="2f\\+3"):
            await mk.aggregate([_u("a", 1.0), _u("b", 2.0), _u("c", 3.0), _u("d", 4.0)])

    @pytest.mark.asyncio
    async def test_minimum_valid_n(self):
        """n = 2f+3 is the minimum valid input (should NOT raise)."""
        mk = MultiKrum(num_byzantines=1)
        # n=5 = 2*1+3 → valid
        updates = [_u(f"p{i}", float(i)) for i in range(5)]
        result = await mk.aggregate(updates)
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_raises(self):
        mk = MultiKrum(num_byzantines=1)
        with pytest.raises(ValueError):
            await mk.aggregate([])


class TestMultiKrumSelection:
    @pytest.mark.asyncio
    async def test_excludes_byzantine_outlier(self):
        """n=6, f=1 → selects 4 most central; outliers [100, 200] should be excluded."""
        mk = MultiKrum(num_byzantines=1)
        updates = [
            _u("honest_1", 1.0),
            _u("honest_2", 1.1),
            _u("honest_3", 0.9),
            _u("honest_4", 1.05),
            _u("byz_1",  100.0),
            _u("byz_2",  200.0),
        ]
        result = await mk.aggregate(updates)
        # Result should be close to 1.0, not skewed toward 100/200
        assert abs(float(result.weights[0]) - 1.0) < 0.2

    @pytest.mark.asyncio
    async def test_selected_peers_recorded_in_metadata(self):
        mk = MultiKrum(num_byzantines=1)
        updates = [_u(f"p{i}", float(i)) for i in range(5)]
        result = await mk.aggregate(updates)
        assert "selected_peers" in result.metadata
        assert result.metadata["aggregation_method"] == "multikrum"
        assert result.metadata["num_byzantines"] == 1

    @pytest.mark.asyncio
    async def test_num_selected_equals_n_minus_2f(self):
        """MultiKrum selects n - 2f updates."""
        mk = MultiKrum(num_byzantines=1)
        # n=6, f=1 → select 4
        updates = [_u(f"p{i}", float(i)) for i in range(6)]
        result = await mk.aggregate(updates)
        assert len(result.metadata["selected_peers"]) == 4

    @pytest.mark.asyncio
    async def test_result_is_average_not_single_peer(self):
        """MultiKrum averages selected peers, unlike Krum which picks one."""
        mk = MultiKrum(num_byzantines=1)
        updates = [
            _u("a", 0.0),
            _u("b", 2.0),
            _u("c", 2.0),
            _u("d", 2.0),
            _u("e", 100.0),  # outlier
        ]
        result = await mk.aggregate(updates)
        # Selected: a, b, c, d → avg ≈ 1.5
        assert len(result.contributing_peers) > 1

    @pytest.mark.asyncio
    async def test_dict_weights(self):
        mk = MultiKrum(num_byzantines=1)
        updates = [
            ModelUpdate(f"p{i}", {"layer": np.array([float(i)])})
            for i in range(5)
        ]
        result = await mk.aggregate(updates)
        assert "layer" in result.weights

    @pytest.mark.asyncio
    async def test_nan_raises(self):
        mk = MultiKrum(num_byzantines=1)
        updates = [ModelUpdate(f"p{i}", np.array([float("nan")])) for i in range(5)]
        with pytest.raises(ValueError, match="NaN"):
            await mk.aggregate(updates)
