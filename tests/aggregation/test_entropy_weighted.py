"""Tests for the RNEP-inspired EntropyWeightedAvg aggregation strategy."""

import math
import numpy as np
import pytest

from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg, _shannon_entropy
from quinkgl.aggregation.base import ModelUpdate


# ------------------------------------------------------------------ #
# Shannon entropy helper
# ------------------------------------------------------------------ #

class TestShannonEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution over 10 classes → max entropy."""
        dist = {str(i): 100 for i in range(10)}
        expected = math.log(10)  # ln(10)
        assert math.isclose(_shannon_entropy(dist), expected, rel_tol=1e-6)

    def test_single_class(self):
        """All data in one class → entropy = 0."""
        dist = {"a": 1000}
        assert _shannon_entropy(dist) == 0.0

    def test_skewed_distribution(self):
        """Skewed distribution → low entropy."""
        dist = {"a": 900, "b": 50, "c": 50}
        uniform_entropy = math.log(3)
        assert 0 < _shannon_entropy(dist) < uniform_entropy

    def test_empty_distribution(self):
        assert _shannon_entropy({}) == 0.0

    def test_zero_values(self):
        dist = {"a": 0, "b": 0}
        assert _shannon_entropy(dist) == 0.0


# ------------------------------------------------------------------ #
# EntropyWeightedAvg — numpy arrays
# ------------------------------------------------------------------ #

class TestEntropyWeightedAvgNumpy:
    @pytest.mark.asyncio
    async def test_balanced_vs_skewed(self):
        """Balanced peer should dominate the aggregated model."""
        agg = EntropyWeightedAvg()

        balanced_dist = {str(i): 100 for i in range(10)}  # H ≈ 2.30
        skewed_dist = {"0": 900, "1": 50, "2": 50}        # H ≈ 0.47

        updates = [
            ModelUpdate(
                "balanced", np.array([10.0]),
                metadata={"label_distribution": balanced_dist},
            ),
            ModelUpdate(
                "skewed", np.array([0.0]),
                metadata={"label_distribution": skewed_dist},
            ),
        ]

        result = await agg.aggregate(updates)
        # Balanced peer has ~83% weight → result should be closer to 10.0
        assert result.weights[0] > 7.0
        assert "balanced" in result.contributing_peers
        assert "skewed" in result.contributing_peers

    @pytest.mark.asyncio
    async def test_two_balanced_peers(self):
        """Two equally balanced peers → ~50/50 average."""
        agg = EntropyWeightedAvg()

        dist = {str(i): 100 for i in range(10)}
        updates = [
            ModelUpdate("a", np.array([0.0]), metadata={"label_distribution": dist}),
            ModelUpdate("b", np.array([10.0]), metadata={"label_distribution": dist}),
        ]

        result = await agg.aggregate(updates)
        assert np.allclose(result.weights, np.array([5.0]), atol=0.01)

    @pytest.mark.asyncio
    async def test_fallback_when_no_distribution(self):
        """Peers without label_distribution get fallback_weight."""
        agg = EntropyWeightedAvg(fallback_weight=1.0)

        updates = [
            ModelUpdate("a", np.array([2.0]), metadata={}),
            ModelUpdate("b", np.array([8.0]), metadata={}),
        ]

        result = await agg.aggregate(updates)
        # Both have equal fallback → simple average
        assert np.allclose(result.weights, np.array([5.0]), atol=0.01)

    @pytest.mark.asyncio
    async def test_entropy_floor(self):
        """Single-class peer should get entropy_floor, not zero."""
        agg = EntropyWeightedAvg(entropy_floor=0.1)

        updates = [
            ModelUpdate(
                "diverse", np.array([10.0]),
                metadata={"label_distribution": {str(i): 100 for i in range(10)}},
            ),
            ModelUpdate(
                "single", np.array([0.0]),
                metadata={"label_distribution": {"0": 1000}},
            ),
        ]

        result = await agg.aggregate(updates)
        # Single-class peer gets floor 0.1; diverse gets ~2.30
        # diverse weight ≈ 2.30 / 2.40 ≈ 0.958 → result close to 10.0
        assert result.weights[0] > 9.0

    @pytest.mark.asyncio
    async def test_metadata_records_weights(self):
        """Aggregated metadata should contain per-peer entropy weights."""
        agg = EntropyWeightedAvg()
        dist = {str(i): 100 for i in range(10)}

        updates = [
            ModelUpdate("a", np.array([1.0]), metadata={"label_distribution": dist}),
            ModelUpdate("b", np.array([2.0]), metadata={"label_distribution": dist}),
        ]

        result = await agg.aggregate(updates)
        assert result.metadata["aggregation_method"] == "entropy_weighted_avg"
        ew = result.metadata["entropy_weights"]
        assert "a" in ew and "b" in ew
        assert math.isclose(ew["a"] + ew["b"], 1.0, rel_tol=1e-4)


# ------------------------------------------------------------------ #
# EntropyWeightedAvg — dict weights (PyTorch state_dict style)
# ------------------------------------------------------------------ #

class TestEntropyWeightedAvgDict:
    @pytest.mark.asyncio
    async def test_dict_weights(self):
        """Dict-style weights (like PyTorch state_dict) should work."""
        agg = EntropyWeightedAvg()

        balanced = {str(i): 100 for i in range(10)}
        skewed = {"0": 900, "1": 50, "2": 50}

        updates = [
            ModelUpdate(
                "a",
                {"fc1.weight": np.array([10.0, 20.0]), "fc1.bias": np.array([1.0])},
                metadata={"label_distribution": balanced},
            ),
            ModelUpdate(
                "b",
                {"fc1.weight": np.array([0.0, 0.0]), "fc1.bias": np.array([0.0])},
                metadata={"label_distribution": skewed},
            ),
        ]

        result = await agg.aggregate(updates)
        # Balanced peer dominates
        assert result.weights["fc1.weight"][0] > 7.0
        assert result.weights["fc1.weight"][1] > 14.0
