"""AGG-TASK-17: Test persistence invariants for all aggregation strategies.

Verifies that state_dict() → load_state_dict() round-trip preserves
all mutable state and that a restored strategy produces identical
aggregation results.
"""

import numpy as np
import pytest

from quinkgl.aggregation.base import AggregationStrategy, ModelUpdate, AggregatedModel
from quinkgl.aggregation.fedavg import FedAvg
from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.scaffold import Scaffold
from quinkgl.aggregation.krum import Krum, MultiKrum
from quinkgl.aggregation.trimmed_mean import TrimmedMean
from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg


def _make_updates(n=3, weight_type="dict"):
    updates = []
    for i in range(n):
        if weight_type == "dict":
            weights = {"w1": np.array([float(i + 1), float(i + 2)]), "w2": np.array([float(i + 3)])}
        else:
            weights = np.array([float(i + 1), float(i + 2), float(i + 3)])
        updates.append(ModelUpdate(
            peer_id=f"peer_{i}",
            weights=weights,
            sample_count=100 * (i + 1),
            metadata={"loss": 0.5 - i * 0.1},
        ))
    return updates


def _round_trip(strategy):
    """Save and restore strategy state."""
    state = strategy.state_dict()
    clone = strategy.__class__(**strategy.config)
    # For FedAvgM/FedProx, also pass constructor args
    if isinstance(strategy, FedAvgM):
        clone = FedAvgM(server_momentum=strategy.server_momentum)
    elif isinstance(strategy, FedProx):
        clone = FedProx(mu=strategy.mu, mode=strategy.mode)
    elif isinstance(strategy, Scaffold):
        clone = Scaffold(
            learning_rate=strategy.learning_rate,
            global_learning_rate=strategy.global_learning_rate,
            control_momentum=strategy.control_momentum,
        )
    elif isinstance(strategy, Krum):
        clone = Krum(num_byzantines=strategy.num_byzantines)
    elif isinstance(strategy, MultiKrum):
        clone = MultiKrum(num_byzantines=strategy.num_byzantines)
    elif isinstance(strategy, TrimmedMean):
        clone = TrimmedMean(trim_ratio=strategy.trim_ratio)
    elif isinstance(strategy, EntropyWeightedAvg):
        clone = EntropyWeightedAvg(
            fallback_weight=strategy.fallback_weight,
            entropy_floor=strategy.entropy_floor,
            normalize=strategy.normalize,
        )
    clone.load_state_dict(state)
    return state, clone


# -----------------------------------------------------------------------
# FedAvg
# -----------------------------------------------------------------------
class TestFedAvgPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = FedAvg(weight_by="data_size")
        updates = _make_updates(3)
        await s.aggregate(updates)
        state, clone = _round_trip(s)
        assert clone.weight_by == s.weight_by

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = FedAvg(weight_by="data_size")
        updates = _make_updates(3)
        r1 = await s.aggregate(updates)
        _, clone = _round_trip(s)
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_equal(r1.weights["w1"], r2.weights["w1"])


# -----------------------------------------------------------------------
# FedAvgM
# -----------------------------------------------------------------------
class TestFedAvgMPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = FedAvgM(server_momentum=0.9)
        updates = _make_updates(3)
        await s.aggregate(updates)
        state, clone = _round_trip(s)
        assert clone.server_momentum == pytest.approx(0.9)
        assert clone.global_weights is not None
        assert clone.momentum_buffer is not None
        np.testing.assert_array_almost_equal(
            clone.global_weights["w1"], s.global_weights["w1"]
        )

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = FedAvgM(server_momentum=0.9)
        updates = _make_updates(3)
        await s.aggregate(updates)  # round 1: plain avg
        r1 = await s.aggregate(updates)  # round 2: with momentum
        _, clone = _round_trip(s)
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_almost_equal(r1.weights["w1"], r2.weights["w1"])

    @pytest.mark.asyncio
    async def test_momentum_buffer_preserved(self):
        s = FedAvgM(server_momentum=0.8)
        updates = _make_updates(3)
        await s.aggregate(updates)
        state = s.state_dict()
        assert "momentum_buffer" in state
        s2 = FedAvgM(server_momentum=0.8)
        s2.load_state_dict(state)
        np.testing.assert_array_almost_equal(
            s2.momentum_buffer["w1"], s.momentum_buffer["w1"]
        )


# -----------------------------------------------------------------------
# FedProx
# -----------------------------------------------------------------------
class TestFedProxPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = FedProx(mu=0.05, mode="training_time")
        updates = _make_updates(3)
        await s.aggregate(updates)
        state, clone = _round_trip(s)
        assert clone.mu == pytest.approx(0.05)
        assert clone.mode == "training_time"
        assert clone.global_weights is not None

    @pytest.mark.asyncio
    async def test_global_weights_preserved(self):
        s = FedProx(mu=0.05)
        updates = _make_updates(3)
        await s.aggregate(updates)
        state = s.state_dict()
        assert "global_weights" in state
        s2 = FedProx(mu=0.05)
        s2.load_state_dict(state)
        np.testing.assert_array_almost_equal(
            s2.global_weights["w1"], s.global_weights["w1"]
        )

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = FedProx(mu=0.05)
        updates = _make_updates(3)
        await s.aggregate(updates)
        r1 = await s.aggregate(updates)
        _, clone = _round_trip(s)
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_almost_equal(r1.weights["w1"], r2.weights["w1"])


# -----------------------------------------------------------------------
# Scaffold
# -----------------------------------------------------------------------
class TestScaffoldPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = Scaffold(learning_rate=0.01, control_momentum=0.0)
        updates = _make_updates(3)
        # Add control variates
        for u in updates:
            u.metadata["control_variate"] = {
                "w1": np.random.randn(2) * 0.01,
                "w2": np.random.randn(1) * 0.01,
            }
        await s.aggregate(updates)
        state, clone = _round_trip(s)
        assert clone._round == s._round
        assert clone._c_global is not None
        for key in s._c_global:
            np.testing.assert_array_almost_equal(clone._c_global[key], s._c_global[key])

    @pytest.mark.asyncio
    async def test_round_counter_preserved(self):
        s = Scaffold()
        updates = _make_updates(3)
        for u in updates:
            u.metadata["control_variate"] = {
                "w1": np.zeros(2), "w2": np.zeros(1),
            }
        await s.aggregate(updates)
        assert s._round == 1
        await s.aggregate(updates)
        assert s._round == 2
        state = s.state_dict()
        s2 = Scaffold()
        s2.load_state_dict(state)
        assert s2._round == 2

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = Scaffold(learning_rate=0.01)
        updates = _make_updates(3)
        for u in updates:
            u.metadata["control_variate"] = {
                "w1": np.zeros(2), "w2": np.zeros(1),
            }
        await s.aggregate(updates)
        r1 = await s.aggregate(updates)
        _, clone = _round_trip(s)
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_almost_equal(r1.weights["w1"], r2.weights["w1"])


# -----------------------------------------------------------------------
# Krum / MultiKrum
# -----------------------------------------------------------------------
class TestKrumPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = Krum(num_byzantines=1)
        state, clone = _round_trip(s)
        assert clone.num_byzantines == 1

    @pytest.mark.asyncio
    async def test_multikrum_round_trip(self):
        s = MultiKrum(num_byzantines=1)
        state, clone = _round_trip(s)
        assert clone.num_byzantines == 1


# -----------------------------------------------------------------------
# TrimmedMean
# -----------------------------------------------------------------------
class TestTrimmedMeanPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = TrimmedMean(trim_ratio=0.2)
        state, clone = _round_trip(s)
        assert clone.trim_ratio == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = TrimmedMean(trim_ratio=0.1)
        updates = _make_updates(5)
        r1 = await s.aggregate(updates)
        _, clone = _round_trip(s)
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_almost_equal(r1.weights["w1"], r2.weights["w1"])


# -----------------------------------------------------------------------
# EntropyWeightedAvg
# -----------------------------------------------------------------------
class TestEntropyWeightedAvgPersistence:
    @pytest.mark.asyncio
    async def test_state_dict_round_trip(self):
        s = EntropyWeightedAvg(fallback_weight=2.0, entropy_floor=0.05, normalize=False)
        state, clone = _round_trip(s)
        assert clone.fallback_weight == pytest.approx(2.0)
        assert clone.entropy_floor == pytest.approx(0.05)
        assert clone.normalize is False

    @pytest.mark.asyncio
    async def test_restored_produces_same_result(self):
        s = EntropyWeightedAvg()
        updates = _make_updates(3)
        for u in updates:
            u.metadata["label_distribution"] = {"0": 50, "1": 50}
        r1 = await s.aggregate(updates)
        _, clone = _round_trip(s)
        for u in updates:
            u.metadata["label_distribution"] = {"0": 50, "1": 50}
        r2 = await clone.aggregate(updates)
        np.testing.assert_array_almost_equal(r1.weights["w1"], r2.weights["w1"])


# -----------------------------------------------------------------------
# Cross-strategy: numpy array weights
# -----------------------------------------------------------------------
class TestNumpyArrayPersistence:
    @pytest.mark.asyncio
    async def test_fedavgm_numpy_round_trip(self):
        s = FedAvgM(server_momentum=0.9)
        updates = _make_updates(3, weight_type="numpy")
        await s.aggregate(updates)
        state = s.state_dict()
        assert "global_weights" in state
        s2 = FedAvgM(server_momentum=0.9)
        s2.load_state_dict(state)
        np.testing.assert_array_almost_equal(s2.global_weights, s.global_weights)
        np.testing.assert_array_almost_equal(s2.momentum_buffer, s.momentum_buffer)

    @pytest.mark.asyncio
    async def test_fedprox_numpy_round_trip(self):
        s = FedProx(mu=0.05)
        updates = _make_updates(3, weight_type="numpy")
        await s.aggregate(updates)
        state = s.state_dict()
        s2 = FedProx(mu=0.05)
        s2.load_state_dict(state)
        np.testing.assert_array_almost_equal(s2.global_weights, s.global_weights)
