"""
Section C — Aggregation data-flow integrity.

Covers:
  C1. Every strategy consumes ModelUpdate and returns AggregatedModel (no struct bypass)
  C2. No strategy mutates incoming ModelUpdate objects in place
  C3. Determinism: same inputs → same output (flags hidden RNG)
"""

import copy
import pytest
import numpy as np

from quinkgl.aggregation.base import ModelUpdate, AggregatedModel
from quinkgl.aggregation.fedavg import FedAvg
from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.scaffold import Scaffold
from quinkgl.aggregation.krum import Krum, MultiKrum
from quinkgl.aggregation.trimmed_mean import TrimmedMean


def _updates(n=5, base=1.0):
    return [
        ModelUpdate(
            peer_id=f"p{i}",
            weights=np.array([base + i * 0.1], dtype=np.float32),
            sample_count=i + 1,
        )
        for i in range(n)
    ]


def _updates_with_label(n=5):
    updates = _updates(n)
    for i, u in enumerate(updates):
        u.metadata["label_distribution"] = {"A": i + 1, "B": n - i}
    return updates


# ---------------------------------------------------------------------------
# C1 — Return type is always AggregatedModel
# ---------------------------------------------------------------------------

class TestReturnType:
    @pytest.mark.asyncio
    async def test_fedavg_returns_aggregated_model(self):
        result = await FedAvg().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_fedavgm_returns_aggregated_model(self):
        result = await FedAvgM().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_fedprox_returns_aggregated_model(self):
        result = await FedProx().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_entropy_weighted_returns_aggregated_model(self):
        result = await EntropyWeightedAvg().aggregate(_updates_with_label())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_staleness_returns_aggregated_model(self):
        result = await StalenessWeightedFedAvg().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_scaffold_returns_aggregated_model(self):
        result = await Scaffold().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_krum_returns_aggregated_model(self):
        result = await Krum(num_byzantines=1).aggregate(_updates(n=5))
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_multikrum_returns_aggregated_model(self):
        result = await MultiKrum(num_byzantines=1).aggregate(_updates(n=5))
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_trimmed_mean_returns_aggregated_model(self):
        result = await TrimmedMean().aggregate(_updates())
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", [
        FedAvg, FedAvgM, FedProx, Scaffold, TrimmedMean, StalenessWeightedFedAvg
    ])
    async def test_result_has_required_fields(self, StratCls):
        result = await StratCls().aggregate(_updates())
        assert isinstance(result.weights, np.ndarray)
        assert isinstance(result.contributing_peers, list)
        assert len(result.contributing_peers) > 0
        assert isinstance(result.metadata, dict)
        assert "aggregation_method" in result.metadata


# ---------------------------------------------------------------------------
# C2 — No strategy mutates incoming ModelUpdate objects
# ---------------------------------------------------------------------------

class TestNoMutation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", [FedAvg, FedAvgM, FedProx, EntropyWeightedAvg,
                                           StalenessWeightedFedAvg, Scaffold, TrimmedMean])
    async def test_input_weights_not_mutated(self, StratCls):
        updates = _updates_with_label()
        originals = [u.weights.copy() for u in updates]
        await StratCls().aggregate(updates)
        for u, original in zip(updates, originals):
            assert np.array_equal(u.weights, original), (
                f"{StratCls.__name__} mutated weights of peer {u.peer_id}"
            )

    @pytest.mark.asyncio
    async def test_krum_does_not_mutate_input(self):
        updates = _updates(n=5)
        originals = [u.weights.copy() for u in updates]
        await Krum(num_byzantines=1).aggregate(updates)
        for u, original in zip(updates, originals):
            assert np.array_equal(u.weights, original)

    @pytest.mark.asyncio
    async def test_multikrum_does_not_mutate_input(self):
        updates = _updates(n=5)
        originals = [u.weights.copy() for u in updates]
        await MultiKrum(num_byzantines=1).aggregate(updates)
        for u, original in zip(updates, originals):
            assert np.array_equal(u.weights, original)

    @pytest.mark.asyncio
    async def test_fedprox_weight_interpolation_does_not_mutate(self):
        fp = FedProx(mu=0.1, mode="weight_interpolation")
        updates = _updates()
        await fp.aggregate(updates)  # first round: stores global_weights
        originals = [u.weights.copy() for u in updates]
        await fp.aggregate(updates)  # second round: applies correction
        for u, original in zip(updates, originals):
            assert np.array_equal(u.weights, original)

    @pytest.mark.asyncio
    async def test_metadata_not_mutated(self):
        strat = FedAvg()
        updates = _updates()
        original_meta = [dict(u.metadata) for u in updates]
        await strat.aggregate(updates)
        for u, original in zip(updates, original_meta):
            assert u.metadata == original


# ---------------------------------------------------------------------------
# C3 — Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.asyncio
    async def test_fedavg_deterministic(self):
        updates = _updates()
        r1 = await FedAvg().aggregate(copy.deepcopy(updates))
        r2 = await FedAvg().aggregate(copy.deepcopy(updates))
        assert np.allclose(r1.weights, r2.weights)

    @pytest.mark.asyncio
    async def test_trimmed_mean_deterministic(self):
        updates = _updates()
        r1 = await TrimmedMean().aggregate(copy.deepcopy(updates))
        r2 = await TrimmedMean().aggregate(copy.deepcopy(updates))
        assert np.allclose(r1.weights, r2.weights)

    @pytest.mark.asyncio
    async def test_krum_deterministic(self):
        updates = _updates(n=5)
        r1 = await Krum(num_byzantines=1).aggregate(copy.deepcopy(updates))
        r2 = await Krum(num_byzantines=1).aggregate(copy.deepcopy(updates))
        assert r1.metadata["selected_peer"] == r2.metadata["selected_peer"]

    @pytest.mark.asyncio
    async def test_entropy_weighted_deterministic(self):
        updates = _updates_with_label()
        r1 = await EntropyWeightedAvg().aggregate(copy.deepcopy(updates))
        r2 = await EntropyWeightedAvg().aggregate(copy.deepcopy(updates))
        assert np.allclose(r1.weights, r2.weights)

    @pytest.mark.asyncio
    async def test_scaffold_deterministic_given_same_state(self):
        """Two fresh Scaffold instances with same inputs produce same output."""
        updates = _updates()
        r1 = await Scaffold().aggregate(copy.deepcopy(updates))
        r2 = await Scaffold().aggregate(copy.deepcopy(updates))
        assert np.allclose(r1.weights, r2.weights)

    @pytest.mark.asyncio
    async def test_fedavgm_deterministic_given_same_state(self):
        updates = _updates()
        agg1, agg2 = FedAvgM(server_momentum=0.9), FedAvgM(server_momentum=0.9)
        r1 = await agg1.aggregate(copy.deepcopy(updates))
        r2 = await agg2.aggregate(copy.deepcopy(updates))
        assert np.allclose(r1.weights, r2.weights)
