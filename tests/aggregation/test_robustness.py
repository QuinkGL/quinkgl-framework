"""
Section B — Aggregation robustness edge cases.

Covers every [ ] checklist item:
  - empty peer-update list
  - single peer update
  - all-zero tensor values  (numerical stability)
  - all-zero sample counts  (total_weight = 0)
  - NaN / Inf in weights
  - shape mismatch (numpy)
  - dict per-value shape mismatch
  - dtype mismatch
  - staleness field missing / negative
  - Byzantine tolerance bounds (Krum, MultiKrum, TrimmedMean)
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _u(pid, value, sample_count=1, round_number=0, dtype=np.float32):
    return ModelUpdate(
        peer_id=pid,
        weights=np.array([value], dtype=dtype),
        sample_count=sample_count,
        round_number=round_number,
    )


def _make_enough(n=6, value=1.0):
    """Return n updates suitable for Krum (needs n >= 2f+3 with f=1 → n>=5)."""
    return [_u(f"p{i}", value + i * 0.01) for i in range(n)]


# Strategies that work with a single update
SINGLE_OK = [FedAvg, FedAvgM, FedProx, EntropyWeightedAvg, Scaffold, TrimmedMean]
# Strategies that need many updates (Byzantine)
BYZANTINE = [Krum, MultiKrum]
# All non-Byzantine strategies
STANDARD = SINGLE_OK + [StalenessWeightedFedAvg]


# ---------------------------------------------------------------------------
# B1 — Empty peer-update list
# ---------------------------------------------------------------------------

class TestEmptyUpdateList:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", STANDARD)
    async def test_empty_raises_for_standard(self, StratCls):
        strat = StratCls()
        with pytest.raises(ValueError, match="[Ee]mpty|empty"):
            await strat.aggregate([])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", BYZANTINE)
    async def test_empty_raises_for_byzantine(self, StratCls):
        strat = StratCls(num_byzantines=1)
        with pytest.raises(ValueError):
            await strat.aggregate([])


# ---------------------------------------------------------------------------
# B2 — Single peer update
# ---------------------------------------------------------------------------

class TestSingleUpdate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK)
    async def test_single_update_returns_aggregated_model(self, StratCls):
        strat = StratCls()
        result = await strat.aggregate([_u("solo", 5.0)])
        assert isinstance(result, AggregatedModel)
        assert np.allclose(result.weights, [5.0], atol=0.1)

    @pytest.mark.asyncio
    async def test_single_update_staleness(self):
        strat = StalenessWeightedFedAvg()
        result = await strat.aggregate([_u("solo", 3.0)])
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_single_update_krum_raises(self):
        """Krum requires n >= 2f+3; single update always fails."""
        krum = Krum(num_byzantines=1)
        with pytest.raises(ValueError, match="2f"):
            await krum.aggregate([_u("solo", 1.0)])

    @pytest.mark.asyncio
    async def test_single_update_multikrum_raises(self):
        mk = MultiKrum(num_byzantines=1)
        with pytest.raises(ValueError, match="2f"):
            await mk.aggregate([_u("solo", 1.0)])

    @pytest.mark.asyncio
    async def test_trimmed_mean_fewer_than_3_falls_back(self):
        """TrimmedMean falls back to FedAvg when n < 3."""
        strat = TrimmedMean(trim_ratio=0.2)
        result = await strat.aggregate([_u("a", 2.0), _u("b", 4.0)])
        assert result.metadata["aggregation_method"] == "trimmed_mean_fallback"
        assert np.allclose(result.weights, [3.0], atol=1e-4)


# ---------------------------------------------------------------------------
# B3 — All-zero tensor values (numerical stability)
# ---------------------------------------------------------------------------

class TestAllZeroTensors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK)
    async def test_zero_tensors_produce_zero(self, StratCls):
        strat = StratCls()
        updates = [_u(f"p{i}", 0.0, sample_count=i + 1) for i in range(3)]
        result = await strat.aggregate(updates)
        assert np.allclose(result.weights, [0.0], atol=1e-7)

    @pytest.mark.asyncio
    async def test_zero_tensors_krum(self):
        krum = Krum(num_byzantines=1)
        updates = [_u(f"p{i}", 0.0) for i in range(5)]
        result = await krum.aggregate(updates)
        assert np.allclose(result.weights, [0.0], atol=1e-7)


# ---------------------------------------------------------------------------
# B4 — All-zero sample counts (total_weight = 0)
# ---------------------------------------------------------------------------

class TestZeroSampleCount:
    @pytest.mark.asyncio
    async def test_fedavg_data_size_zero_sample_count_raises(self):
        strat = FedAvg(weight_by="data_size")
        updates = [_u(f"p{i}", 1.0, sample_count=0) for i in range(3)]
        with pytest.raises(ValueError, match="[Tt]otal weight"):
            await strat.aggregate(updates)

    @pytest.mark.asyncio
    async def test_fedavg_uniform_zero_sample_count_ok(self):
        """Uniform mode ignores sample_count — should succeed."""
        strat = FedAvg(weight_by="uniform")
        updates = [_u(f"p{i}", float(i), sample_count=0) for i in range(3)]
        result = await strat.aggregate(updates)
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_fedavgm_zero_sample_count_raises(self):
        strat = FedAvgM()
        updates = [_u(f"p{i}", 1.0, sample_count=0) for i in range(3)]
        with pytest.raises(ValueError, match="[Tt]otal weight"):
            await strat.aggregate(updates)

    @pytest.mark.asyncio
    async def test_scaffold_zero_sample_count_falls_back_to_equal_weight(self):
        """Scaffold uses 1/n when total_samples=0."""
        strat = Scaffold()
        updates = [_u(f"p{i}", float(i), sample_count=0) for i in range(3)]
        result = await strat.aggregate(updates)
        assert isinstance(result, AggregatedModel)


# ---------------------------------------------------------------------------
# B5 — NaN / Inf in weights
# ---------------------------------------------------------------------------

class TestNaNInfWeights:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK + [StalenessWeightedFedAvg])
    async def test_nan_raises(self, StratCls):
        strat = StratCls()
        updates = [
            ModelUpdate("clean", np.array([1.0])),
            ModelUpdate("bad",   np.array([float("nan")])),
        ]
        with pytest.raises(ValueError, match="NaN"):
            await strat.aggregate(updates)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK + [StalenessWeightedFedAvg])
    async def test_inf_raises(self, StratCls):
        strat = StratCls()
        updates = [
            ModelUpdate("clean", np.array([1.0])),
            ModelUpdate("bad",   np.array([float("inf")])),
        ]
        with pytest.raises(ValueError, match="Inf"):
            await strat.aggregate(updates)

    @pytest.mark.asyncio
    async def test_nan_krum_raises(self):
        krum = Krum(num_byzantines=1)
        updates = [ModelUpdate(f"p{i}", np.array([float("nan")])) for i in range(5)]
        with pytest.raises(ValueError, match="NaN"):
            await krum.aggregate(updates)


# ---------------------------------------------------------------------------
# B6 — Weight-tensor shape mismatch (numpy)
# ---------------------------------------------------------------------------

class TestNumpyShapeMismatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK + [StalenessWeightedFedAvg])
    async def test_numpy_shape_mismatch_raises(self, StratCls):
        strat = StratCls()
        updates = [
            ModelUpdate("a", np.zeros((4,))),
            ModelUpdate("b", np.zeros((6,))),  # different shape
        ]
        with pytest.raises(ValueError):
            await strat.aggregate(updates)

    @pytest.mark.asyncio
    async def test_dict_per_value_shape_mismatch_raises(self):
        """After fix (M-2): dict with same keys but different per-value shapes must raise."""
        strat = FedAvg()
        updates = [
            ModelUpdate("a", {"w": np.zeros((10, 5))}),
            ModelUpdate("b", {"w": np.zeros((8, 4))}),  # same key, different shape
        ]
        with pytest.raises(ValueError):
            await strat.aggregate(updates)


# ---------------------------------------------------------------------------
# B7 — Dtype mismatch (should be handled gracefully, not raise)
# ---------------------------------------------------------------------------

class TestDtypeMismatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("StratCls", SINGLE_OK + [StalenessWeightedFedAvg])
    async def test_dtype_mismatch_handled_gracefully(self, StratCls):
        """float32 + float64 peers should aggregate without error."""
        strat = StratCls()
        updates = [
            ModelUpdate("a", np.array([1.0], dtype=np.float32), sample_count=1),
            ModelUpdate("b", np.array([3.0], dtype=np.float64), sample_count=1),
        ]
        result = await strat.aggregate(updates)
        assert isinstance(result, AggregatedModel)
        assert not np.isnan(result.weights).any()

    @pytest.mark.asyncio
    async def test_int_dtype_handled(self):
        """Integer dtype should be cast internally without error."""
        strat = FedAvg()
        updates = [
            ModelUpdate("a", np.array([2], dtype=np.int32), sample_count=1),
            ModelUpdate("b", np.array([4], dtype=np.int32), sample_count=1),
        ]
        result = await strat.aggregate(updates)
        assert np.allclose(result.weights, [3], atol=1)


# ---------------------------------------------------------------------------
# B8 — Staleness field missing or negative
# ---------------------------------------------------------------------------

class TestStalenessEdgeCases:
    def test_round_number_defaults_to_zero(self):
        """round_number has a default of 0 — 'missing' is impossible."""
        u = ModelUpdate("p", np.array([1.0]))
        assert u.round_number == 0

    def test_negative_staleness_clamped_to_zero(self):
        """update.round_number > current_round → staleness clamped to 0."""
        sw = StalenessWeightedFedAvg(staleness_coefficient=0.5)
        update = ModelUpdate("p", np.array([1.0]), round_number=10)
        weight = sw.compute_staleness_weight(update, current_round=5)
        # staleness = max(0, 5-10) = 0 → factor = 1.0
        assert weight == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_update_with_default_round_aggregates_ok(self):
        sw = StalenessWeightedFedAvg()
        updates = [
            ModelUpdate("a", np.array([1.0])),  # round_number=0 by default
            ModelUpdate("b", np.array([3.0])),
        ]
        result = await sw.aggregate(updates)
        assert isinstance(result, AggregatedModel)


# ---------------------------------------------------------------------------
# B9 — Byzantine tolerance bounds
# ---------------------------------------------------------------------------

class TestByzantineBounds:
    # Krum / MultiKrum — H-1 fix
    @pytest.mark.asyncio
    async def test_krum_n_equals_2f_plus_2_raises(self):
        krum = Krum(num_byzantines=1)
        # n=4 < 2*1+3=5 → must raise
        with pytest.raises(ValueError, match="2f"):
            await krum.aggregate([_u(f"p{i}", float(i)) for i in range(4)])

    @pytest.mark.asyncio
    async def test_krum_n_equals_2f_plus_3_ok(self):
        krum = Krum(num_byzantines=1)
        updates = [_u(f"p{i}", float(i)) for i in range(5)]
        result = await krum.aggregate(updates)
        assert isinstance(result, AggregatedModel)

    @pytest.mark.asyncio
    async def test_multikrum_n_equals_2f_plus_2_raises(self):
        mk = MultiKrum(num_byzantines=1)
        with pytest.raises(ValueError, match="2f"):
            await mk.aggregate([_u(f"p{i}", float(i)) for i in range(4)])

    # TrimmedMean — trim_ratio bounds
    def test_trimmed_mean_ratio_zero_allowed(self):
        strat = TrimmedMean(trim_ratio=0.0)
        assert strat.trim_ratio == 0.0

    def test_trimmed_mean_ratio_half_raises(self):
        with pytest.raises(ValueError):
            TrimmedMean(trim_ratio=0.5)

    def test_trimmed_mean_negative_ratio_raises(self):
        with pytest.raises(ValueError):
            TrimmedMean(trim_ratio=-0.1)

    @pytest.mark.asyncio
    async def test_trimmed_mean_ratio_just_below_half_ok(self):
        strat = TrimmedMean(trim_ratio=0.49)
        updates = [_u(f"p{i}", float(i)) for i in range(6)]
        result = await strat.aggregate(updates)
        assert isinstance(result, AggregatedModel)
