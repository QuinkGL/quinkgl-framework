import numpy as np
import pytest

from quinkgl.aggregation import TrimmedMean
from quinkgl.aggregation.base import ModelUpdate


@pytest.mark.asyncio
async def test_trimmed_mean_trims_both_tails_and_ignores_outlier():
    agg = TrimmedMean(trim_ratio=0.2)
    updates = [
        ModelUpdate("a", np.array([1.0])),
        ModelUpdate("b", np.array([1.1])),
        ModelUpdate("c", np.array([1.2])),
        ModelUpdate("d", np.array([50.0])),
        ModelUpdate("e", np.array([0.9])),
    ]

    result = await agg.aggregate(updates)

    assert result.metadata["aggregation_method"] == "trimmed_mean"
    assert result.metadata["trim_ratio"] == 0.2
    assert np.allclose(result.weights, np.array([1.1]))


@pytest.mark.asyncio
async def test_trimmed_mean_requires_at_least_three_updates():
    agg = TrimmedMean(trim_ratio=0.2)
    updates = [
        ModelUpdate("a", np.array([1.0])),
        ModelUpdate("b", np.array([2.0])),
    ]

    with pytest.raises(ValueError, match="requires at least 3 updates"):
        await agg.aggregate(updates)


@pytest.mark.asyncio
async def test_trimmed_mean_rejects_positive_trim_ratio_when_effective_trim_count_is_zero():
    agg = TrimmedMean(trim_ratio=0.1)
    updates = [
        ModelUpdate("a", np.array([1.0])),
        ModelUpdate("b", np.array([1.1])),
        ModelUpdate("c", np.array([1.2])),
        ModelUpdate("d", np.array([1.3])),
        ModelUpdate("e", np.array([50.0])),
    ]

    with pytest.raises(ValueError, match="no values would be trimmed"):
        await agg.aggregate(updates)
