import numpy as np
import pytest

from quinkgl.aggregation import Krum, MultiKrum
from quinkgl.aggregation.base import ModelUpdate


def test_krum_rejects_invalid_num_byzantines_values():
    with pytest.raises(ValueError, match="non-negative integer"):
        Krum(num_byzantines=-1)

    with pytest.raises(ValueError, match="non-negative integer"):
        Krum(num_byzantines=1.5)

    with pytest.raises(ValueError, match="non-negative integer"):
        MultiKrum(num_byzantines=True)


@pytest.mark.asyncio
async def test_krum_selects_update_near_majority():
    agg = Krum(num_byzantines=1)
    updates = [
        ModelUpdate("a", np.array([0.0])),
        ModelUpdate("b", np.array([0.2])),
        ModelUpdate("c", np.array([0.6])),
        ModelUpdate("d", np.array([10.0])),
        ModelUpdate("e", np.array([11.0])),
        ModelUpdate("f", np.array([12.0])),
    ]

    result = await agg.aggregate(updates)

    assert result.metadata["selected_peer"] == "b"
    assert np.allclose(result.weights, np.array([0.2]))


@pytest.mark.asyncio
async def test_krum_with_two_byzantines_uses_n_minus_f_minus_2_neighbors():
    agg = Krum(num_byzantines=2)
    updates = [
        ModelUpdate("a", np.array([0.0])),
        ModelUpdate("b", np.array([0.1])),
        ModelUpdate("c", np.array([0.2])),
        ModelUpdate("d", np.array([0.3])),
        ModelUpdate("e", np.array([0.4])),
        ModelUpdate("x", np.array([50.0])),
        ModelUpdate("y", np.array([60.0])),
    ]

    result = await agg.aggregate(updates)

    assert result.metadata["selected_peer"] == "b"
    assert np.allclose(result.weights, np.array([0.1]))


@pytest.mark.asyncio
async def test_krum_requires_n_at_least_two_f_plus_three():
    agg = Krum(num_byzantines=2)
    updates = [
        ModelUpdate("a", np.array([0.0])),
        ModelUpdate("b", np.array([0.1])),
        ModelUpdate("c", np.array([0.2])),
        ModelUpdate("d", np.array([0.3])),
        ModelUpdate("e", np.array([0.4])),
        ModelUpdate("f", np.array([0.5])),
    ]

    with pytest.raises(ValueError, match=r"n >= 2\*f \+ 3"):
        await agg.aggregate(updates)


@pytest.mark.asyncio
async def test_multikrum_averages_selected_updates_uniformly():
    agg = MultiKrum(num_byzantines=1)
    updates = [
        ModelUpdate("a", np.array([0.0]), sample_count=1),
        ModelUpdate("b", np.array([0.2]), sample_count=1),
        ModelUpdate("c", np.array([0.4]), sample_count=1000),
        ModelUpdate("x", np.array([10.0]), sample_count=1),
        ModelUpdate("y", np.array([11.0]), sample_count=1),
    ]

    result = await agg.aggregate(updates)

    assert set(result.metadata["selected_peers"]) == {"a", "b", "c"}
    assert result.metadata["weight_by"] == "uniform"
    assert np.allclose(result.weights, np.array([0.2]))
