import numpy as np
import pytest

from quinkgl.aggregation import Krum
from quinkgl.aggregation.base import ModelUpdate


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
