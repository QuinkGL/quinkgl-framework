import numpy as np
import pytest

from quinkgl.aggregation import FedAvg
from quinkgl.aggregation.base import ModelUpdate


@pytest.mark.asyncio
async def test_fedavg_weights_by_sample_count():
    agg = FedAvg()
    updates = [
        ModelUpdate("a", np.array([0.0]), sample_count=1),
        ModelUpdate("b", np.array([2.0]), sample_count=3),
    ]

    result = await agg.aggregate(updates)

    assert np.allclose(result.weights, np.array([1.5]))
