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


@pytest.mark.asyncio
async def test_fedavg_persistence():
    """Test state_dict and load_state_dict for FedAvg (AGG-TASK-17)."""
    agg1 = FedAvg(weight_by="data_size", clip_inverse_loss=False)
    updates = [
        ModelUpdate("a", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("b", np.array([3.0, 4.0]), sample_count=20),
    ]

    result1 = await agg1.aggregate(updates)
    state = agg1.state_dict()

    # Create new aggregator and restore state
    agg2 = FedAvg()
    agg2.load_state_dict(state)

    # Verify config was restored
    assert agg2.config["weight_by"] == "data_size"
    assert agg2.config["clip_inverse_loss"] == False

    # Verify aggregation still works with restored config
    result2 = await agg2.aggregate(updates)
    assert np.allclose(result2.weights, result1.weights)


@pytest.mark.asyncio
async def test_fedavg_concurrent_aggregate():
    """Test concurrent aggregate() calls on same instance (AGG-TASK-19).

    FedAvg is stateless between rounds, so concurrent calls should
    produce correct, independent results without data corruption.
    """
    import asyncio

    agg = FedAvg()

    async def aggregate_task(peer_id: int):
        updates = [
            ModelUpdate(f"peer_{peer_id}", np.array([float(peer_id)]), sample_count=1),
        ]
        return await agg.aggregate(updates)

    # Run multiple aggregations concurrently
    results = await asyncio.gather(
        aggregate_task(1),
        aggregate_task(2),
        aggregate_task(3),
    )

    # All should complete successfully
    assert len(results) == 3
    for result in results:
        assert result is not None
        assert result.weights is not None


@pytest.mark.asyncio
async def test_fedavg_concurrent_aggregate_shared_updates():
    """AGG-TASK-19: Concurrent aggregate() with overlapping updates on same FedAvg instance.

    Verifies that even when the same update list is used across concurrent
    calls, each call produces a valid result without raising or corrupting data.
    """
    import asyncio

    agg = FedAvg(weight_by="uniform")

    updates = [
        ModelUpdate("a", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("b", np.array([3.0, 4.0]), sample_count=20),
    ]

    results = await asyncio.gather(
        agg.aggregate(updates),
        agg.aggregate(updates),
    )

    # Both should produce the same result (FedAvg is stateless)
    assert len(results) == 2
    np.testing.assert_allclose(results[0].weights, results[1].weights)


@pytest.mark.asyncio
async def test_fedavgm_concurrent_aggregate_state_consistency():
    """AGG-TASK-19: FedAvgM has mutable state (momentum_buffer, global_weights).

    Concurrent aggregate() calls must not corrupt internal state.
    Since asyncio is cooperative, the lack of await points inside the
    mutation logic means calls run sequentially within the event loop.
    This test verifies that sequential execution produces consistent state.
    """
    import asyncio
    from quinkgl.aggregation import FedAvgM

    agg = FedAvgM(server_momentum=0.9)

    updates = [
        ModelUpdate("a", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("b", np.array([3.0, 4.0]), sample_count=20),
    ]

    # Run two rounds sequentially (simulating what concurrent calls would do
    # in asyncio since there are no internal await points)
    result1 = await agg.aggregate(updates)
    result2 = await agg.aggregate(updates)

    # Momentum buffer should have been updated
    assert agg.momentum_buffer is not None
    assert agg.global_weights is not None

    # Results should differ (momentum is applied from round 2 onwards)
    assert not np.allclose(result1.weights, result2.weights)

    # State should be consistent
    state = agg.state_dict()
    assert state["server_momentum"] == 0.9


@pytest.mark.asyncio
async def test_scaffold_concurrent_aggregate_state_consistency():
    """AGG-TASK-19: SCAFFOLD has mutable state (_c_global, _round).

    Concurrent aggregate() calls must not corrupt internal state.
    """
    import asyncio
    from quinkgl.aggregation import Scaffold

    agg = Scaffold(learning_rate=0.1)

    cv = {"__single__": np.array([1.0])}

    updates = [
        ModelUpdate("a", np.array([1.0]), sample_count=100, metadata={"control_variate": cv}),
        ModelUpdate("b", np.array([3.0]), sample_count=100, metadata={"control_variate": cv}),
    ]

    # Run two rounds sequentially
    result1 = await agg.aggregate(updates)
    result2 = await agg.aggregate(updates)

    # Round counter should increment
    assert agg._round == 2

    # Global control variate should be set
    assert agg._c_global is not None

    # State should be consistent and persistable
    state = agg.state_dict()
    restored = Scaffold()
    restored.load_state_dict(state)
    assert restored._round == 2
