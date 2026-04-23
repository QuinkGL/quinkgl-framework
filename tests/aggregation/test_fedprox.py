"""
Tests for FedProx aggregation strategy.
"""

import numpy as np
import pytest

from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.base import ModelUpdate


@pytest.mark.asyncio
async def test_fedprox_basic_aggregation():
    """Test basic FedProx aggregation without proximal term."""
    prox = FedProx(mu=0.01)

    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=20),
    ]

    result = await prox.aggregate(updates)
    
    # Without proximal term, should behave like FedAvg
    expected = np.array([2.33333333, 3.33333333])  # Weighted average
    np.testing.assert_allclose(result.weights, expected, atol=1e-5)


@pytest.mark.asyncio
async def test_fedprox_persistence():
    """Test FedProx state persistence (T20)."""
    prox1 = FedProx(mu=0.05, mode="weight_interpolation")

    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
    ]
    
    result1 = await prox1.aggregate(updates)
    state = prox1.state_dict()

    # Create new FedProx and restore state
    prox2 = FedProx()
    prox2.load_state_dict(state)

    # Verify config was restored
    assert prox2.config["mu"] == 0.05
    assert prox2.config["mode"] == "weight_interpolation"

    # Verify aggregation still works with restored config
    result2 = await prox2.aggregate(updates)
    assert np.allclose(result2.weights, result1.weights)


@pytest.mark.asyncio
async def test_fedprox_global_weights_persistence():
    """Test that FedProx global weights are persisted (T20)."""
    prox = FedProx(mu=0.01, mode="weight_interpolation")

    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=20),
    ]

    await prox.aggregate(updates)
    
    # Save state after aggregation
    state = prox.state_dict()
    
    # Verify global weights are in state
    assert "global_weights" in state or "_global_weights" in state


@pytest.mark.asyncio
async def test_fedprox_mode_weight_interpolation():
    """Test FedProx in weight_interpolation mode."""
    prox = FedProx(mu=0.1, mode="weight_interpolation")

    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=10),
    ]

    result = await prox.aggregate(updates)
    
    # In weight_interpolation mode, result should be between local and global
    assert result.weights is not None
    assert result.weights.shape == (2,)
