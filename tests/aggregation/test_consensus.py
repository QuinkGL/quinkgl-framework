"""
Tests for consensus mechanisms.
"""

import numpy as np
import pytest

from quinkgl.aggregation.base import ModelUpdate


def test_byzantine_resistance_with_krum():
    """Test that Krum aggregation resists Byzantine failures."""
    try:
        from quinkgl.aggregation.krum import Krum
    except ImportError:
        pytest.skip("Krum not available")
    
    krum = Krum(num_byzantine=1)
    
    # Normal updates
    normal_updates = [
        ModelUpdate("peer1", np.array([1.0, 1.0]), sample_count=10),
        ModelUpdate("peer2", np.array([1.0, 1.0]), sample_count=10),
        ModelUpdate("peer3", np.array([1.0, 1.0]), sample_count=10),
    ]
    
    # Byzantine update (malicious)
    byzantine_update = ModelUpdate("malicious", np.array([100.0, 100.0]), sample_count=10)
    
    all_updates = normal_updates + [byzantine_update]
    
    # Krum should reject the Byzantine update
    result = krum.aggregate(all_updates)
    
    # Result should be close to the normal updates, not the malicious one
    assert result.weights[0] < 10.0  # Not 100.0


def test_multikrum_aggregation():
    """Test MultiKrum aggregation with multiple good updates."""
    try:
        from quinkgl.aggregation.krum import MultiKrum
    except ImportError:
        pytest.skip("MultiKrum not available")
    
    multikrum = MultiKrum(num_byzantine=1, m=2)
    
    updates = [
        ModelUpdate("peer1", np.array([1.0, 1.0]), sample_count=10),
        ModelUpdate("peer2", np.array([1.0, 1.0]), sample_count=10),
        ModelUpdate("peer3", np.array([1.0, 1.0]), sample_count=10),
        ModelUpdate("malicious", np.array([100.0, 100.0]), sample_count=10),
    ]
    
    result = multikrum.aggregate(updates)
    
    # Result should average the good updates
    assert result.weights[0] < 10.0


def test_entropy_weighted_aggregation():
    """Test entropy-weighted aggregation for diversity."""
    try:
        from quinkgl.aggregation.entropy_weighted import EntropyWeighted
    except ImportError:
        pytest.skip("EntropyWeighted not available")
    
    aggregator = EntropyWeighted()
    
    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=20),
    ]
    
    result = aggregator.aggregate(updates)
    
    assert result.weights is not None
    assert result.weights.shape == (2,)
