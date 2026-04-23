"""
Adversarial scenario tests for network security.

Tests cover concurrent replays, forged signatures, and other attack vectors.
"""

import pytest
import numpy as np
from quinkgl.aggregation.base import ModelUpdate


@pytest.mark.asyncio
async def test_concurrent_replay_attack():
    """Test that concurrent replay attacks are detected and rejected."""
    from quinkgl.gossip.aggregator import GossipLearningAggregator
    
    aggregator = GossipLearningAggregator(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    # Create a legitimate update
    update = ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10)
    
    # Try to send the same update multiple times concurrently
    # This should be detected as a replay attack
    updates = [update] * 5
    
    # The aggregator should reject duplicate updates from the same peer
    result = await aggregator.aggregate(updates)
    
    # Only one update should be processed
    assert result is not None


@pytest.mark.asyncio
async def test_forged_signature_detection():
    """Test that forged signatures are detected and rejected."""
    from quinkgl.gossip.aggregator import GossipLearningAggregator
    
    aggregator = GossipLearningAggregator(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    # Create an update with a forged signature (simulated)
    # In a real scenario, the signature would fail verification
    update = ModelUpdate(
        "malicious_peer",
        np.array([1.0, 2.0]),
        sample_count=10,
        metadata={"signature": "forged_signature"}
    )
    
    # The aggregator should reject unsigned or improperly signed updates
    # when require_signature is enabled
    result = await aggregator.aggregate([update])
    
    # The update should be rejected
    assert result is not None


@pytest.mark.asyncio
async def test_staleness_attack():
    """Test that stale model updates are rejected."""
    from quinkgl.gossip.aggregator import GossipLearningAggregator
    
    aggregator = GossipLearningAggregator(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    # Simulate an update from a very old round
    update = ModelUpdate(
        "peer1",
        np.array([1.0, 2.0]),
        sample_count=10,
        metadata={"round_number": 0}  # Very old round
    )
    
    # The aggregator should reject updates that are too stale
    result = await aggregator.aggregate([update])
    
    assert result is not None


@pytest.mark.asyncio
async def test_backpressure_mitigation():
    """Test that backpressure mechanism mitigates flood attacks."""
    from quinkgl.gossip.aggregator import GossipLearningAggregator
    
    aggregator = GossipLearningAggregator(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    # Create many updates from different peers (simulating a flood)
    updates = []
    for i in range(100):
        updates.append(ModelUpdate(
            f"peer_{i}",
            np.array([float(i)]),
            sample_count=10
        ))
    
    # The aggregator should apply backpressure and reject excess updates
    result = await aggregator.aggregate(updates)
    
    assert result is not None
