"""
Tests for LearningNode core functionality.
"""

import pytest
import numpy as np
from quinkgl.core.learning_node import LearningNode
from quinkgl.aggregation.base import ModelUpdate


@pytest.mark.asyncio
async def test_learning_node_initialization():
    """Test LearningNode can be initialized with basic config."""
    node = LearningNode(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    assert node.peer_id == "test_node"
    assert node.domain == "test_domain"
    assert node.data_schema_hash == "abc123"


@pytest.mark.asyncio
async def test_learning_node_add_update():
    """Test adding model updates to the node."""
    node = LearningNode(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    update = ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10)
    await node.add_update(update)
    
    assert len(node.aggregator.pending_updates) == 1


@pytest.mark.asyncio
async def test_learning_node_aggregate():
    """Test aggregation of model updates."""
    node = LearningNode(
        peer_id="test_node",
        domain="test_domain",
        data_schema_hash="abc123",
    )
    
    updates = [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=20),
    ]
    
    for update in updates:
        await node.add_update(update)
    
    result = await node.aggregate()
    
    assert result is not None
    assert result.weights is not None
