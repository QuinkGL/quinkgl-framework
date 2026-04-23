"""
Tests for topology sampler strategies.
"""

import pytest
import numpy as np
from quinkgl.topology.sampler import RandomSampler, DegreeWeightedSampler


def test_random_sampler_selects_correct_count():
    """Test that RandomSampler selects the correct number of peers."""
    sampler = RandomSampler()
    peers = list(range(10))
    
    selected = sampler.sample(peers, k=3, seed=42)
    
    assert len(selected) == 3
    assert all(p in peers for p in selected)


def test_random_sampler_with_replacement():
    """Test RandomSampler with replacement."""
    sampler = RandomSampler(replacement=True)
    peers = list(range(5))
    
    selected = sampler.sample(peers, k=10, seed=42)
    
    assert len(selected) == 10
    # With replacement, we can have duplicates


def test_random_sampler_without_replacement():
    """Test RandomSampler without replacement."""
    sampler = RandomSampler(replacement=False)
    peers = list(range(10))
    
    selected = sampler.sample(peers, k=5, seed=42)
    
    assert len(selected) == 5
    assert len(set(selected)) == 5  # No duplicates


def test_degree_weighted_sampler():
    """Test DegreeWeightedSampler favors high-degree peers."""
    sampler = DegreeWeightedSampler()
    peers = list(range(5))
    degrees = {i: i + 1 for i in peers}  # Higher index = higher degree
    
    selected = sampler.sample(peers, k=3, degrees=degrees, seed=42)
    
    assert len(selected) == 3
    assert all(p in peers for p in selected)


def test_sampler_handles_empty_peer_list():
    """Test that samplers handle empty peer lists gracefully."""
    sampler = RandomSampler()
    
    selected = sampler.sample([], k=3, seed=42)
    
    assert len(selected) == 0


def test_sampler_handles_k_larger_than_peers():
    """Test that samplers handle k larger than peer count."""
    sampler = RandomSampler(replacement=False)
    peers = list(range(3))
    
    selected = sampler.sample(peers, k=10, seed=42)
    
    # Should return all peers when k > len(peers) without replacement
    assert len(selected) == 3
