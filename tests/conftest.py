"""
Shared pytest fixtures for QuinkGL tests.

This file contains common fixtures used across multiple test files.
"""

import asyncio
import numpy as np
import pytest
import time
from typing import Any, Dict
from unittest.mock import patch


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def simple_model_weights():
    """Provide simple model weights for testing."""
    return {
        "layer1": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "layer2": np.array([0.5, 0.6]),
    }


@pytest.fixture
def simple_model_updates():
    """Provide simple model updates for testing aggregation."""
    from quinkgl.aggregation.base import ModelUpdate

    return [
        ModelUpdate("peer1", np.array([1.0, 2.0]), sample_count=10),
        ModelUpdate("peer2", np.array([3.0, 4.0]), sample_count=20),
        ModelUpdate("peer3", np.array([5.0, 6.0]), sample_count=30),
    ]


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "domain": "test_domain",
        "data_schema_hash": "abc123",
        "model_version": "1.0.0",
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for file operations."""
    yield tmp_path
    # Cleanup is handled by tmp_path fixture


@pytest.fixture
def freeze_time():
    """Fixture to freeze time during tests (T9)."""
    original_time = time.time
    original_monotonic = time.monotonic
    
    frozen_timestamp = 1000.0  # Fixed timestamp
    
    def frozen_time_func():
        return frozen_timestamp
    
    def frozen_monotonic_func():
        return frozen_timestamp
    
    with patch("time.time", side_effect=frozen_time_func):
        with patch("time.monotonic", side_effect=frozen_monotonic_func):
            yield frozen_timestamp
    
    # Restore original functions
    time.time = original_time
    time.monotonic = original_monotonic


@pytest.fixture
def monotonic_now():
    """Fixture to provide controlled monotonic time for staleness tests (T9)."""
    original_monotonic = time.monotonic
    current_time = [0.0]
    
    def advance_time(delta: float = 1.0):
        current_time[0] += delta
    
    def monotonic_func():
        return current_time[0]
    
    with patch("time.monotonic", side_effect=monotonic_func):
        yield {
            "advance": advance_time,
            "get": lambda: current_time[0],
        }
    
    time.monotonic = original_monotonic
