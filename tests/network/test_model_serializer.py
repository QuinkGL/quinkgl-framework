"""
Tests for network model serialization.
"""

import numpy as np
import pytest

from quinkgl.network.model_serializer import serialize_model, deserialize_model, get_model_size_info


def test_serialize_deserialize_numpy_array():
    """Test serialization and deserialization of numpy array weights."""
    weights = {
        "layer1": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "layer2": np.array([0.5, 0.6]),
    }
    
    serialized = serialize_model(weights)
    deserialized = deserialize_model(serialized)
    
    for key in weights:
        assert np.allclose(deserialized[key], weights[key])


def test_serialize_deserialize_with_compression():
    """Test serialization with compression enabled."""
    weights = {
        "layer1": np.random.randn(100, 100),
        "layer2": np.random.randn(50),
    }
    
    serialized = serialize_model(weights, enable_compression=True)
    deserialized = deserialize_model(serialized)
    
    for key in weights:
        assert np.allclose(deserialized[key], weights[key], atol=1e-5)


def test_get_model_size_info():
    """Test getting model size information."""
    weights = {
        "layer1": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "layer2": np.array([0.5, 0.6]),
    }
    
    size_info = get_model_size_info(weights)
    assert "num_parameters" in size_info
    assert "size_bytes" in size_info
    assert size_info["num_parameters"] == 6  # 2x2 + 2 = 6


def test_wire_format_version():
    """Test that wire format version is embedded in serialized data."""
    weights = {"layer1": np.array([1.0, 2.0])}
    
    serialized = serialize_model(weights)
    # First byte should be the version
    version = serialized[0]
    assert version == 1  # Assuming version 1


def test_deserialize_with_wrong_version():
    """Test that deserialization fails with wrong version."""
    weights = {"layer1": np.array([1.0, 2.0])}
    serialized = serialize_model(weights)
    
    # Corrupt the version byte
    corrupted = bytearray(serialized)
    corrupted[0] = 99
    
    with pytest.raises(ValueError, match="Unsupported wire format version"):
        deserialize_model(bytes(corrupted))


def test_size_limit():
    """Test that serialization respects size limits."""
    from quinkgl.network.gossip_community import MAX_INCOMING_MESSAGE_SIZE
    
    weights = {"layer1": np.random.randn(100, 100)}
    serialized = serialize_model(weights)
    
    assert len(serialized) < MAX_INCOMING_MESSAGE_SIZE


def test_pipeline_round_trip_property():
    """Property-based test: serialize -> deserialize should be idempotent (T-14)."""
    import random
    
    # Test with random weight shapes and values
    for _ in range(10):
        shape = (random.randint(1, 20), random.randint(1, 20))
        weights = {"layer1": np.random.randn(*shape).astype(np.float32)}
        
        serialized = serialize_model(weights)
        deserialized = deserialize_model(serialized)
        
        # Should recover the original weights (within compression tolerance)
        assert np.allclose(deserialized["layer1"], weights["layer1"], atol=1e-3)


def test_pipeline_round_trip_with_compression():
    """Property-based test: compression pipeline round-trip (T-14)."""
    from quinkgl.serialization import compress_weights, decompress_weights, CompressionConfig
    
    weights = {"layer1": np.random.randn(50, 50).astype(np.float32)}
    
    # Serialize
    serialized = serialize_model(weights)
    
    # Compress
    config = CompressionConfig(sparsity=0.5, quantization_bits=8)
    compressed = compress_weights(weights, config)
    
    # Decompress
    decompressed = decompress_weights(compressed)
    
    # Should be close to original (within compression tolerance)
    assert np.allclose(decompressed["layer1"], weights["layer1"], atol=0.1)
