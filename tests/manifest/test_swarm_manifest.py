"""Regression tests: SwarmManifest.

Covers:
- SwarmManifest class exists and can be constructed
- Canonical serialization produces consistent hashes
- Different aggregation/topology/compression strategies produce different hashes
- AffinityWeights are in CollaborationPolicy and validated
- Version fields are present on all policies
"""

import pytest

from quinkgl.manifest import SwarmManifest, DataPolicy, CollaborationPolicy


def test_swarm_manifest_construction():
    """SwarmManifest can be constructed with all required fields."""
    manifest = SwarmManifest(
        model_arch_fingerprint="abc123",
        data_schema_hash="def456",
        aggregation_name="FedAvg",
        aggregation_params={"lr": 0.01},
        topology_name="Random",
        topology_params={"view_size": 5},
        compression_enabled=False,
        data_policy=DataPolicy(),
    )
    assert manifest.model_arch_fingerprint == "abc123"
    assert manifest.data_schema_hash == "def456"
    assert manifest.aggregation_name == "FedAvg"


def test_swarm_manifest_dict_roundtrip():
    """SwarmManifest to_dict/from_dict roundtrip preserves data."""
    original = SwarmManifest(
        model_arch_fingerprint="abc123",
        data_schema_hash="def456",
        aggregation_name="FedProx",
        aggregation_params={"mu": 0.1},
        topology_name="Cyclon",
        topology_params={"shuffle_period": 10},
        compression_enabled=True,
        compression_params={"top_k_ratio": 0.5},
    )
    data = original.to_dict()
    restored = SwarmManifest.from_dict(data, strict=False)
    assert restored.model_arch_fingerprint == original.model_arch_fingerprint
    assert restored.data_schema_hash == original.data_schema_hash
    assert restored.aggregation_name == original.aggregation_name
    assert restored.aggregation_params == original.aggregation_params
    assert restored.topology_name == original.topology_name
    assert restored.topology_params == original.topology_params
    assert restored.compression_enabled == original.compression_enabled
    assert restored.compression_params == original.compression_params


def test_swarm_manifest_canonical_hash_deterministic():
    """Identical manifests produce identical hashes."""
    manifest1 = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        aggregation_name="FedAvg",
    )
    manifest2 = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        aggregation_name="FedAvg",
    )
    assert manifest1.manifest_hash() == manifest2.manifest_hash()


def test_swarm_manifest_hash_changes_with_aggregation():
    """Different aggregation strategies produce different hashes."""
    manifest_fedavg = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        aggregation_name="FedAvg",
    )
    manifest_fedprox = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        aggregation_name="FedProx",
        aggregation_params={"mu": 0.1},
    )
    assert manifest_fedavg.manifest_hash() != manifest_fedprox.manifest_hash()


def test_swarm_manifest_hash_changes_with_topology():
    """Different topologies produce different hashes."""
    manifest_random = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        topology_name="Random",
    )
    manifest_cyclon = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        topology_name="Cyclon",
        topology_params={"shuffle_period": 10},
    )
    assert manifest_random.manifest_hash() != manifest_cyclon.manifest_hash()


def test_swarm_manifest_hash_changes_with_compression():
    """Different compression settings produce different hashes."""
    manifest_no_comp = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        compression_enabled=False,
    )
    manifest_with_comp = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        compression_enabled=True,
        compression_params={"top_k_ratio": 0.5},
    )
    assert manifest_no_comp.manifest_hash() != manifest_with_comp.manifest_hash()


def test_collaboration_policy_has_affinity_weights():
    """CollaborationPolicy includes affinity weights."""
    policy = CollaborationPolicy()
    assert hasattr(policy, "affinity_label_w")
    assert hasattr(policy, "affinity_feature_w")
    assert hasattr(policy, "affinity_gradient_w")
    assert hasattr(policy, "affinity_history_w")
    assert policy.affinity_label_w == 0.4
    assert policy.affinity_feature_w == 0.3
    assert policy.affinity_gradient_w == 0.15
    assert policy.affinity_history_w == 0.15


def test_collaboration_policy_affinity_weights_sum_validation():
    """Affinity weights must sum to approximately 1.0."""
    # Valid: sums to 1.0
    policy = CollaborationPolicy(
        affinity_label_w=0.4,
        affinity_feature_w=0.3,
        affinity_gradient_w=0.15,
        affinity_history_w=0.15,
    )
    policy.validate()  # Should not raise

    # Invalid: sums to 0.5
    policy_invalid = CollaborationPolicy(
        affinity_label_w=0.2,
        affinity_feature_w=0.1,
        affinity_gradient_w=0.1,
        affinity_history_w=0.1,
    )
    with pytest.raises(ValueError, match="Affinity weights must sum"):
        policy_invalid.validate()


def test_collaboration_policy_has_version():
    """CollaborationPolicy has version field."""
    policy = CollaborationPolicy()
    assert hasattr(policy, "version")
    assert policy.version == 1


def test_personalization_policy_has_version():
    """PersonalizationPolicy has version field."""
    policy = CollaborationPolicy()
    # Note: PersonalizationPolicy is separate, checking it exists
    from quinkgl.manifest import PersonalizationPolicy
    pp = PersonalizationPolicy()
    assert hasattr(pp, "version")
    assert pp.version == 1


def test_prototype_policy_has_version():
    """PrototypePolicy has version field."""
    from quinkgl.manifest import PrototypePolicy
    pp = PrototypePolicy()
    assert hasattr(pp, "version")
    assert pp.version == 1


def test_swarm_manifest_validation_requires_non_empty_fingerprint():
    """SwarmManifest validation requires non-empty model_arch_fingerprint."""
    manifest = SwarmManifest(
        model_arch_fingerprint="",
        data_schema_hash="def",
    )
    with pytest.raises(ValueError, match="model_arch_fingerprint must be non-empty"):
        manifest.validate()


def test_swarm_manifest_validation_requires_non_empty_data_schema():
    """SwarmManifest validation requires non-empty data_schema_hash."""
    manifest = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="",
    )
    with pytest.raises(ValueError, match="data_schema_hash must be non-empty"):
        manifest.validate()
