from quinkgl.manifest.schema import SwarmManifest
from quinkgl.manifest.strategy_factory import build_topology_from_manifest
from quinkgl.topology import (
    HybridAffinityReliability,
    RandomRegular,
    ReliabilityAware,
    Ring,
    SmallWorld,
)


def _manifest(topology_name):
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="topology-test",
        topology_name=topology_name,
        aggregation_name="FedAvg",
    )


def test_new_topologies_are_buildable_from_manifest_names():
    expected = {
        "RandomRegular": RandomRegular,
        "Expander": RandomRegular,
        "SmallWorld": SmallWorld,
        "Ring": Ring,
        "ReliabilityAware": ReliabilityAware,
        "HybridAffinityReliability": HybridAffinityReliability,
    }

    for name, cls in expected.items():
        topology = build_topology_from_manifest(_manifest(name))
        assert isinstance(topology, cls)
