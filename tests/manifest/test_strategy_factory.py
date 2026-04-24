"""Manifest → topology / aggregation factory."""

from quinkgl.aggregation import EntropyWeightedAvg, FedAvg
from quinkgl.manifest.schema import SwarmManifest, TaskSpec, ModelSpec, ByzantineSpec
from quinkgl.manifest.strategy_factory import (
    ERR_RUN_UNKNOWN_AGGREGATION,
    build_aggregation_from_manifest,
    build_topology_from_manifest,
)
from quinkgl.topology import AffinityTopology, RandomTopology


def _minimal_manifest(**kwargs) -> SwarmManifest:
    base = {
        "name": "t",
        "model_arch_fingerprint": "a" * 64,
        "data_schema_hash": "sha256:" + "0" * 64,
        "aggregation_name": "FedAvg",
        "topology_name": "Random",
    }
    base.update(kwargs)
    return SwarmManifest(
        name=base["name"],
        model_arch_fingerprint=base["model_arch_fingerprint"],
        data_schema_hash=base["data_schema_hash"],
        aggregation_name=base["aggregation_name"],
        topology_name=base["topology_name"],
        task=TaskSpec(
            type="classification",
            input_shape=(1, 28, 28),
            output_shape=(10,),
            label_type="integer",
        ),
        model=ModelSpec(framework="pytorch", arch_hash="sha256:" + "a" * 64),
        byzantine=ByzantineSpec(f=0),
        aggregation_params=base.get("aggregation_params", {}),
        topology_params=base.get("topology_params", {}),
    )


def test_build_default_fedavg_random():
    m = _minimal_manifest()
    agg = build_aggregation_from_manifest(m)
    top = build_topology_from_manifest(m)
    assert isinstance(agg, FedAvg)
    assert isinstance(top, RandomTopology)


def test_build_entropy_affinity_with_params():
    m = _minimal_manifest(
        aggregation_name="EntropyWeightedAvg",
        topology_name="AffinityTopology",
        aggregation_params={"entropy_floor": 0.02, "normalize": True},
        topology_params={"min_affinity": 0.2, "cold_start_rounds": 5},
    )
    agg = build_aggregation_from_manifest(m)
    top = build_topology_from_manifest(m)
    assert isinstance(agg, EntropyWeightedAvg)
    assert agg.entropy_floor == 0.02
    assert isinstance(top, AffinityTopology)
    assert top.min_affinity == 0.2
    assert top.cold_start_rounds == 5


def test_unknown_aggregation():
    m = _minimal_manifest(aggregation_name="NotARealStrategy")
    try:
        build_aggregation_from_manifest(m)
    except ValueError as e:
        assert e.args[0] == ERR_RUN_UNKNOWN_AGGREGATION
    else:
        raise AssertionError("expected ValueError")
