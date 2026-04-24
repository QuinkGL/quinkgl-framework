# Copyright 2026 Ali Seyhan, Baki Turhan
"""Instantiate topology and aggregation strategies from a :class:`SwarmManifest`.

The manifest stores human-facing names (``aggregation_name``, ``topology_name``)
and optional parameter dicts.  This module maps those names to concrete
``quinkgl`` classes and constructs instances for :class:`GossipNode` / CLI ``run``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Type, TypeVar

from quinkgl.aggregation.base import AggregationStrategy
from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg
from quinkgl.aggregation.fedavg import FedAvg
from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.krum import Krum, MultiKrum
from quinkgl.aggregation.scaffold import Scaffold
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.trimmed_mean import TrimmedMean
from quinkgl.manifest.schema import SwarmManifest
from quinkgl.topology.affinity import AffinityTopology
from quinkgl.topology.base import TopologyStrategy
from quinkgl.topology.cyclon import CyclonTopology
from quinkgl.topology.random import RandomTopology

T = TypeVar("T")

ERR_RUN_UNKNOWN_AGGREGATION = "ERR_RUN_UNKNOWN_AGGREGATION"
ERR_RUN_UNKNOWN_TOPOLOGY = "ERR_RUN_UNKNOWN_TOPOLOGY"


def _norm(name: str) -> str:
    return "".join(name.strip().lower().split())


_AGGREGATION_REGISTRY: Dict[str, Type[AggregationStrategy]] = {
    _norm("FedAvg"): FedAvg,
    _norm("FedProx"): FedProx,
    _norm("FedAvgM"): FedAvgM,
    _norm("Krum"): Krum,
    _norm("MultiKrum"): MultiKrum,
    _norm("TrimmedMean"): TrimmedMean,
    _norm("StalenessWeightedFedAvg"): StalenessWeightedFedAvg,
    _norm("EntropyWeightedAvg"): EntropyWeightedAvg,
    _norm("Scaffold"): Scaffold,
}

_TOPOLOGY_REGISTRY: Dict[str, Type[TopologyStrategy]] = {
    _norm("Random"): RandomTopology,
    _norm("RandomTopology"): RandomTopology,
    _norm("Cyclon"): CyclonTopology,
    _norm("CyclonTopology"): CyclonTopology,
    _norm("Affinity"): AffinityTopology,
    _norm("AffinityTopology"): AffinityTopology,
}


def _instantiate(cls: Type[T], params: Dict[str, Any]) -> T:
    """Construct ``cls`` from manifest param dict.

    Strategy ``__init__`` methods accept explicit parameters plus ``**kwargs``
    absorbed into ``config`` on the abstract base classes, so unknown keys are
    tolerated.  If the type system still rejects an argument, surface a clear
    error.
    """
    try:
        sig = inspect.signature(cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return cls(**params)
        kw = {k: v for k, v in params.items() if k in sig.parameters and k != "self"}
        return cls(**kw)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for {cls.__name__}: {exc}"
        ) from exc


def build_aggregation_from_manifest(manifest: SwarmManifest) -> AggregationStrategy:
    """Return a new aggregation strategy from ``manifest.aggregation_name``."""
    name = (manifest.aggregation_name or "FedAvg").strip()
    key = _norm(name)
    cls = _AGGREGATION_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            ERR_RUN_UNKNOWN_AGGREGATION,
            {
                "name": name,
                "known": sorted({c.__name__ for c in _AGGREGATION_REGISTRY.values()}),
            },
        )
    params = dict(manifest.aggregation_params or {})
    return _instantiate(cls, params)


def build_topology_from_manifest(manifest: SwarmManifest) -> TopologyStrategy:
    """Return a new topology strategy from ``manifest.topology_name``."""
    name = (manifest.topology_name or "Random").strip()
    key = _norm(name)
    cls = _TOPOLOGY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            ERR_RUN_UNKNOWN_TOPOLOGY,
            {
                "name": name,
                "known": sorted({c.__name__ for c in _TOPOLOGY_REGISTRY.values()}),
            },
        )
    params = dict(manifest.topology_params or {})
    return _instantiate(cls, params)
