"""
Aggregation package.

This package contains model merge strategies for decentralized learning.
It is responsible for combining peer updates, not for transport,
topology selection, or serialization.

Usage:
    from quinkgl.aggregation import FedAvg, FedProx, TrimmedMean, Krum

    # Basic federated averaging
    aggregator = FedAvg(weight_by="data_size")
    aggregated = await aggregator.aggregate(updates)

    # Byzantine-resilient aggregation
    aggregator = TrimmedMean(trim_ratio=0.1)
    aggregated = await aggregator.aggregate(updates)
"""

from quinkgl.aggregation.base import (
    AggregationStrategy,
    ModelUpdate,
    AggregatedModel
)
from quinkgl.aggregation.fedavg import FedAvg
from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.krum import Krum, MultiKrum
from quinkgl.aggregation.trimmed_mean import TrimmedMean
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg
from quinkgl.aggregation.scaffold import Scaffold

# Export main classes
__all__ = [
    # Base classes
    "AggregationStrategy",
    "ModelUpdate",
    "AggregatedModel",
    # Standard strategies
    "FedAvg",
    # Advanced strategies
    "FedProx",
    "FedAvgM",
    "StalenessWeightedFedAvg",
    # Byzantine-resilient strategies
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    # Entropy-based (RNEP-inspired)
    "EntropyWeightedAvg",
    # Variance reduction (SCAFFOLD)
    "Scaffold",
]
