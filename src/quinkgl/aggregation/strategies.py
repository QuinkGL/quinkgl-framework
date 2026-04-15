"""
Compatibility shim for aggregation strategies.
"""

from quinkgl.aggregation.fedavgm import FedAvgM
from quinkgl.aggregation.fedprox import FedProx
from quinkgl.aggregation.krum import Krum, MultiKrum
from quinkgl.aggregation.trimmed_mean import TrimmedMean

__all__ = ["FedProx", "FedAvgM", "TrimmedMean", "Krum", "MultiKrum"]
