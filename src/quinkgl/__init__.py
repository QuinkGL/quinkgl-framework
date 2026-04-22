# Copyright 2026 Ali Seyhan, Baki Turhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuinkGL: Decentralized Gossip Learning Framework

A Python framework for decentralized machine learning using
gossip-based peer-to-peer communication.

Example:
    from quinkgl import (
        GossipNode,
        PyTorchModel,
        RandomTopology,
        FedAvg,
        TrainingConfig
    )

    model = PyTorchModel(my_pytorch_model)
    node = GossipNode(
        node_id="alice",
        domain="health",
        model=model,
        topology=RandomTopology(),
        aggregation=FedAvg()
    )
    await node.start()
    await node.run_continuous(training_data)
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("quinkgl")
except _PackageNotFoundError:
    __version__ = "0.3.1"

# =============================================================================
# LOGGING — honour QUINKGL_LOG_LEVEL env-var (mirrors Flower's FLWR_LOG_LEVEL)
# =============================================================================
import logging as _logging
import os as _os

_log_level_name = _os.environ.get("QUINKGL_LOG_LEVEL", "WARNING").upper()
_log_level = getattr(_logging, _log_level_name, _logging.WARNING)
_logging.getLogger("quinkgl").setLevel(_log_level)

# Suppress noisy HTTP client loggers (telemetry uses httpx)
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("httpcore").setLevel(_logging.WARNING)

# =============================================================================
# CORE - Main node classes
# =============================================================================
from quinkgl.core.learning_node import LearningNode

# Backward compatibility alias with deprecation warning
class GLNode(LearningNode):
    """Deprecated alias for LearningNode. Use LearningNode instead."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "GLNode is deprecated and will be removed in a future version. "
            "Use LearningNode instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

from quinkgl.network.gossip_node import GossipNode

# =============================================================================
# MODELS - Framework-specific model wrappers
# =============================================================================
from quinkgl.models.base import (
    ModelWrapper,
    TrainingConfig,
    TrainingResult,
    ModelSplit,
    PersonalizedModelWrapper,
    APFLConfig,
    APFLMixin,
)
from quinkgl.models.pytorch import PyTorchModel, PyTorchPersonalizedModel

# TensorFlow is optional - only import if available
try:
    from quinkgl.models.tensorflow import TensorFlowModel
    _tensorflow_available = True
except ImportError:
    _tensorflow_available = False

# =============================================================================
# TOPOLOGY - Peer selection strategies
# =============================================================================
from quinkgl.topology.base import (
    TopologyStrategy,
    PeerInfo,
    SelectionContext
)
from quinkgl.topology.random import RandomTopology
from quinkgl.topology.cyclon import CyclonTopology
from quinkgl.topology.affinity import AffinityTopology
from quinkgl.topology.spectral import (
    SpectralAnalyzer,
    SpectralReport,
    build_ring_adjacency,
    build_complete_adjacency,
    build_random_regular_adjacency,
)

# =============================================================================
# FINGERPRINT - Privacy-preserving data distribution summaries
# =============================================================================
from quinkgl.fingerprint import (
    DataFingerprint,
    FINGERPRINT_SCHEMA_VERSION,
    AffinityWeights,
    FingerprintPrivacyConfig,
    FingerprintComputer,
)

# =============================================================================
# AGGREGATION - Model combining strategies
# =============================================================================
from quinkgl.aggregation.base import (
    AggregationStrategy,
    ModelUpdate,
    AggregatedModel
)
from quinkgl.aggregation.fedavg import FedAvg
from quinkgl.aggregation.strategies import (
    FedProx,
    FedAvgM,
    TrimmedMean,
    Krum,
    MultiKrum
)
from quinkgl.aggregation.staleness_fedavg import StalenessWeightedFedAvg
from quinkgl.aggregation.entropy_weighted import EntropyWeightedAvg
from quinkgl.aggregation.scaffold import Scaffold

# =============================================================================
# MANIFEST - Swarm manifest and data policy
# =============================================================================
from quinkgl.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DataPolicy,
    CollaborationPolicy,
    PersonalizationPolicy,
    PrototypePolicy,
    SwarmManifest,
    check_compatibility,
)

# =============================================================================
# GOSSIP - Model aggregation orchestration
# =============================================================================
from quinkgl.gossip.aggregator import ModelAggregator

# =============================================================================
# OBSERVABILITY - Runtime event primitives and terminal rendering
# =============================================================================
from quinkgl.observability import EventEmitter, RuntimeEvent
from quinkgl.observability.terminal import TerminalObserver, format_runtime_event
from quinkgl.telemetry import TelemetryClient

# =============================================================================
# DATA - Dataset loading and splitting (OPTIONAL - not included in pip package)
# =============================================================================
# Data loading utilities are in scripts/run_gossip_node.py for full functionality
# or use torchvision directly for CIFAR-10
# NOTE: quinkgl.data module is not included in the package; remove these exports
_data_available = False


# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Core
    "LearningNode",
    "GLNode",  # Backward compatibility alias
    "GossipNode",
    
    # Models
    "ModelWrapper",
    "TrainingConfig",
    "TrainingResult",
    "ModelSplit",
    "PersonalizedModelWrapper",
    "APFLConfig",
    "APFLMixin",
    "PyTorchModel",
    "PyTorchPersonalizedModel",
    
    # Topology
    "TopologyStrategy",
    "RandomTopology",
    "CyclonTopology",
    "AffinityTopology",
    "SpectralAnalyzer",
    "SpectralReport",
    "build_ring_adjacency",
    "build_complete_adjacency",
    "build_random_regular_adjacency",
    "PeerInfo",
    "SelectionContext",
    
    # Fingerprint
    "DataFingerprint",
    "FINGERPRINT_SCHEMA_VERSION",
    "AffinityWeights",
    "FingerprintPrivacyConfig",
    "FingerprintComputer",
    
    # Aggregation
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "FedAvgM",
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    "StalenessWeightedFedAvg",
    "EntropyWeightedAvg",
    "Scaffold",
    "ModelUpdate",
    "AggregatedModel",
    
    # Manifest
    "MANIFEST_SCHEMA_VERSION",
    "DataPolicy",
    "CollaborationPolicy",
    "PersonalizationPolicy",
    "PrototypePolicy",
    "SwarmManifest",
    "check_compatibility",

    # Gossip
    "ModelAggregator",

    # Observability
    "EventEmitter",
    "RuntimeEvent",
    "TerminalObserver",
    "format_runtime_event",
    "TelemetryClient",

    # Feature flags
    "_tensorflow_available",
    "_data_available",
]

if _tensorflow_available:
    __all__.append("TensorFlowModel")

# NOTE: _data_available is False (data module not included in package)
# No data exports are added to __all__
