#!/usr/bin/env python3
"""
Single-node experiment script for QuinkGL.

Includes:
- AffinityTopology — like-attracts-like peer selection
- FingerprintComputer — privacy-preserving data fingerprinting
- DataPolicy — manifest-driven collaboration/personalization policy
- PyTorchPersonalizedModel — FedRep/FedBN model split
- APFL adaptive mixing — personalized local/global weight blending
- PrototypeStore / FedPACCollaborator — optional prototype alignment

How to use this file:
1. Edit the CONFIG section below.
2. Run this file in one terminal:

       python script-2.py

3. If you want to observe real gossip traffic, open a second terminal and run
   the same file again after changing at least:
   - NODE_ID
   - PORT

   Keep DOMAIN the same on both nodes if you want them to discover each other.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from quinkgl import (
    __version__,
    AffinityTopology,
    APFLConfig,
    APFLMixin,
    CyclonTopology,
    DataPolicy,
    CollaborationPolicy,
    FedAvg,
    FedAvgM,
    FedProx,
    FingerprintComputer,
    FingerprintPrivacyConfig,
    GossipNode,
    Krum,
    ModelSplit,
    MultiKrum,
    PersonalizationPolicy,
    PrototypePolicy,
    PyTorchModel,
    PyTorchPersonalizedModel,
    RandomTopology,
    TrainingConfig,
    TrimmedMean,
)
from quinkgl.training.prototypes import PrototypeStore, FedPACCollaborator


# =============================================================================
# CONFIG
# =============================================================================

NODE_ID = "ali"
DOMAIN = "demo"
PORT = 7000

# Choose one:
# - "random"
# - "cyclon"
# - "affinity"          (like-attracts-like)
TOPOLOGY_NAME = "affinity"

# Choose one:
# - "fedavg"
# - "fedprox"
# - "fedavgm"
# - "trimmed_mean"
# - "krum"
# - "multikrum"
AGGREGATION_NAME = "fedavg"

# Choose model wrapper:
# - "standard"          — plain PyTorchModel (no personalization)
# - "personalized"      — PyTorchPersonalizedModel with FedRep/FedBN
MODEL_WRAPPER = "personalized"

# APFL adaptive mixing.
# Only effective when MODEL_WRAPPER = "personalized".
APFL_ENABLED = True
APFL_INITIAL_ALPHA = 0.5

# FedProto / FedPAC prototype alignment (experimental).
PROTOTYPES_ENABLED = False
FEDPAC_ENABLED = False

# DataPolicy from manifest.
# These settings control fingerprinting, collaboration, and personalization.
DATA_POLICY = DataPolicy(
    fingerprint_enabled=True,
    min_affinity=0.3,
    privacy_level="standard",
    label_granularity="bucket",
    feature_noise_sigma=0.1,
    gradient_fingerprint=False,
    collaboration=CollaborationPolicy(
        mode="personalized",
        exploration_initial=0.8,
        exploration_decay=0.95,
        exploration_min=0.1,
        ema_alpha=0.2,
        edge_decay_factor=0.95,
        eviction_min_weight=0.05,
        cold_start_rounds=3,
    ),
    personalization=PersonalizationPolicy(
        model_split="auto",
        apfl_enabled=APFL_ENABLED,
        apfl_initial_alpha=APFL_INITIAL_ALPHA,
        fedbn_enabled=True,
    ),
    prototypes=PrototypePolicy(
        enabled=PROTOTYPES_ENABLED,
        alignment_weight=0.1,
        fedpac_enabled=FEDPAC_ENABLED,
    ),
)

# Training/runtime settings.
GOSSIP_INTERVAL_SECONDS = 10.0
EPOCHS_PER_ROUND = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Synthetic dataset settings.
NUM_SAMPLES = 512
NUM_FEATURES = 10
NUM_CLASSES = 2

# Optional wall-clock stop.
MAX_RUNTIME_SECONDS = None

ENABLE_FALLBACK = False

# Telemetry settings.
TELEMETRY_URL = "https://141-147-36-24.sslip.io"
TELEMETRY_HEARTBEAT_SECONDS = 5.0


# =============================================================================
# LOGGING
# =============================================================================


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("ipv8").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


# =============================================================================
# DATA GENERATION
# =============================================================================


def make_synthetic_classification_data(
    num_samples: int,
    num_features: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_samples, num_features)).astype(np.float32)

    score = features[:, 0]
    if num_features > 1:
        score = score + 0.5 * features[:, 1]

    if num_classes == 2:
        labels = (score > 0).astype(np.int64)
    else:
        quantiles = np.linspace(0, 100, num_classes + 1)[1:-1]
        bins = np.percentile(score, quantiles)
        labels = np.digitize(score, bins).astype(np.int64)

    return torch.from_numpy(features), torch.from_numpy(labels)


def compute_fingerprint_from_data(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    policy: DataPolicy,
) -> "DataFingerprint":
    """Compute a privacy-preserving fingerprint from local training data."""
    privacy_config = FingerprintPrivacyConfig(
        label_granularity=policy.label_granularity,
        feature_noise_sigma=policy.feature_noise_sigma,
        gradient_enabled=policy.gradient_fingerprint,
    )
    computer = FingerprintComputer(privacy_config=privacy_config)

    label_counts: dict[str, int] = {}
    feature_moments: dict[str, tuple[float, float]] = {}
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        count = int(mask.sum())
        if count > 0:
            label_counts[str(cls_idx)] = count
            cls_features = features[mask]
            feature_moments[str(cls_idx)] = (
                float(np.mean(cls_features)),
                float(np.var(cls_features)),
            )

    return computer.compute_from_label_counts(
        label_counts=label_counts,
        feature_moments=feature_moments,
    )


# =============================================================================
# MODEL
# =============================================================================


class SimpleMLP(nn.Module):
    """A tiny MLP that is easy to understand and fast to train."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SplitMLP(nn.Module):
    """MLP with explicit backbone/head split for personalized learning.

    Backbone (shared): first two linear layers + ReLU
    Head (personal):   final classification layer
    """

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# =============================================================================
# TOPOLOGY FACTORY
# =============================================================================


def build_topology(name: str, policy: DataPolicy | None = None):
    normalized = name.strip().lower()

    if normalized == "random":
        return RandomTopology(seed=42)

    if normalized == "cyclon":
        return CyclonTopology(
            view_size=20,
            shuffle_length=8,
            shuffle_interval=10.0,
            seed=42,
        )

    if normalized == "affinity":
        collab = (policy or DataPolicy()).collaboration
        return AffinityTopology(
            min_affinity=(policy or DataPolicy()).min_affinity,
            exploration_initial=collab.exploration_initial,
            exploration_decay=collab.exploration_decay,
            exploration_min=collab.exploration_min,
            ema_alpha=collab.ema_alpha,
            edge_decay_factor=collab.edge_decay_factor,
            eviction_min_weight=collab.eviction_min_weight,
            cold_start_rounds=collab.cold_start_rounds,
        )

    raise ValueError(
        f"Unsupported topology '{name}'. "
        "Expected one of: random, cyclon, affinity."
    )


# =============================================================================
# AGGREGATION FACTORY
# =============================================================================


def build_aggregation(name: str):
    normalized = name.strip().lower()

    if normalized == "fedavg":
        return FedAvg(weight_by="data_size")

    if normalized == "fedprox":
        return FedProx(mu=0.01, weight_by="data_size")

    if normalized == "fedavgm":
        return FedAvgM(server_momentum=0.9)

    if normalized == "trimmed_mean":
        return TrimmedMean(trim_ratio=0.1)

    if normalized == "krum":
        return Krum(num_byzantines=1)

    if normalized == "multikrum":
        return MultiKrum(num_byzantines=1)

    raise ValueError(
        f"Unsupported aggregation '{name}'. "
        "Expected one of: fedavg, fedprox, fedavgm, trimmed_mean, krum, multikrum."
    )


# =============================================================================
# MODEL FACTORY
# =============================================================================


def build_model(
    wrapper: str,
    input_size: int,
    num_classes: int,
    policy: DataPolicy,
):
    """Build model wrapper based on config choice."""
    normalized = wrapper.strip().lower()

    if normalized == "standard":
        pytorch_model = SimpleMLP(input_size=input_size, num_classes=num_classes)
        return PyTorchModel(pytorch_model)

    if normalized == "personalized":
        pytorch_model = SplitMLP(input_size=input_size, num_classes=num_classes)
        return PyTorchPersonalizedModel(
            model=pytorch_model,
        )

    raise ValueError(
        f"Unsupported model wrapper '{wrapper}'. "
        "Expected one of: standard, personalized."
    )


# =============================================================================
# RUNTIME HELPERS
# =============================================================================


@dataclass
class ExperimentConfig:
    """A small runtime snapshot that we print at startup."""

    node_id: str
    domain: str
    port: int
    topology: str
    aggregation: str
    model_wrapper: str
    apfl_enabled: bool
    apfl_initial_alpha: float
    prototypes_enabled: bool
    fedpac_enabled: bool
    gossip_interval_seconds: float
    epochs_per_round: int
    batch_size: int
    learning_rate: float
    telemetry_url: str | None
    telemetry_heartbeat_seconds: float
    max_runtime_seconds: float | None
    num_samples: int
    num_features: int
    num_classes: int
    data_policy: DataPolicy


async def run_experiment(node: GossipNode, training_data) -> None:
    if MAX_RUNTIME_SECONDS is None:
        await node.run_continuous(training_data)
        return

    try:
        await asyncio.wait_for(node.run_continuous(training_data), timeout=MAX_RUNTIME_SECONDS)
    except asyncio.TimeoutError:
        print()
        print(f"Reached MAX_RUNTIME_SECONDS={MAX_RUNTIME_SECONDS}. Stopping node.")
        node.stop()


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    configure_logging()

    # Validate data policy (apply_join_policy)
    DATA_POLICY.apply_join_policy()

    config = ExperimentConfig(
        node_id=NODE_ID,
        domain=DOMAIN,
        port=PORT,
        topology=TOPOLOGY_NAME,
        aggregation=AGGREGATION_NAME,
        model_wrapper=MODEL_WRAPPER,
        apfl_enabled=APFL_ENABLED,
        apfl_initial_alpha=APFL_INITIAL_ALPHA,
        prototypes_enabled=PROTOTYPES_ENABLED,
        fedpac_enabled=FEDPAC_ENABLED,
        gossip_interval_seconds=GOSSIP_INTERVAL_SECONDS,
        epochs_per_round=EPOCHS_PER_ROUND,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        telemetry_url=TELEMETRY_URL,
        telemetry_heartbeat_seconds=TELEMETRY_HEARTBEAT_SECONDS,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        data_policy=DATA_POLICY,
    )
    # Synthetic data is prepared once and reused every round.
    training_data = make_synthetic_classification_data(
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )

    # Compute local fingerprint from training data.
    features_np = training_data[0].numpy()
    labels_np = training_data[1].numpy()
    fingerprint = compute_fingerprint_from_data(
        features=features_np,
        labels=labels_np,
        num_classes=NUM_CLASSES,
        policy=DATA_POLICY,
    )
    # Build model wrapper.
    model_wrapper = build_model(
        wrapper=MODEL_WRAPPER,
        input_size=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        policy=DATA_POLICY,
    )

    training_config = TrainingConfig(
        epochs=EPOCHS_PER_ROUND,
        batch_size=min(BATCH_SIZE, NUM_SAMPLES),
        learning_rate=LEARNING_RATE,
        verbose=False,
    )

    # Build topology (AffinityTopology uses data policy params).
    topology = build_topology(TOPOLOGY_NAME, policy=DATA_POLICY)
    aggregation = build_aggregation(AGGREGATION_NAME)

    # Prototype store (optional).
    prototype_store: PrototypeStore | None = None
    fedpac_collaborator: FedPACCollaborator | None = None
    if DATA_POLICY.prototypes.enabled:
        prototype_store = PrototypeStore()
        if DATA_POLICY.prototypes.fedpac_enabled:
            fedpac_collaborator = FedPACCollaborator()
        print(f"Prototype alignment active (weight={DATA_POLICY.prototypes.alignment_weight}, "
              f"fedpac={DATA_POLICY.prototypes.fedpac_enabled})")

    node = GossipNode(
        node_id=NODE_ID,
        domain=DOMAIN,
        model=model_wrapper,
        port=PORT,
        topology=topology,
        aggregation=aggregation,
        gossip_interval=GOSSIP_INTERVAL_SECONDS,
        training_config=training_config,
        enable_fallback=ENABLE_FALLBACK,
        data_policy=DATA_POLICY,
        fingerprint=fingerprint,
    )

    try:
        await node.start()

        if TELEMETRY_URL:
            node.attach_telemetry_client(
                base_url=TELEMETRY_URL,
                heartbeat_interval=TELEMETRY_HEARTBEAT_SECONDS,
            )

        await run_experiment(node, training_data)

    except KeyboardInterrupt:
        print()
        print("Interrupted by user. Shutting down...")
    finally:
        await node.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
