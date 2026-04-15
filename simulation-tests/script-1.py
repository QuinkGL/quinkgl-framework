#!/usr/bin/env python3
"""
Single-node experiment script for QuinkGL.

This file is intentionally verbose and comment-heavy.
The goal is not to be the shortest possible script, but to be the easiest
place to edit while you are experimenting with:

- topology strategies
- aggregation strategies
- training hyperparameters
- synthetic data size / shape
- node identity / domain / port

How to use this file:
1. Edit the CONFIG section below.
2. Run this file in one terminal:

       python script.py

3. If you want to observe real gossip traffic, open a second terminal and run
   the same file again after changing at least:
   - NODE_ID
   - PORT

   Keep DOMAIN the same on both nodes if you want them to discover each other.

Important note about "single node" testing:
- With only one running node, you can still verify that:
  - the node starts correctly
  - local training works
  - runtime observability output appears in the terminal
  - topology selection runs
- But aggregation differences are limited with a single node, because there may
  be no remote model updates to aggregate.
- To really compare aggregation behavior, run two or more nodes with the same
  DOMAIN and different NODE_ID / PORT values.
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
    CyclonTopology,
    FedAvg,
    FedAvgM,
    FedProx,
    GossipNode,
    Krum,
    MultiKrum,
    PyTorchModel,
    RandomTopology,
    TerminalObserver,
    TrainingConfig,
    TrimmedMean,
)


# =============================================================================
# CONFIG
# =============================================================================
#
# This is the main area you will edit during experiments.
# The rest of the file is mostly helper code.
#
# Recommendation:
# - Change one thing at a time.
# - Keep notes about which topology / aggregation combination you ran.
# - If two nodes should talk to each other, keep DOMAIN the same.
# - If you want two nodes not to talk to each other, change DOMAIN.
# =============================================================================

NODE_ID = "node_b"
DOMAIN = "demo"
PORT = 7001

# Choose one:
# - "random"
# - "cyclon"
TOPOLOGY_NAME = "random"

# Choose one:
# - "fedavg"
# - "fedprox"
# - "fedavgm"
# - "trimmed_mean"
# - "krum"
# - "multikrum"
AGGREGATION_NAME = "fedavg"

# Training/runtime settings.
GOSSIP_INTERVAL_SECONDS = 10.0
EPOCHS_PER_ROUND = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Synthetic dataset settings.
NUM_SAMPLES = 512
NUM_FEATURES = 10
NUM_CLASSES = 2

# Trace events use the shared observability system added to the project.
# Leave this True if you want to clearly see:
# - training started / completed
# - selected peers
# - model send / receive events
# - aggregation completion
TRACE_EVENTS = True

# Optional wall-clock stop.
# - Set to None to run until you press Ctrl+C.
# - Set to a number like 60 to stop after ~60 seconds.
#
# This is NOT the same as "run exactly N rounds".
# The framework currently exposes a continuous loop API, so wall-clock timeout
# is the simplest stable control here.
MAX_RUNTIME_SECONDS = None

# Set this to True only if you explicitly want fallback mode and you have
# tunnel infrastructure configured. For most local experiments, False is safer.
ENABLE_FALLBACK = False


# =============================================================================
# LOGGING
# =============================================================================
#
# The project already emits structured runtime event lines through the
# TerminalObserver. Here we keep logging simple:
# - INFO for QuinkGL logs
# - WARNING for noisier dependencies
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
#
# This script uses synthetic data on purpose:
# - no external dataset download is needed
# - easy to change feature count / class count
# - faster for experimentation
#
# If you later want to switch to a real dataset, this is the section to replace.
# =============================================================================


def make_synthetic_classification_data(
    num_samples: int,
    num_features: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a small synthetic classification dataset.

    The label generation is deliberately simple:
    - for binary classification, we threshold a linear score
    - for multi-class classification, we bin a score into quantiles

    This is enough to validate the project flow without adding dataset
    dependencies to the experiment script.
    """
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


# =============================================================================
# MODEL
# =============================================================================
#
# This is intentionally small and readable.
# You can replace it with a deeper network if you want to stress-test
# serialization, communication, or convergence.
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


# =============================================================================
# TOPOLOGY FACTORY
# =============================================================================
#
# Keeping topology creation in one function makes experimentation easier.
# If you want to add a new topology later, you only need to extend this switch.
# =============================================================================


def build_topology(name: str):
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

    raise ValueError(
        f"Unsupported topology '{name}'. "
        "Expected one of: random, cyclon."
    )


# =============================================================================
# AGGREGATION FACTORY
# =============================================================================
#
# Same idea as the topology factory:
# one central place where you can swap strategies.
#
# Some strategies such as Krum / MultiKrum are meaningful only when multiple
# model updates exist. With one node, they may not show their full behavior.
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
# RUNTIME HELPERS
# =============================================================================
#
# These helpers keep the main() function readable.
# =============================================================================


@dataclass
class ExperimentConfig:
    """A small runtime snapshot that we print at startup."""

    node_id: str
    domain: str
    port: int
    topology: str
    aggregation: str
    gossip_interval_seconds: float
    epochs_per_round: int
    batch_size: int
    learning_rate: float
    trace_events: bool
    max_runtime_seconds: float | None
    num_samples: int
    num_features: int
    num_classes: int


def print_experiment_summary(config: ExperimentConfig) -> None:
    """Print a compact summary of the current experiment configuration."""
    print()
    print("=" * 72)
    print("QuinkGL Single-Node Experiment")
    print("=" * 72)
    print(f"Version          : {__version__}")
    print(f"Node ID          : {config.node_id}")
    print(f"Domain           : {config.domain}")
    print(f"Port             : {config.port}")
    print(f"Topology         : {config.topology}")
    print(f"Aggregation      : {config.aggregation}")
    print(f"Gossip Interval  : {config.gossip_interval_seconds}s")
    print(f"Epochs / Round   : {config.epochs_per_round}")
    print(f"Batch Size       : {config.batch_size}")
    print(f"Learning Rate    : {config.learning_rate}")
    print(f"Trace Events     : {config.trace_events}")
    print(f"Max Runtime      : {config.max_runtime_seconds}")
    print(f"Samples          : {config.num_samples}")
    print(f"Features         : {config.num_features}")
    print(f"Classes          : {config.num_classes}")
    print("=" * 72)
    print()
    print(
        "Tip: To test peer discovery and real model exchange, run this same file "
        "again in another terminal with a different NODE_ID and PORT."
    )
    print()


async def run_experiment(node: GossipNode, training_data) -> None:
    """
    Run the node either:
    - forever, until Ctrl+C
    - or until a wall-clock timeout is reached
    """
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
#
# This is the end-to-end assembly of the experiment:
# 1. configure logging
# 2. build data
# 3. build model
# 4. build topology + aggregation
# 5. create GossipNode
# 6. start it
# 7. optionally attach runtime event tracing
# 8. run the continuous learning loop
# =============================================================================


async def main() -> None:
    configure_logging()

    config = ExperimentConfig(
        node_id=NODE_ID,
        domain=DOMAIN,
        port=PORT,
        topology=TOPOLOGY_NAME,
        aggregation=AGGREGATION_NAME,
        gossip_interval_seconds=GOSSIP_INTERVAL_SECONDS,
        epochs_per_round=EPOCHS_PER_ROUND,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        trace_events=TRACE_EVENTS,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )
    print_experiment_summary(config)

    # Synthetic data is prepared once and reused every round.
    # For quick experiments this is usually enough.
    training_data = make_synthetic_classification_data(
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
    )

    # PyTorchModel is the framework wrapper QuinkGL expects.
    pytorch_model = SimpleMLP(input_size=NUM_FEATURES, num_classes=NUM_CLASSES)
    model_wrapper = PyTorchModel(pytorch_model)

    training_config = TrainingConfig(
        epochs=EPOCHS_PER_ROUND,
        batch_size=min(BATCH_SIZE, NUM_SAMPLES),
        learning_rate=LEARNING_RATE,
        verbose=False,
    )

    topology = build_topology(TOPOLOGY_NAME)
    aggregation = build_aggregation(AGGREGATION_NAME)

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
    )

    if TRACE_EVENTS:
        # This is the observability layer added during the refactor.
        # It prints clear terminal lines for:
        # - training start / completion
        # - target selection
        # - model send / receive
        # - aggregation completion
        #
        # You can pass a custom TerminalObserver here later if you want to
        # redirect output to a file or custom logger.
        node.attach_terminal_observer(TerminalObserver())

    try:
        print("Starting node...")
        await node.start()

        stats = node.get_stats()
        print(f"Connection mode : {stats['connection_mode']}")
        print(f"Known peers     : {stats['known_peers']}")
        print()
        print("Node is running. Press Ctrl+C to stop.")
        print()

        await run_experiment(node, training_data)

    except KeyboardInterrupt:
        print()
        print("Interrupted by user. Shutting down...")
    finally:
        await node.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
