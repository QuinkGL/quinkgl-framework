# QuinkGL: Decentralized Gossip Learning Framework

[![PyPI version](https://badge.fury.io/py/quinkgl.svg)](https://badge.fury.io/py/quinkgl)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**QuinkGL** is a fully **decentralized, peer-to-peer (P2P) federated learning framework** that enables collaborative model training across distributed devices without relying on a central parameter server. Built on gossip-based protocols, QuinkGL addresses the core challenges of decentralized learning: **communication efficiency**, **non-IID data heterogeneity**, and **Byzantine fault tolerance**.

---

## Motivation

Centralized federated learning (FL) architectures such as FedAvg [[McMahan et al., 2017]](#references) depend on a parameter server for global aggregation, introducing a single point of failure and a communication bottleneck. As edge computing scales — driven by IoT proliferation and privacy-sensitive domains like healthcare — decentralized alternatives become essential.

QuinkGL draws from the gossip learning paradigm [[Ormándi et al., 2013]](#references), where nodes exchange model updates directly with randomly selected peers. This eliminates server dependency and enables organic convergence through repeated local interactions. The framework extends this foundation with:

- **Data-aware peer selection** via privacy-preserving fingerprints
- **Entropy-weighted aggregation** inspired by RNEP [[Kang & Lee, 2024]](#references)
- **Byzantine-resilient strategies** including Krum [[Blanchard et al., 2017]](#references) and TrimmedMean
- **Pluggable architecture** for topology, aggregation, and model strategies

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Fully Decentralized** | No central server — pure P2P gossip protocol |
| **Non-IID Resilient** | AffinityTopology + EntropyWeightedAvg + FedProx + SCAFFOLD for heterogeneous data |
| **Privacy-Preserving Fingerprints** | Quantized, noised, schema-validated data summaries with per-round binding for peer matching |
| **Byzantine Fault Tolerance** | Krum, MultiKrum, TrimmedMean aggregation strategies |
| **NAT Traversal** | IPv8 with UDP hole punching + automatic tunnel fallback |
| **Framework Agnostic** | PyTorch, TensorFlow, or custom model wrappers |
| **Swarm Manifest** | Canonical SHA-256 commitment to training protocol and privacy policy |
| **Personalized FL** | APFL adaptive mixing, FedRep-style backbone/head split |
| **Staleness-Aware** | StalenessWeightedFedAvg for asynchronous environments |
| **Variance Reduction** | SCAFFOLD with gossip-adapted control variates (Karimireddy et al., 2020) |
| **Spectral Analysis** | Runtime algebraic connectivity (λ₂) and spectral gap measurement for topology evaluation |
| **Observability** | Event-driven telemetry with terminal rendering |

---

## Installation

```bash
pip install quinkgl
```

For development:

```bash
git clone https://github.com/QuinkGL/quinkgl-framework.git
cd quinkgl-framework
pip install -e ".[dev]"
```

---

## Quick Start

### CLI (New in Phase 1)

```bash
# Install
pip install quinkgl

# 1. Create a manifest (this is the swarm blueprint, not the swarm itself)
quinkgl manifest create \
  --name my-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --output swarm.qgl

# 2. Verify the manifest
quinkgl manifest verify swarm.qgl

# 3. Get a shareable magnet URI
quinkgl manifest magnet swarm.qgl

# 4. Scaffold a custom peer project
quinkgl init --output-dir my-peer --template pytorch-vision --manifest swarm.qgl

# 5. Start a peer — the swarm is born when the first peer runs
quinkgl run --manifest swarm.qgl --script my-peer/peer_script.py --dry-run
```

> **Note:** Creating the manifest does **not** start a swarm. The manifest is
> only a static blueprint. A swarm comes into existence when the first peer
> calls `quinkgl run` with that manifest.

### Python API

```python
import asyncio
import torch.nn as nn
from quinkgl import GossipNode, PyTorchModel, AffinityTopology, EntropyWeightedAvg

# 1. Define your model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))

# 2. Wrap the model
model = PyTorchModel(SimpleNet(), device="cpu")

# 3. Create and run the node
async def main():
    node = GossipNode(
        node_id="alice",
        domain="mnist",
        model=model,
        port=7000,
        topology=AffinityTopology(min_affinity=0.3),
        aggregation=EntropyWeightedAvg(),
    )

    await node.start()
    await node.run_continuous(training_data)
    await node.shutdown()

asyncio.run(main())
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          GossipNode                              │
│    (Production-ready node with P2P networking + fallback)        │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │ PyTorchModel │  │ RandomTopology │  │      FedAvg          │ │
│  │ TensorFlow   │  │ CyclonTopology │  │ FedProx  │ FedAvgM  │ │
│  │ CustomModel  │  │ AffinityTopol. │  │ Krum │ TrimmedMean  │ │
│  │              │  │                │  │ EntropyWeightedAvg   │ │
│  │              │  │                │  │ StalenessWeighted    │ │
│  └──────────────┘  └────────────────┘  └──────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │    DataFingerprint ─► AffinityScore ─► Peer Selection     │  │
│  │    (Privacy-preserving data distribution summaries)       │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           ModelAggregator (Train → Gossip → Aggregate)    │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         IPv8 Network Layer + Tunnel Fallback              │  │
│  │      (P2P, NAT Traversal, UDP Hole Punching, Relay)      │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │    Observability: EventEmitter → TelemetryClient          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
QuinkGL/
├── src/quinkgl/
│   ├── core/                  # LearningNode (network-agnostic abstraction)
│   ├── models/                # PyTorch, TensorFlow, personalized model wrappers
│   ├── topology/              # RandomTopology, CyclonTopology, AffinityTopology, SpectralAnalyzer
│   ├── aggregation/           # FedAvg, FedProx, FedAvgM, Krum, TrimmedMean,
│   │                          # EntropyWeightedAvg, StalenessWeightedFedAvg, Scaffold
│   ├── fingerprint/           # DataFingerprint, AffinityWeights, FingerprintComputer
│   ├── manifest/              # SwarmManifest, DataPolicy, CollaborationPolicy
│   ├── gossip/                # Protocol primitives, ModelAggregator orchestration
│   ├── network/               # GossipNode, IPv8 manager, gossip community
│   ├── training/              # Convergence monitoring, prototype-based alignment
│   ├── serialization/         # Model weight serialization, compression pipeline, Error Feedback
│   ├── storage/               # Model checkpointing
│   ├── observability/         # EventEmitter, RuntimeEvent, TerminalObserver
│   ├── telemetry/             # TelemetryClient
│   └── utils/                 # Shared utilities
├── tests/                     # 364+ unit tests
└── docs/                      # Deployment guides, research notes
```

### Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| `core` | Public node abstraction without transport concerns |
| `gossip` | Round orchestration and protocol primitives |
| `network` | IPv8 transport, NAT traversal, and wire delivery |
| `aggregation` | Model merge strategies (pluggable) |
| `topology` | Peer selection, partial-view management, spectral analysis |
| `fingerprint` | Privacy-preserving data distribution summaries |
| `manifest` | Cryptographic swarm identity and policy declaration |
| `training` | Convergence monitoring, prototype alignment (FedProto/FedPAC) |
| `serialization` | Model weight serialization, compression pipeline, error feedback |
| `observability` | Event-driven runtime telemetry |

---

## Topology Strategies

QuinkGL provides pluggable peer selection strategies that determine *which* peers to exchange models with each round.

| Strategy | Approach | Literature |
|----------|----------|-----------|
| `RandomTopology` | Uniform random peer selection | Ormándi et al., 2013 |
| `CyclonTopology` | Periodic shuffling for network exploration | Voulgaris et al., 2005 |
| `AffinityTopology` | **Data-aware** peer selection via fingerprint similarity with exploration–exploitation balancing | Domain-aware collaboration (this work) |
| `Ring` | Stable logical-ring neighbors for low-bandwidth baselines | Decentralized SGD / gossip baselines |
| `RandomRegular` | Fixed-degree random-regular/expander-style peer sampling | Expander overlays for DFL |
| `SmallWorld` | Ring-local peers plus random long-range shortcuts | Watts–Strogatz small-world networks |
| `ReliabilityAware` | Transfer-success, timeout, and latency-aware peer selection | P2P reliability-aware overlays |
| `HybridAffinityReliability` | Blends fingerprint affinity with transfer reliability | QuinkGL hybrid strategy |

### Spectral Analysis

The `SpectralAnalyzer` provides **runtime measurement** of topology quality through algebraic connectivity and spectral gap — quantities that directly determine gossip convergence speed [[Koloskova et al., 2020]](#references).

```python
from quinkgl.topology import SpectralAnalyzer, build_ring_adjacency

analyzer = SpectralAnalyzer()
report = analyzer.analyze(build_ring_adjacency(10))
print(report.summary())
# n=10 e=10 λ₂=0.3820 gap=0.1315 connected=True mix_time≤17.5
```

| Metric | Meaning |
|--------|--------|
| `algebraic_connectivity` (λ₂) | Fiedler value — positive ↔ connected graph |
| `spectral_gap` (1−\|λ₂(W)\|) | Larger gap → faster gossip convergence |
| `mixing_time_upper` | Upper bound: `log(n) / spectral_gap` |
| `is_connected` | Whether the graph is fully connected |

### AffinityTopology — Like-Attracts-Like

`AffinityTopology` selects peers based on **data distribution similarity** using privacy-preserving fingerprints. It incorporates:

- **Multi-signal affinity** — label buckets (40%), feature moments (30%), gradient similarity (15%), collaboration history (15%)
- **Cold-start resilience** — three phases (blind → learning → exploiting) with decaying exploration ratio
- **Adaptive collaboration graph** — EMA-blended edge weights with automatic decay and eviction of stale edges

---

## Communication Efficiency — Error Feedback

QuinkGL's compression pipeline (Delta → Sparsify → Quantize → Serialize → Zlib) uses **biased compressors** (Top-k, QSGD). Without correction, these break convergence guarantees. The `ErrorFeedbackState` module implements the **Error Feedback** mechanism [[Alistarh et al., 2018]](#references) that accumulates the compression residual and re-injects it in the next round:

```python
from quinkgl.serialization import CompressionConfig, SparsificationConfig

config = CompressionConfig(
    sparsification=SparsificationConfig(top_k_ratio=0.01),
    error_feedback=True,   # activate EF — turns biased compressor effectively unbiased
)
```

**Key property**: Over K rounds, `Σ compressed_outputs + final_residual = Σ raw_deltas` (information conservation, verified by unit tests). Supports EF21-style momentum blending and optional residual norm capping.

## Aggregation Strategies

All strategies implement the `AggregationStrategy` interface and are hot-swappable.

| Strategy | Type | Description | Reference |
|----------|------|-------------|-----------|
| `FedAvg` | Standard | Weighted averaging by sample count | McMahan et al., 2017 |
| `FedProx` | Non-IID | Proximal term to limit client drift | Li et al., 2020 |
| `FedAvgM` | Stability | Server momentum for smoother convergence | Hsu et al., 2019 |
| `EntropyWeightedAvg` | Non-IID | Shannon entropy–based weighting (RNEP-inspired) | Kang & Lee, 2024 |
| `StalenessWeightedFedAvg` | Async | Exponential penalty for stale updates | — |
| `Scaffold` | Non-IID | Control-variate drift correction (gossip variant) | Karimireddy et al., 2020 |
| `TrimmedMean` | Byzantine | Trim extreme values before averaging | Yin et al., 2018 |
| `Krum` / `MultiKrum` | Byzantine | Select most central update(s) | Blanchard et al., 2017 |

### EntropyWeightedAvg — RNEP-Inspired Aggregation

Weights each peer's contribution by the **Shannon entropy** of its local label distribution. Peers with diverse (high-entropy) data exert more influence on the aggregated model, while skewed (low-entropy) peers contribute less — preventing overfitting to biased local distributions.

```python
from quinkgl import EntropyWeightedAvg

aggregation = EntropyWeightedAvg(
    entropy_floor=0.01,    # minimum weight for single-class peers
    fallback_weight=1.0,   # weight when no distribution metadata available
)
```

### Scaffold — Variance Reduction via Control Variates

Implements the SCAFFOLD algorithm [[Karimireddy et al., 2020]](#references) adapted for gossip topology. Each node maintains a *control variate* that estimates its local gradient drift. The gossip variant replaces the central server's global control variate with a running EMA of peer control variates.

```python
from quinkgl import Scaffold

aggregation = Scaffold(
    learning_rate=0.01,       # local SGD learning rate
    global_learning_rate=1.0, # aggregation-side scaling
    control_momentum=0.0,     # 0.0 = classic EF, 0.9 = EF21 momentum
)
```

Key property: SCAFFOLD provably reduces the gradient variance caused by non-IID data, unlike FedProx which only adds a proximal penalty.

---

## Privacy-Preserving Data Fingerprints

Each node computes a lightweight, **privacy-preserving summary** of its local data distribution. Raw statistics are never shared — all fields are transformed before transmission.

| Raw Field | Privacy Transform | Output |
|-----------|-------------------|--------|
| Label distribution | Quantize into buckets (low/medium/high) | `label_buckets` |
| Feature moments (mean, var) | Add calibrated Gaussian noise | `noised_moments` |
| Sample count | Bucket into ranges (e.g., "1k–10k") | `sample_bucket` |
| Gradient moments | Noise + **disabled by default** (gradient inversion risk) | `gradient_moments` |

Fingerprints are exchanged during peer discovery and used by `AffinityTopology` to compute affinity scores.

Fingerprint payloads are schema-versioned, strictly validated on parse, and can be refreshed with a per-round nonce during long-running gossip sessions to reduce cross-round linkability.

---

## Swarm Manifest

The **Swarm Manifest** (`.qgl` file) is the canonical protocol-identity layer
that binds swarm compatibility to a description of the training protocol,
model architecture, aggregation strategy, topology, and trust boundary.

A manifest is **not** a running swarm — it is only a static blueprint.  The
swarm comes into existence when peers call `quinkgl run --manifest swarm.qgl`.

Manifests are:

- **Canonically hashed** (SHA-256 over deterministic JSON) so any change to
  policy or architecture produces a new swarm identity.
- **Schema-versioned** and strictly validated to avoid silent field drops or
  incompatible policy mixes.
- **Optionally signed** with Ed25519 so peers can verify creator identity
  before joining.

To create a manifest you need the architecture hash of your model, which is a
fingerprint of layer names, shapes, and dtypes (not weights).  Compute it with
`quinkgl.manifest.compute_arch_hash(model)` and pass it to
`quinkgl manifest create --model-arch-hash <hash>`.

---

## Personalized Federated Learning

QuinkGL supports personalization techniques to handle statistical heterogeneity:

| Technique | Description |
|-----------|-------------|
| **APFL** (Adaptive Personalized FL) | Adaptive mixing coefficient between local and global models |
| **FedRep-style split** | Shared backbone + personalized head via `ModelSplit` |
| **FedProto / FedPAC** | Prototype-based alignment and classifier collaboration |

---

## Public API Overview

### Core

| Class | Description |
|-------|-------------|
| `LearningNode` | Framework node without networking (bring your own transport) |
| `GossipNode` | Production node with IPv8 P2P + automatic tunnel fallback |

### Models

| Class | Description |
|-------|-------------|
| `PyTorchModel` | Wrapper for PyTorch `nn.Module` with NaN validation, gradient clipping |
| `TensorFlowModel` | Wrapper for TensorFlow/Keras models |
| `ModelWrapper` | Base class for custom framework wrappers |
| `PersonalizedModelWrapper` | Base for APFL-style personalized models |
| `TrainingConfig` | Training configuration (epochs, batch_size, lr, grad_clip, optimizer) |

### Fingerprint

| Class | Description |
|-------|-------------|
| `DataFingerprint` | Privacy-preserving data distribution summary |
| `FingerprintComputer` | Computes fingerprints from raw data with configurable privacy |
| `AffinityWeights` | Weights for multi-signal affinity computation |
| `FingerprintPrivacyConfig` | ε-DP budget, noise levels, bucket granularity |

### Manifest & Policy

| Class | Description |
|-------|-------------|
| `DataPolicy` | Minimum affinity, privacy level, cold-start rounds |
| `CollaborationPolicy` | Aggregation and topology parameters |
| `PersonalizationPolicy` | APFL, FedRep configuration |
| `PrototypePolicy` | FedProto/FedPAC alignment settings |

### Observability

| Class | Description |
|-------|-------------|
| `EventEmitter` | Publish/subscribe runtime events |
| `RuntimeEvent` | Structured event payload |
| `TerminalObserver` | Human-readable terminal rendering |
| `TelemetryClient` | Telemetry data collection |

---

## Requirements

- Python 3.10+
- PyTorch 1.9+ (optional, for `PyTorchModel`)
- TensorFlow 2.x (optional, for `TensorFlowModel`)
- IPv8 2.0+ (for P2P networking)
- NumPy

---

## Documentation

The canonical documentation set lives under [`docs/`](docs/). Use **[`docs/index.md`](docs/index.md)** as the entry point: it has a short decision tree and a table of contents that mirrors the book layout (Sphinx toctree).

### Quick entry

| Document | Description |
|----------|-------------|
| [`docs/index.md`](docs/index.md) | Hub: decision tree and links into all sections |
| [`docs/quickstart.md`](docs/quickstart.md) | Minimal “get running” path |
| [`docs/getting-started.md`](docs/getting-started.md) | Full getting started (English) |
| [`docs/getting-started-tr.md`](docs/getting-started-tr.md) | Full getting started (Turkish) |
| [`docs/faq.md`](docs/faq.md) | Frequently asked questions |

### By section

| Section | Start here |
|---------|------------|
| User guide | [`docs/user-guide/index.md`](docs/user-guide/index.md) (manifest, peer script, trust, telemetry, troubleshooting) |
| CLI | [`docs/cli/index.md`](docs/cli/index.md) (`manifest`, `run`, `init`, `keygen`, …) |
| Tutorials | [`docs/tutorials/index.md`](docs/tutorials/index.md) (T1–T6) |
| Concepts | [`docs/concepts/index.md`](docs/concepts/index.md) (gossip, swarm, fingerprints) |
| Reference | [`docs/reference/index.md`](docs/reference/index.md) (API, manifest schema, error codes) |
| Security | [`docs/security/index.md`](docs/security/index.md) (threat model, signing, TOFU, rate limits) |
| Cookbook | [`docs/cookbook/index.md`](docs/cookbook/index.md) (local swarm, multi-peer testing, custom wrappers) |
| Migration | [`docs/migration/index.md`](docs/migration/index.md) |

---

## References

- **McMahan et al.** (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS. (FedAvg)
- **Ormándi et al.** (2013). *Gossip Learning with Linear Models on Fully Distributed Data.* Concurrency and Computation.
- **Blanchard et al.** (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.* NeurIPS. (Krum)
- **Yin et al.** (2018). *Byzantine-Robust Distributed Learning.* ICML. (TrimmedMean)
- **Li et al.** (2020). *Federated Optimization in Heterogeneous Networks.* MLSys. (FedProx)
- **Hsu et al.** (2019). *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* (FedAvgM)
- **Kang & Lee** (2024). *RNEP: Random Node Entropy Pairing for Efficient Decentralized Training with Non-IID Local Data.* Electronics, 13(21), 4193. (EntropyWeightedAvg)
- **Karimireddy et al.** (2020). *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.* ICML. (Scaffold)
- **Alistarh et al.** (2018). *The Convergence of Sparsified Gradient Methods.* NeurIPS. (Error Feedback)
- **Richtárik et al.** (2021). *EF21: A New, Simpler, Theoretically Better.* NeurIPS. (EF21 momentum)
- **Koloskova et al.** (2020). *Unified Theory of Decentralized SGD with Changing Topology and Local Updates.* ICML. (Spectral Gap)
- **Boyd et al.** (2006). *Randomized Gossip Algorithms.* IEEE Trans. Inf. Theory. (Metropolis–Hastings mixing)
- **Voulgaris et al.** (2005). *Cyclon: Inexpensive Membership Management for Unstructured P2P Overlays.* JNSM. (CyclonTopology)
- **Deng et al.** (2021). *Adaptive Personalized Federated Learning.* (APFL)

---

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2026 Ali Seyhan, Baki Turhan

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main repository.
