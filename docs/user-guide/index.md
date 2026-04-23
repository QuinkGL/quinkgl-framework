# User Guide

Welcome to the QuinkGL User Guide. This section covers everything you need to create, join, and manage QuinkGL swarms.

## Contents

```{toctree}
:maxdepth: 2

manifest
peer-script
trust
telemetry
troubleshooting
```

## What is QuinkGL?

QuinkGL is a decentralized, peer-to-peer federated learning framework. Unlike traditional federated learning that relies on a central parameter server, QuinkGL uses **gossip-based protocols** where nodes exchange model updates directly with randomly selected peers.

### Key Concepts

- **Swarm** — A group of peers training the same model architecture on compatible data.
- **Manifest** — A canonical JSON document (`.qgl` file) describing the swarm's training protocol, model architecture, data schema, and policies.
- **Peer** — An individual node participating in the swarm.
- **Gossip** — The peer-to-peer communication pattern where nodes randomly exchange model updates.

## Getting Started

### Installation

```bash
pip install quinkgl
```

For development with all extras:

```bash
pip install -e ".[dev,docs]"
```

### Your First Swarm

1. **Create a manifest**:

```bash
quinkgl manifest create \
  --name my-first-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --output my-swarm.qgl
```

2. **Verify the manifest**:

```bash
quinkgl manifest verify my-swarm.qgl
```

3. **Get a magnet URI** for distribution:

```bash
quinkgl manifest magnet my-swarm.qgl
```

4. **Scaffold a peer project** (if you need custom model/data loaders):

```bash
quinkgl init --output-dir my-peer --template pytorch-vision --manifest my-swarm.qgl
cd my-peer
pytest  # verify stubs are in place
```

5. **Run a peer**:

```bash
# Mode A: built-in model loader (if available)
quinkgl run --manifest my-swarm.qgl --data ./my_data

# Mode B: custom script
quinkgl run --manifest my-swarm.qgl --script peer_script.py
```

## Next Steps

- Read [Working with Manifests](manifest.md) for detailed manifest creation and validation.
- Read [Peer Scripts](peer-script.md) to understand Mode B callable contracts.
- Read [Trust Policies](trust.md) for security configuration (Phase 2).
- Read [Telemetry](telemetry.md) for monitoring and observability.
- Read [Troubleshooting](troubleshooting.md) for common errors and solutions.
