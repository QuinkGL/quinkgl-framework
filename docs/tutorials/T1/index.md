# Tutorial T1 — Your First QuinkGL Swarm in 5 Minutes

This tutorial walks you through creating your first swarm and running a peer.

## Prerequisites

- Python 3.10+
- QuinkGL installed: `pip install quinkgl`

## Step 1: Create a Manifest

A manifest describes your swarm's training protocol:

```bash
quinkgl manifest create \
  --name t1-demo \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --output t1-demo.qgl
```

## Step 2: Verify the Manifest

```bash
quinkgl manifest verify t1-demo.qgl
```

Expected output: `Manifest is valid.`

## Step 3: Get a Magnet URI

```bash
quinkgl manifest magnet t1-demo.qgl
```

This prints a shareable `quinkgl:?xt=urn:qgl:...` URI.

## Step 4: Scaffold a Peer Project

```bash
quinkgl init --output-dir t1-peer --template minimal --manifest t1-demo.qgl
cd t1-peer
```

Inspect the generated files:

```bash
ls -la
# peer_script.py    — build_model and build_loaders stubs
# peer_main.py      — canonical Mode C script
# tests/            — pytest tests
# pyproject.toml    — project config
```

## Step 5: Run Tests

```bash
pytest
```

Expected: 2 tests pass (stub NotImplementedError tests), 1 passes (integration placeholder).

## Next Steps

- **Tutorial T2** — Creating and publishing a signed swarm (Phase 2)
- **Tutorial T4** — Writing a custom peer script for PyTorch models
- Read the [User Guide](../../user-guide/index.md) for detailed concepts
