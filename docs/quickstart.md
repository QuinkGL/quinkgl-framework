# Quickstart

Get a QuinkGL swarm running in under five minutes.

## Install

```bash
pip install quinkgl
```

## Create a Manifest

```bash
quinkgl manifest create \
  --name demo \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --output demo.qgl
```

## Verify

```bash
quinkgl manifest verify demo.qgl
```

## Scaffold a Peer

```bash
quinkgl init --output-dir my-peer --template minimal --manifest demo.qgl
cd my-peer
```

## Run (Dry Mode)

```bash
quinkgl run --manifest demo.qgl --dry-run
```

## Next Steps

- [Tutorial T1](tutorials/T1/index.md) — Full walkthrough
- [CLI Reference](reference/cli-reference.md) — All commands
- [User Guide](user-guide/index.md) — Concepts and trust policies
