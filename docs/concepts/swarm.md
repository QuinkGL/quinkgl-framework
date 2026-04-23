# Swarms

A **swarm** is a group of peers that collaboratively train a machine learning model using the QuinkGL gossip protocol.

## What Defines a Swarm?

A swarm is identified by its **manifest** — a canonical JSON document that commits to:

- **Model architecture** — framework, input/output shapes, architecture hash
- **Task specification** — classification, regression, segmentation, or detection
- **Aggregation strategy** — how peer updates are combined (FedAvg, Krum, etc.)
- **Topology strategy** — how peers are selected (Random, Cyclon, Affinity)
- **Data policy** — privacy settings, fingerprinting rules, collaboration parameters
- **Byzantine tolerance** — maximum number of faulty peers tolerated

## Swarm Identity

Every swarm has a unique **swarm ID** derived from the manifest's canonical bytes:

```
swarm_id = SHA-256(canonical_bytes(manifest))
```

The swarm ID is:
- **Deterministic** — same manifest always produces the same ID
- **Immutable** — any change to the manifest changes the ID
- **Verifiable** — peers can recompute and compare

## Community ID

For IPv8 networking, the swarm ID is truncated to 20 bytes to form the **community ID**:

```
community_id = swarm_id[:20]
```

Only peers with the same community ID can discover and communicate with each other.

## Lifecycle

```
INIT → MANIFEST_RESOLVED → COMMUNITY_STARTED → PEERS_DISCOVERED → TRAINING
```

1. **INIT** — peer starts, manifest is loaded
2. **MANIFEST_RESOLVED** — manifest validated, swarm ID computed
3. **COMMUNITY_STARTED** — IPv8 community joined, DHT announced
4. **PEERS_DISCOVERED** — first compatible peer found
5. **TRAINING** — gossip training loop active

## Joining a Swarm

Peers can join a swarm in three ways:

### Mode A — Standard Model Loader

```bash
quinkgl run --manifest swarm.qgl --data ./my_data
```

The framework builds the model from the manifest's `model.arch_spec` and loads data from the given directory.

### Mode B — User Script

```bash
quinkgl run --manifest swarm.qgl --script peer_script.py --script-arg epochs=10
```

The user script provides `build_model()` and `build_loaders()` callables.

### Mode C — Direct Python

```python
import asyncio
from quinkgl import GossipNode
from quinkgl.manifest import SwarmManifest

async def main():
    manifest = SwarmManifest.from_file("swarm.qgl")
    node = GossipNode(node_id="alice", manifest=manifest, model=my_model)
    await node.start()
    await node.train(rounds=1000)
    await node.stop()

asyncio.run(main())
```

## Leaving a Swarm

Peers leave gracefully by:
1. Stopping the training loop
2. Closing IPv8 connections
3. Removing DHT announcements

Use `Ctrl-C` (SIGINT) for graceful shutdown:

```bash
quinkgl run --manifest swarm.qgl --data ./my_data
# Press Ctrl-C to stop
```
