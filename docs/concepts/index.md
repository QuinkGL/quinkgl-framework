# Concepts

Core concepts behind the QuinkGL framework.

```{toctree}
:maxdepth: 1

swarm
gossip
fingerprint
```

## Swarm

A **swarm** is a decentralised federation of peers that agree to train the
same model architecture on their local data and periodically exchange weight
updates.  Membership is defined by a [swarm manifest](swarm.md) rather than
by a central coordinator.

## Gossip Learning

**Gossip learning** is the asynchronous, P2P variant of federated learning
used by QuinkGL.  Instead of a parameter server, each peer:

1. Trains locally for one round.
2. Selects a random subset of neighbours (gossip targets).
3. Sends its model update to those targets.
4. Receives incoming updates from other peers.
5. Aggregates received updates into its local model.

See [Gossip Learning](gossip.md) for the full protocol.

## Data Fingerprint

A **data fingerprint** is a privacy-preserving sketch of a peer's local
dataset.  It is used to compute affinity scores between swarms and local
data without exposing raw records.  See [Data Fingerprints](fingerprint.md).

## Trust Policies

Peers decide whether to accept a manifest based on three trust policies:

- **open** — accept anything (dev only)
- **tofu** — trust the first creator pubkey seen for a swarm hash
- **pinned** — only accept explicitly listed creator pubkeys

See [Trust Policies](../user-guide/trust.md) for details.

## Model Wrappers

QuinkGL wraps raw framework models (`nn.Module`, `tf.keras.Model`) in a
`ModelWrapper` that provides `get_weights`, `set_weights`, `train`, and
`evaluate`.  This lets the framework checkpoint, aggregate, and resume models
without knowing framework internals.

See [Peer Scripts](../user-guide/peer-script.md) and
[Custom Model Wrapper](../cookbook/custom-model-wrapper.md).
