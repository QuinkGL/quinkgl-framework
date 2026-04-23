# Gossip Learning

**Gossip learning** is the peer-to-peer communication pattern at the heart of QuinkGL.

## How It Works

1. **Local Training** — each peer trains its model on local data for a few epochs
2. **Peer Selection** — the topology strategy selects random or affinity-matched peers
3. **Model Exchange** — peers send their model updates to selected peers
4. **Aggregation** — received updates are combined using the swarm's aggregation strategy
5. **Repeat** — the process continues for the configured number of rounds

## Advantages Over Centralized FL

| Aspect | Centralized FL | Gossip Learning |
|--------|---------------|-----------------|
| Server | Required | None |
| Single point of failure | Yes | No |
| Communication bottleneck | Server link | Distributed |
| Scalability | Limited by server | Organic |
| NAT traversal | Complex | Built-in (IPv8) |

## Message Types

QuinkGL peers exchange several message types:

- **Discovery Announce** — broadcast presence and compatibility
- **Model Update** — send trained model weights
- **Heartbeat** — maintain peer liveness
- **Manifest Request/Response** — fetch swarm manifests (Phase 1)

## Convergence

Gossip learning converges under mild conditions:
- The network graph remains connected over time
- Peers communicate sufficiently often
- Learning rates are appropriately scheduled

The **spectral gap** of the communication graph directly determines convergence speed:
- Larger gap → faster convergence
- `SpectralAnalyzer` measures this at runtime

## Fault Tolerance

QuinkGL handles several failure modes:

- **Peer churn** — peers joining/leaving dynamically
- **Network partitions** — temporary disconnections
- **Byzantine peers** — malicious or buggy peers sending bad updates

Byzantine strategies (Krum, MultiKrum, TrimmedMean) filter out harmful updates.
