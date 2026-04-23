# Recipe: Multi-Peer Testing

Beyond the basic 3-peer fixture, this recipe covers adversarial testing,
peer churn, and partition healing.

## Testing a Byzantine Peer

Inject a peer that sends poisoned updates:

```python
from quinkgl.testing import local_swarm_fixture

def poisoned_script():
    """Return a build_model whose weights are all NaN after round 3."""
    # See Tutorial T4 for the normal script structure.
    pass

with local_swarm_fixture(
    manifest_path="my-swarm.qgl",
    peer_script_path="peer_script.py",
    num_peers=4,
    rounds=10,
    byzantine_indices={3},  # peer #3 is malicious
) as swarm:
    honest = [swarm.nodes[i] for i in range(3)]
    # Assert honest peers still converge despite the malicious update.
```

## Simulating Peer Churn

Kill a peer mid-run and verify the remaining peers continue:

```python
import asyncio

with local_swarm_fixture(...) as swarm:
    # Let peer 2 run for 3 rounds, then stop it.
    await asyncio.sleep(5)
    await swarm.nodes[2].shutdown()

    # The other peers should still reach round 10.
    await swarm.wait_for_round(10)
```

## Network Partition

Block gossip between two groups and later heal the partition:

```python
with local_swarm_fixture(num_peers=4, rounds=20) as swarm:
    # Partition: {0,1} <-> {2,3}
    swarm.partition([{0, 1}, {2, 3}])
    await swarm.wait_for_round(5)

    # Heal
    swarm.heal_partition()
    await swarm.wait_for_round(10)
```

## Custom Assertions

Collect per-peer loss curves and assert statistical properties:

```python
def test_variance_reduces(swarm):
    losses = [
        n.gl_node.aggregator.last_loss
        for n in swarm.nodes
        if n.gl_node.aggregator.last_loss is not None
    ]
    assert len(losses) == len(swarm.nodes)
    assert max(losses) - min(losses) < 0.5  # rough consensus
```

## See Also

- [Tutorial T5](../tutorials/T5/index.md) — Basic fixture usage
- [Running a Local Swarm](running-local-swarm.md) — One-liner quick start
