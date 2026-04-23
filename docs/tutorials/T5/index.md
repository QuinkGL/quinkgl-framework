# Tutorial T5 — Local Multi-Peer Testing with quinkgl.testing

This tutorial shows how to spin up three peers on a single machine, verify
that gossip learning converges, and tear everything down cleanly.

## Prerequisites

- QuinkGL installed with test extras: `pip install quinkgl[test]`
- A manifest and peer script (see [Tutorial T4](../T4/index.md))

## Step 1: Write the Test

```python
# tests/test_local_swarm.py
import pytest
from quinkgl.testing import local_swarm_fixture

@pytest.fixture
def swarm(manifest_path, peer_script_path):
    yield from local_swarm_fixture(
        manifest_path=manifest_path,
        peer_script_path=peer_script_path,
        num_peers=3,
        rounds=5,
    )

class TestLocalSwarm:
    def test_all_peers_reach_round_five(self, swarm):
        for node in swarm.nodes:
            assert node.gl_node.current_round >= 5

    def test_at_least_one_aggregation_occurred(self, swarm):
        updates = sum(
            len(n.gl_node.aggregator.known_peers)
            for n in swarm.nodes
        )
        assert updates > 0
```

## Step 2: Understand What `local_swarm_fixture` Does

1. Creates `num_peers` `GossipNode` instances in separate IPv8 overlays
2. Connects them on `127.0.0.1` with auto-incremented ports
3. Runs the gossip loop for `rounds` iterations
4. Yields a `SwarmTestbed` object with `.nodes`, `.manifest`, and `.metrics`
5. Tears down IPv8, tunnels, and temporary directories on exit

## Step 3: Run the Test

```bash
pytest tests/test_local_swarm.py -v
```

Expected: 2 passed, 0 failed.

## Step 4: Inspect Metrics

The fixture attaches a `MetricsCollector` to every node.  After the test:

```python
def test_loss_decreases(self, swarm):
    for node in swarm.nodes:
        losses = node._test_metrics.get("loss", [])
        if len(losses) >= 2:
            assert losses[-1] <= losses[0]  # rough sanity check
```

## Step 5: Custom Peer Arguments

Pass extra script arguments through the fixture:

```python
yield from local_swarm_fixture(
    manifest_path=manifest_path,
    peer_script_path=peer_script_path,
    num_peers=3,
    rounds=5,
    script_args={"data_dir": "/tmp/fake_data"},
)
```

## Step 6: Debugging a Failing Swarm

If a peer crashes inside the fixture, the exception is re-raised after
teardown so you can read the traceback.  Enable debug logging:

```bash
pytest tests/test_local_swarm.py -v --log-cli-level=debug
```

## Next Steps

- **Tutorial T6** — Streaming metrics to a telemetry server
- [Cookbook: Multi-Peer Testing](../../cookbook/multi-peer-testing.md)
- [Testing Fixtures API](../../reference/api/index.md)
