# Recipe: Running a Local Swarm

Spin up a complete 3-peer gossip-learning swarm on `localhost` for rapid
iteration before deploying to real machines.

## One-Liner

```bash
quinkgl init --output-dir local-swarm --template pytorch-vision --manifest my-swarm.qgl
cd local-swarm
pytest -v
```

The generated `tests/` folder already contains a `local_swarm_fixture` test
that runs 3 peers for 5 rounds.

## Manual Control

If you want to drive the peers from a Python script instead of pytest:

```python
from quinkgl.testing import local_swarm_fixture

with local_swarm_fixture(
    manifest_path="my-swarm.qgl",
    peer_script_path="peer_script.py",
    num_peers=3,
    rounds=10,
) as swarm:
    for node in swarm.nodes:
        print(node.node_id, node.gl_node.current_round)
```

## Port Allocation

`local_swarm_fixture` automatically picks free ports starting at `7000`.  If
you already have services on those ports, set an offset:

```python
local_swarm_fixture(..., base_port=8000)
```

## Checkpoints

Each peer in the fixture writes checkpoints to a temporary directory.  You
can inspect the latest checkpoint after the test:

```python
latest = swarm.nodes[0].gl_node.aggregator.model_store.get_latest_checkpoint()
print(latest.round_number, latest.metrics)
```

## Clean Teardown

The fixture uses `yield` + context-manager semantics.  Even if a test
assertion fails or an exception is raised, IPv8 communities, sockets, and
temporary directories are cleaned up automatically.

## See Also

- [Tutorial T5](../tutorials/T5/index.md) — Full walkthrough with assertions
- [Multi-Peer Testing](multi-peer-testing.md) — Advanced scenarios (Byzantine peers, churn)
