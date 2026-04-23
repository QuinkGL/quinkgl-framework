# Peer Scripts

A peer script is a plain Python file that exports callables consumed by
`quinkgl run --script <path>` (Mode B).  The CLI imports the module,
validates the required exports, and wires them into the `GossipNode`
lifecycle.

## Required Exports

### `build_model(manifest, **kwargs) -> model`

Return a model instance.  For PyTorch this is typically a `torch.nn.Module`;
for TensorFlow a `tf.keras.Model`.  The CLI auto-wraps bare framework models
into the appropriate `ModelWrapper` subclass, so you do not need to import
`PyTorchModel` unless you want custom behaviour.

```python
import torch.nn as nn

def build_model(manifest, **kwargs):
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
```

### `build_loaders(manifest, **kwargs) -> (train_loader, val_loader)`

Return data loaders.  The training loader is required; the validation loader
may be `None`.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

def build_loaders(manifest, **kwargs):
    x = torch.randn(128, 784)
    y = torch.randint(0, 10, (128,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16), None
```

## Optional Exports

### `build_optimizer(manifest, model) -> optimizer`

Override the default optimizer (Adam).  Return a concrete optimizer instance.

```python
import torch

def build_optimizer(manifest, model):
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### `build_scheduler(optimizer, manifest) -> scheduler`

Return an LR scheduler.  Its `.step()` method is called once per gossip round.

```python
def build_scheduler(optimizer, manifest):
    import torch
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### Lifecycle Hooks

Attach callbacks for observability or side effects:

| Hook | Signature | When |
|------|-----------|------|
| `on_round_end` | `(round_idx, metrics) -> None` | After each gossip round |
| `on_model_received` | `(peer_id, round_number) -> None` | After receiving a remote update |
| `on_aggregation_done` | `(peer_ids, sample_count) -> None` | After local aggregation |
| `on_peer_discovered` | `(peer_id) -> None` | When a new peer is discovered |
| `on_fingerprint_ready` | `(fingerprint) -> None` | When local data fingerprint is computed |

```python
def on_round_end(round_idx, metrics):
    print(f"Round {round_idx}: loss={metrics.get('loss')}")
```

## Script Arguments

Pass extra arguments via `--script-arg`:

```bash
quinkgl run ... --script-arg data_dir=/tmp/data --script-arg epochs=5
```

These arrive as `kwargs` in `build_model`, `build_loaders`, and optional
`build_optimizer` / `build_scheduler`.

Reserved keys (`node_id`, `manifest`, `trust_policy`, `trusted_creator_pubkeys`)
are rejected by the CLI to prevent accidental shadowing.

## Testing

The `quinkgl.testing` module provides `local_swarm_fixture` for end-to-end
peer tests:

```python
from quinkgl.testing import local_swarm_fixture

def test_peer():
    with local_swarm_fixture(
        manifest_path="demo.qgl",
        peer_script_path="peer_script.py",
        num_peers=3,
        rounds=5,
    ) as swarm:
        assert all(n.gl_node.current_round >= 5 for n in swarm.nodes)
```

## See Also

- [Tutorial T4](../tutorials/T4/index.md) — Full PyTorch walkthrough
- [CLI Reference: run](../cli/run.md)
- [Cookbook: Custom Model Wrapper](../cookbook/custom-model-wrapper.md)
