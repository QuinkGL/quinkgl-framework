# Tutorial T4 — Writing a Custom Peer Script for PyTorch Models

This tutorial covers the Mode-B peer script API: `build_model`,
`build_loaders`, optional `build_optimizer` / `build_scheduler`, and
lifecycle hooks.

## Prerequisites

- PyTorch installed (`pip install torch`)
- A manifest that declares `"model_framework": "pytorch"`
- Familiarity with [Tutorial T1](../T1/index.md)

## Step 1: Scaffold the Script

```bash
quinkgl init --output-dir t4-peer --template pytorch-vision --manifest my-swarm.qgl
cd t4-peer
```

Open `peer_script.py`.  You will see four stub functions.

## Step 2: Implement `build_model`

Return a `torch.nn.Module` (or a `PyTorchModel` wrapper).  The CLI will
auto-wrap bare modules for you.

```python
import torch
import torch.nn as nn

def build_model(manifest, **kwargs):
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Linear(16 * 112 * 112, 10)

        def forward(self, x):
            x = self.conv(x)
            return self.fc(x.view(x.size(0), -1))

    return TinyCNN()
```

## Step 3: Implement `build_loaders`

Return `(train_loader, val_loader)` or just `train_loader`.

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

def build_loaders(manifest, **kwargs):
    # In production, load your real dataset here.
    features = torch.randn(128, 3, 224, 224)
    labels = torch.randint(0, 10, (128,))
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return train_loader, None
```

## Step 4: Optional — Custom Optimizer & Scheduler

If you need control over the optimizer, export `build_optimizer`:

```python
def build_optimizer(manifest, model):
    import torch
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def build_scheduler(optimizer, manifest):
    import torch
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

The CLI injects the returned optimizer into `TrainingConfig` and calls
`scheduler.step()` once per gossip round.

## Step 5: Optional — Lifecycle Hooks

Attach callbacks for debugging or logging:

```python
def on_round_end(round_idx, metrics):
    loss = metrics.get("loss")
    print(f"Round {round_idx}: loss={loss:.4f}")

def on_model_received(peer_id, round_number):
    print(f"Received update from {peer_id} at round {round_number}")
```

## Step 6: Run the Peer

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --checkpoint-dir ./checkpoints
```

## Next Steps

- **Tutorial T5** — Test multiple peers locally before going to production
- [Peer Script Guide](../../user-guide/peer-script.md) — Full API reference
- [Cookbook: Custom Model Wrapper](../../cookbook/custom-model-wrapper.md)
