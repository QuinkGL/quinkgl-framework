# QuinkGL Getting Started Guide

This guide walks you through creating your first decentralized learning swarm with QuinkGL. It answers every practical question: what is a manifest, what is a creator key, how to get the model hash, do you need a script, and how to connect real data.

---

## 1. Core Concepts

### 1.1 What Is a Swarm?

A **swarm** is a group of peers that share the same manifest (training protocol) and communicate directly P2P to train a model together. There is no central parameter server; each peer sends its model weights to others and receives weights in return.

### 1.2 What Is a Manifest?

A **manifest** (`.qgl` file) is the "constitution" of the swarm. It contains:

- Task type (classification, regression, segmentation, detection)
- Model architecture hash (which architectures are allowed)
- Aggregation strategy (FedAvg, EntropyWeightedAvg, Krum, etc.)
- Topology strategy (RandomTopology, AffinityTopology, CyclonTopology)
- Data policy (fingerprint, privacy level, collaboration mode)
- Task shape (input/output shape, label type)
- Creator signature (proof of who created the manifest)

> **Important:** The manifest itself is **not** a swarm. It is a blueprint that defines the rules for joining a swarm. The swarm is the actual running peers that use the manifest.

### 1.3 What Is a Creator Key and Why Do You Need It?

A **creator key** is the **Ed25519 private key** of the person or organization that signs the manifest.

**What is it used for?**

1. **Identity verification:** The `creator_pubkey` field is written into the manifest. Peers verify that the manifest was signed by this key.
2. **Trust On First Use (TOFU):** When `--trust-policy tofu` is used, a peer caches the creator pubkey the first time it sees a manifest. If the same manifest name later appears with a different creator key, the peer rejects it.
3. **Pinning:** When `--trust-policy pinned` is used, a peer only accepts manifests signed by the specific `--trusted-pubkey` you provide.

**When should you use it?**

- **Always use it in production.** Otherwise anyone can create a fake manifest and join your swarm.
- For local tests (`--trust-policy open` or `tofu`) you can skip keygen, but the manifest will remain unsigned.

### 1.4 Is a Script (peer_script.py) Mandatory?

**Yes, in Mode B it is mandatory.** QuinkGL has operating modes:

- **Mode A:** You use `--data` with a standard model and dataset. (Limited support at the moment.)
- **Mode B:** You use `--script` to define your own model, data loaders, and optimizer. **This is the real-world usage.**

Required functions in `peer_script.py`:

```python
def build_model(manifest, **kwargs):
    ...

def build_loaders(manifest, **kwargs):
    ...  # returns (train_loader, val_loader) tuple
```

Optional functions:

```python
def build_optimizer(manifest, model):
    ...

def on_round_end(round_idx, metrics):
    ...

def on_peer_discovered(peer_id):
    ...

def on_aggregation_done(peer_ids, sample_count):
    ...
```

---

## 2. End-to-End Example

The following example sets up a 5-peer local test swarm. Each peer trains on its own device and shares models with the others. We use **real torchvision datasets** (MNIST and CIFAR-10) instead of mock data.

### 2.1 Generate a Creator Key

```bash
quinkgl keygen --output creator.key
```

Output:
```
Private key written with 0600 permissions. Treat this file as a secret...
ed25519:bfc5819e0264e22be8f1363794aa152a468123f7a797fa57decd88bdd21c0518
```

> **Security:** Never commit `creator.key` to git. Add it to `.gitignore`.

### 2.2 Get the Model Architecture Hash

The model architecture hash is the **SHA-256** of your model's structure. It tells the swarm "only models with this architecture are allowed."

**What is hashed?**

- Layer types and order
- Dimensions (input/output feature counts)
- Activation functions
- **Not the weights** — only the structure

**How to get it:**

QuinkGL provides a built-in helper:

```python
from quinkgl.manifest import compute_arch_hash
import torch.nn as nn

class MyModel(nn.Module):
    ...

model = MyModel()
arch_hash = compute_arch_hash(model)
print(arch_hash)  # sha256:40f4a106...
```

Or from the CLI you can compute it inline:

```bash
python -c "
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel
print(compute_arch_hash(MyModel()))
"
```

> **Important:** If you change the model architecture (e.g., add a new layer), the hash changes and the old manifest becomes invalid. You must create a new manifest.

### 2.3 Create the Manifest

```bash
quinkgl manifest create \
  --name demo-5peer \
  --task-type class \
  --input-shape 1,28,28 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:40f4a106862aa557fdbeb62a0daaa87f2b031acf93a2f9d2028e481c9607b3a5 \
  --aggregation EntropyWeightedAvg \
  --topology AffinityTopology \
  --sign-with creator.key \
  --output demo.qgl
```

**Parameter meanings:**

| Parameter | Description |
|-----------|-------------|
| `--name` | Name of the swarm |
| `--task-type` | Task: `class` (classification), `regr`, `seg`, `det` |
| `--input-shape` | Model input: channels, height, width |
| `--output-shape` | Model output: number of classes |
| `--label-type` | Label type: `integer`, `float`, `one_hot` |
| `--model-framework` | `pytorch`, `tensorflow`, `custom` |
| `--model-arch-hash` | SHA-256 hash of the model architecture |
| `--aggregation` | Aggregation strategy |
| `--topology` | Peer selection strategy |
| `--sign-with` | Creator private key (PEM file) |
| `--output` | Output manifest file |

**Optional parameters:**

```bash
  --round-limit 100              # Maximum number of rounds
  --byzantine-f 1                # Number of Byzantine peers (tolerance)
  --expires-at 2025-12-31        # Manifest expiration date
  --bootstrap-peer 192.168.1.5:7001  # Bootstrap peers
```

### 2.4 Verify the Manifest

```bash
quinkgl manifest verify demo.qgl --trusted-pubkey ed25519:bfc5819e...
```

Output:
```
Manifest valid.
Swarm ID: sha256:3a8f2e...
Signature: valid
```

### 2.5 Write the Peer Script

#### Example A: MNIST (28×28 grayscale, 10 classes)

`mnist_peer_script.py`:

```python
"""Peer script for MNIST classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(manifest, **kwargs):
    return MNISTNet()


def build_loaders(manifest, **kwargs):
    batch_size = int(kwargs.get("batch_size", 32))
    data_root = kwargs.get("data_root", "./data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    val_ds = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_optimizer(manifest, model):
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def on_round_end(round_idx, metrics):
    loss = metrics.get("loss")
    acc = metrics.get("val_accuracy") or metrics.get("accuracy")
    tag = f"round={round_idx:03d}"
    if loss is not None:
        tag += f" loss={loss:.4f}"
    if acc is not None:
        tag += f" acc={acc:.3f}"
    print(tag, flush=True)


def on_peer_discovered(peer_id):
    print(f"[peer-discovered] {peer_id}", flush=True)


def on_aggregation_done(peer_ids, sample_count):
    print(f"[aggregated] peers={list(peer_ids)} samples={sample_count}", flush=True)
```

#### Example B: CIFAR-10 (32×32 RGB, 10 classes)

`cifar10_peer_script.py`:

```python
"""Peer script for CIFAR-10 classification."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(manifest, **kwargs):
    return CIFAR10Net()


def build_loaders(manifest, **kwargs):
    batch_size = int(kwargs.get("batch_size", 32))
    data_root = kwargs.get("data_root", "./data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    val_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_optimizer(manifest, model):
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def on_round_end(round_idx, metrics):
    loss = metrics.get("loss")
    acc = metrics.get("val_accuracy") or metrics.get("accuracy")
    tag = f"round={round_idx:03d}"
    if loss is not None:
        tag += f" loss={loss:.4f}"
    if acc is not None:
        tag += f" acc={acc:.3f}"
    print(tag, flush=True)


def on_peer_discovered(peer_id):
    print(f"[peer-discovered] {peer_id}", flush=True)


def on_aggregation_done(peer_ids, sample_count):
    print(f"[aggregated] peers={list(peer_ids)} samples={sample_count}", flush=True)
```

### 2.6 Start a Single Peer

Open a terminal and run a single peer directly:

```bash
quinkgl run \
  --manifest demo.qgl \
  --script mnist_peer_script.py \
  --node-id peer-1 \
  --port 7001 \
  --rounds 15 \
  --gossip-interval 12.0 \
  --trust-policy tofu \
  --script-arg data_root=./data \
  --checkpoint-dir ./ckpt/peer-1
```

**What this does:**
- Loads `demo.qgl` manifest
- Loads model and data from `mnist_peer_script.py`
- Starts IPv8 on UDP port 7001
- Runs 15 gossip learning rounds
- Saves checkpoints to `./ckpt/peer-1`
- Prints round metrics to stdout

**First run note:** `torchvision` will auto-download MNIST/CIFAR-10 to `./data` on first launch.

**To run a second peer** (open a second terminal):

```bash
quinkgl run \
  --manifest demo.qgl \
  --script mnist_peer_script.py \
  --node-id peer-2 \
  --port 7002 \
  --rounds 15 \
  --gossip-interval 12.0 \
  --trust-policy tofu \
  --script-arg data_root=./data \
  --checkpoint-dir ./ckpt/peer-2
```

### 2.7 Monitor the Run

Since the peer prints directly to stdout, you see output live in the terminal. To save logs to a file while still watching:

```bash
quinkgl run \
  --manifest demo.qgl \
  --script mnist_peer_script.py \
  --node-id peer-1 \
  --port 7001 \
  --rounds 15 \
  --gossip-interval 12.0 \
  --trust-policy tofu \
  --script-arg data_root=./data \
  --checkpoint-dir ./ckpt/peer-1 2>&1 | tee peer-1.log
```

Then in another terminal:

```bash
# Aggregation logs
grep 'aggregated models' peer-1.log

# Per-round accuracy
grep 'round=' peer-1.log

# Discovered peers
grep 'peer-discovered' peer-1.log

# Stop all peers
pkill -f 'quinkgl run'
```

---

## 3. Hashes in Depth

### 3.1 Model Architecture Hash (`model_arch_hash`)

**What is hashed?**

The **structure** of the model. Weights are **not** included; only:
- Layer types and order
- Dimensions (input/output feature count)
- Activation functions

**Why does it matter?**

All peers in a swarm must use the same architecture. Otherwise aggregation (weight averaging) is meaningless.

**How to get it:**

Use QuinkGL's built-in helper:

```python
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel

model = MyModel()
arch_hash = compute_arch_hash(model)
print(arch_hash)  # sha256:...
```

### 3.2 Data Schema Hash (`data_schema_hash`)

**What is hashed?**

The shape of the dataset: input size, channel count, label type, etc.

**Why does it matter?**

Peers automatically reject other peers whose data schema does not match (security + compatibility).

**How to get it:**

QuinkGL generates it automatically. If you want to set it manually:

```python
from quinkgl.models import PyTorchModel

model_wrapper = PyTorchModel(MyModel())
schema_hash = model_wrapper.get_data_schema_hash()
print(schema_hash)  # sha256:0000... format
```

---

## 4. Trust Policies

| Policy | Behavior | Use case |
|--------|----------|----------|
| `open` | Accepts every manifest, no signature check | Quick tests |
| `tofu` | Caches the first seen creator key; rejects if it changes later | Production (recommended) |
| `pinned` | Only accepts manifests signed by the specific `--trusted-pubkey` | High security |

**TOFU example:**

```bash
quinkgl run --manifest demo.qgl --trust-policy tofu ...
```

**Pinned example:**

```bash
quinkgl run --manifest demo.qgl \
  --trust-policy pinned \
  --trusted-pubkey ed25519:bfc5819e0264... \
  ...
```

---

## 5. Frequently Asked Questions

**Q: I changed the manifest. Will old peers accept the new one?**

A: The manifest hash (swarm ID) changes, so old peers see the new manifest as a different swarm. You must restart peers with the new manifest.

**Q: Can I create multiple manifests with the same creator key?**

A: Yes. You can create a separate manifest for each different training task.

**Q: I changed my model architecture but don't want to update the manifest.**

A: You can run with `--strict-manifest false`, but this is **not recommended**. You may encounter aggregation errors or security issues.

**Q: My dataset is huge. Do I have to copy it to every peer?**

A: No. Each peer has its own local data. This is the essence of federated learning. The `build_loaders` function on each peer loads data from its own path.

**Q: When is `on_aggregation_done` called?**

A: It is called when a peer receives models from other peers and completes aggregation. If you see `[aggregated] peers=[...]` in the logs, aggregation succeeded.

**Q: What happens if I set port to 0?**

A: The operating system assigns a random free port. Use a fixed port in production; it is needed for discovery.

---

## 6. Quick Reference Card

```bash
# Generate a key
quinkgl keygen --output creator.key

# Compute model architecture hash
python -c "
from quinkgl.manifest import compute_arch_hash
from my_model import MyModel
print(compute_arch_hash(MyModel()))
"

# Create a manifest
quinkgl manifest create \
  --name <name> --task-type class \
  --input-shape <C,H,W> --output-shape <classes> \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:<hash> \
  --aggregation <FedAvg|EntropyWeightedAvg|...> \
  --topology <RandomTopology|AffinityTopology|...> \
  --sign-with creator.key --output swarm.qgl

# Verify the manifest
quinkgl manifest verify swarm.qgl --trusted-pubkey ed25519:<pubkey>

# Start a peer
quinkgl run --manifest swarm.qgl --script peer_script.py \
  --node-id peer-1 --port 7001 \
  --trust-policy tofu \
  --script-arg data_root=./data

# Show info
quinkgl info
```

---

*This guide is written for QuinkGL v0.3.3.*
