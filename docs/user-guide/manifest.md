# Working with Manifests

A manifest (`.qgl` file) is the single source of truth for a swarm.  It
describes the task, model architecture, aggregation strategy, topology,
byzantine tolerance, and trust boundary.

## Creating a Manifest

### Computing the architecture hash

`--model-arch-hash` requires a `sha256:<64-hex>` fingerprint of your model's
architecture (layer names, shapes, and dtypes).  **It does not include
weights.**

If you already have a model definition, open a terminal in the same directory as your model file and run:

```bash
python3 -c "
import torch.nn as nn
from quinkgl.manifest import compute_arch_hash

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x)

print(compute_arch_hash(MyModel()))
"
```

The terminal will print the hash. Copy the full `sha256:…` string and pass it to `--model-arch-hash`.

If your model lives in a separate file, you can import it directly:

```bash
python3 -c "
from peer_script import build_model
from quinkgl.manifest import compute_arch_hash
from quinkgl.manifest import SwarmManifest

model = build_model(SwarmManifest())
print(compute_arch_hash(model))
"
```

If you do **not** have a model yet, you can start with a dummy hash and
regenerate the manifest later when the model is ready:

```bash
quinkgl manifest create \
  --name my-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:0000000000000000000000000000000000000000000000000000000000000000 \
  --aggregation FedAvg \
  --topology Random \
  --output my-swarm.qgl
```

### Building the file

```bash
quinkgl manifest create \
  --name my-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --output my-swarm.qgl
```

## Manifest vs. Swarm

Creating a manifest does **not** start a swarm. The manifest is only a
static blueprint. A swarm comes into existence the moment the first peer
calls `quinkgl run` with that manifest.

Typical workflow:

1. **Design the model** (or prototype it).
2. **Compute the architecture hash** with `compute_arch_hash`.
3. **Create the manifest** (`quinkgl manifest create …`).
4. **Share the manifest** (file, magnet URI, or directory advertisement).
5. **Peers join** by running `quinkgl run --manifest my-swarm.qgl …`.

If you are iterating quickly and do not have the final model yet, you can
use a dummy hash (`sha256:0…0`) in step 3, then regenerate the manifest
with the real hash before step 5.

## Signing

Generate a key and sign at creation time:

```bash
quinkgl keygen --output creator.key
quinkgl manifest create ... --sign-with creator.key --output signed.qgl
```

## Verifying

```bash
quinkgl manifest verify signed.qgl
quinkgl manifest verify signed.qgl --trusted-pubkey ed25519:<hex>
```

## Inspecting

```bash
quinkgl manifest show my-swarm.qgl
quinkgl manifest show my-swarm.qgl --json
```

## Magnet URI

Derive a shareable URI:

```bash
quinkgl manifest magnet my-swarm.qgl
```

Peers can fetch the manifest directly from a magnet URI when the directory
infrastructure is available.

## Schema Highlights

| Field | Purpose |
|-------|---------|
| `name` | Human-readable swarm identifier |
| `task` | Input/output shapes, label type, tags |
| `model` | Framework tag + architecture hash |
| `aggregation` | Strategy name and params (e.g. `FedAvg`) |
| `topology` | Peer-selection strategy (e.g. `Random`) |
| `byzantine` | `f` tolerance and enforcement flags |
| `round_limit` | Hard cap on training rounds |
| `data_policy` | Privacy constraints (gradient clipping, DP) |
| `creator_pubkey` / `signature` | Ed25519 identity (Phase 2) |

## Distribution

After creation you can:

1. Share the `.qgl` file directly.
2. Share the magnet URI.
3. Publish a signed `SwarmAdvertisement` via `quinkgl publish`.

## See Also

- [Manifest Schema Reference](../reference/manifest-schema.md)
- [Trust Policies](trust.md)
- [Signing Reference](../security/signing.md)
- `quinkgl manifest --help`
