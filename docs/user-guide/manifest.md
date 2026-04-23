# Working with Manifests

A manifest (`.qgl` file) is the single source of truth for a swarm.  It
describes the task, model architecture, aggregation strategy, topology,
byzantine tolerance, and trust boundary.

## Creating a Manifest

```bash
quinkgl manifest create \
  --name my-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:<64-hex> \
  --aggregation FedAvg \
  --topology Random \
  --output my-swarm.qgl
```

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
