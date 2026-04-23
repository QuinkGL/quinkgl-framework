# `quinkgl manifest`

Create, inspect, verify, and share swarm manifests.

## Subcommands

### `quinkgl manifest create`

Build a new `.qgl` manifest file.

| Flag | Required | Description |
|------|----------|-------------|
| `--name` | yes | Swarm name |
| `--task-type` | yes | `class`, `regr`, `seg`, or `det` |
| `--input-shape` | yes | Comma-separated shape (e.g. `3,224,224`) |
| `--output-shape` | yes | Comma-separated shape (e.g. `10`) |
| `--label-type` | yes | Label encoding name |
| `--tags` | no | Comma-separated tags |
| `--model-framework` | yes | `pytorch`, `tensorflow`, or `custom` |
| `--model-arch-hash` | yes | Architecture digest (`sha256:<64-hex>`) |
| `--model-arch-file` | no | Optional architecture spec JSON |
| `--aggregation` | yes | Aggregation strategy name |
| `--aggregation-param` | no | `key=value` strategy params (repeatable) |
| `--topology` | yes | Topology strategy name |
| `--topology-param` | no | `key=value` strategy params (repeatable) |
| `--data-policy` | no | Path to a JSON data-policy file |
| `--byzantine-f` | `0` | Byzantine peer tolerance |
| `--round-limit` | no | Maximum training rounds |
| `--expires-at` | no | ISO-8601 expiry timestamp |
| `--bootstrap-peer` | no | `host:port` bootstrap (repeatable) |
| `--tracker-tier` | no | Comma-separated tracker URLs per tier (repeatable) |
| `--sign-with` | no | Path to Ed25519 private key PEM |
| `--output` | yes | Destination `.qgl` path |

Example:

```bash
quinkgl manifest create \
  --name health-xray \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 2 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --sign-with creator.key \
  --output health-xray.qgl
```

### `quinkgl manifest show`

Pretty-print a `.qgl` file.

```bash
quinkgl manifest show my-swarm.qgl
quinkgl manifest show my-swarm.qgl --json
```

### `quinkgl manifest verify`

Validate schema, hash, and optionally signature.

| Flag | Description |
|------|-------------|
| `--trusted-pubkey` | Accept only manifests signed by this pubkey (repeatable) |
| `--expected-swarm-id` | Fail if manifest hash does not match |

```bash
quinkgl manifest verify my-swarm.qgl
quinkgl manifest verify my-swarm.qgl --trusted-pubkey ed25519:3b6a...
```

### `quinkgl manifest magnet`

Derive a magnet URI from a `.qgl` file.

```bash
quinkgl manifest magnet my-swarm.qgl
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation error (bad schema, missing keys) |
| `2` | I/O error (file not found) |
| `3` | Crypto error (signature failure) |
| `4` | Trust error (creator not in trusted set) |
| `5` | Hash mismatch |

## See Also

- [Working with Manifests](../user-guide/manifest.md)
- [Trust Policies](../user-guide/trust.md)
