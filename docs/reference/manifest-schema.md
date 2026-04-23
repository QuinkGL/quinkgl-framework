# Manifest Schema Reference

QuinkGL manifests are JSON objects with the following top-level fields.

## Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | `int` | yes | Currently `4` |
| `name` | `string` | yes | Swarm identifier |
| `description` | `string` | no | Human-readable summary |
| `created_at` | `string` | yes | ISO-8601 timestamp |
| `expires_at` | `string` | no | ISO-8601 expiry or `null` |
| `task` | `object` | yes | See [TaskSpec](#taskspec) |
| `model` | `object` | yes | See [ModelSpec](#modelspec) |
| `aggregation` | `object` | yes | See [Aggregation](#aggregation) |
| `topology` | `object` | yes | See [Topology](#topology) |
| `compression` | `object` | no | See [Compression](#compression) |
| `data_policy` | `object` | no | See [DataPolicy](#datapolicy) |
| `byzantine` | `object` | no | See [ByzantineSpec](#byzantinespec) |
| `round_limit` | `int` | no | Maximum rounds or `null` |
| `bootstrap_peers` | `array` | no | List of `{kind, address}` objects |
| `tracker_urls` | `array` | no | Nested array of tracker URL tiers |
| `creator_pubkey` | `string` | no | Ed25519 pubkey hex (with `ed25519:` prefix) |
| `signature` | `string` | no | Base64 Ed25519 signature |

## TaskSpec

```json
{
  "type": "classification",
  "input_shape": [3, 224, 224],
  "output_shape": [10],
  "label_type": "integer",
  "tags": ["medical", "xray"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | `classification`, `regression`, `segmentation`, `detection` |
| `input_shape` | `int[]` | Model input dimensions |
| `output_shape` | `int[]` | Model output dimensions |
| `label_type` | `string` | Label encoding hint |
| `tags` | `string[]` | Arbitrary category tags |

## ModelSpec

```json
{
  "framework": "pytorch",
  "arch_hash": "sha256:7f2c1a9b...",
  "arch_spec": null,
  "genesis_weights_hash": null,
  "genesis_weights_url": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `framework` | `string` | `pytorch`, `tensorflow`, or `custom` |
| `arch_hash` | `string` | `sha256:<64-hex>` architecture digest |
| `arch_spec` | `object` | Optional free-form architecture details |
| `genesis_weights_hash` | `string` | Optional initial weights hash |
| `genesis_weights_url` | `string` | Optional initial weights URL |

## Aggregation

```json
{
  "name": "FedAvg",
  "params": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Strategy name |
| `params` | `object` | Strategy-specific key/value map |

## Topology

```json
{
  "name": "Random",
  "params": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Strategy name |
| `params` | `object` | Strategy-specific key/value map |

## Compression

```json
{
  "enabled": false,
  "params": {}
}
```

## DataPolicy

```json
{
  "min_peers": 1,
  "max_peers": 10,
  "gradient_clip_norm": null,
  "local_dp_epsilon": null,
  "local_dp_delta": null,
  "local_dp_max_grad_norm": null,
  "share_raw_updates": true
}
```

## ByzantineSpec

```json
{
  "f": 0,
  "enforce_n_gt_2f_plus_2": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `f` | `int` | Maximum expected Byzantine peers |
| `enforce_n_gt_2f_plus_2` | `bool` | Reject swarms where `N <= 2f+2` |

## Validation Rules

- `schema_version` must be `4`.
- `model.arch_hash` must match `^sha256:[0-9a-f]{64}$`.
- `model.framework` must be one of `pytorch`, `tensorflow`, `custom`.
- `aggregation.name` and `topology.name` must be non-empty strings.
- If `signature` is present, `creator_pubkey` must also be present.

## See Also

- [Working with Manifests](../user-guide/manifest.md)
- `quinkgl.manifest.SwarmManifest` API
