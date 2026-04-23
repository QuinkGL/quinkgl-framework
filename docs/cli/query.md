# `quinkgl query`

Query a local swarm advertisement cache (Phase 3).

## Synopsis

```
quinkgl query --cache <path> [--tag <tag>] [--tags <csv>]
              [--input-shape <shape>] [--label-type <type>]
              [--trusted-pubkey <hex>]
```

## Description

Filters a JSON cache of `SwarmAdvertisement` objects by tags, shape, label type, or trusted creator.

## Flags

| Flag | Type | Required | Description |
|---|---|---|---|
| `--cache` | path | YES | JSON cache file path |
| `--tag` | string | NO | Repeatable tag filter |
| `--tags` | string | NO | Comma-separated tags |
| `--input-shape` | string | NO | e.g. `3,224,224` |
| `--label-type` | string | NO | `integer`, `float`, `binary`, etc. |
| `--trusted-pubkey` | string | NO | Repeatable trusted pubkey (ed25519:hex or raw hex) |

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | I/O error |

## Example

```bash
quinkgl query --cache cache.json --tag vision --trusted-pubkey ed25519:abc...
```
