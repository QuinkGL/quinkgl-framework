# `quinkgl discover`

Discover swarms by affinity ranking (Phase 3).

## Synopsis

```
quinkgl discover --cache <path> --fingerprint <path>
                 [--min-affinity <float>] [--max-swarms <int>]
```

## Description

Ranks cached `SwarmAdvertisement` objects by data fingerprint affinity against a local fingerprint.

## Flags

| Flag | Type | Required | Description |
|---|---|---|---|
| `--cache` | path | YES | JSON cache file path |
| `--fingerprint` | path | YES | Local fingerprint JSON file |
| `--min-affinity` | float | NO | Minimum affinity score (default 0.5) |
| `--max-swarms` | int | NO | Maximum results (default 1) |

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | I/O error |

## Example

```bash
quinkgl discover \
  --cache cache.json \
  --fingerprint my_data.json \
  --min-affinity 0.3 \
  --max-swarms 5
```
