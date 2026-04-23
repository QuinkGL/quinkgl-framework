# `quinkgl publish`

Publish a swarm advertisement to the directory (Phase 3).

## Synopsis

```
quinkgl publish --manifest <path> --sign-with <pem> --output <path>
                [--reference-fingerprint <path>] [--tags <csv>]
```

## Description

Creates a signed `SwarmAdvertisement` JSON from a manifest and private key.

## Flags

| Flag | Type | Required | Description |
|---|---|---|---|
| `--manifest` | path | YES | Path to `.qgl` manifest file |
| `--sign-with` | path | YES | Path to Ed25519 private key PEM |
| `--output` | path | YES | Destination JSON file for the advertisement |
| `--reference-fingerprint` | path | NO | JSON fingerprint file for affinity matching |
| `--tags` | string | NO | Comma-separated tags |

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | I/O error (missing file) |
| 3 | Cryptographic error |

## Example

```bash
quinkgl publish \
  --manifest swarm.qgl \
  --sign-with creator.pem \
  --reference-fingerprint ref.json \
  --tags vision,pytorch \
  --output ad.json
```
