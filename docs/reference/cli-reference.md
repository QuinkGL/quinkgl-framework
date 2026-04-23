# CLI Reference Overview

High-level overview of the `quinkgl` CLI.

## Commands

| Command | Purpose |
|---------|---------|
| `quinkgl manifest create` | Build a `.qgl` swarm manifest |
| `quinkgl manifest show` | Pretty-print a manifest |
| `quinkgl manifest verify` | Validate schema, hash, and signature |
| `quinkgl manifest magnet` | Derive a magnet URI |
| `quinkgl keygen` | Generate an Ed25519 signing key |
| `quinkgl run` | Start a peer node |
| `quinkgl status` | Inspect a running local peer |
| `quinkgl info` | Print framework version and registered strategies |
| `quinkgl init` | Scaffold a peer-script project |
| `quinkgl publish` | Sign and serialise a swarm advertisement |
| `quinkgl query` | Filter an advertisement cache |
| `quinkgl discover` | Rank ads by data-fingerprint affinity |

## Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--log-level` | `debug`, `info`, `warn`, `error` |
| `--work-dir` | Runtime directory |
| `--config` | TOML config file |
| `--no-color` | Disable ANSI colours |
| `-q`, `--quiet` | Suppress non-error output |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation error |
| `2` | I/O error |
| `3` | Crypto error |
| `4` | Trust error |
| `5` | Hash mismatch |
| `6` | Wire error |
| `7` | Node configuration error |
| `130` | Interrupted (Ctrl-C) |

## See Also

- [CLI subcommand pages](../cli/index.md)
- [Error Codes](error-codes.md)
