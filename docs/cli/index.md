# CLI Reference

Complete reference for the `quinkgl` command-line interface.

```{toctree}
:maxdepth: 1

manifest
run
telemetry
status
info
init
keygen
```

## Global Flags

These flags are accepted by every subcommand:

| Flag | Description |
|------|-------------|
| `--json` | Emit machine-readable JSON to stdout |
| `--log-level` | `debug`, `info`, `warn`, or `error` (default: `info`) |
| `--work-dir` | Runtime directory for sockets, TOFU cache, etc. |
| `--config` | Path to an optional TOML config file |
| `--no-color` | Disable ANSI colours |
| `-q`, `--quiet` | Suppress non-error stderr output |

## Exit Code Reference

| Code | Name | Typical Cause |
|------|------|---------------|
| `0` | `SUCCESS` | Command completed normally |
| `1` | `VALIDATION_ERROR` | Bad arguments, invalid manifest schema, missing keys |
| `2` | `IO_ERROR` | File not found, permission denied, directory already exists |
| `3` | `CRYPTO_ERROR` | Signature failure, key missing, cryptography not installed |
| `4` | `TRUST_ERROR` | TOFU conflict, creator not in trusted set, no running node |
| `5` | `HASH_MISMATCH` | Manifest hash does not match `--expected-swarm-id` |
| `6` | `WIRE_ERROR` | Network-level failure (reserved) |
| `7` | `NODE_CONFIG_ERROR` | Script missing callables, model build failure |
| `130` | `INTERRUPTED` | Ctrl-C (`SIGINT`) |

## Directory Commands

In addition to the documented subcommands above, the CLI also supports:

- `quinkgl publish` — Sign and serialise a `SwarmAdvertisement`
- `quinkgl query` — Filter a local advertisement cache
- `quinkgl discover` — Rank cached ads by data-fingerprint affinity

See the [Directory](../user-guide/index.md) section of the User Guide for
usage examples.
