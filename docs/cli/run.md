# `quinkgl run`

Start a QuinkGL peer node.

## Synopsis

```bash
quinkgl run --manifest <path> [--data <dir> | --script <path>] [options]
```

## Description

`quinkgl run` constructs a `GossipNode`, attaches telemetry if configured,
and enters the gossip-learning loop.  It supports three modes:

- **Mode A** (`--data`) — standard model + data directory (not yet implemented).
- **Mode B** (`--script`) — user-provided `build_model` / `build_loaders` script.
- **Mode C** — fully custom script that constructs the node manually.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` | **required** | Path, URL, or magnet URI to the swarm `.qgl` file |
| `--data` | — | Data directory (Mode A) |
| `--script` | — | User script path (Mode B) |
| `--script-arg` | — | `key=value` pairs passed to script callables (repeatable) |
| `--node-id` | auto-generated | Peer identifier (`peer-<random>` if omitted) |
| `--port` | `0` | UDP port for IPv8 (0 = random) |
| `--trust-policy` | `open` | `open`, `tofu`, or `pinned` |
| `--trusted-pubkey` | — | Ed25519 hex pubkey for `pinned` policy (repeatable) |
| `--rounds` | manifest limit or `1000` | Training round count |
| `--telemetry-url` | — | Telemetry server base URL |
| `--telemetry-secret` | `QUINKGL_TELEMETRY_SECRET` env | Auth secret for telemetry ingest |
| `--telemetry-heartbeat-interval` | `5.0` | Seconds between heartbeats |
| `--checkpoint-dir` | — | Directory for periodic model checkpoints |
| `--resume` | false | Load latest checkpoint before training |
| `--dry-run` | false | Verify manifest and exit without starting IPv8 |

## Examples

### Mode B with a custom script

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --trust-policy tofu
```

### With telemetry

```bash
export QUINKGL_TELEMETRY_SECRET="change-me"
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --telemetry-url https://dash.example.com/api
```

### Resume from checkpoint

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --checkpoint-dir ./checkpoints \
  --resume
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation error (bad manifest, missing arguments) |
| `2` | I/O error |
| `3` | Crypto error (signing/verification failure) |
| `4` | Trust error (pinned/tofu rejection) |
| `7` | Node configuration error (script missing callables, model build failure) |

## See Also

- [Peer Scripts](../user-guide/peer-script.md)
- [Telemetry](../user-guide/telemetry.md)
- `quinkgl status`
