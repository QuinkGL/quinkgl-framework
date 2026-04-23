# `quinkgl status`

Inspect the runtime state of a local peer.

## Synopsis

```bash
quinkgl status [--node-id <id>] [--watch]
```

## Description

`quinkgl status` discovers running peers by looking for unix sockets and
JSON snapshots in `<work-dir>/running/`.  It prefers the socket transport
and falls back to the JSON file if the socket is unreachable.

## Flags

| Flag | Description |
|------|-------------|
| `--node-id` | Select a specific peer (required when multiple peers are running) |
| `--watch` | Refresh every 2 seconds until Ctrl-C |

## Examples

Show a single running node:

```bash
quinkgl status
```

Watch a specific node:

```bash
quinkgl status --node-id peer-a3f1 --watch
```

JSON output:

```bash
quinkgl --json status --node-id peer-a3f1
```

## Output Fields

- `node_id` — Peer identifier
- `status` — `INIT`, `MANIFEST_RESOLVED`, `COMMUNITY_STARTED`, `TRAINING`, etc.
- `since` — ISO timestamp when the peer started
- `swarm_name` / `swarm_id_short` — Swarm identity
- `ipv8_port` — Bound UDP port
- `peers_connected` / `peers_discovered` — Peer counts
- `current_round` — Last completed gossip round

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | I/O error (cannot read state) |
| `4` | Trust error (no matching node found) |

## See Also

- [Telemetry](../user-guide/telemetry.md) — Fleet-wide observability
- `quinkgl run`
