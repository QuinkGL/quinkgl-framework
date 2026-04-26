# `quinkgl telemetry`

Run telemetry backend utilities for the QuinkGL dashboard.

## Synopsis

```bash
quinkgl telemetry serve [options]
quinkgl telemetry enroll <manifest.qgl> [options]
```

## Description

`quinkgl telemetry serve` starts the FastAPI telemetry backend used by the
React dashboard. It exposes REST snapshots, WebSocket updates, open enrollment,
and authenticated ingest endpoints.

`quinkgl telemetry enroll` registers a manifest with the telemetry backend and
writes a private `.telemetry.qglkey` file next to the manifest. That key lets
`quinkgl run` send telemetry for the matching swarm without manually exporting
a global secret.

## `serve` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host interface for uvicorn. Use loopback behind Caddy. |
| `--port` | `8765` | Port for the telemetry API. |
| `--auth-secret` | `QUINKGL_TELEMETRY_SECRET` env | Legacy global ingest secret. |
| `--cors-origin` | — | Allowed dashboard origin. Repeat for multiple origins. |
| `--token-file` | — | JSON file where swarm-scoped token hashes are stored. Enables open enrollment. |
| `--max-request-bytes` | `65536` | Maximum ingest request size. |
| `--rate-limit-max-requests` | `120` | Maximum ingest requests per window. |
| `--rate-limit-window-seconds` | `60.0` | Rate-limit window length. |

## `enroll` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `<manifest.qgl>` | required | Manifest to enroll. |
| `--dashboard-url` | manifest telemetry URL or hosted default | Dashboard origin. Do not include `/api`. |
| `--output` | `<manifest>.telemetry.qglkey` | Output key file path. |
| `--overwrite` | false | Replace an existing `.qglkey` file. |

## Examples

Start the telemetry backend on a VPS behind Caddy:

```bash
sudo mkdir -p /etc/quinkgl
sudo touch /etc/quinkgl/telemetry-tokens.json
sudo chmod 600 /etc/quinkgl/telemetry-tokens.json

quinkgl telemetry serve \
  --host 127.0.0.1 \
  --port 8765 \
  --cors-origin https://dash.example.com \
  --token-file /etc/quinkgl/telemetry-tokens.json
```

Enroll a swarm manifest:

```bash
quinkgl telemetry enroll my-swarm.qgl \
  --dashboard-url https://dash.example.com
```

This writes:

```text
my-swarm.telemetry.qglkey
```

Run the peer normally after enrollment:

```bash
quinkgl run --manifest my-swarm.qgl --script peer_script.py
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation or enrollment error |

## See Also

- [Telemetry](../user-guide/telemetry.md)
- [`quinkgl run`](run.md)
