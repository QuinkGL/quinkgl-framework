# Recipe: Telemetry Setup

Connect your peers to the live QuinkGL dashboard.

## What You Need

1. **Running telemetry backend** — hosted at the default QuinkGL dashboard origin
2. **Enrolled manifest** — `quinkgl telemetry enroll` creates the private `.qglkey`

The dashboard is a separate React application already running on an Oracle
VPS.  As a peer operator you **do not** install, build, or host it.

## Start Peers with Telemetry

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py
```

Enroll once after creating the manifest:

```bash
quinkgl telemetry enroll my-swarm.qgl --dashboard-url https://dash.quinkgl.io
```

The command writes a private key file next to the manifest:

```text
my-swarm.qgl
my-swarm.telemetry.qglkey
```

The `.qgl` file is shareable and contains only secret-free telemetry metadata.
The `.qglkey` file is private, should be distributed out-of-band, and should
not be committed to source control.

Change the heartbeat interval (default 60.0 s):

```bash
quinkgl run ... --telemetry-heartbeat-interval 30.0
```

## What the Dashboard Provides

| Endpoint | Description |
|----------|-------------|
| `GET /api/session` | Fleet-wide session summary |
| `GET /api/nodes` | All known nodes with last-seen timestamps |
| `GET /api/nodes/{id}` | Single node snapshot |
| `GET /api/network/graph` | Nodes + edges for topology visualisation |
| `GET /api/network/stats` | Aggregate stats (isolated nodes, message volume) |
| `WS /api/stream` | Live WebSocket fan-out of every update |

## Querying the Fleet

```bash
curl -s https://dash.quinkgl.io/api/nodes | python -m json.tool
curl -s https://dash.quinkgl.io/api/session | python -m json.tool
```

## Running a Private Telemetry Server (Advanced)

If you need a completely private instance, deploy the FastAPI server yourself:

```bash
sudo mkdir -p /etc/quinkgl
sudo touch /etc/quinkgl/telemetry-tokens.json
sudo chmod 600 /etc/quinkgl/telemetry-tokens.json
quinkgl telemetry serve --host 127.0.0.1 --port 8765 \
  --cors-origin https://dash.quinkgl.io \
  --token-file /etc/quinkgl/telemetry-tokens.json
```

`POST /api/telemetry/enroll` appends new swarm token hashes to the token file.
You do not manually edit it for each swarm. A generated token can only write
telemetry for its registered `swarm_id`.

Keep uvicorn on loopback and let Caddy terminate TLS:

```caddyfile
dash.quinkgl.io {
  handle /api/* {
    reverse_proxy 127.0.0.1:8765
  }

  handle {
    root * /var/www/quinkgl-web
    try_files {path} /index.html
    file_server
  }
}
```

Caddy proxies WebSocket upgrades automatically, so `/api/stream` and `/api/ws`
use the same upstream.

For systemd:

```ini
[Unit]
Description=QuinkGL telemetry backend
After=network-online.target

[Service]
WorkingDirectory=/opt/QuinkGL
ExecStart=/opt/QuinkGL/.venv/bin/quinkgl telemetry serve --host 127.0.0.1 --port 8765 --cors-origin https://dash.quinkgl.io --token-file /etc/quinkgl/telemetry-tokens.json
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Validate both layers before starting peers:

```bash
curl -i http://127.0.0.1:8765/api/session
curl -i https://dash.quinkgl.io/api/session
```

If the public curl returns `502 Bad Gateway`, Caddy is reachable but cannot
connect to the FastAPI upstream. Check that the `quinkgl telemetry serve`
process is running and that the Caddy `reverse_proxy` port matches it.

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QUINKGL_TELEMETRY_SECRET` | — | Legacy global auth secret for ingest |
| `QUINKGL_TELEMETRY_SECRET_HEADER` | `X-QuinkGL-Telemetry-Secret` | Header name |
| `QUINKGL_TELEMETRY_CORS_ALLOW_ORIGINS` | — | Comma-separated origins |

Resource limits (in-memory, session-scoped):

| Limit | Default |
|-------|---------|
| `max_nodes` | 256 |
| `max_events_per_node` | 500 |
| `max_rounds_per_node` | 128 |
| `max_edges` | 2048 |

## See Also

- [Tutorial T6](../tutorials/T6/index.md) — Full walkthrough
- [User Guide: Telemetry](../user-guide/telemetry.md) — Architecture
