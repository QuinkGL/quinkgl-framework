# Tutorial T6 — Monitoring a Fleet with the Telemetry Server

This tutorial shows how to stream per-round metrics from each peer to the
live QuinkGL dashboard.

## Prerequisites

- A running swarm (see [Tutorial T5](../T5/index.md) for local testing)
- A manifest enrolled with `quinkgl telemetry enroll`

## Architecture Overview

```
Peer A,B,C  ──HTTP POST──►  Oracle VPS
(TelemetryClient)          (FastAPI + Dashboard)
                            ├─ /api/telemetry/events
                            ├─ /api/telemetry/heartbeats
                            ├─ /api/nodes   (query)
                            └─ /api/stream  (WebSocket)
```

The dashboard is a separate React application already hosted on an Oracle
VPS.  You do **not** need to install or run it yourself.  Enroll the manifest
once to create the private `.telemetry.qglkey` file for that swarm.

## Step 1: Enable Peer Telemetry

Enroll the manifest:

```bash
quinkgl telemetry enroll my-swarm.qgl --dashboard-url https://dash.quinkgl.io
```

This writes:

```text
my-swarm.qgl
my-swarm.telemetry.qglkey
```

Then start the peer normally:

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py
```

The CLI verifies that the `.qglkey` `swarm_id` matches the manifest hash before
sending telemetry. Legacy deployments can still use `QUINKGL_TELEMETRY_SECRET`.

By default the heartbeat interval is 5 s.  Change it with
`--telemetry-heartbeat-interval 10.0`.

## Step 2: Inspect the Fleet via REST

While the dashboard UI renders the live view, you can also query the raw
REST API directly:

```bash
curl -s https://dash.quinkgl.io/api/nodes | python -m json.tool
curl -s https://dash.quinkgl.io/api/session | python -m json.tool
curl -s https://dash.quinkgl.io/api/network/stats | python -m json.tool
```

## Step 3: Watch Local Node State

On the peer machine itself you do not need the dashboard:

```bash
quinkgl status --node-id peer-a3f1
```

Add `--watch` for a 2-second refresh loop until you press Ctrl-C:

```bash
quinkgl status --node-id peer-a3f1 --watch
```

## Step 4: Live WebSocket Stream (Optional)

Connect to the dashboard's WebSocket endpoint for real-time updates:

```python
import asyncio
import websockets
import json

async def stream():
    uri = "wss://dash.quinkgl.io/api/stream"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            data = json.loads(msg)
            print(f"{data['type']} — node={data.get('payload', {}).get('node_id')}")

asyncio.run(stream())
```

Message types: `node_snapshot_updated`, `session_stats_updated`,
`node_event_received`, `network_edge_updated`, `connection_status_updated`.

## Security Notes

- Use a private `.telemetry.qglkey` file for production swarms. Legacy
  deployments can still use `--telemetry-secret` or `QUINKGL_TELEMETRY_SECRET`.
  The server rejects missing or invalid ingest credentials with HTTP 401.
- The telemetry endpoint stores only aggregate metrics and event metadata,
  **never** model weights or raw training data.

## Next Steps

- [User Guide: Telemetry](../../user-guide/telemetry.md) — Architecture and scaling
- [CLI Reference: run](../../cli/run.md) — All telemetry flags
