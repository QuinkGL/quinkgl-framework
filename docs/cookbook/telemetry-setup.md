# Recipe: Telemetry Setup

Connect your peers to the live QuinkGL dashboard.

## What You Need

1. **Dashboard URL** — e.g. `https://dash.quinkgl.io/api`
2. **Telemetry secret** — provided by the swarm operator

The dashboard is a separate React application already running on an Oracle
VPS.  As a peer operator you **do not** install, build, or host it.

## Point Peers at the Dashboard

```bash
export QUINKGL_TELEMETRY_SECRET="<secret-from-operator>"
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --telemetry-url https://dash.quinkgl.io/api
```

Change the heartbeat interval (default 5.0 s):

```bash
quinkgl run ... --telemetry-heartbeat-interval 10.0
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

If you need a completely private instance, deploy the FastAPI server
yourself:

```python
from quinkgl.telemetry import create_telemetry_app
import uvicorn

app = create_telemetry_app(auth_secret="change-me")
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `QUINKGL_TELEMETRY_SECRET` | — | Auth secret for ingest |
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
