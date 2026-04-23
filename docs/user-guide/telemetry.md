# Telemetry

QuinkGL includes a built-in telemetry subsystem for fleet-wide observability.
Each peer forwards runtime events and periodic heartbeats to a central
dashboard.  The dashboard is a separate React application hosted on an Oracle
VPS; peer operators only need the URL and a shared secret to connect.

## Architecture

```
┌─────────────┐     POST /api/telemetry/events      ┌─────────────────┐
│   Peer A    │ ───────────────────────────────────> │                 │
│  (GossipNode)    POST /api/telemetry/heartbeats   │ Oracle VPS      │
└─────────────┘ ───────────────────────────────────> │                 │
                                                     │ FastAPI backend │
┌─────────────┐     POST /api/telemetry/events      │ + React dash    │
│   Peer B    │ ───────────────────────────────────> │                 │
│  (GossipNode)    POST /api/telemetry/heartbeats   │  • TelemetryStore│
└─────────────┘ ───────────────────────────────────> │  • TelemetryStreamHub
                                                     │                 │
                                                     └─────────────────┘
                                                              │
                                                              │ WS /api/stream
                                                              ▼
                                                       ┌─────────────┐
                                                       │  Dashboard  │
                                                       │   (React)   │
                                                       └─────────────┘
```

### Components

- **TelemetryClient** (`quinkgl.telemetry.client`) — Lives inside each peer.
  Subscribes to the `GossipNode` event emitter, serialises events, and
  forwards them over HTTP with retry and backoff.
- **TelemetryServer** (`quinkgl.telemetry.server`) — FastAPI app running on
  the VPS.  Exposes REST query endpoints, ingest endpoints, and a WebSocket
  fan-out stream.
- **Dashboard** (`quinkgl-web`) — Separate React + Vite SPA that consumes
  the REST API and WebSocket stream.  **Not part of the QuinkGL Python
  package**.

## Peer-Side Wiring

The CLI attaches a `TelemetryClient` automatically when `--telemetry-url` is
set:

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py \
  --telemetry-url https://dash.quinkgl.io/api \
  --telemetry-secret "$QUINKGL_TELEMETRY_SECRET"
```

The client:

1. Listens to every `RuntimeEvent` emitted by the peer (training completed,
   model sent/received, aggregation completed, peer discovered, etc.).
2. Serialises and `POST`s them to `/api/telemetry/events`.
3. Starts a background heartbeat loop that polls `node.get_stats()` and
   `POST`s the snapshot to `/api/telemetry/heartbeats` every
   `--telemetry-heartbeat-interval` seconds (default 5.0).
4. Retries failed deliveries with exponential backoff (up to 3 attempts).
5. Emits `telemetry.disconnected` / `telemetry.delivery_failed` events
   locally so the terminal UI reflects connectivity issues.

## Server Endpoints

All query endpoints are read-only and require no authentication.  Ingest
endpoints require the `X-QuinkGL-Telemetry-Secret` header when
`auth_secret` is set.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/session` | No | Session metadata |
| `GET /api/nodes` | No | All node snapshots |
| `GET /api/nodes/{id}` | No | Single node |
| `GET /api/nodes/{id}/events` | No | Event history |
| `GET /api/events` | No | All events |
| `GET /api/nodes/{id}/rounds` | No | Per-round summaries |
| `GET /api/rounds` | No | All rounds |
| `GET /api/network/graph` | No | Topology graph |
| `GET /api/network/stats` | No | Aggregate stats |
| `POST /api/telemetry/events` | Yes | Ingest event |
| `POST /api/telemetry/heartbeats` | Yes | Ingest heartbeat |
| `POST /api/telemetry/connection-status` | Yes | Ingest connection state |
| `WS /api/stream` or `/api/ws` | No | Live updates |

## Scaling Considerations

The default server is **in-memory and single-process**.  For small-to-medium
fleets (up to a few hundred peers) this is sufficient because:

- Events are small JSON payloads (hundreds of bytes).
- Pruning keeps memory bounded (`max_nodes=256`, `max_events_per_node=500`).
- Rate limiting prevents accidental overload.

For larger deployments you can run multiple server instances behind a load
balancer or replace `TelemetryStore` with a persistent backend by subclassing
and injecting it into `create_telemetry_app`.

## Security

- **Auth**: Set `auth_secret` (or `QUINKGL_TELEMETRY_SECRET`) in production.
  Without it, anyone who can reach the ingest endpoints can pollute your
  telemetry data.
- **TLS**: Run the server behind HTTPS.  The secret header is sent on every
  request.
- **CORS**: Restrict `cors_allow_origins` to your dashboard domain.
- **Privacy**: The telemetry server stores only metadata (event types, round
  numbers, loss values, peer IDs).  Model weights and raw training data
  never leave the peer.

## See Also

- [Tutorial T6](../tutorials/T6/index.md) — Step-by-step walkthrough
- [Cookbook: Telemetry Setup](../cookbook/telemetry-setup.md) — Connection recipe
