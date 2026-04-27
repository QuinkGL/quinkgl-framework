# Telemetry

QuinkGL includes a built-in telemetry subsystem for fleet-wide observability.
Each peer forwards runtime events and periodic heartbeats to a central
dashboard backend. The dashboard is a separate React application; peer
operators enroll a manifest once, run peers with the generated `.qglkey`, and
paste a short-lived terminal code into the website to view that swarm.

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

The CLI attaches a `TelemetryClient` automatically unless `--no-telemetry` or
`QUINKGL_TELEMETRY_DISABLE=1` is set:

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer_script.py
```

The default telemetry origin is built into QuinkGL. A manifest can declare
secret-free telemetry metadata:

```json
{
  "telemetry": {
    "dashboard_url": "https://dash.quinkgl.io",
    "enrollment": "invite-required"
  }
}
```

The private ingest credential is separate from the manifest. For a manifest
named `my-swarm.qgl`, enroll once:

```bash
quinkgl telemetry enroll my-swarm.qgl --dashboard-url https://dash.quinkgl.io
```

The enrollment response is written to `my-swarm.telemetry.qglkey`:

```json
{
  "schema_version": 1,
  "swarm_id": "<manifest-hash>",
  "dashboard_url": "https://dash.quinkgl.io",
  "ingest_token": "qgl_live_<private-token>"
}
```

`quinkgl run` verifies that the `.qglkey` `swarm_id` matches the manifest hash
before using the token. Advanced deployments can still override the origin
with `QUINKGL_TELEMETRY_URL`; do not include `/api` because the client adds
`/api/telemetry/events` and `/api/telemetry/heartbeats` internally.

When the `.qglkey` is present, `quinkgl run` asks the backend for a dashboard
login code and prints it in the terminal:

```text
Dashboard code: QGL-ABCD-1234
Open the telemetry dashboard login page and paste this code.
```

The code is short-lived and single-use. It exchanges for a read-only viewer
token scoped to the manifest hash (`swarm_id`). The website can then show all
peers, events, rounds, topology, and aggregation activity for that swarm, but
not any other swarm on the same telemetry backend. The browser never receives
the private `ingest_token`.

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

When `--token-file` is configured, ingest endpoints require the
`X-QuinkGL-Telemetry-Secret` header with a swarm-scoped ingest token, and
dashboard read endpoints require `Authorization: Bearer <viewer-token>`.
The viewer token is obtained by posting a terminal dashboard code to
`/api/dashboard/login`.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /api/dashboard/codes` | Ingest token | Create a short-lived dashboard code |
| `POST /api/dashboard/login` | Dashboard code | Exchange code for a viewer token |
| `GET /api/session` | Viewer token | Swarm-scoped session metadata |
| `GET /api/nodes` | Viewer token | Swarm-scoped node snapshots |
| `GET /api/nodes/{id}` | Viewer token | Single authorized node |
| `GET /api/nodes/{id}/events` | Viewer token | Event history |
| `GET /api/events` | Viewer token | Swarm-scoped events |
| `GET /api/nodes/{id}/rounds` | Viewer token | Per-round summaries |
| `GET /api/rounds` | Viewer token | Swarm-scoped rounds |
| `GET /api/network/graph` | Viewer token | Swarm topology graph |
| `GET /api/network/stats` | Viewer token | Swarm aggregate stats |
| `POST /api/telemetry/events` | Yes | Ingest event |
| `POST /api/telemetry/heartbeats` | Yes | Ingest heartbeat |
| `POST /api/telemetry/connection-status` | Yes | Ingest connection state |
| `WS /api/stream` or `/api/ws` | Viewer token | Live swarm updates |

## Running the Backend

Run the FastAPI telemetry backend on loopback and put Caddy, nginx, or another
TLS proxy in front of it:

```bash
quinkgl telemetry serve --host 127.0.0.1 --port 8765 \
  --cors-origin https://dash.quinkgl.io \
  --token-file /etc/quinkgl/telemetry-tokens.json
```

For swarm-scoped tokens, provide a backend token file:

Start the backend with `--token-file /etc/quinkgl/telemetry-tokens.json`.
Open enrollment appends token hashes to that file automatically through
`POST /api/telemetry/enroll`. When a token file is configured, ingest requests
must present a token that matches the payload `swarm_id`; mismatches are
rejected with `403`.

For Caddy, route `/api/*` to the telemetry process and serve the React build
for every other path. A public `502 Bad Gateway` from Caddy means the proxy is
alive but the upstream process or port is wrong.

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

- **Auth**: Use a swarm-scoped token file in production. Legacy deployments can
  still set `auth_secret` or `QUINKGL_TELEMETRY_SECRET`; without credentials,
  anyone who can reach the ingest endpoints can pollute your telemetry data.
- **TLS**: Run the server behind HTTPS.  The secret header is sent on every
  request.
- **CORS**: Restrict `cors_allow_origins` to your dashboard domain.
- **Privacy**: The telemetry server stores only metadata (event types, round
  numbers, loss values, peer IDs).  Model weights and raw training data
  never leave the peer.

## See Also

- [Tutorial T6](../tutorials/T6/index.md) — Step-by-step walkthrough
- [Cookbook: Telemetry Setup](../cookbook/telemetry-setup.md) — Connection recipe
