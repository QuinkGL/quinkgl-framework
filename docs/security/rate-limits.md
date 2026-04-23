# Rate Limits

QuinkGL implements rate limits to prevent denial-of-service attacks.

## Manifest Exchange

| Limit | Value | Behavior |
|---|---|---|
| Max requests per peer per minute | 4 | Excess → `NACK(RATE_LIMITED)` |
| Request timeout | 30 seconds | → `ERR_WIRE_TIMEOUT` |
| Max chunk size | 1200 bytes | UDP headroom |
| Max total chunks | Unbounded | `ERR_WIRE_CHUNK_INCONSISTENT` on mismatch |

## Directory Community

| Limit | Value | Scope |
|---|---|---|
| Max ads per creator per day | 100 | Per `creator_pubkey` |
| Max ads per peer session | 10 | Per TCP/IPv8 session |
| Max cache size | 10,000 ads | Local eviction |
| Max ad age | 30 days | `expires_at` enforced |

## Telemetry

| Limit | Value | Behavior |
|---|---|---|
| Server rate limit | 120 req / 60s / IP | HTTP 429 |
| Max request size | 64 KiB | HTTP 413 |
| Client retry initial delay | 0.5s | Exponential backoff |
| Client retry max delay | 5.0s | Cap |
| Max delivery attempts | 3 | Then drop |
| Max pending queue | 256 events | Oldest dropped on overflow |

## Rationale

| Limit | Reasoning |
|---|---|
| Manifest: 4 req/min | Prevents amplification; legitimate peers rarely need more |
| Directory: 100 ads/day | Prevents spam; legitimate creators publish infrequently |
| Telemetry: 120 req/min | Balances observability vs. server load |
| Max request: 64 KiB | Prevents memory exhaustion from oversized payloads |

## Customization

Rate limits are **not** user-configurable in Phase 2/3. They are hard-coded constants to prevent accidental misconfiguration that weakens the network.

Future phases may expose tuning knobs for private deployments.
