# Rate Limits

QuinkGL implements rate limits to prevent denial-of-service attacks.

## Model Chunk Transfer

| Limit | Value | Behavior |
|---|---|---|
| Per-round model fanout | 3 targets up to 100 compatible peers; 5 up to 250; 7 up to 500; 10 above 500 | Caps concurrent outbound model transfers per peer |
| Chunk payload size | 1024 bytes | Keeps UDP datagrams below typical MTU ceilings |
| Transfer timeout | 900 seconds | Incomplete transfer is abandoned |
| Max chunks per transfer | 300,000 | Oversized transfer rejected |
| Initial ACK window | 32 chunks | Sender limits in-flight chunks |
| Max ACK window | 128 chunks | Upper bound for future window growth |
| ACK timeout | 8 seconds | Unacked sent chunks become retry candidates |
| Max send attempts per chunk | 8 | Transfer fails after repeated missed ACKs |
| Receiver NACK report interval | 30 seconds | Repeated identical missing-chunk reports are suppressed |
| Legacy NACK resend budget | 20 reports per transfer | Applies to inactive cached transfers only |
| NACK peer bucket | 40 tokens, refill 1 token / 2s | Applies to legacy NACK resends |
| NACK transfer bucket | 20 tokens, refill 1 token / 2s | Applies to legacy NACK resends |

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
| Chunk window | Prevents UDP burst loss while keeping large model transfers moving |
| NACK throttling | Prevents repeated missing-chunk reports from becoming log or resend spam |
| Directory: 100 ads/day | Prevents spam; legitimate creators publish infrequently |
| Telemetry: 120 req/min | Balances observability vs. server load |
| Max request: 64 KiB | Prevents memory exhaustion from oversized payloads |

## Customization

Rate limits are **not** user-configurable in Phase 2/3. They are hard-coded constants to prevent accidental misconfiguration that weakens the network.

Future phases may expose tuning knobs for private deployments.
