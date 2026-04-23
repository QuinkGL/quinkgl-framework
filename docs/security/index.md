# Security Model

This section covers QuinkGL's security architecture, threat model, and operational security practices.

```{toctree}
:maxdepth: 2

threat-model
signing
trust-policies
tofu-cache
rate-limits
key-rotation
incident-response
```

## Overview

QuinkGL's security model is built around **manifest-based identity** and **creator-controlled trust**. Every swarm is defined by a cryptographically signed manifest that binds together:

- **Model architecture** — what is being trained
- **Data schema** — what shape of data is expected
- **Training protocol** — aggregation and topology strategies
- **Privacy policy** — fingerprinting and collaboration rules

The manifest is signed by its **creator** using Ed25519. Peers decide whether to trust a manifest based on the active **trust policy**.

## Security Guarantees

| Guarantee | Mechanism |
|---|---|
| Manifest integrity | SHA-256 canonical bytes + Ed25519 signature |
| Creator authenticity | Ed25519 public key in `creator_pubkey` |
| Replay resistance | `created_at` + `expires_at` fields |
| Peer isolation | `community_id` derived from `swarm_id` |
| Byzantine tolerance | Krum, MultiKrum, TrimmedMean aggregation |

## Non-Goals

QuinkGL explicitly does **not** protect against:

- **Sybil attacks** without `pinned` trust policy — an attacker can create unlimited peers
- **Data extraction** — peers voluntarily share model weights; differential privacy is the user's responsibility
- **Network eavesdropping** — use TLS or IPv8's built-in encryption for transport-layer confidentiality

## Quick Reference

| Trust Policy | Use When |
|---|---|
| `open` | Private networks, rapid prototyping |
| `tofu` | Small consortiums with occasional churn |
| `pinned` | Production deployments with known creators |

See [Trust Policies](trust-policies.md) for detailed guidance.
