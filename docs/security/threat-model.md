# Threat Model

## What QuinkGL Defends Against

### 1. Manifest Tampering

**Threat:** An attacker modifies a swarm manifest to change the model architecture, aggregation strategy, or data policy.

**Defense:**
- Manifests are canonically serialized and hashed (`swarm_id = SHA-256(canonical_bytes)`)
- The creator signs the canonical bytes with Ed25519
- Peers recompute the hash and verify the signature before joining

### 2. Impersonation

**Threat:** An attacker creates a manifest claiming to be from a trusted creator.

**Defense:**
- `pinned` trust policy: only accept manifests from a pre-configured set of `creator_pubkey` values
- `tofu` trust policy: cache the first-seen creator per swarm and reject changes

### 3. Replay Attacks

**Threat:** An attacker re-uses an old manifest for a new swarm.

**Defense:**
- `created_at` and `expires_at` fields in the manifest
- Peers reject expired manifests (`ERR_MANIFEST_EXPIRED`)

### 4. Byzantine Peers

**Threat:** Malicious peers send corrupted model updates to poison the aggregated model.

**Defense:**
- Aggregation strategies: Krum, MultiKrum, TrimmedMean
- `byzantine.f` parameter bounds tolerated faulty peers
- `enforce_n_gt_2f_plus_2` ensures quorum safety

### 5. Fingerprint Leakage

**Threat:** Data fingerprints reveal too much about private training data.

**Defense:**
- Quantized buckets (not exact values)
- Calibrated Gaussian noise (ε-DP)
- Gradient fingerprinting disabled by default
- Per-round nonces reduce linkability

## What QuinkGL Does NOT Defend Against

| Threat | Reason | Mitigation |
|---|---|---|
| **Sybil attacks** | No global identity provider | Use `pinned` trust policy |
| **Model inversion** | Weights are shared by design | Apply differential privacy locally |
| **Eavesdropping** | IPv8 transport encryption is optional | Deploy over VPN/TLS |
| **Denial of service** | Gossip is open by design | Rate limits + firewall rules |

## Attack Scenarios

### Scenario A: Creator Key Compromise

An attacker steals the creator's private key and publishes a malicious manifest.

**Impact:** Peers with `pinned` or `tofu` policy may accept the malicious manifest.

**Response:** See [Incident Response](incident-response.md).

### Scenario B: Rogue Peer in Open Swarm

An attacker joins an `open`-policy swarm and sends bad updates.

**Impact:** Model quality degrades; Byzantine aggregation limits but does not eliminate damage.

**Response:** Switch to `pinned` policy; identify and blacklist the attacker.

### Scenario C: Fingerprint Enumeration

An attacker probes many peers to reconstruct approximate data distributions.

**Impact:** Indirect information leakage about private datasets.

**Response:** Enable stricter privacy config; disable gradient fingerprinting.
