# Incident Response

This guide describes how to respond to security incidents involving QuinkGL swarms.

## Incident Types

### Type 1: Creator Key Compromise

**Symptoms:**
- Unexpected manifests signed with the creator's key
- Peers report `ERR_TRUST_TOFU_CONFLICT` (if using tofu)
- Model convergence degrades suddenly

**Response:**

1. **Stop all peers immediately**
   ```bash
   # Find running peers
   ps aux | grep quinkgl
   # Kill gracefully with SIGINT
   kill -INT <pid>
   ```

2. **Revoke trust for the compromised key**
   ```bash
   # Remove from pinned sets
   # Edit peer startup scripts to exclude the old pubkey
   ```

3. **Rotate the key** — see [Key Rotation](key-rotation.md)

4. **Create a new swarm** with the new key

5. **Notify all peers** with the new manifest and pubkey

### Type 2: Rogue Peer in Open Swarm

**Symptoms:**
- Model divergence or accuracy degradation
- Byzantine aggregation triggers frequently
- Telemetry shows anomalous peer behavior

**Response:**

1. **Switch to `pinned` trust policy**
   ```bash
   quinkgl run \
     --manifest swarm.qgl \
     --trust-policy pinned \
     --trusted-pubkey ed25519:<legitimate-creator>
   ```

2. **Identify the rogue peer** via telemetry logs

3. **Firewall the rogue peer's IP** if known

### Type 3: Manifest Tampering Detected

**Symptoms:**
- `quinkgl manifest verify` reports `INVALID` signature
- Hash mismatches during manifest exchange

**Response:**

1. **Do not join the tampered manifest**
2. **Obtain the authentic manifest** from a trusted source
3. **Verify the authentic manifest's signature**
4. **Report the tampered manifest's source** (tracker, peer, etc.)

### Type 4: TOFU Cache Poisoning

**Symptoms:**
- `ERR_TRUST_TOFU_CONFLICT` on legitimate manifests
- TOFU cache contains unexpected entries

**Response:**

1. **Inspect the cache**
   ```bash
   cat ~/.local/state/quinkgl/tofu_creators.json
   ```

2. **Clear the cache if compromised**
   ```bash
   rm ~/.local/state/quinkgl/tofu_creators.json
   ```

3. **Re-join with the legitimate manifest** (this re-populates the cache)

## Prevention Checklist

- [ ] Use `pinned` trust policy in production
- [ ] Rotate creator keys every 90 days
- [ ] Monitor telemetry for anomalies
- [ ] Restrict filesystem access to `$work_dir`
- [ ] Keep private keys offline (HSM, air-gapped)
- [ ] Maintain a peer contact list for emergency notifications

## Contact

For security issues, contact the swarm creator or QuinkGL maintainers directly. Do not disclose sensitive details in public channels until the incident is contained.
