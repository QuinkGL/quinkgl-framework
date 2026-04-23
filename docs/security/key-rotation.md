# Key Rotation

QuinkGL's current design treats manifest signing keys as **long-lived but not permanent**.

## Current Policy

**There is no in-place key rotation.** A new keypair means a new creator identity, which means a new swarm.

This is intentional:
- Manifests are immutable by design
- `swarm_id` binds to the exact manifest content
- Changing the creator would change the canonical bytes → new `swarm_id`

## Rotation Workflow

When a creator needs to rotate keys:

1. **Generate a new keypair**
   ```bash
   quinkgl keygen --output new-creator.pem
   ```

2. **Create a new manifest** with the new key
   ```bash
   quinkgl manifest create \
     --name my-swarm-v2 \
     --task-type class \
     --input-shape 3,224,224 \
     --output-shape 10 \
     --label-type integer \
     --model-framework pytorch \
     --model-arch-hash sha256:... \
     --aggregation FedAvg \
     --topology Random \
     --sign-with new-creator.pem \
     --output swarm-v2.qgl
   ```

3. **Distribute the new manifest** to peers

4. **Update peer configurations** with the new trusted pubkey
   ```bash
   quinkgl run \
     --manifest swarm-v2.qgl \
     --trust-policy pinned \
     --trusted-pubkey ed25519:<new-pubkey>
   ```

5. **Deprecate the old swarm** by letting `expires_at` pass

## Recommendations

- **Rotate keys every 90 days** for production swarms
- **Document the rotation schedule** in the swarm's README
- **Maintain an overlap period** where both old and new swarms are active
- **Never reuse a compromised key** — generate entirely new key material

## Future Work

A future spec version may introduce:
- **Key delegation** — authorized sub-keys
- **Revocation lists** — explicitly revoked pubkeys
- **Threshold signatures** — multi-sig creator committees

These require schema version bumps and are not planned for Phase 2/3.
