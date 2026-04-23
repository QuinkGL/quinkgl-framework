# Tutorial T2 — Creating and Publishing a Signed Swarm

This tutorial walks you through generating an Ed25519 identity, signing a
manifest, and distributing the swarm to peers.

## Prerequisites

- QuinkGL CLI installed (`pip install quinkgl`)
- A manifest file (see [Tutorial T1](../T1/index.md))

## Step 1: Generate a Creator Key

```bash
quinkgl keygen --output creator.key
```

This writes a 64-byte hex private key to `creator.key`. **Keep this file
secure** — anyone who holds it can publish updates that existing peers will
trust.

The public half is printed to stdout; you will need it later:

```
ed25519:3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
```

## Step 2: Create a Signed Manifest

Reuse the manifest from T1 and add a signature at creation time:

```bash
quinkgl manifest create \
  --name t2-signed \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:7f2c1a9b3e4d0123456789abcdef0123456789abcdef0123456789abcdef0123 \
  --aggregation FedAvg \
  --topology Random \
  --sign-with creator.key \
  --output t2-signed.qgl
```

The manifest now contains a `signature` and `creator_pubkey` field.

## Step 3: Verify the Signature

```bash
quinkgl manifest verify t2-signed.qgl \
  --trusted-pubkey ed25519:3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
```

Expected: `Manifest signature is valid.`

## Step 4: Publish to the Directory

```bash
quinkgl publish t2-signed.qgl --endpoint https://dir.example.com/v1
```

Peers can now discover the swarm via `quinkgl discover` or the magnet URI.

## Step 5: Share the Magnet URI

```bash
quinkgl manifest magnet t2-signed.qgl
```

Send the printed `quinkgl:?xt=urn:qgl:...` URI to your peers.  Because the
manifest is signed, peers running `--trust-policy tofu` or `pinned` will
automatically validate the creator identity before joining.

## Next Steps

- **Tutorial T3** — Learn how peers join with strict trust policies
- [Trust Policies](../../security/trust-policies.md) — Deep dive on open, tofu, and pinned
- [Signing Reference](../../security/signing.md) — Key rotation and revocation
