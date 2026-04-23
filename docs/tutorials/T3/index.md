# Tutorial T3 — Joining a Production Swarm with Pinned Trust

This tutorial shows how to join a signed swarm safely using the `pinned`
trust policy so that only manifests signed by an explicitly listed creator
are accepted.

## Prerequisites

- The swarm creator's Ed25519 public key (from [Tutorial T2](../T2/index.md))
- A valid `.qgl` manifest or magnet URI

## Step 1: Obtain the Manifest

Download the manifest from the creator or the directory:

```bash
quinkgl query --endpoint https://dir.example.com/v1 t2-signed
```

Or fetch directly from a magnet URI:

```bash
quinkgl manifest magnet "quinkgl:?xt=urn:qgl:..." --output t2-signed.qgl
```

## Step 2: Inspect the Signature

```bash
quinkgl manifest verify t2-signed.qgl
```

Note the `creator_pubkey` printed in the output.  This is the key you will
pin.

## Step 3: Run with Pinned Trust

```bash
quinkgl run \
  --manifest t2-signed.qgl \
  --script peer_script.py \
  --trust-policy pinned \
  --trusted-pubkey ed25519:3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
```

If the manifest signature does not match the supplied pubkey, the node exits
immediately with `ERR_NODE_UNSIGNED_MANIFEST_REJECTED`.

## Step 4: TOFU as a Softer Alternative

If you do not know the creator key in advance but want to lock it after the
first encounter, use `--trust-policy tofu`:

```bash
quinkgl run \
  --manifest t2-signed.qgl \
  --script peer_script.py \
  --trust-policy tofu
```

The first successful manifest pins the creator pubkey locally in
`~/.local/state/quinkgl/tofu_cache.json`.  Any later manifest with a
different creator for the same swarm hash triggers
`ERR_TRUST_TOFU_CONFLICT`.

## Step 5: Multiple Trusted Keys

You can list several pubkeys if the swarm has multiple authorised creators:

```bash
quinkgl run \
  --manifest t2-signed.qgl \
  --script peer_script.py \
  --trust-policy pinned \
  --trusted-pubkey ed25519:AAA... \
  --trusted-pubkey ed25519:BBB...
```

## Next Steps

- **Tutorial T4** — Writing a custom peer script for real models
- [Trust Policies](../../security/trust-policies.md) — Detailed comparison of all three policies
- [TOFU Cache](../../security/tofu-cache.md) — Cache format and manual clearing
