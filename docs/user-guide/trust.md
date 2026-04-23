# Trust Policies

QuinkGL supports three trust policies that govern how a peer evaluates the
creator identity attached to a manifest.

## `open` (Default)

No signature verification.  The peer accepts any manifest that parses
successfully.

Use this for local development and testing only.

```bash
quinkgl run --manifest my-swarm.qgl --script peer.py --trust-policy open
```

## `tofu` — Trust On First Use

The first time a peer sees a manifest for a given swarm hash, it records the
creator pubkey in a local cache (`~/.local/state/quinkgl/tofu_cache.json`).
Any later manifest with a **different** creator for the same swarm triggers
`ERR_TRUST_TOFU_CONFLICT` and the node refuses to start.

This gives you signature validation without having to distribute pubkeys
out-of-band.

```bash
quinkgl run --manifest my-swarm.qgl --script peer.py --trust-policy tofu
```

## `pinned` — Explicit Trust List

Only manifests signed by one of the pubkeys supplied via `--trusted-pubkey`
are accepted.  If the manifest is unsigned or signed by an unknown creator,
the node exits immediately with `ERR_TRUST_TOFU_CONFLICT`.

Use this in production where you know the creator key(s) in advance.

```bash
quinkgl run \
  --manifest my-swarm.qgl \
  --script peer.py \
  --trust-policy pinned \
  --trusted-pubkey ed25519:3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
```

You can list multiple trusted keys:

```bash
quinkgl run ... \
  --trusted-pubkey ed25519:AAA... \
  --trusted-pubkey ed25519:BBB...
```

## Comparison

| Policy | Signature Required | Out-of-Band Key Distribution | Use Case |
|--------|-------------------|------------------------------|----------|
| `open` | No | None | Local dev |
| `tofu` | Yes | None | Small teams, first deployment |
| `pinned` | Yes | Yes | Production, audited swarms |

## See Also

- [TOFU Cache](../security/tofu-cache.md)
- [Signing Reference](../security/signing.md)
- [Tutorial T3](../tutorials/T3/index.md)
