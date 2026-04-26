# Frequently Asked Questions

## General

### What is QuinkGL?

QuinkGL is a decentralized, peer-to-peer federated learning framework using gossip-based protocols.

### Do I need a central server?

No. QuinkGL is fully decentralized. Peers communicate directly with each other.

### What Python versions are supported?

Python 3.10+.

## Manifests

### What is a `.qgl` file?

A UTF-8 JSON file describing a swarm's training protocol, model architecture, and policies. It is the canonical identity of a swarm.

### How do I get a swarm's ID?

```bash
quinkgl manifest show swarm.qgl | grep "Swarm ID"
```

Or programmatically:
```python
from quinkgl.manifest import SwarmManifest
m = SwarmManifest.from_file("swarm.qgl")
print(m.manifest_hash())
```

### Can I edit a manifest after creation?

No. Manifests are immutable. Any change creates a new `swarm_id`.

## Trust

### Which trust policy should I use?

- **Testing / demos:** `open`
- **Small consortiums:** `tofu`
- **Production:** `pinned`

### What happens if my creator key is compromised?

See [Incident Response](security/incident-response.md). Short answer: rotate keys and create a new swarm.

## Running Peers

### How do I stop a running peer?

Press `Ctrl-C` (SIGINT) for graceful shutdown.

### Can I run multiple peers on the same machine?

Yes, use different `--port` values and `--node-id` names.

## Troubleshooting

### `ERR_MANIFEST_HASH_MISMATCH`

The manifest was tampered with or corrupted. Re-download from a trusted source.

### `ERR_NODE_UNSIGNED_MANIFEST_REJECTED`

You are using `pinned` trust policy with an unsigned manifest. Either sign the manifest or switch to `open`/`tofu`.

### `ERR_TRUST_TOFU_CONFLICT`

The manifest's creator differs from the cached creator for this swarm. Clear the TOFU cache if you trust the new creator.
