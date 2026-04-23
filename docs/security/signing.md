# Ed25519 Signing

QuinkGL uses **Ed25519** for manifest and advertisement signing.

## Why Ed25519?

- **Fast** — signing and verification are computationally cheap
- **Compact** — 32-byte public keys, 64-byte signatures
- **Deterministic** — no randomness required during signing (no nonce reuse risk)
- **Standard** — widely supported in the `cryptography` library

## Key Format

### Public Key

```
ed25519:<64-hex-chars>
```

Example:
```
ed25519:5a5e0985d6b5d499bb6dcfe7c0a4df2f38ec92eb12dae05f8bdeffcc699ccc22
```

The raw key is 32 bytes (64 hex characters).

### Private Key

Stored as **PKCS#8 PEM** on disk with `0600` permissions on POSIX systems.

```bash
-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIA...
-----END PRIVATE KEY-----
```

## Generating a Keypair

```bash
quinkgl keygen --output creator.pem
```

Output:
```
ed25519:5a5e0985d6b5d499bb6dcfe7c0a4df2f38ec92eb12dae05f8bdeffcc699ccc22
```

To print the public key without writing a file:
```bash
quinkgl keygen --print-public-only
```

## Signing a Manifest

```bash
quinkgl manifest create \
  --name my-swarm \
  --task-type class \
  --input-shape 3,224,224 \
  --output-shape 10 \
  --label-type integer \
  --model-framework pytorch \
  --model-arch-hash sha256:... \
  --aggregation FedAvg \
  --topology Random \
  --sign-with creator.pem \
  --output signed-swarm.qgl
```

The signature covers **canonical bytes** of the manifest with `signature` excluded (§5.3).

## Verifying a Manifest

```bash
quinkgl manifest verify signed-swarm.qgl
```

If the manifest is signed and the signature is valid:
```
Manifest is valid.
```

If tampered:
```
Signature check failed: manifest tampered or creator_pubkey/signature mismatch.
```

## Programmatic API

```python
from quinkgl.manifest import sign_manifest, verify_manifest
from quinkgl.manifest import SwarmManifest

# Load manifest
manifest = SwarmManifest.from_file("swarm.qgl")

# Sign
with open("creator.pem", "rb") as f:
    key_pem = f.read()
signed = sign_manifest(manifest, key_pem)

# Verify
valid = verify_manifest(signed)  # True or False
```

## Security Recommendations

1. **Store private keys offline** when possible (Hardware Security Module, air-gapped machine)
2. **Never commit `.pem` files** to version control
3. **Rotate keys periodically** — see [Key Rotation](key-rotation.md)
4. **Use `pinned` trust policy** in production — see [Trust Policies](trust-policies.md)
