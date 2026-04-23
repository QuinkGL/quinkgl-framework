# `quinkgl keygen`

Generate an Ed25519 keypair for manifest signing.

## Synopsis

```bash
quinkgl keygen [--output <path>] [--overwrite] [--print-public-only]
```

## Flags

| Flag | Description |
|------|-------------|
| `--output` | Destination path for the PKCS#8 PEM private key |
| `--overwrite` | Allow overwriting an existing key file |
| `--print-public-only` | Do not write a file; print a fresh pubkey and exit |

## Examples

Generate a key and print the pubkey:

```bash
quinkgl keygen --output creator.key
# prints: ed25519:3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
```

Print a one-time pubkey without saving:

```bash
quinkgl keygen --print-public-only
```

## Security Notes

- The private key is written with `0600` permissions.
- Treat the file as a secret: anyone with read access can sign manifests
  that peers will trust.
- Back up the key offline; losing it means you cannot publish manifest
  updates under the same creator identity.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | I/O error (cannot write file) |
| `3` | Crypto error (signing subsystem unavailable, file exists) |

## See Also

- [Signing Reference](../security/signing.md)
- `quinkgl manifest create --sign-with`
