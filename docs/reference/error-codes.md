# Error Codes

## CLI Exit Codes

| Code | Constant | Meaning | Typical Fix |
|------|----------|---------|-------------|
| `0` | `SUCCESS` | Command completed normally | — |
| `1` | `VALIDATION_ERROR` | Bad arguments or invalid manifest schema | Check `--help`; validate manifest with `quinkgl manifest verify` |
| `2` | `IO_ERROR` | File not found or permission denied | Check paths; ensure `--output-dir` does not already exist for `init` |
| `3` | `CRYPTO_ERROR` | Signature/key failure | Install `cryptography>=41.0.0`; verify key file exists and is valid PEM |
| `4` | `TRUST_ERROR` | TOFU conflict or creator not trusted | Use `--trust-policy open` for dev, or add correct `--trusted-pubkey` |
| `5` | `HASH_MISMATCH` | Manifest hash does not match expectation | Regenerate manifest; update `--expected-swarm-id` |
| `6` | `WIRE_ERROR` | Network failure (reserved) | Check connectivity, firewall, IPv8 port |
| `7` | `NODE_CONFIG_ERROR` | Peer script missing callables or model build failed | Ensure `build_model` and `build_loaders` are exported and callable |
| `130` | `INTERRUPTED` | `SIGINT` / Ctrl-C | — |

## Manifest Error Constants

These strings are raised as `ValueError` tags inside the manifest layer:

| Constant | Meaning |
|----------|---------|
| `ERR_MANIFEST_INVALID_JSON` | File is not valid JSON |
| `ERR_MANIFEST_NOT_OBJECT` | JSON root is not an object |
| `ERR_MANIFEST_SCHEMA_VERSION` | `schema_version` is not `4` |
| `ERR_MANIFEST_UNKNOWN_KEYS` | Extra keys outside the schema |
| `ERR_MANIFEST_MISSING_KEYS` | Required keys are absent |
| `ERR_MANIFEST_FIELD_INVALID` | Field type or value is wrong |
| `ERR_MANIFEST_EXPIRED` | `expires_at` is in the past |
| `ERR_MANIFEST_DATA_POLICY` | Data-policy validation failed |
| `ERR_MANIFEST_HASH_MISMATCH` | Canonical hash does not match `expected_swarm_id` |
| `ERR_SIGNATURE_INVALID` | Ed25519 signature verification failed |
| `ERR_SIGNING_UNAVAILABLE` | `cryptography` package not installed |
| `ERR_CREATOR_NOT_TRUSTED` | Creator pubkey not in trusted set |
| `ERR_TRUST_TOFU_CONFLICT` | TOFU cache has a different creator for this swarm |

## Telemetry Error Constants

| Constant | Meaning |
|----------|---------|
| `ERR_NODE_NO_MANIFEST` | `GossipNode` constructed without `domain` or `manifest` |
| `ERR_NODE_AGGREGATION_MISMATCH` | Local aggregation name differs from manifest |
| `ERR_NODE_TOPOLOGY_MISMATCH` | Local topology name differs from manifest |
| `ERR_NODE_UNSIGNED_MANIFEST_REJECTED` | `pinned` policy requires a signed manifest |

## See Also

- [CLI Reference Overview](cli-reference.md)
- `quinkgl.cli.exit_codes`
