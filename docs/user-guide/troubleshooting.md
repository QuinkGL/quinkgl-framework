# Troubleshooting

This page lists every `ERR_*` error code with a brief diagnosis and suggested remediation.

## Manifest Errors

- `ERR_MANIFEST_INVALID_JSON` — The file is not valid JSON. Check syntax with a linter.
- `ERR_MANIFEST_NOT_OBJECT` — The root of the JSON is not an object. Ensure the file starts with `{`.
- `ERR_MANIFEST_SCHEMA_VERSION` — The `schema_version` field does not match the expected version. Upgrade QuinkGL.
- `ERR_MANIFEST_UNKNOWN_KEYS` — Unexpected top-level keys found. Remove them or use `strict=False`.
- `ERR_MANIFEST_MISSING_KEYS` — Required fields are missing. Consult the manifest schema reference.
- `ERR_MANIFEST_FIELD_INVALID` — A field failed type or regex validation. Check field-specific rules.
- `ERR_MANIFEST_EXPIRED` — The manifest's `expires_at` is in the past. Contact the swarm creator.
- `ERR_MANIFEST_DATA_POLICY` — The `data_policy` block failed validation. Check sub-fields.
- `ERR_MANIFEST_HASH_MISMATCH` — Recomputed hash does not match expected `swarm_id`. The manifest may be tampered.
- `ERR_MANIFEST_FETCH_REQUIRED` — A magnet URI was given but no `peer_fetcher` was provided. Supply a fetcher callback.

## Magnet Errors

- `ERR_MAGNET_SCHEME` — The URI does not start with `quinkgl:`. Fix the scheme.
- `ERR_MAGNET_XT` — The `xt` parameter is missing or malformed. It must be `urn:qgl:<64-hex>`.
- `ERR_MAGNET_DUPLICATE` — A parameter appears more than once where only one is allowed (e.g. `dn`).

## Node Errors

- `ERR_NODE_NO_MANIFEST` — `GossipNode` was constructed without `manifest=` or `domain=`. Provide one.
- `ERR_NODE_AGGREGATION_MISMATCH` — The aggregation strategy name does not match the manifest. Check `--aggregation`.
- `ERR_NODE_TOPOLOGY_MISMATCH` — The topology strategy name does not match the manifest. Check `--topology`.
- `ERR_NODE_UNSIGNED_MANIFEST_REJECTED` — `trust_policy=pinned` requires a signed manifest. Obtain a signed `.qgl`.
- `ERR_NODE_ARCH_MISMATCH` — The model's architecture hash does not match `manifest.model.arch_hash`. Verify your model code.
- `ERR_NODE_DATA_SHAPE_MISMATCH` — The first batch shape differs from `manifest.task.input_shape`. Check data loaders.
- `ERR_RUN_NO_STANDARD_MODEL` — Mode A cannot build the model from the manifest. Use Mode B with `--script`.
- `ERR_SCRIPT_CALLABLES_MISSING` — The user script lacks `build_model` or `build_loaders`. Implement both callables.

## Trust / Signing Errors

- `ERR_TRUST_POLICY_VIOLATION` — The active trust policy rejected the manifest. Review policy settings.
- `ERR_TRUST_TOFU_CONFLICT` — TOFU cache shows a different creator for this `swarm_id`. Investigate potential spoofing.
- `ERR_SIGNING_UNAVAILABLE` — The `cryptography` package is not installed. Run `pip install quinkgl[crypto]`.
- `ERR_SIGNATURE_INVALID` — The manifest signature does not verify against `creator_pubkey`. The manifest may be tampered.
- `ERR_CREATOR_NOT_TRUSTED` — `trust_policy=pinned` and the creator is not in the trusted set. Add the pubkey.

## Wire Errors

- `ERR_WIRE_UNKNOWN_SWARM` — The peer does not know the requested swarm ID. Check bootstrap peers.
- `ERR_WIRE_RATE_LIMITED` — Too many manifest requests from this peer. Wait and retry.
- `ERR_WIRE_TIMEOUT` — No response within 30 seconds. Check network connectivity.
- `ERR_WIRE_CHUNK_INCONSISTENT` — Chunk metadata differs across responses. Retry or request from another peer.
