# API Reference

Public Python API modules.

## `quinkgl`

Top-level package exports:

- `GossipNode` — Full P2P gossip-learning node
- `PyTorchModel` — PyTorch `ModelWrapper`
- `TensorFlowModel` — TensorFlow `ModelWrapper`
- `TelemetryClient` — Telemetry forwarder
- `SwarmManifest` — Manifest parser / builder
- `DataFingerprint` — Privacy-preserving dataset sketch

## `quinkgl.manifest`

- `SwarmManifest.from_file(path)` — Load from `.qgl`
- `SwarmManifest.from_dict(data)` — Load from dict
- `SwarmManifest.to_file(path)` — Save to `.qgl`
- `SwarmManifest.manifest_hash()` — SHA-256 canonical hash
- `SwarmManifest.to_magnet()` — Generate magnet URI
- `sign_manifest(manifest, key_bytes)` — Ed25519 sign
- `verify_manifest(manifest)` — Ed25519 verify
- `keygen(output_path)` — Generate Ed25519 keypair

## `quinkgl.models`

- `ModelWrapper` — Abstract base class
- `PyTorchModel(model, device="cpu")` — PyTorch wrapper
- `PyTorchPersonalizedModel` — FedRep / FedBN wrapper
- `TrainingConfig` — Hyperparameter container
- `TrainingResult` — Metrics container

## `quinkgl.telemetry`

- `TelemetryClient(base_url, ...)` — Event/heartbeat forwarder
- `create_telemetry_app(...)` — FastAPI server factory
- `TelemetryStore(...)` — In-memory session store

## `quinkgl.testing`

- `local_swarm_fixture(...)` — Multi-peer test fixture

## See Also

- [Manifest Schema Reference](../manifest-schema.md)
- [CLI Reference](../cli-reference.md)
