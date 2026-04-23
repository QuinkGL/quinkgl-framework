# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Checkpoint and resume** (`quinkgl run`):
  - `--checkpoint-dir <path>` saves model weights every 10 rounds via `ModelStore` (msgpack-serialised, with SHA-256 checksums).
  - `--resume` seeds the model from the latest checkpoint before training begins.
  - Final best-effort checkpoint is written when the training loop exits cleanly.
  - `_ensure_model_wrapper()` auto-wraps bare `torch.nn.Module` / `tf.keras.Model` objects returned by Mode-B user scripts so that checkpoint `get_weights` / `set_weights` work transparently.
  - Periodic checkpoints use `start_round + round_idx` numbering so resumed runs produce correctly sequenced checkpoint IDs.

- **Unix-socket status introspection** (`quinkgl status`):
  - `StatusServer` binds an async one-shot JSON server at `<work-dir>/running/<node-id>.sock` (mode `0600`).
  - `_discover_nodes()` deduplicates `.sock` and `.json` artefacts so a single peer does not appear as two entries.
  - `_read_state()` falls back from the socket to the sibling `.json` snapshot when the socket is missing or the server has crashed.
  - `status --watch` refreshes every 2 seconds until Ctrl-C.

- **Telemetry wiring** (`quinkgl run --telemetry-url`):
  - `TelemetryClient` forwards `RuntimeEvent`s to `POST /api/telemetry/events` and polls `node.get_stats()` for `POST /api/telemetry/heartbeats`.
  - Retry with exponential backoff (3 attempts, initial delay 0.5 s, max 5.0 s).
  - Reads `QUINKGL_TELEMETRY_SECRET` from the environment as a fallback for `--telemetry-secret`.

- **Tutorials**:
  - **T2** — Creating and Publishing a Signed Swarm (`keygen`, `manifest create --sign-with`, `publish`, magnet URI sharing).
  - **T3** — Joining a Production Swarm with Pinned Trust (`--trust-policy pinned`, `--trusted-pubkey`, TOFU as a softer alternative).
  - **T4** — Writing a Custom Peer Script for PyTorch Models (`build_model`, `build_loaders`, optional `build_optimizer` / `build_scheduler`, lifecycle hooks).
  - **T5** — Local Multi-Peer Testing with `quinkgl.testing` (`local_swarm_fixture`, port allocation, metrics inspection, debug logging).
  - **T6** — Monitoring a Fleet with the Telemetry Server (dashboard connection, REST queries, WebSocket subscription, `status --watch`).

- **Cookbook recipes**:
  - Running a Local Swarm — one-liner `quinkgl init` + `pytest` workflow.
  - Multi-Peer Testing — Byzantine peer injection, churn simulation, network partition/healing, custom assertions.
  - Custom Model Wrapper — minimal `ModelWrapper` subclass, FedRep/FedBN personalised layers.
  - Telemetry Setup — dashboard connection recipe, environment variables, resource limits, security notes.

- **User guides**:
  - Trust Policies — deep dive on `open`, `tofu`, and `pinned` with comparison table.
  - Peer Scripts — complete Mode-B API reference (`build_model`, `build_loaders`, hooks, script arguments, testing).
  - Working with Manifests — creation, signing, verification, inspection, distribution.
  - Telemetry — architecture diagram, peer-side wiring, server endpoints, scaling and security notes.

- **Reference pages**:
  - Manifest Schema Reference — field-by-field `.qgl` reference (TaskSpec, ModelSpec, Aggregation, Topology, DataPolicy, ByzantineSpec, validation rules).
  - CLI Reference Overview — command table, global flags, exit code reference, directory commands.
  - Error Codes — exit codes with diagnosis and remediation, manifest error constants, telemetry error constants.
  - API Reference — public Python API modules (`quinkgl`, `quinkgl.manifest`, `quinkgl.models`, `quinkgl.telemetry`, `quinkgl.testing`).

- **CLI reference pages**:
  - `quinkgl run` — all flags, Mode A/B/C description, examples, exit codes.
  - `quinkgl manifest` — `create`, `show`, `verify`, `magnet` subcommands with full flag tables.
  - `quinkgl keygen` — key generation, `--overwrite`, `--print-public-only`, security notes.
  - `quinkgl status` — socket/JSON dual transport, `--watch`, output fields.
  - `quinkgl info` — version and strategy output.
  - `quinkgl init` — four templates (`minimal`, `pytorch-vision`, `pytorch-tabular`, `custom`), generated file listing.

- **Concepts index** — swarm, gossip learning, data fingerprint, trust policies, model wrappers.

- **Quickstart** — 5-minute install, manifest creation, verification, scaffolding, dry-run.

### Changed

- Removed `Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0` HTML comment headers from all documentation files; the spec reference is now implied by project context rather than repeated per-file metadata.
- Telemetry documentation updated to reflect the actual implementation: endpoint paths are `/api/telemetry/events`, `/api/telemetry/heartbeats`, and `/api/stream` (not the older `/v1/…` paths); auth uses `X-QuinkGL-Telemetry-Secret` by default; dashboard is a separate hosted application rather than a user-installed component.

### Removed

- `scripts/verify_release_docs.py` — release gate script (superseded by CI test suite).
- `.github/workflows/docs.yml` — GitHub Actions docs workflow.
- `.github/pull_request_template.md`.
- `.pre-commit-config.yaml`.

## [0.1.0] — 2026-04-23

### Added

- **CLI** (`quinkgl` command-line interface):
  - `quinkgl manifest create` — build `.qgl` swarm manifests from flags.
  - `quinkgl manifest show` — pretty-print or JSON-dump a manifest.
  - `quinkgl manifest verify` — validate schema + hash + optional expected swarm-id.
  - `quinkgl manifest magnet` — derive a `quinkgl:?xt=urn:qgl:…` magnet URI.
  - `quinkgl run` — start a peer node in Mode A (standard model), Mode B (user script hooks), or dry-run.
  - `quinkgl status` — introspect a running local peer via its state file/socket.
  - `quinkgl info` — print framework version, registered strategies, and dependency versions.
  - `quinkgl init` — scaffold a user peer-script project with 4 templates (`minimal`, `pytorch-vision`, `pytorch-tabular`, `custom`).
  - Global flags: `--json`, `--log-level`, `--work-dir`, `--config`, `--no-color`, `--quiet`, `--version`.
  - Exit-code mapping (stable small integers: 0 success, 1 validation, 2 I/O, 3 crypto, 4 trust, 5 hash mismatch, 7 node config, 130 interrupted).

- **Testing helpers** (`quinkgl.testing`):
  - `local_swarm_fixture(size, manifest_path)` — async context manager spinning up N in-process peers.
  - `make_dummy_manifest(**overrides)` — produce a valid `SwarmManifest` for tests.
  - `DummyDataLoader(shape, num_batches, label_type)` — synthetic data loader yielding correctly-shaped tensors.

- **Documentation infrastructure**:
  - Sphinx/MyST skeleton (`docs/conf.py`, `docs/index.md`) with pages for tutorials, user guides, reference, concepts, security, cookbook, and CLI.
  - `tests/docs/` — standalone test modules enforcing documentation requirements (docstring coverage, CLI page parity, error-code coverage, spec version sync, link integrity, tutorial execution).

- **Project metadata**:
  - `[project.scripts] quinkgl = "quinkgl.cli.__main__:main"` in `pyproject.toml`.
  - `[project.optional-dependencies] docs` group (`sphinx`, `myst-parser`, `sphinx-autodoc2`).
  - `[tool.setuptools.package-data]` including Jinja2 template files.
