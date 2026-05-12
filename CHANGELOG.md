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
- `quinkgl run` telemetry heartbeats now default to 60 seconds to reduce routine HTTP log noise during long-running peer sessions.
- Chunked model transfer now uses ACK/window flow control for large payloads, longer transfer timeouts, sender-side retry state, and throttled receiver NACK reports. Active ACK/window transfers no longer consume the legacy NACK resend budget for normal missing-chunk reports.

### Removed

- `scripts/verify_release_docs.py` — release gate script (superseded by CI test suite).
- `.github/workflows/docs.yml` — GitHub Actions docs workflow.
- `.github/pull_request_template.md`.
- `.pre-commit-config.yaml`.

## [0.3.4] — 2026-04-28

### Breaking / expectations

- **Telemetry dashboard URL semantics** — Manifest `telemetry.dashboard_url` must be a bare HTTP(S) **origin** (no trailing `/api`). Invalid URLs are rejected at manifest validation time.

### Added — Open telemetry enrollment and swarm-scoped auth

- **`TelemetryConfig` on `.qgl`** — Swarm manifests may include optional `telemetry` metadata: `dashboard_url` and `enrollment` (`invite-required` | `none`), validated in `SwarmManifest.validate()`.
- **`quinkgl manifest create`** — `--telemetry-dashboard-url` and `--telemetry-enrollment` embed that metadata when issuing new manifests (no secrets in the `.qgl` file).
- **`quinkgl telemetry enroll <manifest.qgl>`** — POSTs to `POST /api/telemetry/enroll` on the dashboard origin, receives per-swarm credentials, and writes **`<manifest>.telemetry.qglkey`** (JSON: `swarm_id`, `ingest_token`, `dashboard_url`, `schema_version`).
- **`quinkgl telemetry dashboard-code <manifest.qgl>`** — Requests a short-lived dashboard login code without starting a peer; optional `--node-id` for audit context.
- **`quinkgl run` integration** — If `*.telemetry.qglkey` sits next to the manifest path, `quinkgl run` resolves `TelemetryAuth` from that file (verifies `swarm_id` matches the manifest), preferring the key’s `dashboard_url` when set. After wiring telemetry, the CLI may print a **dashboard code** (`QGL-XXXX-XXXX`) and expiry so operators can paste it into the hosted dashboard.
- **Telemetry server (FastAPI)** — Open enrollment when `--token-file` is configured: issues ingest tokens per swarm, persists SHA-256 token hashes via `TelemetryTokenRegistry`.
- **Swarm-scoped dashboard read path** — `DashboardAccessRegistry` issues one-time dashboard codes and longer-lived viewer tokens; session/streaming/REST snapshot routes can require `viewer_token` so the browser never sees the ingest secret.
- **New REST surface** — Includes `POST /api/telemetry/enroll`, `POST /api/dashboard/codes`, `POST /api/dashboard/login`, and viewer-scoped variants of session/node/event routes (see `telemetry/server.py`).
- **Python modules** — `quinkgl.telemetry.qglkey` (`TelemetryQglKey`, `load_qglkey`, `default_qglkey_path`), `quinkgl.telemetry.tokens`, `quinkgl.telemetry.viewer`; `TelemetryConfig` re-exported from `quinkgl.manifest`.

### Changed

- **Telemetry event ingest** — `TelemetryStore` accepts `round_started` / `round_completed`, `node.state.*` event families, and existing `security.*` prefixes without returning **422** for supported training lifecycle events.
- **Default node selection** — `TelemetryStore._select_default_node_id()` orders running nodes by most recent activity using **numeric timestamps** and breaks ties by **`node_id`** so behaviour is stable across platforms with coarse or skewed clocks.
- **Token file bootstrap** — `TelemetryTokenRegistry.from_file()` treats an **empty** token file as a valid empty registry (typical `touch` before first `serve`), instead of failing JSON decode.

### Documentation

- New **`docs/cli/telemetry.md`** — Full reference for `telemetry serve`, `enroll`, `dashboard-code`, flags, examples, exit codes.
- Updates to **`docs/user-guide/telemetry.md`**, **`docs/cookbook/telemetry-setup.md`**, **`docs/tutorials/T1` / `T6`**, **`docs/cli/run.md`**, **`docs/cli/index.md`**, **Getting started** guides (version line), **`README`**, and **`docs/faq.md`** where telemetry workflow changed.

### Tooling / tests

- **CI** — Windows pytest invocation simplified; `testpaths` configured; IPv8-related test stabilisation; publish workflow aligned with the new gate.
- **Tests** — `tests/cli/test_telemetry_cli.py`, extended `tests/telemetry/test_api.py`, `tests/telemetry/test_qglkey.py`, updates to run telemetry defaults and public API surface tests.
- **Repo hygiene** — `*.telemetry.qglkey` added to `.gitignore` so enrollment artefacts are not committed by mistake.

### Upgrading from v0.3.3

1. Bump dependency / reinstall: `pip install -U quinkgl` or `pip install -e .` from this tag.
2. For fleet telemetry: run `quinkgl telemetry serve ... --token-file /path/to/tokens.json`, then per swarm `quinkgl telemetry enroll your.qgl --dashboard-url https://your-dashboard`.
3. Operators can use `quinkgl telemetry dashboard-code your.qgl` or rely on the code printed when peers start with a `.qglkey` present.
4. Ensure manifest `telemetry.dashboard_url` values are origins only (no `/api` suffix).

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
