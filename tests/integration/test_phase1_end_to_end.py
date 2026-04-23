"""Phase 1 end-to-end integration gate (TASKS_SPLIT.md §2).

Exercises the happy path that both Track A and Track B jointly
guarantee:

    quinkgl --version
    quinkgl init --template pytorch-vision --output-dir <tmp>
    quinkgl manifest create --name ... --output <tmp>/swarm.qgl
    quinkgl manifest show   <tmp>/swarm.qgl
    quinkgl manifest verify <tmp>/swarm.qgl
    quinkgl manifest magnet <tmp>/swarm.qgl
    quinkgl run --manifest <tmp>/swarm.qgl --script <peer.py> --dry-run
    quinkgl keygen + sign-with + manifest verify --trusted-pubkey

Design notes
------------
* We invoke the CLI in-process via ``quinkgl.cli.__main__.main(argv)``
  instead of shelling out.  This keeps the test fast (~1 s), avoids
  depending on which Python interpreter happens to be on $PATH, and
  lets us assert on structured stdout produced in ``--json`` mode.
* No IPv8 reactor is started — ``quinkgl run --dry-run`` (§11.4)
  short-circuits after manifest validation.  End-to-end gossip is
  exercised by the ``tests/network/`` suite separately.
* This file is the single integration gate promised by
  TASKS_SPLIT.md §2 ("Integration testini Track A
  tests/integration/test_phase1_end_to_end.py altına yazar").
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main as cli_main


# --- small helpers -------------------------------------------------------


def _run_cli(*argv: str) -> int:
    """Invoke ``quinkgl ...`` in-process and return its exit code.

    argparse's ``--version`` action calls ``sys.exit`` directly, so we
    translate :class:`SystemExit` into the same ``int`` return shape
    used by :func:`quinkgl.cli.__main__.main`.
    """
    try:
        return int(cli_main(list(argv)) or 0)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, bool):
            return int(code)
        try:
            return int(code)
        except (TypeError, ValueError):
            return 1


def _dummy_arch_hash() -> str:
    """A deterministic, spec-shaped ``model_arch_fingerprint`` value.

    The integration happy path does not care about a real architecture
    hash — it just needs a ``sha256:<hex>`` string that the manifest
    validator accepts.
    """
    return "sha256:" + hashlib.sha256(b"integration-test-arch").hexdigest()


@pytest.fixture
def swarm_workdir(tmp_path: Path) -> Path:
    """Isolated working dir so ``quinkgl run`` has no neighbours."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "running").mkdir()
    return work


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "swarm.qgl"


# --- Test body -----------------------------------------------------------


class TestVersionAndInfo:
    def test_version_prints_and_exits_0(self, capsys):
        rc = _run_cli("--version")
        assert rc == 0
        out = capsys.readouterr().out
        assert "quinkgl" in out.lower()

    def test_info_exits_0(self, capsys):
        rc = _run_cli("info")
        assert rc == 0


class TestManifestCreateFlow:
    """Covers ``manifest create → show → verify → magnet`` (spec §11.3-11.6)."""

    def test_create_produces_valid_qgl(
        self, manifest_path: Path, capsys
    ):
        rc = _run_cli(
            "manifest",
            "create",
            "--name",
            "e2e-swarm",
            "--task-type",
            "class",
            "--input-shape",
            "3,224,224",
            "--output-shape",
            "10",
            "--label-type",
            "integer",
            "--tags",
            "e2e,smoke",
            "--model-framework",
            "pytorch",
            "--model-arch-hash",
            _dummy_arch_hash(),
            "--aggregation",
            "FedAvg",
            "--topology",
            "Random",
            "--output",
            str(manifest_path),
        )
        assert rc == 0, capsys.readouterr().err
        assert manifest_path.exists()
        # File MUST parse as JSON and carry schema_version=3.
        payload = json.loads(manifest_path.read_text())
        assert payload.get("schema_version") == 3
        assert payload.get("name") == "e2e-swarm"

    def test_show_emits_swarm_metadata(
        self, manifest_path: Path, capsys
    ):
        """`manifest show` MUST surface swarm name + swarm-id (spec §11.5).

        The human-readable renderer currently uses ``Swarm ID:`` as a
        label; we accept any case so future JSON-mode migrations stay
        detectable without brittle whitespace matching.
        """
        self.test_create_produces_valid_qgl(manifest_path, capsys)
        capsys.readouterr()  # drain previous output
        rc = _run_cli("manifest", "show", str(manifest_path))
        assert rc == 0
        out = capsys.readouterr().out.lower()
        assert "swarm id" in out or "swarm_id" in out
        assert "e2e-swarm" in out

    def test_verify_exits_0_on_valid_manifest(
        self, manifest_path: Path, capsys
    ):
        self.test_create_produces_valid_qgl(manifest_path, capsys)
        rc = _run_cli("manifest", "verify", str(manifest_path))
        assert rc == 0

    def test_verify_exits_non_zero_on_corrupted_manifest(
        self, manifest_path: Path, capsys
    ):
        self.test_create_produces_valid_qgl(manifest_path, capsys)
        data = json.loads(manifest_path.read_text())
        data["schema_version"] = 999  # force unsupported version
        manifest_path.write_text(json.dumps(data))
        rc = _run_cli("manifest", "verify", str(manifest_path))
        assert rc != 0

    def test_magnet_emits_urn_qgl(
        self, manifest_path: Path, capsys
    ):
        self.test_create_produces_valid_qgl(manifest_path, capsys)
        capsys.readouterr()  # drain previous output
        rc = _run_cli("manifest", "magnet", str(manifest_path))
        assert rc == 0
        out = capsys.readouterr().out
        # The CLI emits a ``quinkgl:?xt=urn:qgl:<hex>`` URI (spec §11.6);
        # we also accept the generic ``magnet:?`` prefix in case a
        # future revision switches the scheme.
        assert "xt=urn:qgl:" in out
        assert out.startswith("quinkgl:?") or "magnet:?" in out


class TestRunDryRun:
    """`quinkgl run --dry-run` validates manifest and exits without IPv8."""

    def test_run_dry_run_exits_0_for_valid_manifest(
        self, manifest_path: Path, swarm_workdir: Path, capsys
    ):
        # Build a manifest first.
        creator = TestManifestCreateFlow()
        creator.test_create_produces_valid_qgl(manifest_path, capsys)

        # Minimal user script (Mode B).  The dry-run path short-circuits
        # before calling any of the callables, but we still register
        # them so the "missing callables" guard passes.
        script = swarm_workdir / "peer.py"
        script.write_text(
            "from quinkgl.testing import DummyDataLoader\n"
            "def build_model(manifest, **kw):\n"
            "    return None\n"
            "def build_loaders(manifest, **kw):\n"
            "    return DummyDataLoader([4,3,224,224]), None\n"
        )

        rc = _run_cli(
            "--work-dir",
            str(swarm_workdir),
            "run",
            "--manifest",
            str(manifest_path),
            "--script",
            str(script),
            "--dry-run",
        )
        assert rc == 0, capsys.readouterr().err


class TestKeygenAndSignedVerify:
    """Covers ``keygen → manifest create --sign-with → verify --trusted-pubkey`` (spec §11.7, §16)."""

    def test_full_signing_round_trip(
        self, tmp_path: Path, manifest_path: Path, capsys
    ):
        # cryptography is optional; skip cleanly when absent so CI
        # without it can still run the rest of the integration gate.
        crypto = pytest.importorskip("cryptography")

        # 1. Generate a keypair.  ``--output`` writes the private key
        #    (0600) and echoes the public key on stdout's last line.
        key_path = tmp_path / "creator.key"
        rc = _run_cli("keygen", "--output", str(key_path))
        assert rc == 0, capsys.readouterr().err
        stdout = capsys.readouterr().out
        pubkey_line = stdout.strip().splitlines()[-1].strip()
        assert pubkey_line.startswith("ed25519:"), stdout
        assert key_path.exists()
        # Private key MUST be 0600 on POSIX.
        if os.name == "posix":
            assert (key_path.stat().st_mode & 0o777) == 0o600

        # 2. Create a signed manifest.
        rc = _run_cli(
            "manifest",
            "create",
            "--name",
            "signed-swarm",
            "--task-type",
            "class",
            "--input-shape",
            "3,32,32",
            "--output-shape",
            "10",
            "--label-type",
            "integer",
            "--model-framework",
            "pytorch",
            "--model-arch-hash",
            _dummy_arch_hash(),
            "--aggregation",
            "FedAvg",
            "--topology",
            "Random",
            "--sign-with",
            str(key_path),
            "--output",
            str(manifest_path),
        )
        assert rc == 0, capsys.readouterr().err
        data = json.loads(manifest_path.read_text())
        assert data.get("signature", "").startswith("ed25519:")
        assert data.get("creator_pubkey") == pubkey_line

        # 3. Verify with --trusted-pubkey succeeds.
        rc = _run_cli(
            "manifest",
            "verify",
            str(manifest_path),
            "--trusted-pubkey",
            pubkey_line,
        )
        assert rc == 0

        # 4. Verify with an unrelated trusted pubkey → exits with the
        #    TRUST_ERROR exit code (4, per §11.11).
        unrelated = "ed25519:" + "cc" * 32
        rc = _run_cli(
            "manifest",
            "verify",
            str(manifest_path),
            "--trusted-pubkey",
            unrelated,
        )
        assert rc == 4


class TestInitScaffolder:
    """`quinkgl init` produces a ready-to-edit project layout (Appendix D.1)."""

    def test_minimal_template_produces_expected_files(
        self, tmp_path: Path, capsys
    ):
        out_dir = tmp_path / "demo-swarm"
        rc = _run_cli(
            "init",
            "--template",
            "minimal",
            "--output-dir",
            str(out_dir),
        )
        assert rc == 0, capsys.readouterr().err
        # Minimum contract per Appendix D.1.1: an entry-point peer
        # script, a README, a pyproject, and a tests/ directory.
        assert (out_dir / "README.md").exists()
        assert (out_dir / "pyproject.toml").exists()
        assert (out_dir / "tests").is_dir()
        assert (out_dir / "peer_main.py").exists() or (
            out_dir / "peer.py"
        ).exists()
        assert (out_dir / "peer_script.py").exists() or (
            out_dir / "peer.py"
        ).exists()

    def test_init_refuses_existing_dir(self, tmp_path: Path, capsys):
        out_dir = tmp_path / "already-here"
        out_dir.mkdir()
        rc = _run_cli(
            "init",
            "--template",
            "minimal",
            "--output-dir",
            str(out_dir),
        )
        # Appendix D.1.5 requires a non-zero exit when output-dir exists.
        assert rc != 0
