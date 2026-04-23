# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""CLI signing surface: ``quinkgl keygen`` + ``manifest create --sign-with`` +
``manifest verify`` against the exit-code contract in §11.7 and §11.11.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import (
    CRYPTO_ERROR,
    SUCCESS,
    TRUST_ERROR,
)


@pytest.fixture
def base_create_args() -> list[str]:
    """Minimal ``manifest create`` args; caller appends ``--output <path>``."""
    return [
        "manifest", "create",
        "--name", "signed-swarm",
        "--task-type", "class",
        "--input-shape", "3",
        "--output-shape", "2",
        "--label-type", "integer",
        "--model-framework", "custom",
        "--model-arch-hash", "sha256:" + "a" * 64,
        "--aggregation", "FedAvg",
        "--topology", "Random",
    ]


# --- `quinkgl keygen` ------------------------------------------------------


class TestKeygenCLI:
    def test_writes_pem_and_prints_pubkey(self, tmp_path: Path, capsys):
        pem_path = tmp_path / "peer.pem"
        rc = main(["keygen", "--output", str(pem_path)])
        assert rc == SUCCESS

        captured = capsys.readouterr()
        assert pem_path.exists()
        assert pem_path.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
        # stdout ends with ed25519:<64-hex> so operators can pipe it.
        last_line = captured.out.strip().splitlines()[-1]
        assert last_line.startswith("ed25519:")
        assert len(last_line) == len("ed25519:") + 64
        # Security notice lives on stderr (never stdout).
        assert "0600" in captured.err or "secret" in captured.err.lower()

    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="POSIX permissions only"
    )
    def test_pem_permissions_are_0600(self, tmp_path: Path):
        pem_path = tmp_path / "peer.pem"
        assert main(["keygen", "--output", str(pem_path)]) == SUCCESS
        mode = os.stat(pem_path).st_mode & 0o777
        assert mode == 0o600

    def test_refuses_overwrite_without_flag(self, tmp_path: Path):
        pem_path = tmp_path / "peer.pem"
        assert main(["keygen", "--output", str(pem_path)]) == SUCCESS
        # §11.7 exit 3 for "file exists without overwrite".
        rc = main(["keygen", "--output", str(pem_path)])
        assert rc == CRYPTO_ERROR

    def test_overwrite_flag_replaces_key(self, tmp_path: Path, capsys):
        pem_path = tmp_path / "peer.pem"
        assert main(["keygen", "--output", str(pem_path)]) == SUCCESS
        rc = main(["keygen", "--output", str(pem_path), "--overwrite"])
        assert rc == SUCCESS

    def test_print_public_only_writes_no_file(self, tmp_path: Path, capsys):
        pem_path = tmp_path / "peer.pem"
        rc = main(["keygen", "--print-public-only"])
        assert rc == SUCCESS
        captured = capsys.readouterr()
        assert not pem_path.exists()
        last_line = captured.out.strip().splitlines()[-1]
        assert last_line.startswith("ed25519:")


# --- `quinkgl manifest create --sign-with` ---------------------------------


class TestManifestCreateWithSigning:
    def _keygen(self, tmp_path: Path) -> Path:
        pem_path = tmp_path / "creator.pem"
        assert main(["keygen", "--output", str(pem_path)]) == SUCCESS
        return pem_path

    def test_sign_with_produces_signed_manifest(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        pem = self._keygen(tmp_path)
        out = tmp_path / "signed.qgl"
        rc = main(base_create_args + ["--sign-with", str(pem), "--output", str(out)])
        # Drain captured output from the keygen + create runs.
        capsys.readouterr()
        assert rc == SUCCESS
        data = json.loads(out.read_text())
        assert data["creator_pubkey"] is not None
        assert data["signature"] is not None
        assert data["signature"].startswith("ed25519:")
        assert data["creator_pubkey"].startswith("ed25519:")

    def test_sign_with_missing_key_fails_cleanly(
        self, tmp_path: Path, base_create_args: list[str]
    ):
        out = tmp_path / "signed.qgl"
        bogus = tmp_path / "does-not-exist.pem"
        rc = main(base_create_args + ["--sign-with", str(bogus), "--output", str(out)])
        # FileNotFound → IO_ERROR (exit 2); file never written.
        assert rc == 2
        assert not out.exists()


# --- `quinkgl manifest verify` ---------------------------------------------


class TestManifestVerifyCLI:
    def _make_signed(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ) -> Path:
        pem_path = tmp_path / "creator.pem"
        assert main(["keygen", "--output", str(pem_path)]) == SUCCESS
        out = tmp_path / "signed.qgl"
        assert main(
            base_create_args + ["--sign-with", str(pem_path), "--output", str(out)]
        ) == SUCCESS
        capsys.readouterr()
        return out

    def test_verify_accepts_good_signature(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        path = self._make_signed(tmp_path, base_create_args, capsys)
        assert main(["manifest", "verify", str(path)]) == SUCCESS

    def test_verify_rejects_tampered_manifest(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        path = self._make_signed(tmp_path, base_create_args, capsys)
        # Mutate name inside the signed manifest — signature must fail.
        data = json.loads(path.read_text())
        data["name"] = "hijacked"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        rc = main(["manifest", "verify", str(path)])
        assert rc == CRYPTO_ERROR

    def test_verify_trusted_pubkey_accepts_known_creator(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        path = self._make_signed(tmp_path, base_create_args, capsys)
        data = json.loads(path.read_text())
        trusted = data["creator_pubkey"]
        rc = main(
            ["manifest", "verify", str(path), "--trusted-pubkey", trusted]
        )
        assert rc == SUCCESS

    def test_verify_trusted_pubkey_rejects_unknown_creator(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        path = self._make_signed(tmp_path, base_create_args, capsys)
        foreign = "ed25519:" + "0" * 64
        rc = main(
            ["manifest", "verify", str(path), "--trusted-pubkey", foreign]
        )
        assert rc == TRUST_ERROR

    def test_verify_unsigned_manifest_still_valid(
        self, tmp_path: Path, base_create_args: list[str], capsys
    ):
        """An unsigned manifest without trust flags must still exit 0 — the
        CLI is backwards-compatible with Phase-1 workflows."""
        out = tmp_path / "unsigned.qgl"
        assert main(base_create_args + ["--output", str(out)]) == SUCCESS
        capsys.readouterr()
        assert main(["manifest", "verify", str(out)]) == SUCCESS
