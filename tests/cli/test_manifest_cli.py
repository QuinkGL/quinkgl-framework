# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Manifest CLI tests (B-2 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import (
    HASH_MISMATCH,
    IO_ERROR,
    SUCCESS,
    VALIDATION_ERROR,
)


@pytest.fixture
def sample_manifest_args() -> list[str]:
    return [
        "manifest", "create",
        "--name", "test-swarm",
        "--task-type", "class",
        "--input-shape", "3,224,224",
        "--output-shape", "10",
        "--label-type", "integer",
        "--model-framework", "pytorch",
        "--model-arch-hash", "sha256:" + "a" * 64,
        "--aggregation", "FedAvg",
        "--topology", "Random",
        "--output",
    ]


class TestManifestCreate:
    def test_create_minimal(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        args = sample_manifest_args + [str(out)]
        assert main(args) == SUCCESS
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["name"] == "test-swarm"
        assert data["schema_version"] == 3

    def test_create_invalid_arch_hash(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        args = sample_manifest_args.copy()
        # Replace the arch-hash value (which is two positions before --output)
        args[args.index("--model-arch-hash") + 1] = "bad-hash"
        args.append(str(out))
        assert main(args) == VALIDATION_ERROR


class TestManifestShow:
    def test_show_human(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["manifest", "show", str(out)]) == SUCCESS

    def test_show_json(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["--json", "manifest", "show", str(out)]) == SUCCESS

    def test_show_missing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "missing.qgl"
        assert main(["manifest", "show", str(out)]) == IO_ERROR


class TestManifestVerify:
    def test_verify_ok(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["manifest", "verify", str(out)]) == SUCCESS

    def test_verify_bad_expected_id(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["manifest", "verify", str(out), "--expected-swarm-id", "bad"]) == HASH_MISMATCH


class TestManifestMagnet:
    def test_magnet_roundtrip(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["manifest", "magnet", str(out)]) == SUCCESS

    def test_magnet_json(self, tmp_path: Path, sample_manifest_args: list[str]) -> None:
        out = tmp_path / "swarm.qgl"
        assert main(sample_manifest_args + [str(out)]) == SUCCESS
        assert main(["--json", "manifest", "magnet", str(out)]) == SUCCESS
