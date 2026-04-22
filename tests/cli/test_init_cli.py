# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Init scaffolder tests (B-7 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import IO_ERROR, SUCCESS


class TestInitMinimal:
    def test_init_creates_expected_layout(self, tmp_path: Path) -> None:
        out = tmp_path / "my-peer"
        assert main(["init", "--output-dir", str(out), "--template", "minimal"]) == SUCCESS
        expected = [
            "pyproject.toml",
            "README.md",
            ".gitignore",
            "peer_script.py",
            "peer_main.py",
            "conftest.py",
            "tests/test_build_model.py",
            "tests/test_build_loaders.py",
            "tests/test_integration.py",
        ]
        for rel in expected:
            assert (out / rel).exists(), f"Missing {rel}"

    def test_init_existing_dir_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "exists"
        out.mkdir()
        assert main(["init", "--output-dir", str(out)]) == IO_ERROR

    def test_init_no_unrendered_placeholders(self, tmp_path: Path) -> None:
        out = tmp_path / "my-peer"
        assert main(["init", "--output-dir", str(out), "--template", "minimal"]) == SUCCESS
        for p in out.rglob("*"):
            if p.is_file() and p.suffix not in {".pyc", ".pyo"}:
                text = p.read_text()
                assert "{" not in text or "}" not in text, f"Unrendered placeholder in {p.name}"


@pytest.mark.parametrize("template", ["minimal", "pytorch-vision", "pytorch-tabular", "custom"])
def test_init_all_templates(template: str, tmp_path: Path) -> None:
    out = tmp_path / f"peer-{template}"
    assert main(["init", "--output-dir", str(out), "--template", template]) == SUCCESS
    assert (out / "pyproject.toml").exists()
    assert (out / "peer_script.py").exists()
    assert (out / "peer_main.py").exists()
    assert (out / "tests" / "test_build_model.py").exists()
