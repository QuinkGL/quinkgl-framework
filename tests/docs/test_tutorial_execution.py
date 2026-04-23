# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Execute runnable tutorial code repositories in CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import DOCS_DIR, REPO_ROOT


def _discover_tutorials() -> list[Path]:
    """Yield tutorial directories containing a run.sh script."""
    tutorials_dir = DOCS_DIR / "tutorials"
    if not tutorials_dir.exists():
        return []
    results = []
    for subdir in tutorials_dir.iterdir():
        run_script = subdir / "run.sh"
        if run_script.exists():
            results.append(run_script)
    return results


@pytest.mark.slow
@pytest.mark.parametrize("run_script", _discover_tutorials(), ids=lambda p: p.parent.name)
def test_tutorial_execution(run_script: Path) -> None:
    """§22.11.4 — each tutorial's code repository runs to completion."""
    result = subprocess.run(
        ["bash", str(run_script)],
        capture_output=True,
        text=True,
        cwd=str(run_script.parent),
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Tutorial {run_script.parent.name} failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_no_tutorials_is_not_a_failure() -> None:
    """If no tutorials exist yet, the suite should still pass."""
    tutorials = _discover_tutorials()
    if not tutorials:
        pytest.skip("No tutorials discovered yet")
