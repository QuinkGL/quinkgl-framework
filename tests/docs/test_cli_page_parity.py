# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Verify every CLI subcommand has a reference page under docs/cli/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import DOCS_DIR, REPO_ROOT


def _discover_cli_subcommands() -> list[str]:
    """Return all subcommands by running `quinkgl --help`."""
    # If the CLI entry point is not installed yet, return empty list.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "quinkgl.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        if result.returncode != 0:
            # Try installed script
            result = subprocess.run(
                ["quinkgl", "--help"],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                check=False,
            )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    text = result.stdout
    # Heuristic: argparse prints subcommands under "positional arguments:"
    # as a single line like "{manifest,run,status,info,init,keygen}".
    # We look for that specific pattern after the positional arguments header.
    subcommands = []
    in_positional = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower() == "positional arguments:":
            in_positional = True
            continue
        if in_positional:
            if not stripped:
                break
            if "{" in stripped and "}" in stripped:
                inner = stripped[stripped.find("{") + 1 : stripped.find("}")]
                subcommands.extend(s.strip() for s in inner.split(","))
                break
    return sorted(set(subcommands))


def test_cli_page_parity() -> None:
    """§22.11.2 — every CLI subcommand has docs/cli/<subcmd>.md."""
    subcommands = _discover_cli_subcommands()
    if not subcommands:
        pytest.skip("No CLI subcommands discovered (CLI not yet installed)")

    missing: list[str] = []
    for cmd in subcommands:
        page = DOCS_DIR / "cli" / f"{cmd}.md"
        if not page.exists():
            missing.append(str(page.relative_to(REPO_ROOT)))

    if missing:
        pytest.fail(f"Missing CLI reference pages: {', '.join(missing)}")
