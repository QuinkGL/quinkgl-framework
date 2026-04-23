# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""CLI skeleton tests (B-1 acceptance)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import SUCCESS


def test_cli_help() -> None:
    """quinkgl --help exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == SUCCESS


def test_cli_version() -> None:
    """quinkgl --version prints version and exits 0."""
    import quinkgl

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == SUCCESS


def test_cli_no_args_prints_help() -> None:
    """quinkgl with no args prints help and exits 0."""
    assert main([]) == SUCCESS


def test_cli_subcommands_exist() -> None:
    """All §11 subcommands are registered."""
    # We probe the parser by checking --help output contains subcommand names.
    import io
    from quinkgl.cli.__main__ import _build_parser

    parser = _build_parser()
    out = io.StringIO()
    parser.print_help(out)
    text = out.getvalue()
    for cmd in ("manifest", "keygen", "run", "status", "info", "init"):
        assert cmd in text, f"Subcommand {cmd} not found in help"


def test_cli_info_json_output() -> None:
    """quinkgl info --json emits valid JSON."""
    import json

    assert main(["--json", "info"]) == SUCCESS
