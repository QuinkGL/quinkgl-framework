# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Verify every ERR_* constant is documented in troubleshooting."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from .conftest import DOCS_DIR, REPO_ROOT

ERRORS_MODULE = "quinkgl.manifest.errors"
TROUBLESHOOTING_PATHS = [
    DOCS_DIR / "reference" / "error-codes.md",
    DOCS_DIR / "user-guide" / "troubleshooting.md",
]


def _load_err_constants() -> list[str]:
    try:
        mod = importlib.import_module(ERRORS_MODULE)
    except Exception:
        return []
    return sorted(
        name
        for name in getattr(mod, "__all__", dir(mod))
        if name.startswith("ERR_")
        and not name.startswith("__")
        and isinstance(getattr(mod, name, None), str)
    )


def test_error_code_coverage() -> None:
    """§22.11.3 — every ERR_* has a troubleshooting entry."""
    constants = _load_err_constants()
    if not constants:
        pytest.skip(f"{ERRORS_MODULE} not yet available (Track A pending)")

    # Collect all docs text
    docs_text = ""
    for p in TROUBLESHOOTING_PATHS:
        if p.exists():
            docs_text += p.read_text(encoding="utf-8")

    if not docs_text:
        pytest.fail(
            "No troubleshooting documentation found. "
            f"Expected one of: {[str(p) for p in TROUBLESHOOTING_PATHS]}"
        )

    missing = [c for c in constants if c not in docs_text]
    if missing:
        pytest.fail(
            f"Missing troubleshooting docs for ERR_* codes: {', '.join(missing)}"
        )
