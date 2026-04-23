# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Shared helpers for documentation lint tests."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
SPEC_PATH = REPO_ROOT / "SWARM_ARCHITECTURE_SPEC.md"


def get_spec_version() -> str | None:
    """Extract the spec version string from SWARM_ARCHITECTURE_SPEC.md."""
    if not SPEC_PATH.exists():
        return None
    text = SPEC_PATH.read_text(encoding="utf-8")
    m = re.search(r"\*\*Version:\*\*\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    if m:
        return m.group(1)
    return None


def walk_docs() -> Iterator[Path]:
    """Yield all markdown / rst / Python files under docs/."""
    if not DOCS_DIR.exists():
        return
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".rst", ".py"}:
            yield p


def extract_conforms_lines(path: Path) -> list[str]:
    """Return every 'Conforms to ...' line found in *path*."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return re.findall(r"Conforms to SWARM_ARCHITECTURE_SPEC\.md v([0-9]+\.[0-9]+\.[0-9]+)", text)


@pytest.fixture(scope="session")
def spec_version() -> str | None:
    return get_spec_version()


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return DOCS_DIR
