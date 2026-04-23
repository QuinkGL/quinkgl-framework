# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Check that relative links inside docs/ resolve."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from .conftest import DOCS_DIR, REPO_ROOT


def _extract_relative_links(text: str) -> list[str]:
    """Find all relative markdown / reST link targets (raw)."""
    # Markdown: [text](url) or [text]: url
    md_links = re.findall(r"\[.*?\]\((.*?)\)", text)
    md_ref_links = re.findall(r"^\[.*?\]:\s*(.*?)$", text, re.MULTILINE)
    # reST: `text <url>`_ or .. _label: url
    rst_links = re.findall(r"`[^<`]*<([^>`]+)>`__?", text)
    rst_refs = re.findall(r"\.\. _[^:]+:\s*(.*)", text)

    all_links = md_links + md_ref_links + rst_links + rst_refs
    relative = []
    for link in all_links:
        link = link.split()[0]  # drop title attributes
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative.append(link)
    return relative


def test_internal_links_resolve() -> None:
    """§22.11.5 — relative links in docs must resolve."""
    broken: list[str] = []
    for path in DOCS_DIR.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".rst"}:
            continue
        text = path.read_text(encoding="utf-8")
        for link in _extract_relative_links(text):
            if link.startswith("/"):
                target = DOCS_DIR / link.lstrip("/")
            else:
                target = path.parent / link
            target = target.resolve()
            if not target.exists():
                broken.append(f"{path.relative_to(REPO_ROOT)} -> {link}")

    if broken:
        pytest.fail(f"Broken relative links:\n" + "\n".join(broken))


@pytest.mark.slow
def test_external_links_with_lychee() -> None:
    """Optional: call lychee for external URL validation (non-blocking)."""
    try:
        subprocess.run(["lychee", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("lychee not installed")

    result = subprocess.run(
        ["lychee", "--no-progress", "--exclude", "localhost", str(DOCS_DIR)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # We treat lychee output as informational in this test.
    # A separate CI step can enforce strict mode.
    if result.returncode != 0:
        pytest.fail(f"lychee found broken external links:\n{result.stdout}")
