# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Verify every doc page's 'Conforms to' line matches current spec version."""

from __future__ import annotations

import pytest

from .conftest import DOCS_DIR, REPO_ROOT, extract_conforms_lines, get_spec_version, walk_docs


def test_spec_version_sync() -> None:
    """§22.11.6 — 'Conforms to' lines match current spec version."""
    spec_version = get_spec_version()
    if not spec_version:
        pytest.skip("Could not parse spec version from SWARM_ARCHITECTURE_SPEC.md")

    mismatches: list[str] = []
    for path in walk_docs():
        versions = extract_conforms_lines(path)
        for v in versions:
            if v != spec_version:
                mismatches.append(
                    f"{path.relative_to(REPO_ROOT)} says v{v}, expected v{spec_version}"
                )

    if mismatches:
        pytest.fail("Spec version mismatches:\n" + "\n".join(mismatches))
