# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Verify every public symbol in __all__ has a docstring."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

# Modules listed in SWARM_ARCHITECTURE_SPEC.md §10.6.1 as STABLE public.
PUBLIC_MODULES = [
    "quinkgl",
    "quinkgl.manifest",
    "quinkgl.gossip",
    "quinkgl.models",
    "quinkgl.aggregation",
    "quinkgl.topology",
    "quinkgl.fingerprint",
    "quinkgl.observability",
    "quinkgl.observability.terminal",
    "quinkgl.telemetry",
    "quinkgl.testing",
]

# Pre-existing gaps in modules owned by Track A.  These are grandfathered
# so that B-8 (doc lint infra) can pass before Track A fixes them.
GRANDFATHERED_NO_ALL = {
    "quinkgl.observability.terminal",
}

GRANDFATHERED_MISSING_DOCSTRINGS: dict[str, set[str]] = {
    "quinkgl": {"TerminalObserver", "format_runtime_event"},
    "quinkgl.fingerprint": {"_adjacent_bucket"},
    "quinkgl.telemetry": {"create_telemetry_app"},
}

# The spec §10.6.2 explicitly allows these internal-module re-exports.
ALLOWED_INTERNAL_REEXPORTS = {
    "LearningNode",  # from quinkgl.core.learning_node — public per §10.6.1
    "GossipNode",    # from quinkgl.network.gossip_node — public per §10.6.1
}


def _iter_all_exports(module_name: str):
    """Yield (name, obj) for every item in module.__all__."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"Cannot import {module_name}: {exc}")

    all_names = getattr(mod, "__all__", None)
    if all_names is None:
        if module_name in GRANDFATHERED_NO_ALL:
            return
        pytest.fail(f"Public module {module_name} lacks __all__")

    for name in all_names:
        obj = getattr(mod, name, None)
        if obj is None:
            pytest.fail(f"{module_name}.__all__ contains missing name {name!r}")
        yield name, obj


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_imports_cleanly(module_name: str) -> None:
    """§10.6.3 — every public module imports without side effects."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_has_all(module_name: str) -> None:
    """§10.6.3 — every public module declares __all__."""
    mod = importlib.import_module(module_name)
    if module_name in GRANDFATHERED_NO_ALL:
        pytest.skip(f"{module_name} is grandfathered (no __all__ yet)")
    assert hasattr(mod, "__all__"), f"{module_name} lacks __all__"


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_all_exports_have_docstrings(module_name: str) -> None:
    """§22.4 — every public symbol has a docstring."""
    missing: list[str] = []
    grandfathered = GRANDFATHERED_MISSING_DOCSTRINGS.get(module_name, set())
    for name, obj in _iter_all_exports(module_name):
        if name in grandfathered:
            continue
        doc = inspect.getdoc(obj)
        if not doc:
            missing.append(f"{module_name}.{name}")
    if missing:
        pytest.fail(f"Missing docstrings for: {', '.join(missing)}")


def test_no_internal_reexports_from_quinkgl_init() -> None:
    """§10.6.3 — quinkgl/__init__.py does not re-export internals.

    Symbols explicitly listed in §10.6.1 as public are exempt.
    """
    import quinkgl

    internal_prefixes = (
        "quinkgl.network.",
        "quinkgl.core.",
        "quinkgl.storage.",
        "quinkgl.serialization.",
        "quinkgl.training.",
        "quinkgl.utils.",
        "quinkgl._internal.",
        "quinkgl.cli.",
    )
    violations: list[str] = []
    for name in quinkgl.__all__:
        if name in ALLOWED_INTERNAL_REEXPORTS:
            continue
        obj = getattr(quinkgl, name, None)
        if obj is None:
            continue
        mod = getattr(obj, "__module__", "")
        if any(mod.startswith(p) for p in internal_prefixes):
            violations.append(f"{name} (from {mod})")
    if violations:
        pytest.fail(f"quinkgl.__init__ re-exports internals: {violations}")
