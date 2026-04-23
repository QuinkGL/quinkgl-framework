"""Regression tests for `quinkgl.manifest.errors` (spec §19).

Every constant listed under spec §19 MUST exist with exactly its own name as
the string value.  The `__all__` list MUST cover all of them; ``import *``
MUST expose them verbatim.  This file is the authoritative cross-check between
the normative spec (§19) and the interface contract (TASKS_SPLIT §0.1).
"""

from __future__ import annotations

import importlib

import pytest

# Spec §19 — authoritative list.  Grouped for readability only; order does not
# matter.  If spec §19 gains or loses a constant, update both this tuple and
# `quinkgl.manifest.errors.__all__`.
SPEC_SECTION_19 = (
    # §19.1 Manifest
    "ERR_MANIFEST_INVALID_JSON",
    "ERR_MANIFEST_NOT_OBJECT",
    "ERR_MANIFEST_SCHEMA_VERSION",
    "ERR_MANIFEST_UNKNOWN_KEYS",
    "ERR_MANIFEST_MISSING_KEYS",
    "ERR_MANIFEST_FIELD_INVALID",
    "ERR_MANIFEST_EXPIRED",
    "ERR_MANIFEST_DATA_POLICY",
    "ERR_MANIFEST_HASH_MISMATCH",
    "ERR_MANIFEST_FETCH_REQUIRED",
    # §19.2 Magnet
    "ERR_MAGNET_SCHEME",
    "ERR_MAGNET_XT",
    "ERR_MAGNET_DUPLICATE",
    # §19.3 Node
    "ERR_NODE_NO_MANIFEST",
    "ERR_NODE_AGGREGATION_MISMATCH",
    "ERR_NODE_TOPOLOGY_MISMATCH",
    "ERR_NODE_UNSIGNED_MANIFEST_REJECTED",
    "ERR_NODE_ARCH_MISMATCH",
    "ERR_NODE_DATA_SHAPE_MISMATCH",
    "ERR_RUN_NO_STANDARD_MODEL",
    "ERR_SCRIPT_CALLABLES_MISSING",
    # §19.4 Trust / Signing (Phase 2 consumers, surface ready in Phase 1)
    "ERR_TRUST_POLICY_VIOLATION",
    "ERR_TRUST_TOFU_CONFLICT",
    "ERR_SIGNING_UNAVAILABLE",
    "ERR_SIGNATURE_INVALID",
    "ERR_CREATOR_NOT_TRUSTED",
    # §19.5 Wire
    "ERR_WIRE_UNKNOWN_SWARM",
    "ERR_WIRE_RATE_LIMITED",
    "ERR_WIRE_TIMEOUT",
    "ERR_WIRE_CHUNK_INCONSISTENT",
)

# TASKS_SPLIT §0.1 subset Track B relies on — must be a proper subset of §19.
INTERFACE_CONTRACT_SUBSET = (
    "ERR_MANIFEST_INVALID_JSON",
    "ERR_MANIFEST_NOT_OBJECT",
    "ERR_MANIFEST_SCHEMA_VERSION",
    "ERR_MANIFEST_UNKNOWN_KEYS",
    "ERR_MANIFEST_MISSING_KEYS",
    "ERR_MANIFEST_FIELD_INVALID",
    "ERR_MANIFEST_EXPIRED",
    "ERR_MANIFEST_HASH_MISMATCH",
    "ERR_MANIFEST_FETCH_REQUIRED",
    "ERR_MAGNET_SCHEME",
    "ERR_MAGNET_XT",
    "ERR_MAGNET_DUPLICATE",
    "ERR_NODE_NO_MANIFEST",
    "ERR_NODE_AGGREGATION_MISMATCH",
    "ERR_NODE_TOPOLOGY_MISMATCH",
    "ERR_NODE_UNSIGNED_MANIFEST_REJECTED",
    "ERR_NODE_ARCH_MISMATCH",
    "ERR_NODE_DATA_SHAPE_MISMATCH",
    "ERR_RUN_NO_STANDARD_MODEL",
    "ERR_SCRIPT_CALLABLES_MISSING",
    "ERR_WIRE_UNKNOWN_SWARM",
    "ERR_WIRE_TIMEOUT",
    "ERR_WIRE_CHUNK_INCONSISTENT",
)


@pytest.fixture(scope="module")
def errors_module():
    return importlib.import_module("quinkgl.manifest.errors")


def test_every_spec_constant_is_defined(errors_module) -> None:
    missing = [name for name in SPEC_SECTION_19 if not hasattr(errors_module, name)]
    assert not missing, f"Spec §19 constants missing from module: {missing}"


def test_constants_are_string_and_self_valued(errors_module) -> None:
    for name in SPEC_SECTION_19:
        value = getattr(errors_module, name)
        assert isinstance(value, str), f"{name} must be str, got {type(value)!r}"
        assert value == name, (
            f"{name} value must equal its own identifier (stable telemetry "
            f"tag). Got {value!r}."
        )


def test_constant_values_are_unique(errors_module) -> None:
    values = [getattr(errors_module, n) for n in SPEC_SECTION_19]
    assert len(set(values)) == len(values), "Duplicate ERR_* values detected"


def test_all_list_matches_spec(errors_module) -> None:
    exported = set(errors_module.__all__)
    spec = set(SPEC_SECTION_19)
    assert exported == spec, (
        f"__all__ mismatch. Missing: {spec - exported}. Extra: {exported - spec}."
    )


def test_star_import_exposes_every_constant() -> None:
    ns: dict[str, object] = {}
    exec("from quinkgl.manifest.errors import *", ns)
    for name in SPEC_SECTION_19:
        assert name in ns, f"{name} not exposed via `import *`"


def test_interface_contract_subset_is_contained() -> None:
    spec = set(SPEC_SECTION_19)
    contract = set(INTERFACE_CONTRACT_SUBSET)
    assert contract <= spec, (
        "TASKS_SPLIT §0.1 Track B subset drifted from spec §19. "
        f"Out-of-spec names: {contract - spec}"
    )


def test_raise_round_trip(errors_module) -> None:
    """Canonical raise form from §19 preamble: `ValueError(CODE, extra)`."""
    code = errors_module.ERR_MANIFEST_INVALID_JSON
    with pytest.raises(ValueError) as exc_info:
        raise ValueError(code, {"path": "x.qgl"})
    assert exc_info.value.args[0] == code
    assert exc_info.value.args[1] == {"path": "x.qgl"}
