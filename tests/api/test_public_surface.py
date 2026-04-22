"""Public API surface contract (spec §10.6).

These tests pin down the exact public export set promised by §10.6.1 and
the enforcement rules of §10.6.3:

1. Every module in the stable public surface imports cleanly.
2. Each public module's ``__all__`` matches the spec table (CURRENT +
   PLANNED-Phase-1 additions that are now CURRENT).
3. No internal module path appears as a re-export on ``quinkgl`` itself.
4. ``quinkgl.check_compatibility`` behaves as §10.6.5 mandates.

When a Phase-2 or later export is added, update the corresponding set
below and cite the spec subsection.
"""

from __future__ import annotations

import importlib

import pytest

from quinkgl.manifest import MANIFEST_SCHEMA_VERSION, SwarmManifest
from quinkgl.manifest import errors as E


# ---------------------------------------------------------------------------
# Expected public exports (spec §10.6.1, CURRENT + Phase 1 additions)
# ---------------------------------------------------------------------------

# Top-level `quinkgl`: CURRENT table + the Phase 1 additions that A-3..A-12
# now promote to CURRENT.  Exports not yet implemented (`Phase 2+` rows such
# as TrustPolicy / TrainingMetrics) are deliberately omitted.
EXPECTED_TOPLEVEL = {
    # Core
    "LearningNode",
    "GLNode",  # deprecated alias
    "GossipNode",
    # Models
    "ModelWrapper",
    "TrainingConfig",
    "TrainingResult",
    "ModelSplit",
    "PersonalizedModelWrapper",
    "APFLConfig",
    "APFLMixin",
    "PyTorchModel",
    "PyTorchPersonalizedModel",
    # Topology
    "TopologyStrategy",
    "RandomTopology",
    "CyclonTopology",
    "AffinityTopology",
    "SpectralAnalyzer",
    "SpectralReport",
    "PeerInfo",
    "SelectionContext",
    # Fingerprint
    "DataFingerprint",
    "FINGERPRINT_SCHEMA_VERSION",
    "AffinityWeights",
    "FingerprintPrivacyConfig",
    "FingerprintComputer",
    # Aggregation
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "FedAvgM",
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    "StalenessWeightedFedAvg",
    "EntropyWeightedAvg",
    "Scaffold",
    "ModelUpdate",
    "AggregatedModel",
    # Manifest (CURRENT + Phase 1 re-exports)
    "MANIFEST_SCHEMA_VERSION",
    "DataPolicy",
    "CollaborationPolicy",
    "PersonalizationPolicy",
    "PrototypePolicy",
    "SwarmManifest",
    "check_compatibility",
    # Gossip
    "ModelAggregator",
    # Observability
    "EventEmitter",
    "RuntimeEvent",
    "TerminalObserver",
    "format_runtime_event",
    "TelemetryClient",
    # Feature flags (prefixed underscore is acceptable at package root —
    # these are documented in __all__ to keep ``from quinkgl import *``
    # self-documenting when `hasattr` checks are used downstream).
    "_tensorflow_available",
    "_data_available",
}

# `quinkgl.manifest`: CURRENT + Phase 1 additions now promoted.  The spec
# table doesn't require ``ModelSpec`` / ``TaskSpec`` / ``ByzantineSpec`` as
# public re-exports but the current module exposes them for dataclass
# composition in Mode B user scripts; those are tolerated extras.
EXPECTED_MANIFEST_REQUIRED = {
    "MANIFEST_SCHEMA_VERSION",
    "DataPolicy",
    "CollaborationPolicy",
    "PersonalizationPolicy",
    "PrototypePolicy",
    "SwarmManifest",
    # Phase 1 additions
    "MagnetLink",
    "parse_magnet",
    "format_magnet",
    "load_manifest",
    "compute_arch_hash",
    "check_compatibility",
}

# `quinkgl.manifest.errors`: every ERR_* constant promoted in A-1.
EXPECTED_ERRORS_PREFIX = "ERR_"


# ---------------------------------------------------------------------------
# 1. Module import sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "quinkgl",
        "quinkgl.manifest",
        "quinkgl.manifest.errors",
        "quinkgl.gossip",
        "quinkgl.models",
        "quinkgl.aggregation",
        "quinkgl.topology",
        "quinkgl.fingerprint",
        "quinkgl.observability",
        "quinkgl.observability.terminal",
        "quinkgl.telemetry",
    ],
)
def test_public_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. __all__ enforcement
# ---------------------------------------------------------------------------


def _exported(module_name: str) -> set:
    mod = importlib.import_module(module_name)
    exported = set(getattr(mod, "__all__", []))
    # Conditional (optional-install) exports are tolerated as long as the
    # attribute is present on the module; TensorFlowModel is the canonical
    # example.
    return exported


class TestTopLevelSurface:
    def test_all_declared(self):
        import quinkgl

        assert hasattr(quinkgl, "__all__"), "quinkgl must declare __all__"
        assert isinstance(quinkgl.__all__, list)

    def test_all_matches_expected(self):
        import quinkgl

        actual = set(quinkgl.__all__)
        # TensorFlowModel is optional — only present when TF is installed.
        actual.discard("TensorFlowModel")
        missing = EXPECTED_TOPLEVEL - actual
        extra = actual - EXPECTED_TOPLEVEL
        assert not missing, f"missing from quinkgl.__all__: {sorted(missing)}"
        assert not extra, f"unexpected in quinkgl.__all__: {sorted(extra)}"

    def test_every_entry_resolves(self):
        import quinkgl

        for name in quinkgl.__all__:
            assert hasattr(quinkgl, name), (
                f"{name!r} declared in __all__ but not defined on quinkgl"
            )


class TestManifestSurface:
    def test_phase1_additions_exported(self):
        actual = _exported("quinkgl.manifest")
        missing = EXPECTED_MANIFEST_REQUIRED - actual
        assert not missing, (
            f"quinkgl.manifest is missing required Phase 1 exports: "
            f"{sorted(missing)}"
        )

    def test_every_entry_resolves(self):
        mod = importlib.import_module("quinkgl.manifest")
        for name in mod.__all__:
            assert hasattr(mod, name)


class TestErrorsSurface:
    def test_all_entries_are_err_prefixed(self):
        mod = importlib.import_module("quinkgl.manifest.errors")
        # The `__all__` is a tuple in errors.py — cast to set for comparison.
        exported = set(mod.__all__)
        assert exported, "quinkgl.manifest.errors.__all__ must be non-empty"
        for name in exported:
            assert name.startswith(EXPECTED_ERRORS_PREFIX), (
                f"{name!r} in quinkgl.manifest.errors.__all__ is not ERR_-prefixed"
            )
            assert hasattr(mod, name)


# ---------------------------------------------------------------------------
# 3. Internal modules must not leak into quinkgl re-exports
# ---------------------------------------------------------------------------


INTERNAL_PREFIXES = (
    "quinkgl.network.",  # except GossipNode re-exported via top-level
    "quinkgl.core.",
    "quinkgl.storage.",
    "quinkgl.serialization.",
    "quinkgl.training.",
    "quinkgl.utils.",
    "quinkgl._internal.",
    "quinkgl.cli.",
)


class TestInternalLeakage:
    def test_top_level_names_not_pointed_at_internal_modules(self):
        """Each public symbol on ``quinkgl`` MUST resolve to a class/function
        whose ``__module__`` is either a public module OR a valid internal
        implementation path (the latter is fine: only the *import path* is
        public).  This test catches the accidental case where ``__all__``
        lists a submodule (e.g. ``"quinkgl.network"``) which would encourage
        users to dot-walk into IPv8 internals.
        """
        import quinkgl

        for name in quinkgl.__all__:
            obj = getattr(quinkgl, name)
            # Modules as re-exports are forbidden — only classes, functions,
            # constants, or feature flags.
            import types

            assert not isinstance(obj, types.ModuleType), (
                f"quinkgl.{name} re-exports a module ({obj!r}); users should "
                f"import submodules directly, not via the top-level package"
            )


# ---------------------------------------------------------------------------
# 4. check_compatibility behaviour (§10.6.5)
# ---------------------------------------------------------------------------


class TestCheckCompatibility:
    def test_accepts_current_version_manifest(self):
        import quinkgl

        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
            name="t",
        )
        # MANIFEST_SCHEMA_VERSION is the default; must not raise.
        quinkgl.check_compatibility(m)

    def test_accepts_older_version_manifest(self):
        """Older fixtures (N-1) MUST remain loadable — forward-compat for
        peers that persist manifests to disk."""
        import quinkgl

        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
            name="t",
        )
        if MANIFEST_SCHEMA_VERSION <= 1:
            pytest.skip("no older schema version to exercise on this build")
        m.schema_version = MANIFEST_SCHEMA_VERSION - 1
        quinkgl.check_compatibility(m)

    def test_rejects_future_version_manifest(self):
        import quinkgl

        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
            name="t",
        )
        m.schema_version = MANIFEST_SCHEMA_VERSION + 1
        with pytest.raises(ValueError) as exc_info:
            quinkgl.check_compatibility(m)
        assert exc_info.value.args[0] == E.ERR_MANIFEST_SCHEMA_VERSION
        detail = exc_info.value.args[1]
        assert detail["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION + 1
        assert detail["supported_schema_version"] == MANIFEST_SCHEMA_VERSION
        # Message MUST guide the user to upgrade (spec §10.6.5).
        assert "upgrade" in detail["detail"].lower()

    def test_rejects_missing_schema_version(self):
        import quinkgl

        class _Fake:
            pass

        with pytest.raises(ValueError) as exc_info:
            quinkgl.check_compatibility(_Fake())
        assert exc_info.value.args[0] == E.ERR_MANIFEST_SCHEMA_VERSION
