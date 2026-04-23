"""`GossipNode` manifest-kwarg enforcement surface (spec §10.4).

These tests exercise the *construction-time* contract only: no IPv8 sockets
are opened and the network stack is not started.  They verify that invalid
combinations surface the correct ``ERR_NODE_*`` code so that downstream
tools (CLI, scaffolder) can translate them to user-facing exits.

Intentionally out of scope here: the actual training loop, peer discovery,
IPv8 lifecycle.  Those require a live IPv8 stack and are covered by the
network integration suite.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from quinkgl.manifest import SwarmManifest
from quinkgl.manifest import errors as E
from quinkgl.aggregation import FedAvg, TrimmedMean
from quinkgl.topology import RandomTopology, CyclonTopology

# The enforcement layer lives in ``quinkgl.network.gossip_node`` but the
# spec mandates public import via ``quinkgl.GossipNode``.
from quinkgl import GossipNode


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _DummyModel:
    """Stand-in ``ModelWrapper`` — enough surface to satisfy construction."""

    def __init__(self, schema_hash: str = "deadbeef", version: str = "v1"):
        self._schema = schema_hash
        self._version = version

    def get_data_schema_hash(self) -> str:
        return self._schema

    def get_model_version(self) -> str:
        return self._version


def _manifest(
    *,
    aggregation_name: str = "FedAvg",
    topology_name: str = "Random",
    signature: Optional[Dict[str, Any]] = None,
) -> SwarmManifest:
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="NodeTest",
        aggregation_name=aggregation_name,
        topology_name=topology_name,
        signature=signature,
    )


# Tests attempt to keep expensive subsystems (IPv8, telemetry) silent.
@pytest.fixture(autouse=True)
def _quiet_node(monkeypatch: pytest.MonkeyPatch):
    # Silence the terminal observer side-effect in __init__.
    yield


# ---------------------------------------------------------------------------
# Mutual-exclusion: manifest vs domain
# ---------------------------------------------------------------------------


class TestManifestVsDomain:
    def test_both_manifest_and_domain_rejected(self):
        with pytest.raises(ValueError) as exc:
            GossipNode(
                node_id="n1",
                domain="health",
                model=_DummyModel(),
                manifest=_manifest(),
                quiet=True,
            )
        assert exc.value.args[0] == E.ERR_NODE_NO_MANIFEST

    def test_neither_manifest_nor_domain_rejected(self):
        with pytest.raises(ValueError) as exc:
            GossipNode(node_id="n1", model=_DummyModel(), quiet=True)
        assert exc.value.args[0] == E.ERR_NODE_NO_MANIFEST

    def test_manifest_only_succeeds(self):
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(),
            quiet=True,
        )
        # With manifest, the node still exposes ``domain`` (derived) so that
        # existing downstream callers reading ``self.domain`` keep working.
        assert isinstance(node.domain, str) and node.domain
        assert node.manifest is not None

    def test_domain_only_succeeds_legacy_path(self):
        node = GossipNode(
            node_id="n1",
            domain="health",
            model=_DummyModel(),
            quiet=True,
        )
        assert node.domain == "health"
        assert node.manifest is None


# ---------------------------------------------------------------------------
# Aggregation / topology class-name match (strict)
# ---------------------------------------------------------------------------


class TestStrictAggregationMatch:
    def test_aggregation_name_mismatch_raises(self):
        with pytest.raises(ValueError) as exc:
            GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(aggregation_name="FedAvg"),
                aggregation=TrimmedMean(),
                quiet=True,
            )
        assert exc.value.args[0] == E.ERR_NODE_AGGREGATION_MISMATCH

    def test_aggregation_name_match_succeeds(self):
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(aggregation_name="FedAvg"),
            aggregation=FedAvg(),
            quiet=True,
        )
        assert type(node.gl_node.aggregator.aggregator).__name__ == "FedAvg"

    def test_non_strict_skips_aggregation_check(self):
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(aggregation_name="FedAvg"),
            aggregation=TrimmedMean(),
            strict_manifest=False,
            quiet=True,
        )
        # No raise — non-strict lets the mismatch through.
        assert node is not None


class TestStrictTopologyMatch:
    def test_topology_name_mismatch_raises(self):
        with pytest.raises(ValueError) as exc:
            GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(topology_name="Random"),
                topology=CyclonTopology(),
                quiet=True,
            )
        assert exc.value.args[0] == E.ERR_NODE_TOPOLOGY_MISMATCH

    def test_topology_name_match_succeeds(self):
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(topology_name="Random"),
            topology=RandomTopology(),
            quiet=True,
        )
        assert type(node.gl_node.aggregator.topology).__name__ == "RandomTopology"


# ---------------------------------------------------------------------------
# Trust policy
# ---------------------------------------------------------------------------


class TestTrustPolicy:
    def test_invalid_trust_policy_raises_value_error(self):
        with pytest.raises(ValueError):
            GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(),
                trust_policy="paranoid",
                quiet=True,
            )

    @pytest.mark.parametrize("policy", ["open", "tofu", "pinned"])
    def test_valid_trust_policies_accepted(self, policy: str):
        if policy == "pinned":
            # Pinned with a signature is fine.
            node = GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(
                    signature={
                        "algorithm": "ed25519",
                        "value": "a" * 128,
                        "signer_pubkey": "b" * 64,
                    }
                ),
                trust_policy=policy,
                trusted_creator_pubkeys={bytes.fromhex("cd" * 32)},
                quiet=True,
            )
        else:
            node = GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(),
                trust_policy=policy,
                quiet=True,
            )
        assert node.trust_policy == policy

    def test_pinned_without_signature_rejected(self):
        with pytest.raises(ValueError) as exc:
            GossipNode(
                node_id="n1",
                model=_DummyModel(),
                manifest=_manifest(signature=None),
                trust_policy="pinned",
                quiet=True,
            )
        assert exc.value.args[0] == E.ERR_NODE_UNSIGNED_MANIFEST_REJECTED


# ---------------------------------------------------------------------------
# from_domain legacy shim
# ---------------------------------------------------------------------------


class TestFromDomain:
    def test_from_domain_classmethod_builds_legacy_node(self):
        node = GossipNode.from_domain(
            node_id="n1",
            domain="health",
            model=_DummyModel(),
            quiet=True,
        )
        assert node.domain == "health"
        assert node.manifest is None

    def test_from_domain_rejects_manifest_kwarg(self):
        with pytest.raises(TypeError):
            GossipNode.from_domain(
                node_id="n1",
                domain="health",
                model=_DummyModel(),
                manifest=_manifest(),
                quiet=True,
            )


# ---------------------------------------------------------------------------
# train() + async context manager surfaces
# ---------------------------------------------------------------------------


class TestTrainSurface:
    def test_train_requires_positive_rounds(self):
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(),
            quiet=True,
        )

        async def _run():
            with pytest.raises(ValueError):
                await node.train(rounds=0)
            with pytest.raises(ValueError):
                await node.train(rounds=-1)

        asyncio.run(_run())

    def test_async_context_manager_exposes_start_and_stop(self):
        """The protocol surface (``__aenter__``/``__aexit__``) must exist."""
        node = GossipNode(
            node_id="n1",
            model=_DummyModel(),
            manifest=_manifest(),
            quiet=True,
        )
        assert hasattr(node, "__aenter__")
        assert hasattr(node, "__aexit__")
        assert asyncio.iscoroutinefunction(node.__aenter__)
        assert asyncio.iscoroutinefunction(node.__aexit__)
        assert asyncio.iscoroutinefunction(node.train)
