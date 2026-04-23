"""Join-flow state machine tests (spec §14).

These tests drive the state transitions directly on a constructed
:class:`GossipNode` without standing up an IPv8 listener.  The
full-stack "start → discover → train" path is covered by the CLI run
integration tests; here we pin down:

* initial-state invariants (with/without manifest),
* the legal transition graph,
* event emission on every transition, and
* error/failure paths at every step.
"""

from __future__ import annotations

import pytest

from quinkgl.manifest import SwarmManifest
from quinkgl.models import PyTorchModel
from quinkgl.network.gossip_node import GossipNode, NodeState


def _manifest() -> SwarmManifest:
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="JoinTest",
    )


def _make_node(with_manifest: bool = True) -> GossipNode:
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    model = PyTorchModel(nn.Linear(3, 2))
    if with_manifest:
        return GossipNode(
            node_id="alice",
            model=model,
            manifest=_manifest(),
            port=0,
            enable_fallback=False,
        )
    return GossipNode(
        node_id="alice",
        domain="health",
        model=model,
        port=0,
        enable_fallback=False,
    )


# --- Initial state ---------------------------------------------------------


class TestInitialState:
    def test_manifest_constructed_node_starts_in_manifest_resolved(self):
        node = _make_node(with_manifest=True)
        assert node.state is NodeState.MANIFEST_RESOLVED
        # ``swarm_id`` MUST be populated as the 32-byte digest of the
        # canonical manifest bytes — observers rely on this for DHT work.
        assert isinstance(node.swarm_id, bytes)
        assert len(node.swarm_id) == 32

    def test_legacy_domain_only_node_starts_in_init(self):
        node = _make_node(with_manifest=False)
        assert node.state is NodeState.INIT
        assert node.swarm_id is None


# --- Happy-path transitions ------------------------------------------------


class TestHappyPath:
    def test_full_linear_progression(self):
        node = _make_node(with_manifest=True)
        assert node.state is NodeState.MANIFEST_RESOLVED

        node._transition(NodeState.COMMUNITY_STARTED, mode="ipv8_p2p")
        assert node.state is NodeState.COMMUNITY_STARTED

        node.mark_peer_discovered(peer_id="bob")
        assert node.state is NodeState.PEERS_DISCOVERED

        node.begin_training(rounds=3)
        assert node.state is NodeState.TRAINING

    def test_mark_peer_discovered_is_idempotent(self):
        node = _make_node(with_manifest=True)
        node._transition(NodeState.COMMUNITY_STARTED)
        node.mark_peer_discovered()
        node.mark_peer_discovered()  # second call: no-op
        assert node.state is NodeState.PEERS_DISCOVERED

    def test_begin_training_idempotent(self):
        node = _make_node(with_manifest=True)
        node._transition(NodeState.COMMUNITY_STARTED)
        node.mark_peer_discovered()
        node.begin_training()
        node.begin_training()  # second call: no-op
        assert node.state is NodeState.TRAINING


# --- Error paths -----------------------------------------------------------


class TestErrorPaths:
    def test_mark_failed_from_any_state(self):
        for transitions in (
            [],
            [NodeState.COMMUNITY_STARTED],
            [NodeState.COMMUNITY_STARTED, NodeState.PEERS_DISCOVERED],
            [
                NodeState.COMMUNITY_STARTED,
                NodeState.PEERS_DISCOVERED,
                NodeState.TRAINING,
            ],
        ):
            node = _make_node(with_manifest=True)
            for target in transitions:
                node._transition(target)
            node.mark_failed("boom")
            assert node.state is NodeState.FAILED

    def test_illegal_skip_raises(self):
        """Must not be able to jump directly from MANIFEST_RESOLVED to
        TRAINING — the two-step community_started / peers_discovered gate
        exists to guarantee the listener is up and at least one compatible
        peer has been observed."""
        node = _make_node(with_manifest=True)
        with pytest.raises(RuntimeError):
            node._transition(NodeState.TRAINING)

    def test_begin_training_without_peers_discovered_raises(self):
        node = _make_node(with_manifest=True)
        node._transition(NodeState.COMMUNITY_STARTED)
        with pytest.raises(RuntimeError):
            node.begin_training()

    def test_failed_is_terminal_except_for_init(self):
        node = _make_node(with_manifest=True)
        node.mark_failed("manifest_load")
        # FAILED → INIT is the only permitted exit (graceful shutdown).
        with pytest.raises(RuntimeError):
            node._transition(NodeState.COMMUNITY_STARTED)
        node._transition(NodeState.INIT, reason="reset")
        assert node.state is NodeState.INIT


# --- Event emission --------------------------------------------------------


class TestEventEmission:
    def test_each_transition_emits_node_state_event(self):
        node = _make_node(with_manifest=True)
        events = []
        node.gl_node.aggregator.event_emitter.subscribe(events.append)

        node._transition(NodeState.COMMUNITY_STARTED, mode="ipv8_p2p")
        node.mark_peer_discovered(peer_id="bob")
        node.begin_training(rounds=5)

        types = [e.event_type for e in events if e.event_type.startswith("node.state.")]
        assert types == [
            "node.state.community_started",
            "node.state.peers_discovered",
            "node.state.training",
        ]

    def test_failed_transition_emits_reason(self):
        node = _make_node(with_manifest=True)
        events = []
        node.gl_node.aggregator.event_emitter.subscribe(events.append)

        node.mark_failed("manifest_load")

        failed = [e for e in events if e.event_type == "node.state.failed"]
        assert len(failed) == 1
        assert failed[0].payload.get("reason") == "manifest_load"
        assert failed[0].payload.get("from") == "manifest_resolved"
        assert failed[0].payload.get("to") == "failed"
