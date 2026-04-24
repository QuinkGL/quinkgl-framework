# Copyright 2026 Ali Seyhan, Baki Turhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gossip Learning Node with IPv8 Integration and Fallback Support

Combines LearningNode with GossipLearningCommunity for full P2P gossip learning.
Automatically falls back to tunnel relay when IPv8 P2P fails.
"""

import asyncio
import hashlib
import logging
import struct
import os
import tempfile
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Set
from enum import Enum

from ipv8.keyvault.crypto import default_eccrypto

from quinkgl.core.learning_node import LearningNode
from quinkgl.manifest.errors import (
    ERR_NODE_AGGREGATION_MISMATCH,
    ERR_NODE_NO_MANIFEST,
    ERR_NODE_TOPOLOGY_MISMATCH,
    ERR_NODE_UNSIGNED_MANIFEST_REJECTED,
)
from quinkgl.manifest.schema import SwarmManifest
from quinkgl.models.base import ModelWrapper, TrainingConfig
from quinkgl.topology.base import TopologyStrategy, PeerInfo, SelectionContext
from quinkgl.aggregation.base import AggregationStrategy
from quinkgl.network.ipv8_manager import IPv8Manager
from quinkgl.network.gossip_community import generate_community_id
from quinkgl.network.gossip_community import GossipLearningCommunity
from quinkgl.network.gossip_community import MAX_ROUND_SKIP


_VALID_TRUST_POLICIES = frozenset({"open", "tofu", "pinned"})


def _coerce_trust_policy(value: Any) -> str:
    """Accept either a :class:`quinkgl.gossip.TrustPolicy` or a bare string.

    Returns the lowercase ``str`` form so every downstream check can
    keep comparing against string literals.  Raising up-front yields a
    clearer error than failing deep inside the reactor with a
    ``KeyError`` when someone passes ``TrustPolicy.TOFU`` as an int-like
    object.
    """
    from quinkgl.gossip.trust import TrustPolicy

    if isinstance(value, TrustPolicy):
        return value.value
    if isinstance(value, str):
        return value
    raise ValueError(
        f"invalid trust_policy {value!r}; expected one of "
        f"{{'open','tofu','pinned'}} or a quinkgl.gossip.TrustPolicy member"
    )


def _class_name_matches(instance: Any, expected_name: str) -> bool:
    """Match ``type(instance).__name__`` against a manifest-declared name.

    The manifest uses abbreviated forms (e.g. ``"Random"``) while the
    Python classes carry suffixes (``"RandomTopology"``); we accept both
    ``"Random" == "Random"`` and ``"RandomTopology".removesuffix("Topology")
    == "Random"`` so callers can speak either dialect without translating.
    """
    actual = type(instance).__name__
    if actual == expected_name:
        return True
    for suffix in ("Topology", "Aggregator", "Strategy"):
        if actual.endswith(suffix) and actual[: -len(suffix)] == expected_name:
            return True
    return False


def _raise_node(code: str, **ctx: object) -> None:
    raise ValueError(code, ctx)

logger = logging.getLogger(__name__)


def _tunnel_sign_data(
    sender_id: str,
    domain: str,
    round_number: int,
    data_schema_hash: str,
    sample_count: int,
    loss: float,
    accuracy: float,
    timestamp: int,
    weights_bytes: bytes,
) -> bytes:
    return (
        sender_id.encode("utf-8")
        + domain.encode("utf-8")
        + struct.pack("!I", int(round_number))
        + data_schema_hash.encode("utf-8")
        + struct.pack("!I", int(sample_count))
        + struct.pack("!d", float(loss))
        + struct.pack("!d", float(accuracy))
        + struct.pack("!Q", int(timestamp))
        + hashlib.sha256(weights_bytes).digest()
    )


def _tunnel_sign(
    private_key,
    sender_id: str,
    domain: str,
    round_number: int,
    data_schema_hash: str,
    sample_count: int,
    loss: float,
    accuracy: float,
    timestamp: int,
    weights_bytes: bytes,
) -> bytes:
    msg = _tunnel_sign_data(
        sender_id,
        domain,
        round_number,
        data_schema_hash,
        sample_count,
        loss,
        accuracy,
        timestamp,
        weights_bytes,
    )
    return private_key.signature(msg)


def _tunnel_verify(
    public_key,
    signature: bytes,
    sender_id: str,
    domain: str,
    round_number: int,
    data_schema_hash: str,
    sample_count: int,
    loss: float,
    accuracy: float,
    timestamp: int,
    weights_bytes: bytes,
) -> bool:
    if not signature:
        return False
    try:
        msg = _tunnel_sign_data(
            sender_id,
            domain,
            round_number,
            data_schema_hash,
            sample_count,
            loss,
            accuracy,
            timestamp,
            weights_bytes,
        )
        return public_key.verify(signature, msg)
    except Exception:
        return False


class ConnectionMode(Enum):
    """Connection mode for the node."""
    IPV8_P2P = "ipv8_p2p"       # Direct P2P via IPv8
    TUNNEL_RELAY = "tunnel"     # Tunnel relay fallback


class NodeState(Enum):
    """Join-flow state machine (spec §14).

    A :class:`GossipNode` is always in exactly one of these states.
    Transitions are recorded as ``node.state.<new>`` runtime events so that
    observers (terminal UI, telemetry, tests) can trace the full peer
    lifecycle without reaching into private attributes.
    """

    INIT = "init"
    MANIFEST_RESOLVED = "manifest_resolved"
    COMMUNITY_STARTED = "community_started"
    PEERS_DISCOVERED = "peers_discovered"
    TRAINING = "training"
    FAILED = "failed"


# Forward-only transition graph (§14.2).  ``any → INIT`` on shutdown is
# modelled by allowing every state to go back to INIT explicitly; ``any →
# FAILED`` on error is allowed too.  Unknown transitions raise.
_ALLOWED_TRANSITIONS: Dict[NodeState, frozenset] = {
    NodeState.INIT: frozenset({NodeState.MANIFEST_RESOLVED, NodeState.FAILED}),
    NodeState.MANIFEST_RESOLVED: frozenset(
        {NodeState.COMMUNITY_STARTED, NodeState.FAILED, NodeState.INIT}
    ),
    NodeState.COMMUNITY_STARTED: frozenset(
        {NodeState.PEERS_DISCOVERED, NodeState.FAILED, NodeState.INIT}
    ),
    NodeState.PEERS_DISCOVERED: frozenset(
        {NodeState.TRAINING, NodeState.FAILED, NodeState.INIT}
    ),
    NodeState.TRAINING: frozenset({NodeState.FAILED, NodeState.INIT}),
    NodeState.FAILED: frozenset({NodeState.INIT}),
}


class GossipNode:
    """
    Complete Gossip Learning Node with IPv8 networking and tunnel fallback.

    This class combines the LearningNode framework with IPv8 P2P networking
    for full decentralized gossip learning. Falls back to tunnel relay
    when IPv8 P2P is unavailable.

    Example:
        ```python
        from quinkgl.network import GossipNode
        from quinkgl.models import PyTorchModel

        # Wrap your model
        model = PyTorchModel(my_pytorch_model)

        # Create node with tunnel fallback
        node = GossipNode(
            node_id="alice",
            domain="health",
            model=model,
            port=7000,
            tunnel_server="tunnel.example.com:50051"  # Optional fallback
        )

        # Start and run
        await node.start()
        await node.run_continuous(training_data)
        ```
    """

    def __init__(
        self,
        node_id: str,
        domain: Optional[str] = None,
        model: Optional[ModelWrapper] = None,
        port: int = 0,
        topology: Optional[TopologyStrategy] = None,
        aggregation: Optional[AggregationStrategy] = None,
        gossip_interval: float = 60.0,
        training_config: Optional[TrainingConfig] = None,
        auto_discovery: bool = True,
        tunnel_server: Optional[str] = None,
        enable_fallback: bool = True,
        fallback_timeout: float = 30.0,
        min_peers_before_aggregate: int = 1,
        data_policy: Optional[Any] = None,
        fingerprint: Optional[Any] = None,
        require_signature: bool = True,
        quiet: bool = False,
        *,
        manifest: Optional[SwarmManifest] = None,
        strict_manifest: bool = True,
        trust_policy: str = "open",
        trusted_creator_pubkeys: Optional[Set[bytes]] = None,
    ):
        """
        Initialize GossipNode.

        Args:
            node_id: Unique identifier for this node
            domain: Domain identifier (legacy path; mutually exclusive with
                ``manifest``).  Exactly one of ``domain`` / ``manifest`` MUST
                be provided — else ``ERR_NODE_NO_MANIFEST``.
            model: Wrapped model (PyTorchModel, TensorFlowModel, or custom)
            manifest: Swarm manifest describing topology/aggregation/task
                identity (spec §10.4).  When supplied, the node validates
                that ``aggregation`` and ``topology`` class names match the
                manifest's declared names (strict mode).  ``domain`` is
                derived from ``manifest.task.type`` for legacy downstream
                callers that still read ``self.domain``.
            strict_manifest: If True (default), enforce aggregation/topology
                name matches.  Set False to accept divergent local choices.
            trust_policy: One of {"open", "tofu", "pinned"} — governs how
                creator-pubkey trust is evaluated.  ``"pinned"`` requires
                the manifest to carry a signature.
            trusted_creator_pubkeys: Set of ed25519 pubkeys (32 bytes each)
                accepted under the ``"pinned"`` policy.
            port: UDP port for IPv8 (0 for random)
            topology: Topology strategy (defaults to RandomTopology)
            aggregation: Aggregation strategy (defaults to FedAvg)
            gossip_interval: Seconds between gossip rounds
            training_config: Configuration for local training
            auto_discovery: Enable automatic peer discovery
            tunnel_server: Tunnel server address (host:port) for fallback
            enable_fallback: Enable tunnel fallback when IPv8 fails
            fallback_timeout: Seconds to wait for IPv8 before fallback
            min_peers_before_aggregate: Minimum updates before aggregation
            quiet: If True, no default TerminalObserver is attached.
                Events are still emitted and can be consumed by custom observers.
        """
        # --- Spec §10.4 enforcement (runs before any side-effectful init) --
        if model is None:
            raise TypeError("GossipNode requires a `model` argument")

        has_manifest = manifest is not None
        has_domain = isinstance(domain, str) and domain != ""
        if has_manifest and has_domain:
            _raise_node(
                ERR_NODE_NO_MANIFEST,
                detail="`manifest` and `domain` are mutually exclusive",
                node_id=node_id,
            )
        if not has_manifest and not has_domain:
            _raise_node(
                ERR_NODE_NO_MANIFEST,
                detail=(
                    "GossipNode requires exactly one of `manifest` or "
                    "`domain` — neither was supplied"
                ),
                node_id=node_id,
            )

        trust_policy = _coerce_trust_policy(trust_policy)
        if trust_policy not in _VALID_TRUST_POLICIES:
            raise ValueError(
                f"invalid trust_policy {trust_policy!r}; expected one of "
                f"{sorted(_VALID_TRUST_POLICIES)}"
            )

        # Strict aggregation / topology name checks fire *before* defaults
        # are materialised so the manifest's declared names don't get
        # shadowed by implicit FedAvg / RandomTopology fallbacks.
        if has_manifest and strict_manifest:
            if aggregation is not None and not _class_name_matches(
                aggregation, manifest.aggregation_name
            ):
                _raise_node(
                    ERR_NODE_AGGREGATION_MISMATCH,
                    detail="aggregation class name does not match manifest",
                    expected=manifest.aggregation_name,
                    actual=type(aggregation).__name__,
                    node_id=node_id,
                )
            if topology is not None and not _class_name_matches(
                topology, manifest.topology_name
            ):
                _raise_node(
                    ERR_NODE_TOPOLOGY_MISMATCH,
                    detail="topology class name does not match manifest",
                    expected=manifest.topology_name,
                    actual=type(topology).__name__,
                    node_id=node_id,
                )

        if trust_policy == "pinned" and has_manifest and manifest.signature is None:
            _raise_node(
                ERR_NODE_UNSIGNED_MANIFEST_REJECTED,
                detail=(
                    "trust_policy='pinned' requires a signed manifest "
                    "(manifest.signature must be non-null)"
                ),
                node_id=node_id,
            )

        # TOFU enforcement (spec §15.1).  Only runs when:
        #   1. trust_policy == "tofu"
        #   2. a manifest is present and signed (creator_pubkey set)
        # The cache lookup is synchronous by design — a conflict MUST
        # abort ``__init__`` before IPv8 is spun up so the node never
        # advertises itself in a swarm bound to a different creator.
        self._tofu_cache = None  # late-bound, populated on conflict/record
        if trust_policy == "tofu" and has_manifest and manifest.creator_pubkey:
            from quinkgl.network.tofu import TofuCache, default_tofu_cache_path

            cache_path = default_tofu_cache_path()
            try:
                tofu_cache = TofuCache(cache_path)
                tofu_cache.record_or_validate(
                    manifest.manifest_hash(), manifest.creator_pubkey
                )
                self._tofu_cache = tofu_cache
            except ValueError as exc:
                # TofuCache.record_or_validate raises
                # ``ValueError(ERR_TRUST_TOFU_CONFLICT, {...})`` on
                # creator-key divergence; re-raise via the node-level
                # helper so the error matches the shape of every other
                # ``ERR_NODE_*`` / ``ERR_TRUST_*`` raised from __init__.
                from quinkgl.manifest.errors import ERR_TRUST_TOFU_CONFLICT

                if exc.args and exc.args[0] == ERR_TRUST_TOFU_CONFLICT:
                    raise
                # Something else went wrong (e.g. bad creator_pubkey
                # value); surface it directly.
                raise

        # Derive a legacy `domain` string for code paths that still read
        # ``self.domain`` (community-id generation, tunnel announce).  When
        # the manifest provides a richer identity, we prefer
        # ``manifest.task.type`` ("classification", "regression", ...) and
        # fall back to the manifest name.
        if has_manifest:
            task_type = getattr(manifest.task, "type", None) if manifest.task else None
            effective_domain = task_type or manifest.name or "manifest"
        else:
            effective_domain = domain  # type: ignore[assignment]

        self.manifest = manifest
        self.trust_policy = trust_policy
        self.trusted_creator_pubkeys = set(trusted_creator_pubkeys or set())
        self.strict_manifest = strict_manifest

        # Spec §14.1: peers start at INIT.  When a manifest is supplied up
        # front we advance to MANIFEST_RESOLVED at the end of ``__init__``
        # after all side-effectful subsystem wiring succeeds.
        self.state: NodeState = NodeState.INIT
        self.swarm_id: Optional[bytes] = None

        self.node_id = node_id
        self.domain = effective_domain
        self.port = port
        self.tunnel_server = tunnel_server
        self.enable_fallback = enable_fallback
        self.fallback_timeout = fallback_timeout
        self.model_version = model.get_model_version()

        # Get data schema hash from model
        self.data_schema_hash = model.get_data_schema_hash()

        # Import default strategies if not provided
        if topology is None:
            from quinkgl.topology import RandomTopology
            topology = RandomTopology()

        if aggregation is None:
            from quinkgl.aggregation import FedAvg
            aggregation = FedAvg()

        # Create the framework LearningNode
        # ``domain=`` in the GossipNode constructor is only set for the legacy
        # manifest-free path.  Manifest mode leaves it ``None`` and derives
        # ``effective_domain`` from the manifest task.  The LearningNode and
        # ModelAggregator *must* use that same string — otherwise
        # ``SelectionContext.my_domain`` does not match discovery announcers
        # (``peer_info.domain``) and :meth:`get_compatible_peers` always returns
        # empty, yielding no gossip targets.
        self.gl_node = LearningNode(
            peer_id=node_id,
            domain=effective_domain,
            model=model,
            topology=topology,
            aggregation=aggregation,
            data_schema_hash=self.data_schema_hash,
            gossip_interval=gossip_interval,
            training_config=training_config,
            min_peers_before_aggregate=min_peers_before_aggregate
        )

        # IPv8 manager
        self.ipv8_manager = IPv8Manager(node_id=node_id, port=port)
        self.ipv8_manager.domain = effective_domain
        self.ipv8_manager.data_schema_hash = self.data_schema_hash
        self.ipv8_manager.require_signature = require_signature
        self.ipv8_manager.last_seen_round_state_path = ""
        self.ipv8_manager.max_round_skip = MAX_ROUND_SKIP
        self.require_signature = require_signature

        # Community (set after IPv8 starts)
        self.community: Optional[GossipLearningCommunity] = None

        # Tunnel client (lazy loaded)
        self.tunnel_client = None
        self._tunnel_connected = False

        # Remote peers via tunnel
        self._tunnel_peers: Dict[str, dict] = {}  # peer_id -> {node_id, domain, schema}
        self._tunnel_last_seen_round: Dict[str, int] = {}

        # Connection mode
        self.connection_mode: ConnectionMode = ConnectionMode.IPV8_P2P

        # Auto-discovery flag
        self.auto_discovery = auto_discovery

        # State
        self.running = False
        self._ipv8_failed = False
        # Track the gossip-loop task so shutdown can cancel it
        self._run_task: Optional[asyncio.Task] = None
        self._telemetry_client = None

        # Callbacks for tunnel messages
        self._on_tunnel_peer_discovered: Optional[Callable] = None
        self._on_tunnel_model_update: Optional[Callable] = None

        # Domain-aware collaboration
        self.data_policy = data_policy
        self._fingerprint_source = fingerprint
        self.fingerprint = fingerprint if not callable(fingerprint) else None

        # Lifecycle tracking
        self._start_time: Optional[float] = None
        self._quiet = quiet

        # Auto-attach TerminalObserver unless quiet
        if not quiet:
            self.attach_terminal_observer()

        logger.debug(
            f"GossipNode initialized: node_id={node_id}, "
            f"domain={effective_domain!r} (raw domain arg={domain!r}), "
            f"schema={self.data_schema_hash[:8]}..., "
            f"fallback={'enabled' if enable_fallback else 'disabled'}"
        )

        # §14.2 transition: construction with a manifest in hand already
        # satisfies the "manifest resolved" step — there's nothing left to
        # fetch.  Legacy ``domain=``-only nodes stay at INIT because the
        # swarm identity is implicit and cannot be signed off on here.
        if self.manifest is not None:
            try:
                self.swarm_id = bytes.fromhex(self.manifest.manifest_hash())
            except Exception:
                self.swarm_id = None
            self._transition(NodeState.MANIFEST_RESOLVED, reason="manifest_constructed")

    async def start(self):
        """Start the node and join the P2P network."""
        if self.running:
            logger.warning("Node already running")
            return

        # §14.2: the network layer only comes up once a manifest has been
        # resolved (either via constructor or via :meth:`from_domain` which
        # fabricates an implicit one).  We tolerate legacy ``INIT`` nodes
        # here for back-compat but MUST still emit a transition event so
        # observers see the ``community_started`` milestone.
        if self.state is NodeState.FAILED:
            raise RuntimeError("cannot start a node that is in FAILED state")

        logger.debug(f"Starting GossipNode '{self.node_id}'...")
        logger.debug(f"Attempting P2P connection (timeout: {self.fallback_timeout}s)...")

        try:
            ipv8_started = await self._try_start_ipv8_with_timeout()
        except Exception:
            # Hard failure bringing up the listener — surface it as FAILED
            # so observers don't see a node stuck in MANIFEST_RESOLVED.
            try:
                self._transition(NodeState.FAILED, reason="ipv8_start_exception")
            except Exception:
                pass
            raise

        if ipv8_started:
            self.connection_mode = ConnectionMode.IPV8_P2P
            logger.debug("Using IPv8 P2P mode")
        elif self.enable_fallback and self.tunnel_server:
            # IPv8 failed or timed out, try tunnel fallback
            logger.warning("IPv8 P2P failed/timeout, falling back to tunnel relay...")
            emitter = self.gl_node.aggregator.event_emitter
            if emitter:
                emitter.emit("security.tunnel_downgrade", {
                    "node_id": self.node_id,
                    "domain": self.domain,
                    "reason": "ipv8_failed_or_timed_out",
                })
            self.connection_mode = ConnectionMode.TUNNEL_RELAY
            await self._start_tunnel_fallback()
        else:
            raise RuntimeError(
                "Failed to start networking. IPv8 failed and "
                "fallback is disabled or no tunnel server configured."
            )

        # Mark LearningNode as ready (network layer connected)
        # Note: LearningNode doesn't use _joined flag anymore, this is for GossipNode's internal tracking
        self.running = True
        if self._telemetry_client is not None:
            self._telemetry_client.start(self.get_stats)

        mode_str = "IPv8 P2P" if self.connection_mode == ConnectionMode.IPV8_P2P else "Tunnel Relay"
        logger.debug(f"GossipNode '{self.node_id}' started in {mode_str} mode")

        # §14.2 transition — community is up and the listener is bound.
        # Legacy ``INIT`` nodes (no manifest provided at construction time)
        # must first pass through MANIFEST_RESOLVED to keep the transition
        # graph linear.
        if self.state is NodeState.INIT:
            try:
                self._transition(
                    NodeState.MANIFEST_RESOLVED, reason="legacy_domain_only"
                )
            except Exception:  # pragma: no cover — defensive
                pass
        try:
            self._transition(
                NodeState.COMMUNITY_STARTED, mode=mode_str.lower().replace(" ", "_")
            )
        except Exception:  # pragma: no cover — defensive
            pass

        self._start_time = time.time()

        # Emit lifecycle events so observers render the startup banner
        emitter = self.gl_node.aggregator.event_emitter
        if emitter:
            from quinkgl import __version__ as _ver
            emitter.emit("node.config", {
                "node_id": self.node_id,
                "version": _ver,
                "domain": self.domain,
                "port": self.port,
                "topology": type(self.gl_node.aggregator.topology).__name__,
                "aggregation": type(self.gl_node.aggregator.aggregator).__name__,
                "connection_mode": mode_str,
                "model": type(self.gl_node.aggregator.model).__name__,
                "gossip_interval": self.gl_node.aggregator.gossip_interval,
                "data_policy": {
                    "fingerprint_enabled": getattr(self.data_policy, "fingerprint_enabled", False),
                    "min_affinity": getattr(self.data_policy, "min_affinity", None),
                    "privacy_level": getattr(self.data_policy, "privacy_level", None),
                } if self.data_policy else None,
                "fingerprint_summary": {
                    "label_buckets": len(getattr(self.fingerprint, "label_buckets", {})),
                    "sample_bucket": getattr(self.fingerprint, "sample_bucket", None),
                } if self.fingerprint else None,
            })
            emitter.emit("node.started", {
                "node_id": self.node_id,
                "connection_mode": mode_str,
            })

    async def _try_start_ipv8_with_timeout(self) -> bool:
        """
        Try to start IPv8 P2P networking with timeout.

        Returns:
            True if IPv8 P2P connection successful, False if timeout/error
        """
        try:
            # Step 1: Start IPv8 (with timeout)
            start_time = time.time()

            try:
                await asyncio.wait_for(
                    self.ipv8_manager.start(
                        community_class=GossipLearningCommunity,
                        node_id_param=self.node_id
                    ),
                    timeout=self.fallback_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"IPv8 start timed out after {self.fallback_timeout}s")
                # B8: Clean up half-started IPv8 (ports, tasks, community)
                await self.ipv8_manager.stop()
                return False

            # Get community reference
            self.community = self.ipv8_manager.community

            # Update community with our specific info
            self.community.domain = self.domain
            self.community.data_schema_hash = self.data_schema_hash
            self.community.model_version = self.model_version
            self.community.require_signature = getattr(self, "require_signature", True)
            self.community.event_emitter = self.gl_node.aggregator.event_emitter
            self.community.current_round_provider = lambda: self.gl_node.current_round
            # Derive manifest hash: prefer the attached SwarmManifest (spec
            # §12.3 — the manifest is the authoritative swarm identity) and
            # fall back to the legacy data_policy digest so current tests
            # that construct nodes without a manifest keep working.
            manifest_hash = None
            if self.manifest is not None:
                try:
                    manifest_hash = self.manifest.manifest_hash()
                except Exception:
                    manifest_hash = None
            if manifest_hash is None and self.data_policy is not None:
                try:
                    import json, hashlib
                    manifest_hash = hashlib.sha256(
                        json.dumps(self.data_policy, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                except Exception:
                    pass
            self.community._instance_community_id = generate_community_id(
                self.domain, self.data_schema_hash, manifest_hash=manifest_hash
            )
            # Expose the manifest identity on the community so the
            # DiscoveryAnnounce pre-filter (§12.3) and outgoing announces
            # pick it up.
            self.community.manifest = self.manifest
            self.community.manifest_hash = manifest_hash or ""
            # Wire manifest id into aggregator for downstream topology/selection
            self.gl_node.aggregator._local_manifest_id = manifest_hash

            # Setup callbacks
            self._setup_ipv8_callbacks()

            self._configure_local_fingerprint_runtime()

            # Wire prototype store if data policy enables it
            if self.data_policy is not None and self.data_policy.prototypes.enabled:
                from quinkgl.training.prototypes import PrototypeStore
                self.gl_node.aggregator._prototype_store = PrototypeStore()

            # Step 2: Wait for peer discovery with remaining timeout
            # Guarantee at least MIN_PEER_DISCOVERY_WINDOW seconds for
            # discovery even if IPv8 start consumed most of the budget.
            MIN_PEER_DISCOVERY_WINDOW = 5.0
            elapsed = time.time() - start_time
            remaining_timeout = max(
                MIN_PEER_DISCOVERY_WINDOW,
                self.fallback_timeout - elapsed,
            )
            if remaining_timeout > MIN_PEER_DISCOVERY_WINDOW:
                logger.info(
                    f"Using MIN_PEER_DISCOVERY_WINDOW floor: "
                    f"user fallback_timeout={self.fallback_timeout}s, "
                    f"elapsed={elapsed:.2f}s, using {remaining_timeout}s"
                )

            if remaining_timeout > 0:
                try:
                    await asyncio.wait_for(
                        self._wait_for_peers(),
                        timeout=remaining_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Peer discovery timed out after {self.fallback_timeout}s total")

            # Sync known peers
            self._sync_known_peers()

            peer_count = self.community.get_peer_count()

            # If zero peers discovered and fallback is available, treat
            # this as an IPv8 failure so the caller can trigger tunnel mode
            # instead of running with an empty overlay.
            if peer_count == 0 and self.enable_fallback and self.tunnel_server:
                logger.warning(
                    "IPv8 started but 0 peers discovered — "
                    "treating as failure so tunnel fallback can activate"
                )
                await self.ipv8_manager.stop()
                self._ipv8_failed = True
                return False

            # Start Cyclon topology if applicable
            from quinkgl.topology.cyclon import CyclonTopology
            if isinstance(self.gl_node.topology, CyclonTopology):
                context = SelectionContext(
                    my_peer_id=self.node_id,
                    my_domain=self.domain,
                    my_data_schema_hash=self.data_schema_hash,
                    known_peers=list(self.gl_node.aggregator.known_peers.values()),
                    my_model_version=self.model_version,
                )
                await self.gl_node.topology.start(context)
                logger.debug("Cyclon topology started with periodic shuffle")

            logger.debug(f"IPv8 P2P ready with {peer_count} peers discovered")

            return True

        except Exception as e:
            logger.warning(f"IPv8 P2P failed: {e}")
            # B8: Clean up any partially-started IPv8 resources
            await self.ipv8_manager.stop()
            self._ipv8_failed = True
            return False

    async def _wait_for_peers(self, min_peers: int = 0) -> None:
        """
        Wait for peer discovery.

        Args:
            min_peers: Minimum number of peers required (0 = any is fine)
        """
        # Poll for peer discovery with exponential backoff
        check_interval = 0.5
        max_interval = 2.0

        while True:
            peer_count = self.community.get_peer_count()

            if peer_count >= min_peers:
                logger.debug(f"Discovered {peer_count} peers")
                return

            # Wait before next check
            await asyncio.sleep(check_interval)
            check_interval = min(check_interval * 1.5, max_interval)

    def _setup_ipv8_callbacks(self):
        """Setup callbacks between IPv8 community and LearningNode."""
        from quinkgl.topology.cyclon import CyclonTopology

        # When model update received
        async def on_model_update(
            sender_id: str,
            weights: Any,
            sample_count: int,
            round_number: int,
            loss: float,
            accuracy: float
        ):
            from quinkgl.gossip.protocol import ModelUpdateMessage

            message = ModelUpdateMessage.create(
                sender_id=sender_id,
                weights=weights,
                sample_count=sample_count,
                loss=loss,
                accuracy=accuracy,
                round_number=round_number
            )

            await self.gl_node._handle_network_message(message)
            logger.debug(f"Processed model update from {sender_id}")

        self.community.on_model_update_callback = on_model_update

        # B2: Wire checkpoint broadcast and receive
        async def on_checkpoint_received(
            sender_id: str, round_number: int,
            loss: float, accuracy: float, model_version: str
        ):
            from quinkgl.gossip.protocol import CheckpointAnnounceMessage
            msg = CheckpointAnnounceMessage.create(
                sender_id=sender_id,
                round_number=round_number,
                loss=loss,
                accuracy=accuracy,
                model_version=model_version,
            )
            await self.gl_node._handle_network_message(msg)
            logger.debug(f"Processed checkpoint from {sender_id} (round {round_number})")

        self.community.on_checkpoint_callback = on_checkpoint_received

        async def broadcast_checkpoint_via_ipv8(checkpoint_msg):
            self.community.broadcast_checkpoint(
                sender_id=checkpoint_msg.sender_id,
                round_number=checkpoint_msg.round_number,
                loss=checkpoint_msg.loss,
                accuracy=checkpoint_msg.accuracy,
                model_version=checkpoint_msg.model_version,
            )

        self.gl_node.aggregator.broadcast_callback = broadcast_checkpoint_via_ipv8

        # When peer discovered
        async def on_peer_discovered(peer_info):
            from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo

            framework_peer_info = FrameworkPeerInfo(
                peer_id=peer_info.node_id,
                domain=peer_info.domain,
                data_schema_hash=peer_info.data_schema_hash,
                model_version=peer_info.model_version,
                data_fingerprint=getattr(peer_info, 'data_fingerprint', None),
            )

            self.gl_node.aggregator.add_peer(framework_peer_info)
            logger.debug(f"Added peer {peer_info.node_id} to aggregator")

        self.community.on_peer_discovered_callback = on_peer_discovered

        # When peer leaves
        async def on_peer_left(node_id: str):
            await self.gl_node.aggregator.remove_peer(node_id)
            logger.debug(f"Removed peer {node_id} from aggregator")

        self.community.on_peer_left_callback = on_peer_left

        # Cyclon shuffle callback
        if isinstance(self.gl_node.topology, CyclonTopology):
            async def on_shuffle_request(sender_id: str, remote_peers: list) -> list:
                return await self.gl_node.topology.handle_incoming_shuffle(
                    sender_id, remote_peers
                )

            self.community.on_shuffle_callback = on_shuffle_request

            def make_shuffle_callback(community):
                async def send_shuffle(peer_id: str, peers: list) -> list:
                    peers_bytes = community._serialize_peer_list(peers)
                    if peer_id in community.known_peers:
                        peer_info = community.known_peers[peer_id]
                        community.ez_send(peer_info.peer, ShufflePayload(
                            sender_id=community.node_id,
                            peers_bytes=peers_bytes
                        ))
                    return []
                return send_shuffle

            from quinkgl.network.gossip_community import ShufflePayload
            self.gl_node.topology.set_shuffle_callback(
                make_shuffle_callback(self.community)
            )
            self.gl_node.topology.set_local_peer_info(
                PeerInfo(
                    peer_id=self.node_id,
                    domain=self.domain,
                    data_schema_hash=self.data_schema_hash,
                    model_version=self.model_version,
                )
            )

        # Prototype exchange callback (FedProto/FedPAC — Faz 6e)
        async def on_prototype_received(sender_id: str, prototypes):
            aggregator = self.gl_node.aggregator
            if hasattr(aggregator, '_prototype_store') and aggregator._prototype_store is not None:
                aggregator._prototype_store.merge_global_prototypes({sender_id: prototypes})
                logger.debug(f"Merged prototypes from {sender_id}")

        self.community.on_prototype_callback = on_prototype_received

    def _configure_local_fingerprint_runtime(self) -> None:
        aggregator = self.gl_node.aggregator
        aggregator._local_fingerprint_update_callback = self._apply_local_fingerprint
        source = getattr(self, "_fingerprint_source", getattr(self, "fingerprint", None))
        if source is None:
            aggregator._local_fingerprint_provider = None
            aggregator._set_local_fingerprint(None)
            return
        aggregator._local_fingerprint_provider = self._make_local_fingerprint_provider(source)
        aggregator._refresh_local_fingerprint()

    def _make_local_fingerprint_provider(self, source: Any) -> Callable[[int], Any]:
        from quinkgl.fingerprint.fingerprint import DataFingerprint

        if callable(source):
            def provider(round_number: int) -> Any:
                fingerprint = source(round_number)
                if isinstance(fingerprint, DataFingerprint) and fingerprint.round_nonce is None:
                    return self._bind_fingerprint_to_round(fingerprint, round_number)
                return fingerprint

            return provider

        def provider(round_number: int) -> Any:
            if isinstance(source, DataFingerprint):
                return self._bind_fingerprint_to_round(source, round_number)
            return source

        return provider

    def _bind_fingerprint_to_round(self, fingerprint: Any, round_number: int) -> Any:
        from quinkgl.fingerprint.fingerprint import DataFingerprint

        if not isinstance(fingerprint, DataFingerprint):
            return fingerprint
        payload = fingerprint.to_dict()
        payload["round_nonce"] = hashlib.sha256(
            f"quinkgl-fingerprint-round:{round_number}".encode("utf-8")
        ).hexdigest()[:16]
        return DataFingerprint.from_dict(payload)

    def _apply_local_fingerprint(self, fingerprint: Optional[Any]) -> None:
        self.fingerprint = fingerprint
        if self.community is not None:
            self.community.local_fingerprint = fingerprint

    def _get_tunnel_signing_key(self):
        if self.community is not None and hasattr(self.community, "my_peer"):
            my_peer = getattr(self.community, "my_peer", None)
            if my_peer is not None and getattr(my_peer, "key", None) is not None:
                return my_peer.key

        key_file = os.path.join(tempfile.gettempdir(), f"ipv8_quinkgl_{self.node_id}.pem")
        if not os.path.exists(key_file):
            return None

        try:
            with open(key_file, "rb") as fh:
                pem = fh.read()
            return default_eccrypto.generate_key("medium").key_from_pem(pem)
        except Exception as exc:
            logger.warning(f"Failed to load tunnel signing key for {self.node_id}: {exc}")
            return None

    async def _start_tunnel_fallback(self):
        """Start tunnel relay fallback."""
        if not self.tunnel_server:
            raise ValueError("No tunnel_server configured")
        from quinkgl.network.fallback import TunnelClient

        try:
            self.tunnel_client = TunnelClient(self.tunnel_server, self.node_id)
            await self.tunnel_client.connect()
            self._tunnel_connected = True

            # Setup tunnel message handlers
            self._setup_tunnel_callbacks()

            logger.debug(f"Tunnel fallback connected to {self.tunnel_server}")

        except Exception as e:
            logger.error(f"Tunnel fallback failed: {e}")
            raise

    def _setup_tunnel_callbacks(self):
        """Setup callbacks for tunnel messages."""
        if not self.tunnel_client:
            return

        # B1: concrete handler for MODEL_UPDATE via tunnel
        async def _on_tunnel_model_update(data: dict):
            """Decode, validate, and dispatch a tunnel MODEL_UPDATE payload.

            Mirrors the IPv8 ``on_model_update`` path so updates arriving
            via tunnel are treated identically by the aggregation layer.
            """
            from quinkgl.network.model_serializer import deserialize_model
            from quinkgl.gossip.protocol import ModelUpdateMessage
            from quinkgl.network.gossip_community import MAX_INCOMING_MESSAGE_SIZE

            emitter = self.gl_node.aggregator.event_emitter

            def emit_drop(reason: str, peer_id: Optional[str] = None, security_event: Optional[str] = None):
                if emitter:
                    payload = {
                        "node_id": self.node_id,
                        "reason": reason,
                    }
                    if peer_id is not None:
                        payload["peer_id"] = peer_id
                    if security_event:
                        emitter.emit(security_event, payload)
                    emitter.emit("tunnel_payload_dropped", payload)

            # ── Required-field validation ──────────────────────────
            required = (
                "sender_id",
                "domain",
                "data_schema_hash",
                "round_number",
                "sample_count",
                "loss",
                "accuracy",
                "timestamp",
                "weights",
                "signature",
                "signer_public_key",
            )
            missing = [f for f in required if f not in data]
            if missing:
                logger.warning(f"Tunnel MODEL_UPDATE missing fields: {missing}")
                emit_drop(f"missing fields: {missing}", security_event="security.signature_missing")
                return

            sender_id = data["sender_id"]
            tunnel_sender_id = data.get("_tunnel_sender_id")
            if tunnel_sender_id is None:
                logger.warning(
                    f"Tunnel update rejected: no stream binding for sender={sender_id}"
                )
                emit_drop("stream sender missing", peer_id=sender_id, security_event="security.identity_mismatch")
                return
            if sender_id != tunnel_sender_id:
                logger.warning(
                    f"Tunnel update rejected: sender mismatch payload={sender_id} tunnel={tunnel_sender_id}"
                )
                emit_drop("tunnel sender mismatch", peer_id=sender_id, security_event="security.identity_mismatch")
                return

            # ── Domain / schema compatibility ─────────────────────
            msg_domain = data.get("domain")
            msg_schema = data.get("data_schema_hash")
            if msg_domain != self.domain or msg_schema != self.data_schema_hash:
                logger.debug(
                    f"Tunnel update rejected: domain/schema mismatch "
                    f"(got {msg_domain}/{msg_schema}, "
                    f"expected {self.domain}/{self.data_schema_hash})"
                )
                emit_drop("domain/schema mismatch", peer_id=sender_id)
                return

            # ── Size bound on hex-decoded weights ─────────────────
            weights_hex = data["weights"]
            if len(weights_hex) // 2 > MAX_INCOMING_MESSAGE_SIZE:
                logger.warning(
                    f"Tunnel update rejected: weights too large "
                    f"({len(weights_hex) // 2} bytes)"
                )
                emit_drop("oversized weights", peer_id=sender_id, security_event="security.oversized_message")
                return

            try:
                weights_bytes = bytes.fromhex(weights_hex)
                signature = bytes.fromhex(data["signature"])
                signer_public_key_bytes = bytes.fromhex(data["signer_public_key"])
            except Exception as e:
                logger.warning(f"Tunnel model decoding failed: {e}")
                emit_drop(f"deserialization error: {e}", peer_id=sender_id)
                return

            if not signature:
                emit_drop("missing signature", peer_id=sender_id, security_event="security.signature_missing")
                return

            if not default_eccrypto.is_valid_public_bin(signer_public_key_bytes):
                emit_drop("invalid signer public key", peer_id=sender_id, security_event="security.signature_rejected")
                return

            try:
                signer_public_key = default_eccrypto.key_from_public_bin(signer_public_key_bytes)
            except Exception as e:
                emit_drop(f"invalid signer public key: {e}", peer_id=sender_id, security_event="security.signature_rejected")
                return

            if not _tunnel_verify(
                signer_public_key,
                signature,
                sender_id,
                msg_domain,
                data["round_number"],
                msg_schema,
                data["sample_count"],
                data["loss"],
                data["accuracy"],
                data["timestamp"],
                weights_bytes,
            ):
                logger.warning(f"Tunnel update rejected: invalid signature from {sender_id}")
                emit_drop("invalid signature", peer_id=sender_id, security_event="security.signature_rejected")
                return

            last_round = self._tunnel_last_seen_round.get(sender_id, -1)
            if data["round_number"] <= last_round:
                logger.warning(
                    f"Tunnel update rejected: replayed round {data['round_number']} <= {last_round} from {sender_id}"
                )
                emit_drop("replayed round", peer_id=sender_id, security_event="security.replay_rejected")
                return

            current_round = getattr(self.gl_node.aggregator, "current_round", 0)
            if data["round_number"] > current_round + MAX_ROUND_SKIP:
                emit_drop("future round rejected", peer_id=sender_id, security_event="security.future_round_rejected")
                return

            # ── Deserialize weights ───────────────────────────────
            try:
                weights = deserialize_model(weights_bytes)
            except Exception as e:
                logger.warning(f"Tunnel model deserialization failed: {e}")
                emit_drop(f"deserialization error: {e}", peer_id=sender_id)
                return

            # ── Dispatch to aggregation layer ─────────────────────
            message = ModelUpdateMessage.create(
                sender_id=sender_id,
                weights=weights,
                sample_count=data.get("sample_count", 0),
                loss=data.get("loss", 0.0),
                accuracy=data.get("accuracy", 0.0),
                round_number=data["round_number"],
            )
            await self.gl_node._handle_network_message(message)
            self._tunnel_last_seen_round[sender_id] = data["round_number"]
            logger.debug(f"Processed tunnel model update from {sender_id}")

        self._on_tunnel_model_update = _on_tunnel_model_update

        # Handle incoming chat messages (may contain model updates)
        async def on_tunnel_message(msg):
            try:
                import json

                data = json.loads(msg.text)

                # Null-check required fields
                msg_type = data.get("type")
                if not msg_type:
                    logger.debug("Tunnel message missing 'type' field — ignored")
                    return

                if msg_type == "MODEL_UPDATE":
                    data["_tunnel_sender_id"] = getattr(msg, "_tunnel_sender_id", getattr(msg, "sender_id", None))
                    # Require essential fields
                    if not all(k in data for k in ("sender_id", "round_number", "weights")):
                        logger.warning("MODEL_UPDATE missing required fields — ignored")
                        return
                    await self._on_tunnel_model_update(data)
                elif msg_type == "PEER_ANNOUNCE":
                    peer_info = data.get("peer_info", {})
                    self._tunnel_peers[peer_info.get("node_id")] = peer_info

                    if self._on_tunnel_peer_discovered:
                        await self._on_tunnel_peer_discovered(peer_info)

                    from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo

                    framework_peer_info = FrameworkPeerInfo(
                        peer_id=peer_info.get("node_id"),
                        domain=peer_info.get("domain"),
                        data_schema_hash=peer_info.get("data_schema_hash"),
                        model_version=peer_info.get("model_version", "1.0.0")
                    )

                    self.gl_node.aggregator.add_peer(framework_peer_info)
                    logger.debug(f"Added tunnel peer {peer_info.get('node_id')} to aggregator")

            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.warning(f"Error handling tunnel message: {e}")

        self.tunnel_client.on_chat_message = on_tunnel_message

        # Handle peer list updates from tunnel
        # B16 §5.6: Re-announce when new peers appear, not just at startup
        async def on_peer_list(peer_ids: list):
            logger.debug(f"Tunnel server reports {len(peer_ids)} connected peers")
            new_peers = [p for p in peer_ids if p not in self._tunnel_peers]
            if new_peers:
                logger.debug(f"New tunnel peers detected: {new_peers} — re-announcing")
                await self._announce_to_tunnel()

        self.tunnel_client.on_peer_list = on_peer_list

        # B9: Detect tunnel stream death
        async def on_tunnel_disconnected():
            logger.warning("Tunnel stream died — marking tunnel as disconnected")
            self._tunnel_connected = False

        self.tunnel_client.on_disconnected = on_tunnel_disconnected

    def _sync_known_peers(self):
        """Sync known peers from community to LearningNode."""
        if not self.community:
            return

        from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo

        connected_peer_ids = []
        for peer_info in self.community.get_compatible_peers():
            framework_peer_info = FrameworkPeerInfo(
                peer_id=peer_info.node_id,
                domain=peer_info.domain,
                data_schema_hash=peer_info.data_schema_hash,
                model_version=peer_info.model_version
            )
            self.gl_node.aggregator.add_peer(framework_peer_info)
            connected_peer_ids.append(peer_info.node_id)

        logger.debug(f"Synced {self.community.get_peer_count()} peers to aggregator")

    async def _wire_gl_continuous_loop(self) -> None:
        """Attach send callback, fingerprint, and sync before ``run_continuous``."""
        if not self.running:
            raise RuntimeError("Node must be started before running")

        async def send_to_peer(peer_id: str, message):
            if self.connection_mode == ConnectionMode.IPV8_P2P:
                await self.community.send_model_update(
                    target_node_id=peer_id,
                    weights=message.weights,
                    sample_count=message.sample_count,
                    round_number=message.round_number,
                    loss=message.loss,
                    accuracy=message.accuracy,
                )
            else:
                await self._send_model_update_via_tunnel(peer_id, message)

        self.gl_node.aggregator.send_message_callback = send_to_peer
        self._configure_local_fingerprint_runtime()

        if self.connection_mode == ConnectionMode.IPV8_P2P:
            self._sync_known_peers()

        if self.connection_mode == ConnectionMode.TUNNEL_RELAY:
            await self._announce_to_tunnel()

    async def run_continuous(self, data=None, data_provider=None):
        """
        Run continuous gossip learning.

        Args:
            data: Training data (single dataset)
            data_provider: Callable that returns training data per round
        """
        await self._wire_gl_continuous_loop()

        # B13: Track the gossip-loop task so shutdown() can cancel it
        self._run_task = asyncio.current_task()
        try:
            await self.gl_node.run_continuous(data_provider=data_provider or data)
        finally:
            self._run_task = None

    async def _send_model_update_via_tunnel(self, peer_id: str, message):
        """Send model update via tunnel relay."""
        if not self.tunnel_client or not self._tunnel_connected:
            # B9: Raise so Track A's per-peer try/except logs the failure
            raise ConnectionError(f"Cannot send to {peer_id}: tunnel not connected")

        import json
        from quinkgl.network.model_serializer import serialize_model

        try:
            signing_key = self._get_tunnel_signing_key()
            if signing_key is None:
                raise ValueError("Tunnel signing key unavailable")

            # Serialize weights
            weights_bytes = serialize_model(message.weights)
            loss_val = message.loss if message.loss is not None else 0.0
            acc_val = message.accuracy if message.accuracy is not None else 0.0
            timestamp = int(time.time())
            signature = _tunnel_sign(
                signing_key,
                self.node_id,
                self.domain,
                message.round_number,
                self.data_schema_hash,
                message.sample_count,
                loss_val,
                acc_val,
                timestamp,
                weights_bytes,
            )
            signer_public_key = signing_key.pub().key_to_bin()

            payload = {
                "type": "MODEL_UPDATE",
                "sender_id": self.node_id,
                "domain": self.domain,
                "data_schema_hash": self.data_schema_hash,
                "round_number": message.round_number,
                "sample_count": message.sample_count,
                "loss": loss_val,
                "accuracy": acc_val,
                "weights": weights_bytes.hex(),
                "signature": signature.hex(),
                "signer_public_key": signer_public_key.hex(),
                "timestamp": timestamp,
            }

            # Pre-send size check against gRPC max message length
            GRPC_MAX_MSG_BYTES = 50 * 1024 * 1024  # 50 MB (matches server config)
            payload_json = json.dumps(payload)
            if len(payload_json.encode("utf-8")) > GRPC_MAX_MSG_BYTES:
                raise ValueError(
                    f"Tunnel payload too large ({len(payload_json) / 1024 / 1024:.1f} MB) "
                    f"— exceeds gRPC max message length"
                )

            await self.tunnel_client.send_chat_message(peer_id, payload_json)
            logger.debug(f"Sent model update to {peer_id} via tunnel")

        except Exception as e:
            logger.error(f"Failed to send model update via tunnel: {e}")

    async def _announce_to_tunnel(self):
        """Announce ourselves to other peers via tunnel."""
        if not self.tunnel_client:
            return

        import json

        announcement = {
            "type": "PEER_ANNOUNCE",
            "peer_info": {
                "node_id": self.node_id,
                "domain": self.domain,
                "data_schema_hash": self.data_schema_hash,
                "model_version": self.model_version
            }
        }

        # Broadcast to all known peers
        for peer_id in self._tunnel_peers.keys():
            try:
                await self.tunnel_client.send_chat_message(peer_id, json.dumps(announcement))
            except Exception as e:
                logger.debug(f"Failed to announce to {peer_id}: {e}")

    def stop(self):
        """Stop the node (best-effort, sync path).

        ``LearningNode.stop`` is a coroutine; when called from a running
        event loop we schedule it, otherwise we fall back to
        ``asyncio.run`` so callers on the sync path still get a graceful
        teardown.  Prefer :meth:`shutdown` for the fully async flow.
        """
        if not self.running:
            return

        logger.debug(f"Stopping GossipNode '{self.node_id}'...")

        coro = self.gl_node.stop()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            loop.create_task(coro)
        else:
            try:
                asyncio.run(coro)
            except Exception as exc:  # pragma: no cover — best effort
                logger.debug("Sync stop() teardown raised: %s", exc)

        self.running = False
        if self._telemetry_client is not None:
            self._telemetry_client.pause()

    # ------------------------------------------------------------------
    # Spec §14 — join-flow state machine
    # ------------------------------------------------------------------

    def _transition(self, new_state: NodeState, **event_data: Any) -> None:
        """Advance the state machine and emit a ``node.state.*`` event.

        Invalid transitions raise :class:`RuntimeError` rather than silently
        no-op so that test suites and observers can pin down buggy call
        sites.  The only exception is the no-op self-transition (``old ==
        new``), which is tolerated because multiple code paths (e.g.
        ``start()`` + lazy peer-discovered callback) may race to mark the
        same milestone.
        """
        old = self.state
        if new_state is old:
            return
        allowed = _ALLOWED_TRANSITIONS.get(old, frozenset())
        if new_state not in allowed:
            raise RuntimeError(
                f"illegal GossipNode transition {old.value!r} → "
                f"{new_state.value!r}; allowed: {sorted(s.value for s in allowed)}"
            )
        self.state = new_state
        emitter = None
        try:
            emitter = self.gl_node.aggregator.event_emitter
        except AttributeError:
            pass
        if emitter is not None:
            try:
                emitter.emit(
                    f"node.state.{new_state.value}",
                    {
                        "node_id": self.node_id,
                        "from": old.value,
                        "to": new_state.value,
                        **event_data,
                    },
                )
            except Exception:  # pragma: no cover — observer failures are non-fatal
                pass

    def mark_peer_discovered(self, **event_data: Any) -> None:
        """Signal that at least one compatible peer has been observed.

        Intended to be called from the community's peer-discovered callback
        (or from tests to drive the state machine deterministically).  A
        no-op once the node is already past ``PEERS_DISCOVERED``.
        """
        if self.state in (
            NodeState.PEERS_DISCOVERED,
            NodeState.TRAINING,
        ):
            return
        self._transition(NodeState.PEERS_DISCOVERED, **event_data)

    def begin_training(self, **event_data: Any) -> None:
        """Move the node into ``TRAINING`` after peers have been discovered.

        Called automatically at the top of :meth:`train`; exposed publicly
        so CLI-driven launchers can force the transition when they rely on
        out-of-band peer discovery (e.g. static bootstrap peers).
        """
        if self.state is NodeState.TRAINING:
            return
        if self.state is not NodeState.PEERS_DISCOVERED:
            raise RuntimeError(
                f"begin_training requires state PEERS_DISCOVERED, "
                f"currently {self.state.value!r}"
            )
        self._transition(NodeState.TRAINING, **event_data)

    def mark_failed(self, reason: str, **event_data: Any) -> None:
        """Move the node into the terminal ``FAILED`` state.

        ``reason`` is a short machine-readable string (e.g.
        ``"manifest_load"``) that observers can route on.  Subsequent
        calls to protocol methods (``start``, ``train``) on a failed node
        are the caller's responsibility — this method does not tear the
        node down.
        """
        self._transition(NodeState.FAILED, reason=reason, **event_data)

    # ------------------------------------------------------------------
    # Spec §10.4 / §10.5 public surface
    # ------------------------------------------------------------------

    @classmethod
    def from_domain(cls, node_id: str, domain: str, **kwargs: Any) -> "GossipNode":
        """Legacy shim: construct a manifest-less node from a domain string.

        Kept as an explicit classmethod so call-sites that do NOT yet hold a
        :class:`SwarmManifest` can migrate incrementally.  ``manifest`` is
        forbidden here — use the constructor directly for the manifest-first
        path (spec §10.4).
        """
        if "manifest" in kwargs:
            raise TypeError(
                "GossipNode.from_domain does not accept `manifest` — use "
                "GossipNode(manifest=...) for the manifest-first path"
            )
        return cls(node_id=node_id, domain=domain, **kwargs)

    async def train(
        self,
        *,
        rounds: int,
        data_provider: Any = None,
        eval_data_provider: Any = None,
        stop_condition: Optional[Callable[[int, Dict[str, float]], bool]] = None,
        on_round_end: Optional[
            Callable[[int, Dict[str, float]], Awaitable[None]]
        ] = None,
    ) -> None:
        """Run the gossip training loop for ``rounds`` iterations (spec §10.5.4).

        This is a thin wrapper over the underlying :class:`LearningNode`
        loop.  ``rounds`` MUST be a positive integer; when the manifest
        specifies ``round_limit``, the effective cap is
        ``min(rounds, manifest.round_limit)`` so users cannot accidentally
        exceed the swarm-declared budget.

        ``stop_condition(round_idx, metrics) -> bool`` is consulted after
        every round; a truthy return terminates the loop early.
        ``on_round_end(round_idx, metrics)`` is fired as an async callback
        for observers.  Exceptions raised inside these hooks are logged but
        MUST NOT terminate training (§10.5.5).
        """
        if not isinstance(rounds, int) or rounds <= 0:
            raise ValueError(
                f"rounds must be a positive integer, got {rounds!r}"
            )
        if self.manifest is not None and self.manifest.round_limit:
            rounds = min(rounds, int(self.manifest.round_limit))

        # §14.2 transition — kicking off training implies peers were found.
        # Mirror the CLI/run behaviour where ``train()`` is awaited directly
        # after ``start()`` without an explicit peer-discovered hook (common
        # in static-bootstrap deployments and in tests).
        if self.state is NodeState.COMMUNITY_STARTED:
            self.mark_peer_discovered(reason="train_entered")
        if self.state is NodeState.PEERS_DISCOVERED:
            self.begin_training(rounds=rounds)

        # The full gossip loop lives in ``LearningNode.run_continuous``
        # (which delegates to the aggregator).  We kick it off as a
        # background task and bridge per-round notifications + rounds cap
        # to the caller by polling ``aggregator.current_round`` +
        # ``aggregator.metrics``.
        aggregator = self.gl_node.aggregator

        async def _run_continuous_task() -> None:
            # Match :meth:`run_continuous` (transport wiring + community sync) so
            # ``send_message_callback`` and ``known_peers`` are set before the
            # gossip loop, not only the bare :meth:`LearningNode.run_continuous`
            # path.
            await self._wire_gl_continuous_loop()
            await self.gl_node.run_continuous(
                data_provider=data_provider,
                eval_data_provider=eval_data_provider,
            )

        training_task = asyncio.create_task(_run_continuous_task())

        seen_round = 0
        poll_interval = 0.2
        try:
            while True:
                current = int(getattr(aggregator, "current_round", 0))
                if current > seen_round:
                    for round_idx in range(seen_round, current):
                        raw_metrics = getattr(aggregator, "metrics", {}) or {}
                        metrics: Dict[str, float] = {
                            k: float(v)
                            for k, v in raw_metrics.items()
                            if isinstance(v, (int, float))
                        }
                        if on_round_end is not None:
                            try:
                                await on_round_end(round_idx, metrics)
                            except Exception as exc:
                                logger.warning(
                                    "on_round_end hook raised: %s", exc
                                )
                        if stop_condition is not None:
                            try:
                                if stop_condition(round_idx, metrics):
                                    aggregator.running = False
                            except Exception as exc:  # pragma: no cover
                                logger.warning(
                                    "stop_condition raised, ignoring: %s", exc
                                )
                    seen_round = current

                if seen_round >= rounds:
                    aggregator.running = False

                if training_task.done():
                    break

                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.sleep(poll_interval)),
                        timeout=poll_interval + 0.1,
                    )
                except asyncio.TimeoutError:  # pragma: no cover
                    pass
        finally:
            aggregator.running = False
            if not training_task.done():
                try:
                    await asyncio.wait_for(training_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    training_task.cancel()
                    try:
                        await training_task
                    except (asyncio.CancelledError, Exception):
                        pass
            elif training_task.exception() is not None:
                logger.warning(
                    "run_continuous task raised: %s",
                    training_task.exception(),
                )

    async def __aenter__(self) -> "GossipNode":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # ``shutdown()`` is the full async teardown (IPv8 + tunnel + tasks);
        # ``stop()`` alone would leak sockets on exception paths.
        try:
            await self.shutdown()
        except Exception as err:  # pragma: no cover — defensive
            logger.warning("error during GossipNode shutdown: %s", err)

    async def shutdown(self):
        """Full shutdown including IPv8 and tunnel."""
        # §14.2 graceful stop — drive the state machine back to INIT so the
        # node may be re-started with the same manifest if the caller wants
        # a fresh join attempt (e.g. CLI ``--retry`` mode).  We do this at
        # the top of shutdown rather than the bottom so observers see the
        # transition before any blocking teardown noise.
        if self.state is not NodeState.INIT:
            try:
                self._transition(NodeState.INIT, reason="shutdown")
            except RuntimeError:
                # FAILED → INIT is allowed; any other illegal transition
                # means we were already in INIT or something tore down
                # mid-construction — safe to ignore.
                pass

        # Cancel the gossip-loop task first, then await it
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        self.stop()

        # Stop Cyclon topology if running
        from quinkgl.topology.cyclon import CyclonTopology
        if isinstance(self.gl_node.topology, CyclonTopology):
            await self.gl_node.topology.stop()

        # Stop IPv8
        await self.ipv8_manager.stop()

        # Stop tunnel client
        if self.tunnel_client and self._tunnel_connected:
            try:
                await self.tunnel_client.close()
                logger.debug("Tunnel client disconnected")
            except Exception as e:
                logger.warning(f"Error closing tunnel client: {e}")

        if self._telemetry_client is not None:
            await self._telemetry_client.stop()

        # Emit node.stopped lifecycle event
        emitter = self.gl_node.aggregator.event_emitter
        if emitter:
            uptime = time.time() - self._start_time if self._start_time else 0.0
            emitter.emit("node.stopped", {
                "node_id": self.node_id,
                "total_rounds": self.gl_node.current_round,
                "uptime_seconds": round(uptime, 1),
            })

        logger.debug(f"GossipNode '{self.node_id}' shutdown complete")

    def get_stats(self) -> dict:
        """Get node statistics."""
        ipv8_stats = self.ipv8_manager.get_stats()

        return {
            "node_id": self.node_id,
            "domain": self.domain,
            "data_schema_hash": self.data_schema_hash,
            "running": self.running,
            "current_round": self.gl_node.current_round,
            "connection_mode": self.connection_mode.value,
            "ipv8_peers": self.community.get_peer_count() if self.community else 0,
            "tunnel_peers": len(self._tunnel_peers),
            "ipv8_port": ipv8_stats.get("port"),
            "tunnel_server": self.tunnel_server,
            "known_peers": [
                p.node_id for p in self.community.get_compatible_peers()
            ] if self.community else list(self._tunnel_peers.keys())
        }

    def get_model(self) -> ModelWrapper:
        """Get the underlying model wrapper."""
        return self.gl_node.get_model()

    def attach_terminal_observer(self, observer=None):
        """
        Attach a terminal observer to the node's runtime event stream.

        Args:
            observer: Optional TerminalObserver instance. If omitted, a default
                observer is created.

        Returns:
            The attached observer instance.
        """
        if observer is None:
            from quinkgl.observability.terminal import TerminalObserver

            observer = TerminalObserver()

        self.gl_node.aggregator.event_emitter.subscribe(observer.handle)
        return observer

    def attach_telemetry_client(self, telemetry_client=None, *, base_url: Optional[str] = None, heartbeat_interval: float = 5.0):
        """
        Attach a telemetry client to the node's runtime event stream.

        Args:
            telemetry_client: Optional TelemetryClient instance.
            base_url: Telemetry service base URL used when creating a default client.
            heartbeat_interval: Seconds between heartbeat updates for a default client.

        Returns:
            The attached TelemetryClient instance.
        """
        if telemetry_client is None:
            from quinkgl.telemetry.client import TelemetryClient

            if not base_url:
                raise ValueError("base_url is required when telemetry_client is not provided")
            telemetry_client = TelemetryClient(
                base_url=base_url,
                heartbeat_interval=heartbeat_interval,
            )

        self._telemetry_client = telemetry_client
        telemetry_client.bind_runtime_event_sink(
            lambda event_type, payload: self.gl_node.aggregator.event_emitter.emit(event_type, payload)
        )
        self.gl_node.aggregator.event_emitter.subscribe(telemetry_client.handle)
        if self.running:
            telemetry_client.start(self.get_stats)

        # Emit a one-time event so TerminalObserver can show a single line
        emitter = self.gl_node.aggregator.event_emitter
        if emitter:
            emitter.emit("telemetry.connected", {
                "node_id": self.node_id,
                "base_url": telemetry_client.base_url,
                "heartbeat_interval": telemetry_client.heartbeat_interval,
            })

        return telemetry_client

    def is_using_fallback(self) -> bool:
        """Check if node is using tunnel fallback."""
        return self.connection_mode == ConnectionMode.TUNNEL_RELAY

    def get_connection_mode(self) -> ConnectionMode:
        """Get current connection mode."""
        return self.connection_mode
