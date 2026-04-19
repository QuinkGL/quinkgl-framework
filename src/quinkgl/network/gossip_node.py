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
import time
from typing import Optional, Any, Callable, Dict
from enum import Enum

from quinkgl.core.learning_node import LearningNode
from quinkgl.models.base import ModelWrapper, TrainingConfig
from quinkgl.topology.base import TopologyStrategy, PeerInfo, SelectionContext
from quinkgl.aggregation.base import AggregationStrategy
from quinkgl.network.ipv8_manager import IPv8Manager
from quinkgl.network.gossip_community import generate_community_id
from quinkgl.network.gossip_community import GossipLearningCommunity

logger = logging.getLogger(__name__)


class ConnectionMode(Enum):
    """Connection mode for the node."""
    IPV8_P2P = "ipv8_p2p"       # Direct P2P via IPv8
    TUNNEL_RELAY = "tunnel"     # Tunnel relay fallback


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
        domain: str,
        model: ModelWrapper,
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
        quiet: bool = False,
    ):
        """
        Initialize GossipNode.

        Args:
            node_id: Unique identifier for this node
            domain: Domain identifier (e.g., "health", "agriculture")
            model: Wrapped model (PyTorchModel, TensorFlowModel, or custom)
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
        self.node_id = node_id
        self.domain = domain
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
        self.gl_node = LearningNode(
            peer_id=node_id,
            domain=domain,
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
        self.ipv8_manager.domain = domain
        self.ipv8_manager.data_schema_hash = self.data_schema_hash

        # Community (set after IPv8 starts)
        self.community: Optional[GossipLearningCommunity] = None

        # Tunnel client (lazy loaded)
        self.tunnel_client = None
        self._tunnel_connected = False

        # Remote peers via tunnel
        self._tunnel_peers: Dict[str, dict] = {}  # peer_id -> {node_id, domain, schema}

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
            f"GossipNode initialized: node_id={node_id}, domain={domain}, "
            f"schema={self.data_schema_hash[:8]}..., "
            f"fallback={'enabled' if enable_fallback else 'disabled'}"
        )

    async def start(self):
        """Start the node and join the P2P network."""
        if self.running:
            logger.warning("Node already running")
            return

        logger.debug(f"Starting GossipNode '{self.node_id}'...")
        logger.debug(f"Attempting P2P connection (timeout: {self.fallback_timeout}s)...")

        # Try IPv8 P2P first with timeout
        ipv8_started = await self._try_start_ipv8_with_timeout()

        if ipv8_started:
            self.connection_mode = ConnectionMode.IPV8_P2P
            logger.debug("Using IPv8 P2P mode")
        elif self.enable_fallback and self.tunnel_server:
            # IPv8 failed or timed out, try tunnel fallback
            logger.warning("IPv8 P2P failed/timeout, falling back to tunnel relay...")
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
            self.community._instance_community_id = generate_community_id(
                self.domain, self.data_schema_hash
            )
            type(self.community).community_id = self.community._instance_community_id

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

    async def _start_tunnel_fallback(self):
        """Start tunnel relay fallback."""
        try:
            from quinkgl.network.fallback import TunnelClient

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

            # ── Required-field validation ──────────────────────────
            required = ("sender_id", "round_number", "weights")
            missing = [f for f in required if f not in data]
            if missing:
                logger.warning(f"Tunnel MODEL_UPDATE missing fields: {missing}")
                if emitter:
                    emitter.emit("tunnel_payload_dropped", {
                        "node_id": self.node_id,
                        "reason": f"missing fields: {missing}",
                    })
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
                if emitter:
                    emitter.emit("tunnel_payload_dropped", {
                        "node_id": self.node_id,
                        "peer_id": data.get("sender_id"),
                        "reason": "domain/schema mismatch",
                    })
                return

            # ── Size bound on hex-decoded weights ─────────────────
            weights_hex = data["weights"]
            if len(weights_hex) // 2 > MAX_INCOMING_MESSAGE_SIZE:
                logger.warning(
                    f"Tunnel update rejected: weights too large "
                    f"({len(weights_hex) // 2} bytes)"
                )
                if emitter:
                    emitter.emit("tunnel_payload_dropped", {
                        "node_id": self.node_id,
                        "peer_id": data.get("sender_id"),
                        "reason": "oversized weights",
                    })
                return

            # ── Deserialize weights ───────────────────────────────
            try:
                weights_bytes = bytes.fromhex(weights_hex)
                weights = deserialize_model(weights_bytes)
            except Exception as e:
                logger.warning(f"Tunnel model deserialization failed: {e}")
                if emitter:
                    emitter.emit("tunnel_payload_dropped", {
                        "node_id": self.node_id,
                        "peer_id": data.get("sender_id"),
                        "reason": f"deserialization error: {e}",
                    })
                return

            # ── Dispatch to aggregation layer ─────────────────────
            message = ModelUpdateMessage.create(
                sender_id=data["sender_id"],
                weights=weights,
                sample_count=data.get("sample_count", 0),
                loss=data.get("loss", 0.0),
                accuracy=data.get("accuracy", 0.0),
                round_number=data["round_number"],
            )
            await self.gl_node._handle_network_message(message)
            logger.debug(f"Processed tunnel model update from {data['sender_id']}")

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

    async def run_continuous(self, data=None, data_provider=None):
        """
        Run continuous gossip learning.

        Args:
            data: Training data (single dataset)
            data_provider: Callable that returns training data per round
        """
        if not self.running:
            raise RuntimeError("Node must be started before running")

        # Single dynamic dispatch — reads connection_mode on each call
        # so failback / mode changes take effect immediately.
        async def send_to_peer(peer_id: str, message):
            if self.connection_mode == ConnectionMode.IPV8_P2P:
                await self.community.send_model_update(
                    target_node_id=peer_id,
                    weights=message.weights,
                    sample_count=message.sample_count,
                    round_number=message.round_number,
                    loss=message.loss,
                    accuracy=message.accuracy
                )
            else:
                await self._send_model_update_via_tunnel(peer_id, message)

        self.gl_node.aggregator.send_message_callback = send_to_peer
        self._configure_local_fingerprint_runtime()

        # Sync peers before starting
        if self.connection_mode == ConnectionMode.IPV8_P2P:
            self._sync_known_peers()

        # Announce ourselves to other tunnel peers
        if self.connection_mode == ConnectionMode.TUNNEL_RELAY:
            await self._announce_to_tunnel()

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
            # Serialize weights
            weights_bytes = serialize_model(message.weights)

            payload = {
                "type": "MODEL_UPDATE",
                "sender_id": self.node_id,
                "domain": self.domain,
                "data_schema_hash": self.data_schema_hash,
                "round_number": message.round_number,
                "sample_count": message.sample_count,
                "loss": message.loss,
                "accuracy": message.accuracy,
                "weights": weights_bytes.hex()  # Convert bytes to hex for JSON
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
        """Stop the node."""
        if not self.running:
            return

        logger.debug(f"Stopping GossipNode '{self.node_id}'...")

        self.gl_node.stop()
        self.running = False
        if self._telemetry_client is not None:
            self._telemetry_client.pause()

    async def shutdown(self):
        """Full shutdown including IPv8 and tunnel."""
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
