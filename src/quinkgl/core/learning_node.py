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
LearningNode - Framework Node for Gossip Learning

The primary interface for users to participate in gossip learning.
This is the framework layer - network layer is handled separately.
"""

import logging
import asyncio
from typing import Optional, Any, Callable, List

from quinkgl.models.base import ModelWrapper, TrainingConfig
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
from quinkgl.aggregation.base import AggregationStrategy
from quinkgl.gossip.aggregator import ModelAggregator
from quinkgl.storage.model_store import ModelStore

logger = logging.getLogger(__name__)


class LearningNode:
    """
    Framework node for Gossip Learning.

    This is the primary interface for users to participate in
    decentralized gossip learning. It handles the training, gossip,
    and aggregation cycles. Network layer must be connected separately.

    Example:
        ```python
        from quinkgl import LearningNode, PyTorchModel, RandomTopology, FedAvg

        # Wrap your model
        model = PyTorchModel(my_pytorch_model)

        # Create node
        node = LearningNode(
            peer_id="my-peer-1",
            domain="health",
            model=model,
            topology=RandomTopology(),
            aggregation=FedAvg()
        )

        # Join and run
        await node.join()
        await node.run_continuous(training_data)
        ```

    Note: For production use with built-in P2P networking, use GossipNode instead.
    """

    def __init__(
        self,
        peer_id: str,
        domain: str,
        model: ModelWrapper,
        topology: TopologyStrategy,
        aggregation: AggregationStrategy,
        data_schema_hash: Optional[str] = None,
        storage_dir: Optional[str] = None,
        gossip_interval: float = 60.0,
        training_config: Optional[TrainingConfig] = None,
        min_peers_before_aggregate: int = 1
    ):
        """
        Initialize a LearningNode.

        Args:
            peer_id: Unique identifier for this node
            domain: Domain identifier (e.g., "health", "agriculture")
            model: Wrapped model (PyTorchModel, TensorFlowModel, or custom)
            topology: Topology strategy for peer selection
            aggregation: Aggregation strategy for combining models
            data_schema_hash: Optional schema hash (auto-generated if None)
            storage_dir: Optional directory for model checkpoints
            gossip_interval: Seconds between gossip rounds
            training_config: Configuration for local training
            min_peers_before_aggregate: Minimum pending updates before aggregation
        """
        self.peer_id = peer_id
        self.domain = domain

        # Model
        self.model = model

        # Auto-generate schema hash if not provided
        if data_schema_hash is None:
            data_schema_hash = model.get_data_schema_hash()
        self.data_schema_hash = data_schema_hash

        # Strategies
        self.topology = topology
        self.aggregation = aggregation

        # Storage
        self.model_store = ModelStore(storage_dir=storage_dir) if storage_dir else None

        # Create aggregator (manages training→gossip→aggregation cycle)
        self.aggregator = ModelAggregator(
            peer_id=peer_id,
            domain=domain,
            data_schema_hash=data_schema_hash,
            model=model,
            topology=topology,
            aggregator=aggregation,
            gossip_interval=gossip_interval,
            training_config=training_config,
            min_peers_before_aggregate=min_peers_before_aggregate
        )

        # Bootstrap peers for manual peer discovery
        self._bootstrap_peers: List[str] = []

        logger.debug(
            f"LearningNode initialized: peer_id={peer_id}, domain={domain}, "
            f"schema={data_schema_hash[:8]}..."
        )

    async def join(self, bootstrap_peers: Optional[List[str]] = None):
        """
        Join the learning network.

        Args:
            bootstrap_peers: Optional list of bootstrap peer addresses
                             to manually add for discovery.
        """
        if self.is_running:
            logger.warning("Node already joined")
            return

        # Store bootstrap peers for reference
        if bootstrap_peers:
            self._bootstrap_peers = bootstrap_peers
            logger.debug(f"Bootstrap peers provided: {bootstrap_peers}")

        # Note: Actual network connection is handled by transport layer (e.g., GossipNode)
        # This method marks the node as ready for learning
        logger.debug(f"Node {self.peer_id} joined domain '{self.domain}'")

    async def leave(self):
        """Leave the learning network and stop training."""
        if not self.is_running and not self.aggregator.running:
            logger.warning("Node is not active")
            return

        self.aggregator.stop()
        logger.debug(f"Node {self.peer_id} left the network")

    async def run_continuous(self, data=None, data_provider: Optional[Callable] = None):
        """
        Run continuous gossip learning.

        Args:
            data: Training data (single dataset)
            data_provider: Callable that returns training data per round

        Either `data` or `data_provider` should be provided.
        If `data_provider` is given, it's called each round to get fresh data.

        Raises:
            RuntimeError: If run without starting the learning loop first
        """
        if not self.is_running and not self.aggregator.running:
            # Auto-start if not running
            logger.debug("Auto-starting learning loop")

        logger.debug(f"Starting continuous gossip learning for node {self.peer_id}")

        await self.aggregator.run_continuous(
            data_provider=data_provider or data
        )

    def stop(self):
        """Stop the gossip learning loop."""
        self.aggregator.stop()

    def register_hook(self, hook_name: str, callback: Callable):
        """
        Register a lifecycle hook.

        Args:
            hook_name: Name of the hook ("before_train", "after_train", etc.)
            callback: Async or sync function to call

        Available hooks:
            - before_train: Called before local training
            - after_train: Called after local training (receives TrainingResult)
            - before_send: Called before sending model (receives weights)
            - after_receive: Called after receiving model update
            - before_aggregate: Called before aggregation (receives pending updates)
            - after_aggregate: Called after aggregation (receives AggregatedModel)
            - on_training_complete: Called with TrainingResult after training
            - on_model_sent: Called with (peer_ids, model_size) after sending
            - on_aggregation_complete: Called with AggregatedModel after aggregation
        """
        self.aggregator.register_hook(hook_name, callback)

    async def save_checkpoint(self, metrics: Optional[dict] = None):
        """
        Save current model as a checkpoint.

        Args:
            metrics: Optional metrics dict (loss, accuracy, etc.)
        """
        if self.model_store:
            self.model_store.save_checkpoint(
                round_number=self.aggregator.current_round,
                weights=self.model.get_weights(),
                metrics=metrics or {}
            )
        else:
            logger.warning("No model store configured, checkpoint not saved")

    def get_model(self) -> ModelWrapper:
        """Get the underlying model wrapper."""
        return self.model

    def add_peer(self, peer_id: str, domain: str = None, data_schema_hash: str = None):
        """
        Manually add a peer for learning.

        Args:
            peer_id: Peer identifier
            domain: Domain of the peer (defaults to self.domain)
            data_schema_hash: Schema hash of the peer (defaults to self.data_schema_hash)
        """
        peer_info = PeerInfo(
            peer_id=peer_id,
            domain=domain or self.domain,
            data_schema_hash=data_schema_hash or self.data_schema_hash,
            model_version="1.0.0"
        )
        self.aggregator.add_peer(peer_info)
        logger.debug(f"Manually added peer: {peer_id}")

    def remove_peer(self, peer_id: str):
        """
        Manually remove a peer.

        Args:
            peer_id: Peer identifier
        """
        asyncio.create_task(self.aggregator.remove_peer(peer_id))
        logger.debug(f"Manually removed peer: {peer_id}")

    def get_peers(self) -> List[PeerInfo]:
        """Get list of known peers."""
        return list(self.aggregator.known_peers.values())

    @property
    def current_round(self) -> int:
        """Get current training round number."""
        return self.aggregator.current_round

    def increment_round(self):
        """Manually increment the current round number."""
        self.aggregator.increment_round()

    @property
    def is_running(self) -> bool:
        """Check if gossip loop is running."""
        return self.aggregator.running

    @property
    def known_peers(self) -> dict:
        """Get known peers dict (for compatibility)."""
        return self.aggregator.known_peers

    # Network integration methods (to be used by transport layer)

    async def _handle_network_message(self, message):
        """
        Handle an incoming message from the network layer.

        This is called by the transport layer when a message arrives.
        """
        await self.aggregator.handle_incoming_message(message)

    def _set_network_layer(self, network_layer):
        """
        Set the network layer for sending messages.

        This is called by the transport layer during initialization.

        Args:
            network_layer: Object with send_message and broadcast_message methods
        """
        self.aggregator.send_message_callback = network_layer.send_message
        if hasattr(network_layer, 'broadcast_message'):
            self.aggregator.broadcast_callback = network_layer.broadcast_message


# Backward compatibility alias
GLNode = LearningNode
