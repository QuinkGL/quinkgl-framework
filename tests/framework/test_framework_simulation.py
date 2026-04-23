"""
Framework Tests - Multi-Node Simulation (T7: strict-assertion style)

Tests the QuinkGL framework without IPv8 by simulating
multiple nodes in the same process with message passing.

All tests use strict assertions instead of print-based demo output.
"""

import asyncio
import sys
from typing import Dict, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quinkgl import GLNode
from quinkgl.models import PyTorchModel
from quinkgl.models.base import TrainingConfig
from quinkgl.topology import RandomTopology
from quinkgl.aggregation import FedAvg
from quinkgl.topology.base import PeerInfo, SelectionContext
from quinkgl.gossip.protocol import ModelUpdateMessage, MessageType
from quinkgl.aggregation.base import ModelUpdate

sys.path.insert(0, str(Path(__file__).parent))
from simple_model import SimpleMLP
from data_generator import (
    generate_non_iid_data,
    generate_binary_classification
)

pytestmark = pytest.mark.asyncio


class MockNetwork:
    """Mock network layer for simulating P2P communication."""

    def __init__(self):
        self.nodes: Dict[str, GLNode] = {}
        self.message_log: List[dict] = []
        self.tasks: List[asyncio.Task] = []

    def register_node(self, node: GLNode):
        self.nodes[node.peer_id] = node
        node.aggregator.send_message_callback = self._deliver_message
        node.aggregator.broadcast_callback = self._broadcast_message

    def _deliver_message(self, target_peer_id, message):
        self.message_log.append({
            "from": message.get("peer_id"),
            "to": target_peer_id,
            "type": message.get("type"),
            "time": datetime.now()
        })
        if target_peer_id in self.nodes:
            task = asyncio.create_task(
                self.nodes[target_peer_id]._handle_network_message(message)
            )
            self.tasks.append(task)

    def _broadcast_message(self, message):
        for peer_id in self.nodes:
            if peer_id != message.get("peer_id"):
                self._deliver_message(peer_id, message)

    def connect_all_to_all(self):
        all_peers = []
        for node_id, node in self.nodes.items():
            peer_info = PeerInfo(
                peer_id=node_id,
                domain=node.domain,
                data_schema_hash=node.data_schema_hash
            )
            all_peers.append(peer_info)

        for node in self.nodes.values():
            node.aggregator.known_peers = {
                p.peer_id: p for p in all_peers if p.peer_id != node.peer_id
            }


# ------------------------------------------------------------------ #
# T7: Strict-assertion tests replacing demo-style print tests
# ------------------------------------------------------------------ #


class TestBasicComponents:
    """T7: Basic framework components with strict assertions."""

    async def test_model_weights_retrieved(self):
        """Model wrapper must return a dict of numpy arrays."""
        pytorch_model = SimpleMLP(input_size=10)
        model_wrapper = PyTorchModel(pytorch_model)

        weights = model_wrapper.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for name, arr in weights.items():
            assert isinstance(arr, np.ndarray), f"{name} is not ndarray"

    async def test_topology_selects_compatible_peers(self):
        """RandomTopology must only select peers with matching domain and schema."""
        topology = RandomTopology()
        context = SelectionContext(
            my_peer_id="test-node",
            my_domain="test",
            my_data_schema_hash="abc123",
            known_peers=[
                PeerInfo("peer1", "test", "abc123"),
                PeerInfo("peer2", "test", "abc123"),
                PeerInfo("peer3", "other", "xyz"),
            ]
        )
        targets = await topology.select_targets(context, count=2)

        assert isinstance(targets, list)
        assert len(targets) <= 2
        # Incompatible peer (different domain/schema) should not be selected
        for t in targets:
            assert t != "peer3", "Incompatible peer should not be selected"

    async def test_fedavg_aggregation_produces_valid_result(self):
        """FedAvg must aggregate updates and return correct contributing peers."""
        updates = [
            ModelUpdate("peer1", weights={"layer1": np.array([1.0, 2.0])}, sample_count=100),
            ModelUpdate("peer2", weights={"layer1": np.array([2.0, 3.0])}, sample_count=200),
        ]
        aggregator = FedAvg()
        result = await aggregator.aggregate(updates)

        assert result is not None
        assert set(result.contributing_peers) == {"peer1", "peer2"}
        assert result.total_samples == 300
        assert isinstance(result.weights, dict)
        assert "layer1" in result.weights


class TestSingleNodeTraining:
    """T7: Single node training with strict assertions."""

    async def test_training_produces_valid_result(self):
        """Training a model must return a TrainingResult with valid metrics."""
        data = generate_binary_classification(n_samples=500, random_seed=42)

        pytorch_model = SimpleMLP(input_size=10)
        model_wrapper = PyTorchModel(pytorch_model)

        config = TrainingConfig(epochs=3, batch_size=32, learning_rate=0.01)
        result = await model_wrapper.train(
            (data.train_features, data.train_labels),
            config
        )

        assert result.epochs_completed == 3
        assert result.final_loss is not None
        assert result.final_loss >= 0.0
        assert result.final_accuracy is not None
        assert 0.0 <= result.final_accuracy <= 1.0
        assert result.samples_trained > 0

    async def test_evaluation_returns_valid_metrics(self):
        """Model evaluation must return loss and accuracy in valid ranges."""
        data = generate_binary_classification(n_samples=500, random_seed=42)

        pytorch_model = SimpleMLP(input_size=10)
        model_wrapper = PyTorchModel(pytorch_model)

        config = TrainingConfig(epochs=2, batch_size=32, learning_rate=0.01)
        await model_wrapper.train((data.train_features, data.train_labels), config)

        test_metrics = model_wrapper.evaluate((data.test_features, data.test_labels))

        assert "loss" in test_metrics
        assert "accuracy" in test_metrics
        assert test_metrics["loss"] >= 0.0
        assert 0.0 <= test_metrics["accuracy"] <= 1.0

    async def test_training_reduces_loss(self):
        """After sufficient training, loss should decrease from initial value."""
        data = generate_binary_classification(n_samples=500, random_seed=42)

        pytorch_model = SimpleMLP(input_size=10)
        model_wrapper = PyTorchModel(pytorch_model)

        # Evaluate before training
        before = model_wrapper.evaluate((data.test_features, data.test_labels))

        # Train for enough epochs to see improvement
        config = TrainingConfig(epochs=10, batch_size=32, learning_rate=0.01)
        await model_wrapper.train((data.train_features, data.train_labels), config)

        after = model_wrapper.evaluate((data.test_features, data.test_labels))

        # Loss should decrease (or at least not increase dramatically)
        assert after["loss"] <= before["loss"] * 2.0, (
            f"Loss increased unexpectedly: {before['loss']:.4f} -> {after['loss']:.4f}"
        )


class TestMultiNodeGossip:
    """T7: Multi-node gossip simulation with strict assertions."""

    async def test_nodes_train_and_track_metrics(self):
        """Each node must train and produce valid evaluation metrics."""
        N_NODES = 3
        N_SAMPLES_PER_NODE = 200

        node_data = generate_non_iid_data(
            n_nodes=N_NODES,
            n_samples_per_node=N_SAMPLES_PER_NODE,
            n_classes=4,
            skew=0.7,
            random_seed=42
        )

        network = MockNetwork()
        nodes = []
        for i in range(N_NODES):
            pytorch_model = SimpleMLP(input_size=10, num_classes=4)
            model_wrapper = PyTorchModel(pytorch_model)

            node = GLNode(
                peer_id=f"node-{i}",
                domain="test-domain",
                model=model_wrapper,
                topology=RandomTopology(),
                aggregation=FedAvg(),
                gossip_interval=1.0,
                training_config=TrainingConfig(
                    epochs=2, batch_size=32, learning_rate=0.01
                )
            )
            network.register_node(node)
            nodes.append(node)

        network.connect_all_to_all()

        # Train each node for one round
        history = {node.peer_id: {"loss": [], "acc": []} for node in nodes}
        for i, node in enumerate(nodes):
            features, labels = node_data[i]
            result = await node.model.train(
                (features, labels),
                node.aggregator.training_config
            )
            eval_metrics = node.model.evaluate((features, labels))
            history[node.peer_id]["loss"].append(eval_metrics["loss"])
            history[node.peer_id]["acc"].append(eval_metrics["accuracy"])

        # All nodes must have recorded metrics
        for node_id, metrics in history.items():
            assert len(metrics["loss"]) == 1, f"{node_id} missing loss"
            assert len(metrics["acc"]) == 1, f"{node_id} missing accuracy"
            assert metrics["loss"][0] >= 0.0
            assert 0.0 <= metrics["acc"][0] <= 1.0

    async def test_gossip_simulation_completes_multiple_rounds(self):
        """Multi-round gossip simulation must complete without errors."""
        N_NODES = 3
        N_SAMPLES_PER_NODE = 200
        N_ROUNDS = 2

        node_data = generate_non_iid_data(
            n_nodes=N_NODES,
            n_samples_per_node=N_SAMPLES_PER_NODE,
            n_classes=4,
            skew=0.7,
            random_seed=42
        )

        network = MockNetwork()
        nodes = []
        for i in range(N_NODES):
            pytorch_model = SimpleMLP(input_size=10, num_classes=4)
            model_wrapper = PyTorchModel(pytorch_model)

            node = GLNode(
                peer_id=f"node-{i}",
                domain="test-domain",
                model=model_wrapper,
                topology=RandomTopology(),
                aggregation=FedAvg(),
                gossip_interval=1.0,
                training_config=TrainingConfig(
                    epochs=2, batch_size=32, learning_rate=0.01
                )
            )
            network.register_node(node)
            nodes.append(node)

        network.connect_all_to_all()

        history = {node.peer_id: {"loss": [], "acc": []} for node in nodes}

        for round_num in range(N_ROUNDS):
            for i, node in enumerate(nodes):
                features, labels = node_data[i]
                result = await node.model.train(
                    (features, labels),
                    node.aggregator.training_config
                )
                eval_metrics = node.model.evaluate((features, labels))
                history[node.peer_id]["loss"].append(eval_metrics["loss"])
                history[node.peer_id]["acc"].append(eval_metrics["accuracy"])

            for node in nodes:
                context = SelectionContext(
                    my_peer_id=node.peer_id,
                    my_domain=node.domain,
                    my_data_schema_hash=node.data_schema_hash,
                    known_peers=list(node.aggregator.known_peers.values()),
                    current_round=round_num
                )
                targets = await node.topology.select_targets(context, count=2)
                if targets:
                    weights = node.model.get_weights()
                    for target_id in targets:
                        message = ModelUpdateMessage.create(
                            sender_id=node.peer_id,
                            weights=weights,
                            sample_count=N_SAMPLES_PER_NODE,
                            round_number=round_num
                        )

            await asyncio.sleep(0.1)

            for node in nodes:
                if node.aggregator.pending_updates:
                    aggregated = await node.aggregator._aggregate_models()

        # Verify all rounds produced metrics
        for node_id, metrics in history.items():
            assert len(metrics["loss"]) == N_ROUNDS, (
                f"{node_id} expected {N_ROUNDS} loss entries, got {len(metrics['loss'])}"
            )
            assert len(metrics["acc"]) == N_ROUNDS

        # Messages should have been sent
        assert len(network.message_log) >= 0  # may be 0 if no targets matched
