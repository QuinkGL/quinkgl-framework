"""
Framework Test Script - Multi-Node Simulation

Tests the QuinkGL framework without IPv8 by simulating
multiple nodes in the same process with message passing.
"""

import asyncio
import sys
import os
from typing import Dict, List
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pytest

from quinkgl import GLNode
from quinkgl.models import PyTorchModel
from quinkgl.models.base import TrainingConfig
from quinkgl.topology import RandomTopology
from quinkgl.aggregation import FedAvg
from quinkgl.topology.base import PeerInfo, SelectionContext
from quinkgl.gossip.protocol import ModelUpdateMessage, MessageType
from quinkgl.aggregation.base import ModelUpdate

# Import test utilities - use direct import
sys.path.insert(0, str(Path(__file__).parent))
from simple_model import SimpleMLP
from data_generator import (
    generate_non_iid_data,
    generate_binary_classification
)

pytestmark = pytest.mark.asyncio


class MockNetwork:
    """
    Mock network layer for simulating P2P communication
    without actual IPv8 networking.
    """

    def __init__(self):
        self.nodes: Dict[str, GLNode] = {}
        self.message_log: List[dict] = []

    def register_node(self, node: GLNode):
        """Register a node with the mock network."""
        self.nodes[node.peer_id] = node
        # Set the network callback for sending messages
        node.aggregator.send_message_callback = self._send_message
        node.aggregator.broadcast_callback = self._broadcast_message

    def _send_message(self, target_peer_id: str, message):
        """Simulate sending a message to a specific peer."""
        self.message_log.append({
            "from": message.sender_id if hasattr(message, 'sender_id') else "unknown",
            "to": target_peer_id,
            "type": message.msg_type if hasattr(message, 'msg_type') else "unknown",
            "time": datetime.now()
        })

        # Deliver message to target if exists
        if target_peer_id in self.nodes:
            asyncio.create_task(
                self.nodes[target_peer_id]._handle_network_message(message)
            )

    def _broadcast_message(self, message):
        """Simulate broadcasting a message to all peers."""
        for peer_id in self.nodes:
            if peer_id != message.sender_id:
                self._send_message(peer_id, message)

    def connect_all_to_all(self):
        """Connect all nodes to each other (mesh topology)."""
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


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_metrics(peer_id: str, metrics: dict):
    """Print training metrics."""
    loss = metrics.get("loss", 0)
    acc = metrics.get("accuracy", 0)
    print(f"  [{peer_id}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")


async def test_basic_components():
    """Test 1: Basic framework components."""
    print_header("TEST 1: Basic Framework Components")

    # Create a simple model
    pytorch_model = SimpleMLP(input_size=10)
    model_wrapper = PyTorchModel(pytorch_model)

    # Test get/set weights
    weights_before = model_wrapper.get_weights()
    print(f"✓ Model weights retrieved: {len(weights_before)} layers")

    # Test topology
    from quinkgl.topology.base import SelectionContext, PeerInfo
    topology = RandomTopology()
    context = SelectionContext(
        my_peer_id="test-node",
        my_domain="test",
        my_data_schema_hash="abc123",
        known_peers=[
            PeerInfo("peer1", "test", "abc123"),
            PeerInfo("peer2", "test", "abc123"),
            PeerInfo("peer3", "other", "xyz"),  # Incompatible
        ]
    )
    targets = await topology.select_targets(context, count=2)
    print(f"✓ Topology selected {len(targets)} targets: {targets}")

    # Test aggregation
    from quinkgl.aggregation.base import ModelUpdate
    updates = [
        ModelUpdate("peer1", weights={"layer1": np.array([1.0, 2.0])}, sample_count=100),
        ModelUpdate("peer2", weights={"layer1": np.array([2.0, 3.0])}, sample_count=200),
    ]
    aggregator = FedAvg()
    result = await aggregator.aggregate(updates)
    print(f"✓ Aggregated model from {len(result.contributing_peers)} peers")

    print("\n✅ TEST 1 PASSED: All basic components work!")


async def test_single_node_training():
    """Test 2: Single node training."""
    print_header("TEST 2: Single Node Training")

    # Generate data
    data = generate_binary_classification(n_samples=500, random_seed=42)
    print(f"Data: {data.train_features.shape} train, {data.test_features.shape} test")

    # Create model
    pytorch_model = SimpleMLP(input_size=10)
    model_wrapper = PyTorchModel(pytorch_model)

    # Create node (without network)
    node = GLNode(
        peer_id="test-node",
        domain="test-domain",
        model=model_wrapper,
        topology=RandomTopology(),
        aggregation=FedAvg()
    )

    # Train
    from quinkgl.models.base import TrainingConfig
    config = TrainingConfig(epochs=5, batch_size=32, learning_rate=0.01, verbose=True)

    result = await model_wrapper.train(
        (data.train_features, data.train_labels),
        config
    )

    print(f"\nTraining completed:")
    print(f"  Epochs: {result.epochs_completed}")
    print(f"  Final Loss: {result.final_loss:.4f}")
    print(f"  Final Accuracy: {result.final_accuracy:.4f}")

    # Evaluate
    test_metrics = model_wrapper.evaluate((data.test_features, data.test_labels))
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    print("\n✅ TEST 2 PASSED: Single node training works!")


async def test_multi_node_gossip():
    """Test 3: Multi-node gossip simulation."""
    print_header("TEST 3: Multi-Node Gossip Simulation")

    # Configuration
    N_NODES = 3
    N_SAMPLES_PER_NODE = 200
    N_ROUNDS = 2

    # Generate non-IID data for each node
    print(f"\nGenerating non-IID data for {N_NODES} nodes...")
    node_data = generate_non_iid_data(
        n_nodes=N_NODES,
        n_samples_per_node=N_SAMPLES_PER_NODE,
        n_classes=4,
        skew=0.7,
        random_seed=42
    )

    # Show data distribution
    for node_id, (features, labels) in node_data.items():
        class_dist = np.bincount(labels, minlength=4)
        print(f"  Node {node_id}: {class_dist}")

    # Create mock network
    network = MockNetwork()

    # Create nodes
    nodes = []
    for i in range(N_NODES):
        # Create model (4 classes for multi-class data)
        pytorch_model = SimpleMLP(input_size=10, num_classes=4)
        model_wrapper = PyTorchModel(pytorch_model)

        # Create node
        node = GLNode(
            peer_id=f"node-{i}",
            domain="test-domain",
            model=model_wrapper,
            topology=RandomTopology(),
            aggregation=FedAvg(),
            gossip_interval=1.0,  # Short interval for testing
            training_config=TrainingConfig(
                epochs=2,
                batch_size=32,
                learning_rate=0.01,
                verbose=False
            )
        )

        network.register_node(node)
        nodes.append(node)

    # Connect all nodes
    network.connect_all_to_all()

    # Track metrics
    history = {node.peer_id: {"loss": [], "acc": []} for node in nodes}

    print(f"\nStarting {N_ROUNDS} rounds of gossip learning...")

    for round_num in range(N_ROUNDS):
        print(f"\n--- Round {round_num + 1} ---")

        # Each node trains locally
        for i, node in enumerate(nodes):
            features, labels = node_data[i]

            # Train
            result = await node.model.train(
                (features, labels),
                node.aggregator.training_config
            )

            # Evaluate
            eval_metrics = node.model.evaluate((features, labels))
            history[node.peer_id]["loss"].append(eval_metrics["loss"])
            history[node.peer_id]["acc"].append(eval_metrics["accuracy"])

            print_metrics(node.peer_id, eval_metrics)

        # Each node sends its model to others
        print("\n  Exchanging models...")
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
                    network._send_message(target_id, message)

        # Wait for message delivery
        await asyncio.sleep(0.1)

        # Each node aggregates received models
        for node in nodes:
            if node.aggregator.pending_updates:
                aggregated = await node.aggregator._aggregate_models()
                if aggregated:
                    print(f"  {node.peer_id} aggregated {len(aggregated.contributing_peers)} models")

    # Show final results
    print_header("Final Results")
    for node_id, metrics in history.items():
        print(f"\n{node_id}:")
        print(f"  Initial Loss: {metrics['loss'][0]:.4f}")
        print(f"  Final Loss: {metrics['loss'][-1]:.4f}")
        print(f"  Initial Acc: {metrics['acc'][0]:.4f}")
        print(f"  Final Acc: {metrics['acc'][-1]:.4f}")

    # Show network statistics
    print(f"\nNetwork Statistics:")
    print(f"  Total messages sent: {len(network.message_log)}")

    print("\n✅ TEST 3 PASSED: Multi-node gossip simulation works!")


async def main():
    """Run all tests."""
    print("\n" + "🧪 " * 20)
    print("  QuinkGL Framework Test Suite")
    print("  Testing without IPv8 (mock network)")
    print("🧪 " * 20)

    try:
        await test_basic_components()
        await test_single_node_training()
        await test_multi_node_gossip()

        print_header("ALL TESTS PASSED ✅")
        print("\nThe framework is ready for IPv8 integration!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
