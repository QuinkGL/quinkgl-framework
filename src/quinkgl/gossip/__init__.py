"""
Gossip Protocol Module

Continuous gossip protocol for decentralized model exchange.
Handles the communication loop between peers.

Usage:
    from quinkgl.gossip import ModelAggregator

    aggregator = ModelAggregator(peer_id, domain, schema_hash, model, topology, aggregator)
    await aggregator.run_continuous(data_provider)
"""

from quinkgl.gossip.protocol import GossipProtocol, MessageType
from quinkgl.gossip.aggregator import ModelAggregator, GossipOrchestrator

__all__ = [
    "GossipProtocol",
    "MessageType",
    "ModelAggregator",
    "GossipOrchestrator",  # Deprecated alias (use ModelAggregator instead)
]
