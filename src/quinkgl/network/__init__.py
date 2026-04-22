"""
Network package.

This package owns IPv8 transport, peer connectivity, and wire-level model
delivery helpers for GossipNode. It stays focused on network mechanics
and re-exports `serialize_model` / `deserialize_model` for the network's
model payload format. Generic weight-array serialization lives in
`quinkgl.serialization`.

The exported `PeerInfo` is network-layer peer metadata used by the IPv8
community code. It is distinct from `quinkgl.topology.base.PeerInfo`,
which models the framework's partial-view topology state.
"""

from quinkgl.network.ipv8_manager import IPv8Manager
from quinkgl.network.gossip_community import GossipLearningCommunity, PeerInfo, generate_community_id
from quinkgl.network.gossip_node import GossipNode, ConnectionMode
from quinkgl.network.model_serializer import serialize_model, deserialize_model, get_model_size_info
from quinkgl.network.directory import (
    DIRECTORY_COMMUNITY_ID,
    SwarmAdvertisement,
    SwarmAdvertisementPayload,
    SwarmDirectoryCommunity,
    sign_advertisement,
    verify_advertisement,
)

__all__ = [
    "IPv8Manager",
    "GossipLearningCommunity",
    "PeerInfo",
    "GossipNode",
    "ConnectionMode",
    "generate_community_id",
    "serialize_model",
    "deserialize_model",
    "get_model_size_info",
    "DIRECTORY_COMMUNITY_ID",
    "SwarmAdvertisement",
    "SwarmAdvertisementPayload",
    "SwarmDirectoryCommunity",
    "sign_advertisement",
    "verify_advertisement",
]
