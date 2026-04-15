"""
Topology package.

This package provides peer selection and partial-view management
strategies for decentralized learning. It does not own transport,
aggregation, or model serialization concerns, which live in
`quinkgl.network` and `quinkgl.serialization` respectively.

Usage:
    from quinkgl.topology import RandomTopology, CyclonTopology

    # Use built-in strategy
    topology = CyclonTopology(view_size=20)

    # Or implement custom
    from quinkgl.topology.base import TopologyStrategy
    class MyTopology(TopologyStrategy):
        async def select_targets(self, context, count=3):
            return [peer.peer_id for peer in context.get_compatible_peers()[:count]]

        async def should_accept_connection(self, context, peer_info):
            return peer_info.domain == context.my_domain

    topology = MyTopology()
"""

from typing import TYPE_CHECKING

from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
from quinkgl.topology.random import RandomTopology
from quinkgl.topology.cyclon import CyclonTopology
from quinkgl.topology.affinity import AffinityTopology
from quinkgl.topology.spectral import (
    SpectralAnalyzer,
    SpectralReport,
    build_ring_adjacency,
    build_complete_adjacency,
    build_random_regular_adjacency,
)

if TYPE_CHECKING:
    from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
    from quinkgl.topology.random import RandomTopology
    from quinkgl.topology.cyclon import CyclonTopology
    from quinkgl.topology.affinity import AffinityTopology
    from quinkgl.topology.spectral import (
        SpectralAnalyzer,
        SpectralReport,
        build_ring_adjacency,
        build_complete_adjacency,
        build_random_regular_adjacency,
    )

# Export main classes
__all__ = [
    "TopologyStrategy",
    "SelectionContext",
    "PeerInfo",
    "RandomTopology",
    "CyclonTopology",
    "AffinityTopology",
    # Spectral analysis
    "SpectralAnalyzer",
    "SpectralReport",
    "build_ring_adjacency",
    "build_complete_adjacency",
    "build_random_regular_adjacency",
]
