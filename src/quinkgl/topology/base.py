"""
Base Topology Strategy

Abstract base class for all topology strategies.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from quinkgl.fingerprint.fingerprint import DataFingerprint

# Type-safe metadata structure
class PeerMetadata(TypedDict, total=False):
    """Typed dictionary for peer metadata with common fields."""
    capabilities: list[str]
    trust_score: float
    region: str
    last_latency_ms: float
    custom_data: Dict[str, Any]

@dataclass(frozen=False)
class PeerInfo:
    """Information about a peer in the network."""
    peer_id: str
    domain: str
    data_schema_hash: str
    manifest_id: Optional[bytes] = None
    model_version: str = "0.1.0"
    data_fingerprint: Optional["DataFingerprint"] = None
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    age: int = 0

    def update_last_seen(self) -> None:
        """Update the last_seen timestamp to now."""
        self.last_seen = datetime.now()

    def increment_age(self) -> None:
        """Increment age for Cyclon-style eviction."""
        self.age += 1

    def reset_age(self) -> None:
        """Reset age (typically after shuffle)."""
        self.age = 0

def is_version_compatible(local_version: str, remote_version: str) -> bool:
    """
    Check semantic version compatibility between two model versions.

    Peers are compatible if major versions match. Minor and patch
    differences are allowed (backward-compatible changes).

    Args:
        local_version: This node's model version string (e.g., "2.1.3")
        remote_version: Remote peer's model version string (e.g., "2.3.0")

    Returns:
        True if major versions match (compatible), False otherwise
    """
    def _parse_major(ver: str) -> int:
        try:
            return int(ver.split('.')[0])
        except (ValueError, IndexError, AttributeError):
            return 0

    return _parse_major(local_version) == _parse_major(remote_version)


@dataclass
class SelectionContext:
    """Context provided to topology strategy for target selection."""
    my_peer_id: str
    my_domain: str
    my_data_schema_hash: str
    known_peers: List[PeerInfo] = field(default_factory=list)
    current_round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    my_model_version: str = "1.0.0"
    my_manifest_id: Optional[bytes] = None
    my_fingerprint: Optional["DataFingerprint"] = None

    def get_compatible_peers(self, exclude_self: bool = True) -> List[PeerInfo]:
        compatible = []
        for peer in self.known_peers:
            if exclude_self and peer.peer_id == self.my_peer_id:
                continue
            if self.my_manifest_id is not None and peer.manifest_id is not None:
                if peer.manifest_id == self.my_manifest_id:
                    compatible.append(peer)
                continue
            if (peer.domain == self.my_domain and
                peer.data_schema_hash == self.my_data_schema_hash and
                is_version_compatible(self.my_model_version, peer.model_version)):
                compatible.append(peer)
        return compatible

class TopologyStrategy(ABC):
    """
    Abstract base class for topology strategies.

    A topology strategy determines which peers to communicate with
    during the gossip learning process.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the topology strategy with configuration."""
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3
    ) -> List[str]:
        """
        Select peer IDs to send model updates to.

        Args:
            context: Current execution context including known peers
            count: Maximum number of targets to select

        Returns:
            List of peer IDs to send updates to
        """

    @abstractmethod
    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo
    ) -> bool:
        """
        Determine if a connection from a peer should be accepted.

        Default implementation checks compatibility based on:
        - manifest_id (if both have it)
        - domain
        - data_schema_hash
        - model_version compatibility

        Args:
            context: Current execution context
            peer_info: Information about the peer

        Returns:
            True if the connection should be accepted
        """
        # Check manifest_id first if available
        if context.my_manifest_id is not None and peer_info.manifest_id is not None:
            return peer_info.manifest_id == context.my_manifest_id
        
        # Check domain compatibility
        if peer_info.domain != context.my_domain:
            return False
        
        # Check data schema hash compatibility
        if peer_info.data_schema_hash != context.my_data_schema_hash:
            return False
        
        # Check model version compatibility
        if not self._is_version_compatible(context.my_model_version, peer_info.model_version):
            return False
        
        return True
    
    def _is_version_compatible(self, my_version: str, peer_version: str) -> bool:
        """Check if two model versions are compatible (same major version)."""
        try:
            my_major = my_version.split('.')[0]
            peer_major = peer_version.split('.')[0]
            return my_major == peer_major
        except (IndexError, AttributeError):
            return True  # Fallback: assume compatible if version format is unexpected

    async def periodic_maintenance(self, context: SelectionContext) -> None:
        """
        Perform periodic maintenance tasks (e.g., shuffling peer list).

        Args:
            context: Current execution context
        """

    async def start(self, context: SelectionContext) -> None:
        """
        Start the topology strategy (e.g., begin periodic maintenance tasks).

        Args:
            context: Current execution context
        """

    async def stop(self) -> None:
        """
        Stop the topology strategy (e.g., cancel periodic maintenance tasks).
        """

    def get_active_view(self) -> List[PeerInfo]:
        """
        Get the current list of active peers (Partial View).

        Default implementation returns empty list. Subclasses should
        override this if they maintain a partial view.

        Returns:
            List of PeerInfo objects representing the current active view.
        """
        return []

    async def on_peer_disconnected(self, peer_id: str) -> None:
        """
        Called when a peer disconnects.

        Args:
            peer_id: ID of the disconnected peer
        """

    async def on_new_peer_discovered(self, peer_info: PeerInfo) -> None:
        """
        Called when a new peer is discovered.

        Args:
            peer_info: Information about the newly discovered peer
        """
