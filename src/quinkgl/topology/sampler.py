"""
Peer Sampler for Partial View Management.

Manages a fixed-size list of peers (Partial View) for Random Peer Sampling
protocols like Cyclon.

Features:
- Age-based eviction (oldest peer removed when view is full)
- Thread-safe operations using asyncio locks
- Proper Cyclon-style merge logic
"""
import asyncio
import logging
from typing import List, Optional, Dict
from quinkgl.topology.base import PeerInfo

logger = logging.getLogger(__name__)

class PeerSampler:
    """
    Manages a partial view of the network with age-based eviction.

    Attributes:
        view_size: Maximum number of peers to keep in view.
        view: Current active view {peer_id: PeerInfo}.
        _lock: Async lock for thread-safe operations.
    """

    def __init__(self, view_size: int = 20, seed: Optional[int] = None) -> None:
        """
        Initialize PeerSampler.

        Args:
            view_size: Maximum size of the partial view.
            seed: Random seed for reproducibility (used for shuffle tie-breaks).
        """
        self.view_size: int = view_size
        self.view: Dict[str, PeerInfo] = {} 
        self._lock: asyncio.Lock = asyncio.Lock()
        self._seed: Optional[int] = seed

    async def add_peer(self, peer: PeerInfo) -> bool:
        """
        Add a peer to the view. If view is full, evict the oldest peer.

        Args:
            peer: PeerInfo object to add.

        Returns:
            True if peer was added, False if evicted or already present.
        """
        async with self._lock:
            return self._add_peer_unsafe(peer)

    def _add_peer_unsafe(self, peer: PeerInfo) -> bool:
        """
        Internal add_peer without lock (must be called within lock).

        Args:
            peer: PeerInfo object to add.

        Returns:
            True if peer was added, False if evicted or already present.
        """
        if peer.peer_id in self.view:
            # Update existing peer info and reset age (peer refreshed)
            self.view[peer.peer_id] = peer
            self.view[peer.peer_id].reset_age()
            return True

        if len(self.view) >= self.view_size:
            self._evict_oldest_peer()

        self.view[peer.peer_id] = peer
        return True

    async def remove_peer(self, peer_id: str) -> bool:
        """
        Remove a peer from the view.

        Args:
            peer_id: ID of peer to remove.

        Returns:
            True if peer was removed, False if not found.
        """
        async with self._lock:
            if peer_id in self.view:
                del self.view[peer_id]
                return True
            return False

    async def select_random_peers(
        self,
        count: int,
        exclude: Optional[List[str]] = None
    ) -> List[PeerInfo]:
        """
        Select random peers from the view.

        Args:
            count: Number of peers to select.
            exclude: List of peer IDs to exclude.

        Returns:
            List of selected PeerInfo objects.
        """
        async with self._lock:
            candidates = list(self.view.values())

            if exclude:
                candidates = [p for p in candidates if p.peer_id not in exclude]

            if not candidates:
                return []
                
            sample_size = min(count, len(candidates))
            return self._rng.sample(candidates, sample_size)

    def get_view(self) -> List[PeerInfo]:
        """
        Return the current view as a list (snapshot).

        Note: Returns a copy to prevent external modification.

        Returns:
            List of PeerInfo objects.
        """
        return list(self.view.values())

    def get_view_size(self) -> int:
        """Return current view size."""
        return len(self.view)

    def has_peer(self, peer_id: str) -> bool:
        """Check if a peer is in the view."""
        return peer_id in self.view

    async def merge_view(
        self,
        new_peers: List[PeerInfo],
        shuffle_length: Optional[int] = None
    ) -> None:
        """
        Merge a list of new peers into the current view using Cyclon-style logic.

        Cyclon merge algorithm:
        1. Receive peer list L from a remote peer
        2. Remove items from L that are already in local view
        3. Remove oldest items from local view to make space
        4. Add remaining items from L to local view
        5. Reset age of newly added peers

        Args:
            new_peers: List of peers received from shuffle.
            shuffle_length: Number of peers to accept from shuffle (default: all).
        """
        async with self._lock:
            self._merge_view_unsafe(new_peers, shuffle_length)

    def _merge_view_unsafe(
        self,
        new_peers: List[PeerInfo],
        shuffle_length: Optional[int]
    ) -> None:
        """
        Internal merge_view without lock.

        Args:
            new_peers: List of peers received from shuffle.
            shuffle_length: Number of peers to accept from shuffle.
        """
        if not new_peers:
            return

        # Determine how many peers to accept
        if shuffle_length is None:
            shuffle_length = len(new_peers)
        else:
            shuffle_length = min(shuffle_length, len(new_peers))

        # Filter out peers already in our view
        peers_to_add = [p for p in new_peers if p.peer_id not in self.view]

        # If we have more candidates than we can accept, keep youngest
        if len(peers_to_add) > shuffle_length:
            # Sort by age (ascending) and take youngest
            peers_to_add.sort(key=lambda p: p.age)
            peers_to_add = peers_to_add[:shuffle_length]

        # Make space by evicting oldest peers first
        spaces_needed = len(peers_to_add)
        for _ in range(spaces_needed):
            if len(self.view) >= self.view_size:
                self._evict_oldest_peer()

        # Add new peers and reset their age
        for peer in peers_to_add:
            peer.reset_age()
            self.view[peer.peer_id] = peer

    def _evict_oldest_peer(self) -> Optional[str]:
        """
        Evict the peer with the highest age (oldest) from the view.

        This is the proper Cyclon eviction strategy (not random).

        Returns:
            ID of evicted peer, or None if view is empty.
        """
        if not self.view:
            return None

        # Find peer with maximum age
        oldest_peer_id = max(self.view.items(), key=lambda x: x[1].age)[0]

        del self.view[oldest_peer_id]
        return oldest_peer_id

    async def increment_all_ages(self) -> None:
        """
        Increment age of all peers in the view.

        Should be called periodically as part of Cyclon maintenance.
        """
        async with self._lock:
            for peer in self.view.values():
                peer.increment_age()

    async def get_oldest_peers(self, count: int) -> List[PeerInfo]:
        """
        Get the oldest peers from the view (for shuffle initiation).

        Args:
            count: Number of oldest peers to return.

        Returns:
            List of oldest PeerInfo objects, sorted by age descending.
        """
        async with self._lock:
            peers_by_age = sorted(
                self.view.values(),
                key=lambda p: p.age,
                reverse=True
            )
            return peers_by_age[:count]

    async def clear(self) -> None:
        """Clear all peers from the view."""
        async with self._lock:
            self.view.clear()

    def __len__(self) -> int:
        """Return current view size."""
        return len(self.view)

    def __contains__(self, peer_id: str) -> bool:
        """Check if peer is in view."""
        return peer_id in self.view
