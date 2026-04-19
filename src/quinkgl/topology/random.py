"""
Random Topology Strategy

Simplest topology strategy: randomly select k peers from known peers.
"""
import random
from typing import List, Optional, Set, Dict, Any
from datetime import datetime
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo

class RandomTopology(TopologyStrategy):
    """
    Random topology strategy.

    Selects k random peers from the list of known compatible peers.
    Compatible peers are those with matching domain and data schema.

    Attributes:
        rng: Random number generator with optional seed for reproducibility.
        _cache_duration: Seconds to cache compatible peers (0 = no caching).
        _last_cache_time: When the cache was last updated.
        _cached_peer_ids: Cached set of compatible peer IDs.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        cache_duration: float = 5.0,
        **kwargs: Any
    ) -> None:
        """
        Initialize random topology strategy.

        Args:
            seed: Random seed for reproducibility (None = random)
            cache_duration: Seconds to cache compatible peers (0 = no caching)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.rng = random.Random(seed)
        self._cache_duration: float = cache_duration
        self._last_cache_time: float = 0
        self._cached_peer_ids: Set[str] = set()
        self._domain_schema_key: str = "" 

    def _is_cache_valid(self, current_time: float, domain_schema_key: str) -> bool:
        """Check if the cache is still valid."""
        return (
            self._cache_duration > 0
            and domain_schema_key == self._domain_schema_key
            and (current_time - self._last_cache_time) < self._cache_duration
        )

    def _update_cache(self, peer_ids: Set[str], domain_schema_key: str) -> None:
        """Update the cache with new peer IDs."""
        self._cached_peer_ids = peer_ids.copy()
        self._domain_schema_key = domain_schema_key
        self._last_cache_time = datetime.now().timestamp()

    def _get_compatible_peer_ids(self, context: SelectionContext) -> Set[str]:
        """
        Get set of compatible peer IDs with optional caching.

        Args:
            context: Current execution context

        Returns:
            Set of compatible peer IDs
        """
        domain_schema_key = f"{context.my_domain}:{context.my_data_schema_hash}"
        current_time = datetime.now().timestamp()

        if self._is_cache_valid(current_time, domain_schema_key):
            return self._cached_peer_ids.copy()

        compatible = context.get_compatible_peers(exclude_self=True)
        peer_ids = {p.peer_id for p in compatible}

        self._update_cache(peer_ids, domain_schema_key)

        return peer_ids

    async def select_targets(
        self,
        context: SelectionContext,
        count: int = 3
    ) -> List[str]:
        """
        Select random compatible peers as targets.

        Args:
            context: Current execution context
            count: Maximum number of targets to select

        Returns:
            List of peer IDs to send updates to
        """
        compatible_peer_ids = self._get_compatible_peer_ids(context)

        if not compatible_peer_ids:
            return []

        # Select up to count random peers
        selected_count = min(count, len(compatible_peer_ids))
        selected = self.rng.sample(list(compatible_peer_ids), selected_count)

        return selected

    async def should_accept_connection(
        self,
        context: SelectionContext,
        peer_info: PeerInfo
    ) -> bool:
        """
        Accept connection if peer is compatible.

        Args:
            context: Current execution context
            peer_info: Information about the peer

        Returns:
            True if peer has compatible domain, schema, and manifest_id
        """
        if context.my_manifest_id is not None and peer_info.manifest_id is not None:
            return peer_info.manifest_id == context.my_manifest_id
        return (
            peer_info.domain == context.my_domain
            and peer_info.data_schema_hash == context.my_data_schema_hash
        )
