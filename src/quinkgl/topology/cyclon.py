"""
Cyclon Topology Strategy.

Implements a Random Peer Sampling strategy based on the Cyclon algorithm
(Voulgaris et al., 2005).

Algorithm Overview:
- Each node maintains a partial view of peers
- Peers in view have an age that increments each round
- Periodically, a node selects the oldest peer and exchanges views with it
- This creates a continuously mixing random graph

Reference:
    "Cyclon: Inexpensive Membership Management for Unstructured P2P Overlays"
    Spyros Voulgaris, Mark Jelasity, Maarten van Steen (2005)
"""
import asyncio
import logging
from typing import List, Optional, Any
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
from quinkgl.topology.sampler import PeerSampler

logger = logging.getLogger(__name__)

class CyclonTopology(TopologyStrategy):
    """
    Cyclon topology strategy for scalable peer sampling.

    This implementation follows the Cyclon algorithm:
    1. Maintain a partial view with age tracking
    2. Periodically increase age of all peers
    3. Select oldest peer Q for shuffle exchange
    4. Send subset of view to Q, receive Q's subset
    5. Merge using Cyclon's age-based priority

    Attributes:
        sampler: PeerSampler managing the partial view.
        shuffle_length: Number of peers to exchange during shuffle.
        shuffle_interval: Seconds between shuffle operations.
        _shuffle_task: Background task for periodic shuffling.
        _running: Whether the topology is active.
    """

    def __init__(
        self,
        view_size: int = 20,
        shuffle_length: int = 8,
        shuffle_interval: float = 10.0,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize Cyclon topology.

        Args:
            view_size: Maximum size of the partial view (default: 20).
            shuffle_length: Number of peers to exchange during shuffle (default: 8).
            shuffle_interval: Seconds between shuffle operations (default: 10.0).
            seed: Random seed for reproducibility.
            **kwargs: Additional base class arguments.
        """
        super().__init__(**kwargs)
        self.sampler = PeerSampler(view_size=view_size, seed=seed)
        self.shuffle_length = shuffle_length
        self.shuffle_interval = shuffle_interval
        self._shuffle_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._shuffle_peer_callback: Optional[callable] = None

        # For network communication during shuffle
        # Should be set by the caller before starting
        self._send_shuffle: Optional[callable] = None
        self._my_peer_info: Optional[PeerInfo] = None

    def set_shuffle_callback(self, callback: callable) -> None:
        """
        Set the callback for initiating shuffle with a remote peer.

        The callback should have signature:
            async def send_shuffle(peer_id: str, peers: List[PeerInfo]) -> List[PeerInfo]

        Args:
            callback: Async callable to send shuffle request to peer.
        """
        self._send_shuffle = callback

    def set_local_peer_info(self, peer_info: PeerInfo) -> None:
        """
        Set the local peer's info for inclusion in shuffles.

        Args:
            peer_info: This peer's PeerInfo.
        """
        self._my_peer_info = peer_info

    async def select_targets(self, context: SelectionContext, count: int = 3) -> List[str]:
        """
        Select random targets from the current partial view.

        Args:
            context: Execution context.
            count: Number of targets to select.

        Returns:
            List of peer IDs.
        """
        # If view is empty, try to bootstrap from known_peers
        if self.sampler.get_view_size() == 0 and context.known_peers:
            compatible = [
                p for p in context.known_peers
                if p.domain == context.my_domain
                and p.data_schema_hash == context.my_data_schema_hash
                and p.peer_id != context.my_peer_id
            ]
            # Add compatible peers to view
            for peer in compatible[:self.sampler.view_size]:
                await self.sampler.add_peer(peer)

        selected = await self.sampler.select_random_peers(count)
        return [p.peer_id for p in selected]

    async def periodic_maintenance(self, context: SelectionContext) -> None:
        """
        Perform Cyclon shuffle maintenance.

        This method:
        1. Increments age of all peers
        2. Selects oldest peer for shuffle
        3. Initiates shuffle exchange if callback is available

        Args:
            context: Current execution context.
        """
        # Increment age of all peers in view
        await self.sampler.increment_all_ages()

        # Get oldest peer for shuffle
        oldest_peers = await self.sampler.get_oldest_peers(count=1)

        if not oldest_peers:
            # View is empty, try to refill from context
            if context.known_peers:
                compatible = [
                    p for p in context.known_peers
                    if p.domain == context.my_domain
                    and p.data_schema_hash == context.my_data_schema_hash
                    and p.peer_id != context.my_peer_id
                ]
                for peer in compatible[:self.sampler.view_size]:
                    await self.on_new_peer_discovered(peer)
            return

        # Perform shuffle with oldest peer
        shuffle_peer = oldest_peers[0]
        await self._perform_shuffle(shuffle_peer.peer_id)

        logger.debug(
            f"Cyclon maintenance: view_size={self.sampler.get_view_size()}, "
            f"shuffled_with={shuffle_peer.peer_id}"
        )

    async def _perform_shuffle(self, peer_id: str) -> None:
        """
        Perform shuffle exchange with a remote peer following Voulgaris 2005 §3.2.

        Algorithm (initiator side):
        1. Select the oldest peer Q from the view (done by caller).
        2. Select shuffle_length random peers from the view (excluding Q).
        3. Include own descriptor in the subset.
        4. Send the subset to Q; Q responds with its own random subset.
        5. Discard entries sent to Q from the local view (except Q itself).
        6. Merge received entries into the view, evicting oldest if full.

        Args:
            peer_id: ID of peer Q to shuffle with.
        """
        if self._send_shuffle is None:
            logger.debug("No shuffle callback set, skipping shuffle")
            return

        # Step 2: Select shuffle_length random peers from view, excluding Q
        current_view = self.sampler.get_view()
        candidates = [p for p in current_view if p.peer_id != peer_id]

        # Random selection (not youngest/oldest) — per §3.2
        import random
        selected = random.sample(candidates, min(self.shuffle_length, len(candidates)))

        # Step 3: Include own descriptor in the subset
        if self._my_peer_info is not None:
            # Add a fresh copy of self with age 0
            self_copy = PeerInfo(
                peer_id=self._my_peer_info.peer_id,
                domain=self._my_peer_info.domain,
                data_schema_hash=self._my_peer_info.data_schema_hash,
                model_version=self._my_peer_info.model_version,
                age=0,
            )
            selected.append(self_copy)

        # Step 5 (pre): Remember which peers we sent so we can discard them
        my_id = self._my_peer_info.peer_id if self._my_peer_info else None
        sent_peer_ids = {p.peer_id for p in selected if p.peer_id != my_id}

        try:
            # Step 4: Send shuffle request and receive remote peer's subset
            remote_peers = await self._send_shuffle(peer_id, selected)

            # Step 5: Discard entries that were sent to Q (except Q itself)
            for pid in sent_peer_ids:
                if pid in self.sampler and pid != peer_id:
                    await self.sampler.remove_peer(pid)

            # Step 6: Merge received entries into local view
            if remote_peers:
                await self.sampler.merge_view(remote_peers, self.shuffle_length)

            # Reset age of Q after successful exchange
            if peer_id in self.sampler:
                peer_info = self.sampler.view.get(peer_id)
                if peer_info:
                    peer_info.reset_age()

            logger.debug(
                f"Shuffle completed with {peer_id}: "
                f"sent={len(selected)}, received={len(remote_peers) if remote_peers else 0}"
            )
        except Exception as e:
            logger.warning(f"Shuffle with {peer_id} failed: {e}")

    async def should_accept_connection(self, context: SelectionContext, peer_info: PeerInfo) -> bool:
        """
        Accept connection and add to sampler if compatible.

        Args:
            context: Current execution context.
            peer_info: Peer requesting connection.

        Returns:
            True if peer is compatible (manifest ID when available, else domain + schema).
        """
        # Manifest-ID takes priority when both sides have one
        if context.my_manifest_id is not None and peer_info.manifest_id is not None:
            return peer_info.manifest_id == context.my_manifest_id

        if peer_info.domain != context.my_domain:
            return False

        if peer_info.data_schema_hash != context.my_data_schema_hash:
            return False

        return True

    async def on_new_peer_discovered(self, peer_info: PeerInfo) -> None:
        """
        New peer discovered. Add to view with age 0.

        Args:
            peer_info: Information about the newly discovered peer.
        """
        peer_info.reset_age()  # Ensure new peers start with age 0
        await self.sampler.add_peer(peer_info)

    async def on_peer_disconnected(self, peer_id: str) -> None:
        """
        Peer disconnected. Remove from view.

        Args:
            peer_id: ID of the disconnected peer.
        """
        await self.sampler.remove_peer(peer_id)

    def get_active_view(self) -> List[PeerInfo]:
        """
        Return current partial view.

        Returns:
            List of PeerInfo objects in the current view.
        """
        return self.sampler.get_view()

    async def handle_incoming_shuffle(
        self,
        from_peer_id: str,
        remote_peers: List[PeerInfo]
    ) -> List[PeerInfo]:
        """
        Handle incoming shuffle request from a peer (receiver side per §3.2).

        Algorithm (receiver side):
        1. Receive subset S from initiator.
        2. Select shuffle_length random peers from local view (excluding initiator).
        3. Send the selected subset back to the initiator.
        4. Discard entries sent to the initiator from the local view.
        5. Merge received entries S into the view, evicting oldest if full.

        Args:
            from_peer_id: ID of the peer sending the shuffle request.
            remote_peers: List of peers from the remote peer's view.

        Returns:
            List of local peers to send back.
        """
        import random

        # Step 2: Select random peers from local view (excluding initiator)
        current_view = self.sampler.get_view()
        candidates = [p for p in current_view if p.peer_id != from_peer_id]
        response_peers = random.sample(candidates, min(self.shuffle_length, len(candidates)))

        # Remember which peers we're sending back so we can discard them
        sent_peer_ids = {p.peer_id for p in response_peers}

        # Step 4: Discard entries sent to initiator from local view
        for pid in sent_peer_ids:
            if pid in self.sampler:
                await self.sampler.remove_peer(pid)

        # Step 5: Merge received entries into local view
        await self.sampler.merge_view(remote_peers, self.shuffle_length)

        # Reset age of initiator after successful exchange
        if from_peer_id in self.sampler:
            peer_info = self.sampler.view.get(from_peer_id)
            if peer_info:
                peer_info.reset_age()

        return response_peers

    async def start(self, context: SelectionContext) -> None:
        """
        Start the periodic shuffle task.

        Args:
            context: Current execution context.
        """
        if self._running:
            return

        self._running = True

        async def shuffle_loop():
            while self._running:
                try:
                    await asyncio.sleep(self.shuffle_interval)
                    if self._running:
                        await self.periodic_maintenance(context)
                except asyncio.CancelledError:
                    logger.debug("Shuffle loop cancelled")
                    raise  # Re-raise to allow proper cancellation handling
                except Exception as e:
                    logger.error(f"Error in shuffle loop: {e}")

        self._shuffle_task = asyncio.create_task(shuffle_loop())
        logger.info("Cyclon shuffle task started")

    async def stop(self) -> None:
        """Stop the periodic shuffle task."""
        self._running = False

        if self._shuffle_task and not self._shuffle_task.done():
            self._shuffle_task.cancel()
            try:
                await self._shuffle_task
            except asyncio.CancelledError:
                pass

        logger.info("Cyclon shuffle task stopped")

    # TOP-09: Removed unsafe __del__ method - use explicit stop() instead
    # __del__ is unsafe for asyncio task cancellation
