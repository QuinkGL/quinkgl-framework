"""
Gossip Learning Community for IPv8

Implements P2P model exchange and aggregation over IPv8.
Domain isolation ensures only compatible peers communicate.

CHUNKED TRANSFER: Large model updates are split into chunks
to work around UDP MTU limits (~1400 bytes).
"""

import asyncio
import time
import logging
import hashlib
import uuid
from typing import Optional, Callable, List, Any, Dict
from dataclasses import dataclass, field

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload import Payload
from ipv8.peer import Peer

from quinkgl.network.model_serializer import serialize_model, deserialize_model

logger = logging.getLogger(__name__)

# Maximum size for incoming model updates (150 MB) to prevent DoS
MAX_INCOMING_MESSAGE_SIZE = 150 * 1024 * 1024

# Chunk size for large model transfers
# UDP Safe Payload size is ~1400 bytes (MTU 1500 - Headers).
# Using 1024 bytes to be safe and avoid IP fragmentation which causes high packet loss.
CHUNK_SIZE = 1024  # 1KB chunks - safe for MTU

# B7: Inter-chunk send delay (seconds) — prevents UDP buffer overflow on receiver.
# At 1 KB/chunk this yields ~83 KB/s sustained throughput.
CHUNK_SEND_INTERVAL = 0.012

# Timeout for incomplete transfers (300 seconds - increased for Colab/slow networks)
CHUNK_TRANSFER_TIMEOUT = 300

# B4: Chunk-buffer memory caps to prevent DoS
MAX_CONCURRENT_TRANSFERS_PER_PEER = 3
MAX_TOTAL_TRANSFERS = 50
MAX_BUFFERED_BYTES_PER_PEER = 200 * 1024 * 1024  # 200 MB
MAX_CHUNKS_PER_TRANSFER = 300_000  # ~300 MB at 1 KB/chunk

# B5: NACK rate-limiting
NACK_MAX_RESENDS_PER_TRANSFER = 3
NACK_BUCKET_INTERVAL = 10.0  # seconds — refill one token per interval
NACK_BUCKET_MAX_TOKENS = 5   # max burst per peer

# B6: Early NACK gap detection
EARLY_NACK_AGE_THRESHOLD = 5.0  # seconds since buffer creation before probing


def generate_community_id(domain: str, data_schema_hash: str) -> bytes:
    """
    Generate a unique community ID for a domain + schema combination.

    This ensures domain isolation - only peers with matching
    domain and schema can communicate.

    Args:
        domain: Domain identifier (e.g., "health", "agriculture")
        data_schema_hash: Hash of data schema

    Returns:
        20-byte community ID for IPv8
    """
    # Combine domain and schema
    combined = f"QuinkGL-{domain}-{data_schema_hash}".encode('utf-8')

    # Hash to get 20 bytes (SHA-1 produces 20 bytes)
    hashed = hashlib.sha1(combined).digest()

    return hashed


class DiscoveryAnnouncePayload(Payload):
    """
    Payload for peer discovery announcements.

    Peers announce their domain and schema to find compatible peers.
    Optionally includes a fingerprint JSON blob for affinity computation.
    """
    msg_id = 1
    format_list = ['varlenH', 'varlenH', 'varlenH', 'varlenH', 'varlenH']

    def __init__(self, node_id: str, domain: str, data_schema_hash: str,
                 model_version: str, fingerprint_json: str = ""):
        super().__init__()
        self.node_id = node_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model_version = model_version
        self.fingerprint_json = fingerprint_json

    def to_pack_list(self):
        return [
            ('varlenH', self.node_id.encode('utf-8')),
            ('varlenH', self.domain.encode('utf-8')),
            ('varlenH', self.data_schema_hash.encode('utf-8')),
            ('varlenH', self.model_version.encode('utf-8')),
            ('varlenH', self.fingerprint_json.encode('utf-8')),
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        fp_json = ""
        if len(args) > 4:
            fp_json = args[4].decode('utf-8') if args[4] else ""
        return cls(
            args[0].decode('utf-8'),
            args[1].decode('utf-8'),
            args[2].decode('utf-8'),
            args[3].decode('utf-8'),
            fp_json,
        )


class ModelUpdatePayload(Payload):
    """
    Payload for model weight updates.

    Contains serialized model weights and metadata.

    NOTE: Uses 'varlenI' for weights_bytes (large model), 'varlenH' for others
    """
    msg_id = 2
    format_list = ['varlenH', 'varlenI', 'I', 'I', 'varlenH', 'd', 'd', 'I']

    def __init__(
        self,
        sender_id: str,
        weights_bytes: bytes,
        sample_count: int,
        round_number: int,
        data_schema_hash: str,
        loss: float = 0.0,
        accuracy: float = 0.0,
        timestamp: int = 0
    ):
        super().__init__()
        self.sender_id = sender_id
        self.weights_bytes = weights_bytes
        self.sample_count = sample_count
        self.round_number = round_number
        self.data_schema_hash = data_schema_hash
        self.loss = loss
        self.accuracy = accuracy
        self.timestamp = timestamp or int(time.time())

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenI', self.weights_bytes),  # varlenI for large model weights
            ('I', self.sample_count),
            ('I', self.round_number),
            ('varlenH', self.data_schema_hash.encode('utf-8')),
            ('d', self.loss),
            ('d', self.accuracy),
            ('I', self.timestamp)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),
            args[1],
            args[2],
            args[3],
            args[4].decode('utf-8'),
            args[5],
            args[6],
            args[7]
        )


class HeartbeatPayload(Payload):
    """Payload for heartbeat messages."""
    msg_id = 3
    format_list = ['varlenH', 'I']

    def __init__(self, node_id: str, sequence: int):
        super().__init__()
        self.node_id = node_id
        self.sequence = sequence

    def to_pack_list(self):
        return [
            ('varlenH', self.node_id.encode('utf-8')),
            ('I', self.sequence)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(args[0].decode('utf-8'), args[1])


class PrototypeExchangePayload(Payload):
    msg_id = 8
    format_list = ['varlenH', 'varlenH']

    def __init__(self, sender_id: str, prototypes_json: str):
        super().__init__()
        self.sender_id = sender_id
        self.prototypes_json = prototypes_json

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenH', self.prototypes_json.encode('utf-8')),
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(args[0].decode('utf-8'), args[1].decode('utf-8'))


class CheckpointPayload(Payload):
    """
    Payload for checkpoint announcements (B2).

    Mirrors ``CheckpointAnnounceMessage`` fields for wire transmission.
    Consensus is IPv8-only; tunnel fallback does not broadcast checkpoints
    (documented trade-off: tunnel mode is a best-effort relay and checkpoint
    consensus assumes a fully-connected overlay which tunnel does not provide).
    """
    msg_id = 9
    format_list = ['varlenH', 'I', 'd', 'd', 'varlenH']

    def __init__(self, sender_id: str, round_number: int, loss: float,
                 accuracy: float, model_version: str = "1.0.0"):
        super().__init__()
        self.sender_id = sender_id
        self.round_number = round_number
        self.loss = loss
        self.accuracy = accuracy
        self.model_version = model_version

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('I', self.round_number),
            ('d', self.loss),
            ('d', self.accuracy),
            ('varlenH', self.model_version.encode('utf-8')),
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),  # sender_id
            args[1],                   # round_number
            args[2],                   # loss
            args[3],                   # accuracy
            args[4].decode('utf-8'),  # model_version
        )


class ModelChunkPayload(Payload):
    """
    Payload for chunked model transfer.
    
    Large models are split into CHUNK_SIZE chunks and sent individually.
    The receiver buffers chunks and reassembles when all are received.
    
    Fields:
        transfer_id: Unique ID for this transfer (to distinguish multiple transfers)
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks in this transfer
        sender_id: Node ID of the sender  
        data_schema_hash: Schema hash for validation
        round_number: Training round number
        sample_count: Number of training samples
        loss: Training loss
        accuracy: Training accuracy
        chunk_data: The actual chunk bytes
    """
    msg_id = 4
    # varlenH for strings, I for ints, d for floats, varlenH for chunk data
    format_list = ['varlenH', 'I', 'I', 'varlenH', 'varlenH', 'I', 'I', 'd', 'd', 'varlenH']

    def __init__(
        self,
        transfer_id: str,
        chunk_index: int,
        total_chunks: int,
        sender_id: str,
        data_schema_hash: str,
        round_number: int,
        sample_count: int,
        loss: float,
        accuracy: float,
        chunk_data: bytes
    ):
        super().__init__()
        self.transfer_id = transfer_id
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.sender_id = sender_id
        self.data_schema_hash = data_schema_hash
        self.round_number = round_number
        self.sample_count = sample_count
        self.loss = loss
        self.accuracy = accuracy
        self.chunk_data = chunk_data

    def to_pack_list(self):
        return [
            ('varlenH', self.transfer_id.encode('utf-8')),
            ('I', self.chunk_index),
            ('I', self.total_chunks),
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenH', self.data_schema_hash.encode('utf-8')),
            ('I', self.round_number),
            ('I', self.sample_count),
            ('d', self.loss),
            ('d', self.accuracy),
            ('varlenH', self.chunk_data)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),  # transfer_id
            args[1],                   # chunk_index
            args[2],                   # total_chunks
            args[3].decode('utf-8'),  # sender_id
            args[4].decode('utf-8'),  # data_schema_hash
            args[5],                   # round_number
            args[6],                   # sample_count
            args[7],                   # loss
            args[8],                   # accuracy
            args[9]                    # chunk_data (bytes)
        )


class RequestChunksPayload(Payload):
    """
    Payload to request missing chunks (NACK).
    
    Sent by receiver when gaps are detected in a chunked transfer.
    """
    msg_id = 5
    format_list = ['varlenH', 'varlenH', 'varlenI']

    def __init__(self, transfer_id: str, sender_id: str, missing_indices_bytes: bytes):
        super().__init__()
        self.transfer_id = transfer_id
        self.sender_id = sender_id
        self.missing_indices_bytes = missing_indices_bytes

    def to_pack_list(self):
        return [
            ('varlenH', self.transfer_id.encode('utf-8')),
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenI', self.missing_indices_bytes)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),
            args[1].decode('utf-8'),
            args[2]
        )


class ShufflePayload(Payload):
    """
    Payload for Cyclon shuffle request.

    Contains a list of peer descriptors serialized as msgpack bytes
    for efficient transmission.
    """
    msg_id = 6
    format_list = ['varlenH', 'varlenI']

    def __init__(self, sender_id: str, peers_bytes: bytes):
        super().__init__()
        self.sender_id = sender_id
        self.peers_bytes = peers_bytes

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenI', self.peers_bytes)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),
            args[1]
        )


class ShuffleResponsePayload(Payload):
    """
    Payload for Cyclon shuffle response.

    Contains the responding peer's subset of view as msgpack bytes.
    """
    msg_id = 7
    format_list = ['varlenH', 'varlenI']

    def __init__(self, sender_id: str, peers_bytes: bytes):
        super().__init__()
        self.sender_id = sender_id
        self.peers_bytes = peers_bytes

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenI', self.peers_bytes)
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        return cls(
            args[0].decode('utf-8'),
            args[1]
        )


@dataclass
class ChunkBuffer:
    """
    Buffer for reassembling chunked model transfers.
    
    Stores received chunks until all are received, then reassembles.
    """
    transfer_id: str
    sender_id: str
    total_chunks: int
    data_schema_hash: str
    round_number: int
    sample_count: int
    loss: float
    accuracy: float
    created_at: float = field(default_factory=time.time)
    chunks: Dict[int, bytes] = field(default_factory=dict)
    
    def add_chunk(self, chunk_index: int, chunk_data: bytes) -> bool:
        """
        Add a chunk to the buffer.
        
        Returns True if all chunks have been received.
        """
        self.chunks[chunk_index] = chunk_data
        return len(self.chunks) == self.total_chunks
    
    def is_complete(self) -> bool:
        """Check if all chunks have been received."""
        return len(self.chunks) == self.total_chunks
    
    def is_expired(self) -> bool:
        """Check if this transfer has timed out."""
        return time.time() - self.created_at > CHUNK_TRANSFER_TIMEOUT
    
    def reassemble(self) -> bytes:
        """
        Reassemble all chunks into the original data.
        
        Returns the complete serialized model weights.
        """
        if not self.is_complete():
            raise ValueError(f"Cannot reassemble: only {len(self.chunks)}/{self.total_chunks} chunks received")
        
        # Sort chunks by index and concatenate
        sorted_chunks = [self.chunks[i] for i in range(self.total_chunks)]
        return b''.join(sorted_chunks)


class PeerInfo:
    """Information about a discovered peer."""

    def __init__(
        self,
        peer: Peer,
        node_id: str,
        domain: str,
        data_schema_hash: str,
        model_version: str = "1.0.0"
    ):
        self.peer = peer
        self.node_id = node_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model_version = model_version
        self.last_seen = time.time()

    def is_compatible(self, domain: str, data_schema_hash: str) -> bool:
        """Check if peer is compatible (same domain and schema)."""
        return (
            self.domain == domain and
            self.data_schema_hash == data_schema_hash
        )

    def age(self) -> float:
        """Get seconds since last seen."""
        return time.time() - self.last_seen

    def update_seen(self):
        """Update last seen time."""
        self.last_seen = time.time()


class GossipLearningCommunity(Community):
    """
    Gossip Learning Community for P2P model exchange.

    Features:
    - Domain-based isolation (only compatible peers communicate)
    - Model weight exchange with chunked transfer for large models
    - Peer discovery via announce messages
    - Heartbeat for connection tracking
    """

    def __init__(self, *args, node_id: str = "unknown", domain: str = "default",
                 data_schema_hash: str = "", model_version: str = "1.0.0", **kwargs):
        """
        Initialize Gossip Learning Community.

        Args:
            *args: Passed to parent Community (my_peer, my_peer_key, integration_mask)
            node_id: Unique identifier for this node
            domain: Domain identifier (e.g., "health", "agriculture")
            data_schema_hash: Hash of data schema for compatibility
            model_version: Model architecture version
            **kwargs: Additional parameters
        """
        # Store parameters
        self.node_id = node_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model_version = model_version

        # Generate community ID from domain + schema
        self._instance_community_id = generate_community_id(domain, data_schema_hash)

        # Set community_id as class variable for IPv8
        # Note: This affects all instances of this class
        type(self).community_id = self._instance_community_id

        # Initialize parent with all args
        super().__init__(*args, **kwargs)

        # Peer tracking
        self.known_peers: dict[str, PeerInfo] = {}  # node_id -> PeerInfo

        # Heartbeat
        self._heartbeat_sequence = 0

        # Chunk buffer for reassembling large model transfers
        # B3: keyed by (peer.mid, transfer_id) to prevent hijack
        self._chunk_buffers: Dict[tuple, ChunkBuffer] = {}  # (peer_mid, transfer_id) -> ChunkBuffer
        
        # Cache for outgoing transfers to support retry (resending chunks)
        # transfer_id -> { 'weights_bytes': bytes, 'timestamp': float, ... }
        self._outgoing_transfers: Dict[str, Dict] = {}

        # B5: NACK rate-limiting state
        # transfer_id -> resend count
        self._nack_resend_counts: Dict[str, int] = {}
        # peer_mid -> { 'tokens': float, 'last_refill': float }
        self._nack_buckets: Dict[str, Dict] = {}

        # Message handlers
        self.add_message_handler(DiscoveryAnnouncePayload, self.on_discovery_announce)
        self.add_message_handler(ModelUpdatePayload, self.on_model_update)
        self.add_message_handler(HeartbeatPayload, self.on_heartbeat)
        self.add_message_handler(ModelChunkPayload, self.on_model_chunk)
        self.add_message_handler(RequestChunksPayload, self.on_request_chunks)
        self.add_message_handler(ShufflePayload, self.on_shuffle_request)
        self.add_message_handler(ShuffleResponsePayload, self.on_shuffle_response)
        self.add_message_handler(PrototypeExchangePayload, self.on_prototype_exchange)
        self.add_message_handler(CheckpointPayload, self.on_checkpoint)

        # DEBUG: Check if handlers are registered
        logger.debug(f"Registered handlers: {len(self.decode_map)} handlers")

        # Callbacks
        self.on_model_update_callback: Optional[Callable] = None
        self.on_peer_discovered_callback: Optional[Callable] = None
        self.on_peer_left_callback: Optional[Callable] = None
        self.on_shuffle_callback: Optional[Callable] = None
        self.on_prototype_callback: Optional[Callable] = None
        self.on_checkpoint_callback: Optional[Callable] = None

        # Local data fingerprint for affinity-based peer selection
        self.local_fingerprint: Optional[Any] = None

        logger.debug(
            f"GossipLearningCommunity initialized: "
            f"node_id={node_id}, domain={domain}, schema={data_schema_hash[:8] if data_schema_hash else 'unknown'}..."
        )
        logger.debug(f"Community ID: {self._instance_community_id.hex()}")

    def started(self):
        """Called when community is started."""
        logger.debug(f"GossipLearningCommunity STARTED for '{self.node_id}'")
        logger.debug(f"   Domain: {self.domain}")
        logger.debug(f"   Schema: {self.data_schema_hash}")
        logger.debug(f"   My peer: {self.my_peer.address}")

        # Start periodic announcements
        self.register_task(
            "announce_discovery",
            self._announce_discovery,
            interval=15.0,  # Every 15 seconds
            delay=1.0
        )

        self.register_task(
            "send_heartbeat",
            self._send_heartbeat,
            interval=30.0,  # Every 30 seconds
            delay=5.0
        )

        self.register_task(
            "cleanup_stale_peers",
            self._cleanup_stale_peers,
            interval=60.0,  # Every minute
            delay=30.0
        )

        self.register_task(
            "cleanup_stale_transfers",
            self._cleanup_stale_transfers,
            interval=30.0,  # Every 30 seconds
            delay=15.0
        )
        
        self.register_task(
            "cleanup_outgoing_cache",
            self._cleanup_outgoing_cache,
            interval=60.0,
            delay=45.0
        )

        # B6: Early NACK for incomplete buffers with gaps
        self.register_task(
            "nack_incomplete_buffers",
            self._nack_incomplete_buffers,
            interval=5.0,
            delay=10.0
        )

        logger.debug("GossipLearningCommunity tasks registered")

    async def unload(self):
        """Called when community is being unloaded."""
        await super().unload()
        logger.debug(f"GossipLearningCommunity unloaded for '{self.node_id}'")

    async def _announce_discovery(self):
        """Announce our presence to all peers."""
        fingerprint_json = ""
        if self.local_fingerprint is not None:
            try:
                import json as _json
                fingerprint_json = _json.dumps(self.local_fingerprint.to_dict())
            except Exception:
                logger.debug("Could not serialize local fingerprint for announce")

        for peer in self.get_peers():
            self.ez_send(peer, DiscoveryAnnouncePayload(
                self.node_id,
                self.domain,
                self.data_schema_hash,
                self.model_version,
                fingerprint_json,
            ))

    async def _send_heartbeat(self):
        """Send heartbeat to all known peers."""
        self._heartbeat_sequence += 1
        for peer_info in self.known_peers.values():
            self.ez_send(peer_info.peer, HeartbeatPayload(
                self.node_id,
                self._heartbeat_sequence
            ))

    async def _cleanup_stale_peers(self):
        """Remove peers that haven't been seen recently."""
        stale_timeout = 300  # 5 minutes - increased for Colab/slow networks
        stale_peers = []

        for node_id, peer_info in self.known_peers.items():
            if peer_info.age() > stale_timeout:
                stale_peers.append(node_id)

        for node_id in stale_peers:
            peer_info = self.known_peers.pop(node_id)
            logger.debug(f"Removed stale peer: {node_id}")
            if self.on_peer_left_callback:
                await self.on_peer_left_callback(node_id)

    async def _cleanup_stale_transfers(self):
        """Remove incomplete chunk transfers that have timed out."""
        stale_transfers = []
        
        for buf_key, buffer in self._chunk_buffers.items():
            if buffer.is_expired():
                stale_transfers.append(buf_key)
        
        for buf_key in stale_transfers:
            buffer = self._chunk_buffers.pop(buf_key)
            logger.warning(
                f"Removed stale transfer {buffer.transfer_id[:8]}... from {buffer.sender_id}: "
                f"only {len(buffer.chunks)}/{buffer.total_chunks} chunks received"
            )

    async def _cleanup_outgoing_cache(self):
        """Remove old outgoing transfers from cache."""
        current_time = time.time()
        timeout = 600  # Keep cache for 10 minutes
        expired = [tid for tid, data in self._outgoing_transfers.items() if current_time - data['timestamp'] > timeout]
        for tid in expired:
            del self._outgoing_transfers[tid]

    async def _nack_incomplete_buffers(self):
        """B6: Proactively NACK incomplete buffers older than threshold.

        If the last chunk was dropped, the transfer would sit idle until the
        300 s cleanup sweep.  This task detects gaps early and requests the
        missing chunks so transfers recover faster.
        """
        import struct

        now = time.time()
        for (peer_mid, transfer_id), buffer in list(self._chunk_buffers.items()):
            age = now - buffer.created_at
            if age < EARLY_NACK_AGE_THRESHOLD:
                continue
            if buffer.is_complete():
                continue

            # Detect missing chunk indices
            missing = [
                i for i in range(buffer.total_chunks) if i not in buffer.chunks
            ]
            if not missing:
                continue

            # B5 rate-limit: consume a token for this peer
            if not self._nack_try_consume(peer_mid):
                logger.debug(
                    f"Early-NACK rate-limited for peer {peer_mid[:16]}..."
                )
                continue

            # Find the IPv8 Peer object for this peer_mid
            target_peer = None
            for peer_info in self.known_peers.values():
                pmid = peer_info.peer.mid.hex() if isinstance(peer_info.peer.mid, bytes) else str(peer_info.peer.mid)
                if pmid == peer_mid:
                    target_peer = peer_info.peer
                    break

            if target_peer is None:
                continue

            missing_bytes = struct.pack(f'{len(missing)}I', *missing)
            req_payload = RequestChunksPayload(
                transfer_id=transfer_id,
                sender_id=self.node_id,
                missing_indices_bytes=missing_bytes
            )
            logger.debug(
                f"Early-NACK: requesting {len(missing)} missing chunks "
                f"for {transfer_id[:8]}... (age={age:.1f}s)"
            )
            self.ez_send(target_peer, req_payload)

    @lazy_wrapper(DiscoveryAnnouncePayload)
    async def on_discovery_announce(self, peer: Peer, payload: DiscoveryAnnouncePayload):
        if payload.domain != self.domain or payload.data_schema_hash != self.data_schema_hash:
            logger.debug(
                f"Incompatible peer: {payload.node_id} "
                f"(domain={payload.domain}, schema={payload.data_schema_hash[:8]}...)"
            )
            return

        fingerprint = None
        if payload.fingerprint_json:
            try:
                import json
                from quinkgl.fingerprint.fingerprint import DataFingerprint
                fp_dict = json.loads(payload.fingerprint_json)
                fingerprint = DataFingerprint.from_dict(fp_dict)
            except (json.JSONDecodeError, Exception):
                logger.debug(f"Could not parse fingerprint from {payload.node_id}")

        if payload.node_id in self.known_peers:
            self.known_peers[payload.node_id].update_seen()
            if fingerprint is not None:
                self.known_peers[payload.node_id].data_fingerprint = fingerprint
        else:
            peer_info = PeerInfo(
                peer=peer,
                node_id=payload.node_id,
                domain=payload.domain,
                data_schema_hash=payload.data_schema_hash,
                model_version=payload.model_version
            )
            peer_info.data_fingerprint = fingerprint
            self.known_peers[payload.node_id] = peer_info
            logger.debug(f"Peer address discovered: {payload.node_id} @ {peer.address}")
            logger.debug(f"Discovered compatible peer: {payload.node_id}")

            if self.on_peer_discovered_callback:
                await self.on_peer_discovered_callback(peer_info)

    @lazy_wrapper(ModelUpdatePayload)
    async def on_model_update(self, peer: Peer, payload: ModelUpdatePayload):
        """
        Handle incoming model update.

        Deserializes weights and passes to callback.
        """
        logger.debug(f"on_model_update called: sender={payload.sender_id}, round={payload.round_number}")

        # Check message size before deserializing to prevent DoS
        weights_size = len(payload.weights_bytes)
        if weights_size > MAX_INCOMING_MESSAGE_SIZE:
            logger.error(
                f"Rejected oversized model from {payload.sender_id}: "
                f"{weights_size / 1024 / 1024:.2f} MB "
                f"(max {MAX_INCOMING_MESSAGE_SIZE / 1024 / 1024:.0f} MB)"
            )
            return

        logger.debug(
            f"Received model update from {payload.sender_id} "
            f"(round={payload.round_number}, samples={payload.sample_count}, "
            f"size={weights_size / 1024:.1f} KB)"
        )

        # Update peer last seen
        if payload.sender_id in self.known_peers:
            self.known_peers[payload.sender_id].update_seen()

        # Deserialize weights
        try:
            weights = deserialize_model(payload.weights_bytes)
        except ValueError as e:
            # Deserialization validation error (likely from size check in model_serializer)
            logger.error(f"Model validation failed from {payload.sender_id}: {e}")
            return
        except Exception as e:
            logger.error(f"Failed to deserialize model from {payload.sender_id}: {e}")
            return

        # Call callback if registered
        if self.on_model_update_callback:
            await self.on_model_update_callback(
                sender_id=payload.sender_id,
                weights=weights,
                sample_count=payload.sample_count,
                round_number=payload.round_number,
                loss=payload.loss,
                accuracy=payload.accuracy
            )

    @lazy_wrapper(HeartbeatPayload)
    async def on_heartbeat(self, peer: Peer, payload: HeartbeatPayload):
        """Handle heartbeat message."""
        if payload.node_id in self.known_peers:
            self.known_peers[payload.node_id].update_seen()

    @lazy_wrapper(PrototypeExchangePayload)
    async def on_prototype_exchange(self, peer: Peer, payload: PrototypeExchangePayload):
        try:
            from quinkgl.training.prototypes import PrototypeStore
            peer_protos = PrototypeStore.parse_peer_prototypes(
                payload.sender_id, payload.prototypes_json
            )
            if self.on_prototype_callback:
                await self.on_prototype_callback(payload.sender_id, peer_protos)
            else:
                logger.debug(
                    "Received prototype exchange from %s but no callback registered",
                    payload.sender_id,
                )
        except Exception:
            logger.warning(
                "Failed to parse prototype exchange from %s",
                payload.sender_id,
                exc_info=True,
            )

    @lazy_wrapper(CheckpointPayload)
    async def on_checkpoint(self, peer: Peer, payload: CheckpointPayload):
        """Handle incoming checkpoint announcement (B2)."""
        logger.debug(
            f"Received checkpoint from {payload.sender_id} "
            f"(round={payload.round_number}, loss={payload.loss:.4f})"
        )
        if payload.sender_id in self.known_peers:
            self.known_peers[payload.sender_id].update_seen()

        if self.on_checkpoint_callback:
            await self.on_checkpoint_callback(
                sender_id=payload.sender_id,
                round_number=payload.round_number,
                loss=payload.loss,
                accuracy=payload.accuracy,
                model_version=payload.model_version,
            )

    def broadcast_checkpoint(self, sender_id: str, round_number: int,
                             loss: float, accuracy: float,
                             model_version: str = "1.0.0"):
        """Broadcast a checkpoint to all known compatible peers (B2)."""
        payload = CheckpointPayload(
            sender_id=sender_id,
            round_number=round_number,
            loss=loss,
            accuracy=accuracy,
            model_version=model_version,
        )
        for peer_info in self.known_peers.values():
            try:
                self.ez_send(peer_info.peer, payload)
            except Exception as e:
                logger.debug(
                    f"Failed to send checkpoint to {peer_info.node_id}: {e}"
                )
        logger.debug(
            f"Broadcast checkpoint round={round_number} to "
            f"{len(self.known_peers)} peers"
        )

    def _nack_try_consume(self, peer_mid: str) -> bool:
        """B5: Token-bucket rate limiter for NACK handling.

        Returns True if the request is allowed (a token was consumed).
        """
        now = time.time()
        bucket = self._nack_buckets.get(peer_mid)
        if bucket is None:
            bucket = {"tokens": NACK_BUCKET_MAX_TOKENS, "last_refill": now}
            self._nack_buckets[peer_mid] = bucket

        # Refill tokens
        elapsed = now - bucket["last_refill"]
        refill = elapsed / NACK_BUCKET_INTERVAL
        if refill > 0:
            bucket["tokens"] = min(
                bucket["tokens"] + refill, NACK_BUCKET_MAX_TOKENS
            )
            bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    @lazy_wrapper(RequestChunksPayload)
    async def on_request_chunks(self, peer: Peer, payload: RequestChunksPayload):
        """
        Handle request for missing chunks (NACK).
        Resends the requested chunks if available in cache.
        """
        import struct

        tid = payload.transfer_id
        peer_mid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)

        logger.debug(
            f"Chunk request received from {payload.sender_id} "
            f"(transfer={tid[:8]}...)"
        )

        if tid not in self._outgoing_transfers:
            logger.warning(f"Request for unknown/expired transfer {tid} from {payload.sender_id}")
            return

        transfer_data = self._outgoing_transfers[tid]

        # B5: Authorize — only the original recipient may NACK
        expected_mid = transfer_data.get("recipient_mid")
        if expected_mid and peer_mid != expected_mid:
            logger.warning(
                f"NACK rejected: peer {peer_mid[:16]}... is not the "
                f"original recipient of transfer {tid[:8]}..."
            )
            return

        # B5: Per-transfer resend budget
        resend_count = self._nack_resend_counts.get(tid, 0)
        if resend_count >= NACK_MAX_RESENDS_PER_TRANSFER:
            logger.warning(
                f"NACK rejected: resend budget exhausted for transfer {tid[:8]}... "
                f"({resend_count}/{NACK_MAX_RESENDS_PER_TRANSFER})"
            )
            return

        # B5: Per-peer token bucket
        if not self._nack_try_consume(peer_mid):
            logger.warning(
                f"NACK rate-limited for peer {peer_mid[:16]}..."
            )
            return

        # B5: Validate missing_indices_bytes length
        raw_len = len(payload.missing_indices_bytes)
        if raw_len == 0 or raw_len % 4 != 0:
            logger.warning(
                f"Malformed NACK payload from {payload.sender_id}: "
                f"missing_indices_bytes length={raw_len} is not a multiple of 4"
            )
            return

        try:
            missing_count = raw_len // 4
            missing_indices = list(struct.unpack(f'{missing_count}I', payload.missing_indices_bytes))
        except Exception as e:
            logger.error(f"Failed to unpack missing indices: {e}")
            return

        weights_bytes = transfer_data['weights']
        total_chunks = (len(weights_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE

        # B5: Bound indices to total_chunks
        valid_indices = [i for i in missing_indices if i < total_chunks]
        if len(valid_indices) != len(missing_indices):
            logger.debug(
                f"Dropped {len(missing_indices) - len(valid_indices)} "
                f"out-of-range indices from NACK for {tid[:8]}..."
            )

        logger.debug(
            f"Resending {len(valid_indices)} missing chunks to {payload.sender_id} "
            f"(transfer={tid[:8]}...)"
        )

        loss_val = transfer_data.get('loss', 0.0)
        acc_val = transfer_data.get('accuracy', 0.0)
        round_num = transfer_data.get('round', 0)
        sample_cnt = transfer_data.get('samples', 0)

        for i in valid_indices:
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(weights_bytes))
            chunk_data = weights_bytes[start:end]

            chunk_payload = ModelChunkPayload(
                transfer_id=tid,
                chunk_index=i,
                total_chunks=total_chunks,
                sender_id=self.node_id,
                data_schema_hash=self.data_schema_hash,
                round_number=round_num,
                sample_count=sample_cnt,
                loss=loss_val,
                accuracy=acc_val,
                chunk_data=chunk_data
            )

            self.ez_send(peer, chunk_payload)
            await asyncio.sleep(0.002)

        # B5: Increment resend counter
        self._nack_resend_counts[tid] = resend_count + 1

    @lazy_wrapper(ModelChunkPayload)
    async def on_model_chunk(self, peer: Peer, payload: ModelChunkPayload):
        """
        Handle incoming model chunk.
        
        Buffers chunks and triggers model processing when all chunks are received.
        """
        if payload.chunk_index % 50 == 0:
            logger.debug(
                f"on_model_chunk called: chunk {payload.chunk_index}/{payload.total_chunks}"
            )
        transfer_id = payload.transfer_id  # Use the transfer_id from the sender (UUID)
        # B3: key by (peer.mid, transfer_id) so different endpoints cannot hijack
        peer_mid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)
        buf_key = (peer_mid, transfer_id)

        # B4: Validate total_chunks before any allocation
        if payload.total_chunks > MAX_CHUNKS_PER_TRANSFER:
            logger.warning(
                f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                f"total_chunks={payload.total_chunks} > {MAX_CHUNKS_PER_TRANSFER}"
            )
            return

        # B4: Enforce per-peer and global buffer caps on new buffer creation
        if buf_key not in self._chunk_buffers:
            # Per-peer transfer count
            peer_transfers = sum(
                1 for (pmid, _) in self._chunk_buffers if pmid == peer_mid
            )
            if peer_transfers >= MAX_CONCURRENT_TRANSFERS_PER_PEER:
                logger.warning(
                    f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                    f"per-peer limit ({MAX_CONCURRENT_TRANSFERS_PER_PEER}) reached"
                )
                return

            # Global transfer count
            if len(self._chunk_buffers) >= MAX_TOTAL_TRANSFERS:
                logger.warning(
                    f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                    f"global limit ({MAX_TOTAL_TRANSFERS}) reached"
                )
                return

            # Per-peer byte budget
            peer_bytes = sum(
                sum(len(c) for c in buf.chunks.values())
                for (pmid, _), buf in self._chunk_buffers.items()
                if pmid == peer_mid
            )
            if peer_bytes >= MAX_BUFFERED_BYTES_PER_PEER:
                logger.warning(
                    f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                    f"per-peer byte budget ({MAX_BUFFERED_BYTES_PER_PEER // (1024*1024)} MB) exhausted"
                )
                return

        # Log chunk receipt with visible print statements
        if payload.chunk_index == 0:
            logger.debug(
                f"Chunk reception started from {payload.sender_id} "
                f"({payload.total_chunks} chunks)"
            )
            logger.debug(
                f"Receiving chunked model from {payload.sender_id} "
                f"(transfer={transfer_id[:8]}..., chunks={payload.total_chunks}, "
                f"round={payload.round_number})"
            )
        elif (payload.chunk_index + 1) % 50 == 0 or payload.chunk_index == payload.total_chunks - 1:
            progress = (payload.chunk_index + 1) / payload.total_chunks * 100
            logger.debug(
                f"Chunk progress from {payload.sender_id}: "
                f"{payload.chunk_index + 1}/{payload.total_chunks} ({progress:.0f}%)"
            )
        
        if payload.chunk_index == payload.total_chunks - 1:
            logger.debug(f"Received final chunk {payload.chunk_index + 1}/{payload.total_chunks} from {payload.sender_id}")
        
        # Create or get buffer for this transfer (keyed by endpoint identity)
        if buf_key not in self._chunk_buffers:
            logger.debug(
                f"Created chunk buffer {transfer_id[:8]}... "
                f"starting at chunk {payload.chunk_index}"
            )
            self._chunk_buffers[buf_key] = ChunkBuffer(
                transfer_id=transfer_id,
                sender_id=payload.sender_id,
                total_chunks=payload.total_chunks,
                data_schema_hash=payload.data_schema_hash,
                round_number=payload.round_number,
                sample_count=payload.sample_count,
                loss=payload.loss,
                accuracy=payload.accuracy
            )
        
        buffer = self._chunk_buffers[buf_key]
        
        # Add chunk to buffer
        is_complete = buffer.add_chunk(payload.chunk_index, payload.chunk_data)
        
        # Update peer last seen
        if payload.sender_id in self.known_peers:
            self.known_peers[payload.sender_id].update_seen()
            
            # NAT keepalive: Send heartbeat every 5 chunks to keep NAT hole open
            if payload.chunk_index % 5 == 0:
                peer_info = self.known_peers[payload.sender_id]
                self.ez_send(peer_info.peer, HeartbeatPayload(
                    self.node_id,
                    self._heartbeat_sequence
                ))
        
        # If all chunks received, reassemble and process
        if is_complete:
            logger.debug(
                f"All {payload.total_chunks} chunks received from {payload.sender_id}, reassembling..."
            )
            
            try:
                # Reassemble the complete weights
                weights_bytes = buffer.reassemble()
                
                # Remove buffer
                del self._chunk_buffers[buf_key]
                
                # Check size
                if len(weights_bytes) > MAX_INCOMING_MESSAGE_SIZE:
                    logger.error(
                        f"Rejected oversized model from {payload.sender_id}: "
                        f"{len(weights_bytes) / 1024 / 1024:.2f} MB"
                    )
                    return
                
                # Deserialize
                weights = deserialize_model(weights_bytes)
                
                logger.debug(
                    f"Received model update from {payload.sender_id} "
                    f"(round={buffer.round_number}, samples={buffer.sample_count}, "
                    f"size={len(weights_bytes) / 1024:.1f} KB, chunks={payload.total_chunks})"
                )
                
                # Call callback
                if self.on_model_update_callback:
                    logger.debug(
                        f"Model reassembly complete for {payload.sender_id}; invoking callback"
                    )
                    try:
                        await self.on_model_update_callback(
                            sender_id=payload.sender_id,
                            weights=weights,
                            sample_count=buffer.sample_count,
                            round_number=buffer.round_number,
                            loss=buffer.loss,
                            accuracy=buffer.accuracy
                        )
                        logger.debug(
                            f"Model update callback completed for {payload.sender_id}"
                        )
                    except Exception as cb_err:
                        logger.exception(
                            f"Model update callback failed for {payload.sender_id}: {cb_err}"
                        )
                else:
                    logger.warning(
                        f"No model update callback registered for transfer {transfer_id[:8]}..."
                    )
                    
            except Exception as e:
                logger.error(f"Failed to reassemble/process chunked model from {payload.sender_id}: {e}")
                if buf_key in self._chunk_buffers:
                    del self._chunk_buffers[buf_key]
        else:
            # Debug: if this is the final chunk but buffer is not complete, log missing chunks
            if payload.chunk_index == payload.total_chunks - 1:
                actual_chunks = len(buffer.chunks)
                logger.warning(
                    f"Final chunk received for {transfer_id[:8]}... but buffer incomplete: "
                    f"{actual_chunks}/{buffer.total_chunks} chunks"
                )
                missing = [i for i in range(buffer.total_chunks) if i not in buffer.chunks]
                logger.warning(
                    f"Missing chunk indices for {transfer_id[:8]}... "
                    f"(first 10): {missing[:10]} (total={len(missing)})"
                )
                
                # Send NACK (RequestChunksPayload)
                import struct
                # Pack indices as unsigned ints
                missing_bytes = struct.pack(f'{len(missing)}I', *missing)
                
                req_payload = RequestChunksPayload(
                    transfer_id=transfer_id,
                    sender_id=self.node_id,
                    missing_indices_bytes=missing_bytes
                )
                
                # Send request
                logger.debug(
                    f"Requesting {len(missing)} missing chunks for {transfer_id[:8]}..."
                )
                self.ez_send(peer, req_payload)

    async def send_model_update(
        self,
        target_node_id: str,
        weights: Any,
        sample_count: int,
        round_number: int,
        loss: float = 0.0,
        accuracy: float = 0.0
    ) -> bool:
        """
        Send model update to a specific peer.
        
        For large models (> CHUNK_SIZE bytes), uses chunked transfer
        to work around UDP MTU limitations.

        Args:
            target_node_id: Node ID of target peer
            weights: Model weights to send
            sample_count: Number of samples used for training
            round_number: Current round number
            loss: Training loss
            accuracy: Training accuracy

        Returns:
            True if sent successfully
        """
        if target_node_id not in self.known_peers:
            logger.warning(f"Unknown target peer: {target_node_id}")
            return False

        peer_info = self.known_peers[target_node_id]

        try:
            # Serialize weights
            weights_bytes = serialize_model(weights)

            # Convert None to 0.0 for payload packing
            loss_val = loss if loss is not None else 0.0
            acc_val = accuracy if accuracy is not None else 0.0

            # Check if we need chunked transfer
            if len(weights_bytes) <= CHUNK_SIZE:
                # Small payload - use direct transfer (original method)
                payload = ModelUpdatePayload(
                    sender_id=self.node_id,
                    weights_bytes=weights_bytes,
                    sample_count=sample_count,
                    round_number=round_number,
                    data_schema_hash=self.data_schema_hash,
                    loss=loss_val,
                    accuracy=acc_val
                )
                self.ez_send(peer_info.peer, payload)
                logger.debug(f"Sent model update to {target_node_id} ({len(weights_bytes)} bytes)")
            else:
                # Large payload - use chunked transfer
                transfer_id = str(uuid.uuid4())
                total_chunks = (len(weights_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
                
                # Debug: show target peer address
                logger.debug(
                    f"Chunked transfer target: {target_node_id} @ {peer_info.peer.address}"
                )
                
                logger.debug(
                    f"Sending chunked model to {target_node_id} "
                    f"(transfer={transfer_id[:8]}..., size={len(weights_bytes)} bytes, chunks={total_chunks})"
                )
                
                # Cache transfer for retry (B5: include recipient_mid for NACK auth)
                recipient_mid = peer_info.peer.mid.hex() if isinstance(peer_info.peer.mid, bytes) else str(peer_info.peer.mid)
                self._outgoing_transfers[transfer_id] = {
                    'weights': weights_bytes,
                    'loss': loss_val,
                    'accuracy': acc_val,
                    'round': round_number,
                    'samples': sample_count,
                    'timestamp': time.time(),
                    'recipient_mid': recipient_mid,
                }
                
                for i in range(total_chunks):
                    start = i * CHUNK_SIZE
                    end = min(start + CHUNK_SIZE, len(weights_bytes))
                    chunk_data = weights_bytes[start:end]
                    
                    chunk_payload = ModelChunkPayload(
                        transfer_id=transfer_id,
                        chunk_index=i,
                        total_chunks=total_chunks,
                        sender_id=self.node_id,
                        data_schema_hash=self.data_schema_hash,
                        round_number=round_number,
                        sample_count=sample_count,
                        loss=loss_val,
                        accuracy=acc_val,
                        chunk_data=chunk_data
                    )
                    
                    self.ez_send(peer_info.peer, chunk_payload)
                    
                    # Progress logging every 50 chunks (with 10KB chunks, we have ~230 chunks)
                    if (i + 1) % 50 == 0 or i == total_chunks - 1:
                        progress = (i + 1) / total_chunks * 100
                        logger.debug(
                            f"Chunk send progress to {target_node_id}: "
                            f"{i + 1}/{total_chunks} ({progress:.0f}%)"
                        )
                    
                    # B7: Single inter-chunk delay (was two separate sleeps)
                    if i < total_chunks - 1:
                        await asyncio.sleep(CHUNK_SEND_INTERVAL)
                
                logger.debug(f"Sent {total_chunks} chunks to {target_node_id}")
            
            return True

        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
            return False

    def broadcast_model_update(
        self,
        weights: Any,
        sample_count: int,
        round_number: int,
        loss: float = 0.0,
        accuracy: float = 0.0
    ) -> int:
        """
        Broadcast model update to all compatible peers.

        Args:
            weights: Model weights to send
            sample_count: Number of samples used for training
            round_number: Current round number
            loss: Training loss (can be None)
            accuracy: Training accuracy (can be None)

        Returns:
            Number of peers sent to
        """
        sent_count = 0

        for node_id in self.known_peers:
            if self.send_model_update(node_id, weights, sample_count, round_number, loss, accuracy):
                sent_count += 1

        logger.debug(f"Broadcast model update to {sent_count} peers")
        return sent_count

    def get_compatible_peers(self) -> List[PeerInfo]:
        """Get list of all compatible peers."""
        return list(self.known_peers.values())

    def get_peer_count(self) -> int:
        """Get number of connected compatible peers."""
        return len(self.known_peers)

    @staticmethod
    def _serialize_peer_list(peers: list) -> bytes:
        """Serialize a list of framework PeerInfo to msgpack bytes."""
        import msgpack
        data = []
        for p in peers:
            data.append({
                "peer_id": p.peer_id,
                "domain": p.domain,
                "data_schema_hash": p.data_schema_hash,
                "model_version": p.model_version,
                "age": p.age,
            })
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def _deserialize_peer_list(data: bytes) -> list:
        """Deserialize msgpack bytes to a list of framework PeerInfo dicts."""
        import msgpack
        from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo
        items = msgpack.unpackb(data, raw=False)
        result = []
        for item in items:
            p = FrameworkPeerInfo(
                peer_id=item["peer_id"],
                domain=item["domain"],
                data_schema_hash=item["data_schema_hash"],
                model_version=item.get("model_version", "1.0.0"),
            )
            p.age = item.get("age", 0)
            result.append(p)
        return result

    @lazy_wrapper(ShufflePayload)
    async def on_shuffle_request(self, peer: Peer, payload: ShufflePayload):
        """Handle incoming Cyclon shuffle request."""
        remote_peers = self._deserialize_peer_list(payload.peers_bytes)
        sender_id = payload.sender_id

        if self.on_shuffle_callback:
            response_peers = await self.on_shuffle_callback(sender_id, remote_peers)
            response_bytes = self._serialize_peer_list(response_peers)
            self.ez_send(peer, ShuffleResponsePayload(
                sender_id=self.node_id,
                peers_bytes=response_bytes
            ))
        else:
            from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo
            view_peers = [
                FrameworkPeerInfo(
                    peer_id=p.node_id,
                    domain=p.domain,
                    data_schema_hash=p.data_schema_hash,
                    model_version=p.model_version,
                )
                for p in self.known_peers.values()
                if p.node_id != sender_id
            ][:8]
            response_bytes = self._serialize_peer_list(view_peers)
            self.ez_send(peer, ShuffleResponsePayload(
                sender_id=self.node_id,
                peers_bytes=response_bytes
            ))

    @lazy_wrapper(ShuffleResponsePayload)
    async def on_shuffle_response(self, peer: Peer, payload: ShuffleResponsePayload):
        """Handle Cyclon shuffle response."""
        remote_peers = self._deserialize_peer_list(payload.peers_bytes)

        if self.on_shuffle_callback:
            for p in remote_peers:
                if p.peer_id not in self.known_peers:
                    from quinkgl.topology.base import PeerInfo as FrameworkPeerInfo
                    framework_peer_info = FrameworkPeerInfo(
                        peer_id=p.peer_id,
                        domain=p.domain,
                        data_schema_hash=p.data_schema_hash,
                        model_version=p.model_version,
                    )
                    if self.on_peer_discovered_callback:
                        net_peer_info = PeerInfo(
                            peer=peer,
                            node_id=p.peer_id,
                            domain=p.domain,
                            data_schema_hash=p.data_schema_hash,
                            model_version=p.model_version,
                        )
                        self.known_peers[p.peer_id] = net_peer_info
                        await self.on_peer_discovered_callback(net_peer_info)

    async def send_shuffle(self, target_node_id: str, peers_bytes: bytes) -> bytes:
        """
        Send a shuffle request and wait for response.

        This is called by the CyclonTopology via its shuffle callback.
        Returns the response peers_bytes or empty bytes on failure.
        """
        if target_node_id not in self.known_peers:
            return b''

        peer_info = self.known_peers[target_node_id]
        self.ez_send(peer_info.peer, ShufflePayload(
            sender_id=self.node_id,
            peers_bytes=peers_bytes
        ))
        return b''
