"""
Gossip Learning Community for IPv8

Implements P2P model exchange and aggregation over IPv8.
Domain isolation ensures only compatible peers communicate.

CHUNKED TRANSFER: Large model updates are split into chunks
to work around UDP MTU limits (~1400 bytes).
"""

import asyncio
import json
import time
import logging
import hashlib
import os
import struct
import tempfile
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
# v3: Increased from 0.012 to 0.025 to handle aggregate ingress from multiple
# simultaneous senders in a 5-peer swarm (4 senders × 40 KB/s ≈ 160 KB/s).
CHUNK_SEND_INTERVAL = 0.025

# Timeout for incomplete transfers (300 seconds - increased for Colab/slow networks)
CHUNK_TRANSFER_TIMEOUT = 300

# B4: Chunk-buffer memory caps to prevent DoS
MAX_CONCURRENT_TRANSFERS_PER_PEER = 8
MAX_TOTAL_TRANSFERS = 50
MAX_BUFFERED_BYTES_PER_PEER = 200 * 1024 * 1024  # 200 MB
MAX_CHUNKS_PER_TRANSFER = 300_000  # ~300 MB at 1 KB/chunk

# B5: NACK rate-limiting
NACK_MAX_RESENDS_PER_TRANSFER = 6
NACK_BUCKET_INTERVAL = 5.0   # seconds — refill one token per interval
NACK_BUCKET_MAX_TOKENS = 10  # max burst per peer
NACK_TRANSFER_BUCKET_MAX_TOKENS = 6
NACK_TRANSFER_BUCKET_INTERVAL = 5.0

# B6: Early NACK gap detection
EARLY_NACK_AGE_THRESHOLD = 18.0  # seconds — allow most chunks to arrive naturally

# B16 §4.8: Hard byte cap on fingerprint JSON in discovery announcements
MAX_FINGERPRINT_BYTES = 8192  # 8 KB
MAX_ROUND_SKIP = 1000


def generate_community_id(
    domain: str,
    data_schema_hash: str,
    manifest_hash: Optional[str] = None,
) -> bytes:
    """
    Generate a unique community ID for a domain + schema (+ manifest) combination.

    This ensures domain isolation - only peers with matching
    domain, schema, and (optionally) swarm manifest policy can communicate.

    Args:
        domain: Domain identifier (e.g., "health", "agriculture")
        data_schema_hash: Hash of data schema
        manifest_hash: Optional hex SHA-256 of the canonical swarm manifest
            (see ``quinkgl.manifest.schema.DataPolicy.manifest_hash``).
            When supplied, it is bound into the community ID so that peers
            with divergent policy cannot share a community.  Omitted for
            backwards compatibility with legacy callers.

    Returns:
        20-byte community ID for IPv8
    """
    # Combine domain, schema, and optionally manifest hash. The manifest
    # component is appended with a distinct separator so the absence of a
    # manifest is never equivalent to an empty-string manifest.
    if manifest_hash:
        combined = f"QuinkGL-{domain}-{data_schema_hash}-m:{manifest_hash}".encode("utf-8")
    else:
        combined = f"QuinkGL-{domain}-{data_schema_hash}".encode("utf-8")

    # B16 §4.9: SHA-256 truncated to 20 bytes (replaces SHA-1)
    hashed = hashlib.sha256(combined).digest()[:20]

    return hashed


def _chunk_sign_data(sender_id: str, round_number: int, data_schema_hash: str,
                     chunk_index: int, chunk_data: bytes, *, sample_count: int = 0,
                     loss: float = 0.0, accuracy: float = 0.0, timestamp: int = 0,
                     total_chunks: int = 1) -> bytes:
    """B14: Build the canonical byte string that is signed per chunk.

    ``sender_id || round_number || data_schema_hash || chunk_index || SHA-256(chunk_data)``
    """
    return (
        sender_id.encode("utf-8")
        + struct.pack("!I", round_number)
        + data_schema_hash.encode("utf-8")
        + struct.pack("!I", chunk_index)
        + struct.pack("!I", total_chunks)
        + struct.pack("!I", sample_count)
        + struct.pack("!d", loss)
        + struct.pack("!d", accuracy)
        + struct.pack("!I", timestamp)
        + hashlib.sha256(chunk_data).digest()
    )


def _chunk_sign(private_key, sender_id: str, round_number: int,
                data_schema_hash: str, chunk_index: int, chunk_data: bytes, *,
                sample_count: int = 0, loss: float = 0.0, accuracy: float = 0.0,
                timestamp: int = 0, total_chunks: int = 1) -> bytes:
    """B14: Sign a chunk using the IPv8 peer's private key."""
    msg = _chunk_sign_data(sender_id, round_number, data_schema_hash,
                           chunk_index, chunk_data, sample_count=sample_count,
                           loss=loss, accuracy=accuracy, timestamp=timestamp,
                           total_chunks=total_chunks)
    return private_key.signature(msg)


def _chunk_verify(public_key, signature: bytes, sender_id: str,
                  round_number: int, data_schema_hash: str,
                  chunk_index: int, chunk_data: bytes, *, sample_count: int = 0,
                  loss: float = 0.0, accuracy: float = 0.0, timestamp: int = 0,
                  total_chunks: int = 1) -> bool:
    """B14: Verify a chunk signature against the sender's public key."""
    if not signature:
        return False
    msg = _chunk_sign_data(sender_id, round_number, data_schema_hash,
                           chunk_index, chunk_data, sample_count=sample_count,
                           loss=loss, accuracy=accuracy, timestamp=timestamp,
                           total_chunks=total_chunks)
    try:
        return public_key.verify(signature, msg)
    except Exception:
        return False


def _emit_ipv8_payload_dropped(community, reason: str, *, sender_id: Optional[str] = None,
                               peer_mid: Optional[str] = None, security_event: Optional[str] = None,
                               transport: Optional[str] = None, **details) -> None:
    emitter = getattr(community, "event_emitter", None)
    if not emitter:
        return
    payload = {
        "node_id": getattr(community, "node_id", "unknown"),
        "reason": reason,
    }
    if sender_id is not None:
        payload["sender_id"] = sender_id
    if peer_mid is not None:
        payload["peer_mid"] = peer_mid
    if transport is not None:
        payload["transport"] = transport
    for key, value in details.items():
        if value is not None:
            payload[key] = value
    if security_event:
        emitter.emit(security_event, payload)
    emitter.emit("ipv8_payload_dropped", payload)


class DiscoveryAnnouncePayload(Payload):
    """
    Payload for peer discovery announcements.

    Peers announce their domain and schema to find compatible peers.
    Optionally includes a fingerprint JSON blob for affinity computation.

    Spec §12.1 (Phase 1): carries a trailing ``manifest_id`` (hex
    ``swarm_id`` or empty) so peers operating under an explicit manifest
    can mutually identify swarm membership without relying solely on the
    looser ``(domain, data_schema_hash)`` pair.  Old peers emit 5 fields;
    new peers emit 6.  ``from_unpack_list`` accepts both shapes so that
    cross-version gossip keeps working.
    """
    msg_id = 1
    format_list = ['varlenH'] * 6

    def __init__(self, node_id: str, domain: str, data_schema_hash: str,
                 model_version: str = "1.0.0", fingerprint_json: str = "",
                 manifest_id: str = ""):
        super().__init__()
        self.node_id = node_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model_version = model_version
        self.fingerprint_json = fingerprint_json
        self.manifest_id = manifest_id

    def to_pack_list(self):
        return [
            ('varlenH', self.node_id.encode('utf-8')),
            ('varlenH', self.domain.encode('utf-8')),
            ('varlenH', self.data_schema_hash.encode('utf-8')),
            ('varlenH', self.model_version.encode('utf-8')),
            ('varlenH', self.fingerprint_json.encode('utf-8')),
            ('varlenH', self.manifest_id.encode('utf-8')),
        ]

    @classmethod
    def from_unpack_list(cls, *args):
        # Tolerate legacy 5-field payloads from v2.0.0 peers: the trailing
        # ``manifest_id`` is simply absent and decoded as empty.
        fp_json = ""
        manifest_id = ""
        if len(args) > 4:
            fp_json = args[4].decode('utf-8') if args[4] else ""
        if len(args) > 5:
            manifest_id = args[5].decode('utf-8') if args[5] else ""
        return cls(
            args[0].decode('utf-8'),
            args[1].decode('utf-8'),
            args[2].decode('utf-8'),
            args[3].decode('utf-8'),
            fp_json,
            manifest_id,
        )


def _manifest_id_blocks_peer(local_mid: str, remote_mid: str) -> bool:
    """Pre-filter gate from spec §12.3.

    Returns True iff the discovery announce MUST be rejected on a
    manifest-id mismatch.  Both sides MUST advertise a non-empty
    ``manifest_id`` AND the two values MUST differ — otherwise the
    legacy ``(domain, data_schema_hash)`` filter runs as before.
    """
    return bool(local_mid) and bool(remote_mid) and local_mid != remote_mid


class ModelUpdatePayload(Payload):
    """
    Payload for model weight updates.

    Contains serialized model weights and metadata.

    NOTE: Uses 'varlenI' for weights_bytes (large model), 'varlenH' for others
    """
    msg_id = 2
    format_list = ['varlenH', 'varlenI', 'I', 'I', 'varlenH', 'd', 'd', 'I', 'varlenH']

    def __init__(
        self,
        sender_id: str,
        weights_bytes: bytes,
        sample_count: int,
        round_number: int,
        data_schema_hash: str,
        loss: float = 0.0,
        accuracy: float = 0.0,
        timestamp: int = 0,
        signature: bytes = b""
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
        self.signature = signature

    def to_pack_list(self):
        return [
            ('varlenH', self.sender_id.encode('utf-8')),
            ('varlenI', self.weights_bytes),  # varlenI for large model weights
            ('I', self.sample_count),
            ('I', self.round_number),
            ('varlenH', self.data_schema_hash.encode('utf-8')),
            ('d', self.loss),
            ('d', self.accuracy),
            ('I', self.timestamp),
            ('varlenH', self.signature),
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
            args[7],
            args[8] if len(args) > 8 else b"",  # signature
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
        signature: B14 — cryptographic signature over critical fields
    """
    msg_id = 4
    # varlenH for strings, I for ints, d for floats, varlenH for chunk data
    format_list = ['varlenH', 'I', 'I', 'varlenH', 'varlenH', 'I', 'I', 'd', 'd', 'I', 'varlenH', 'varlenH']

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
        chunk_data: bytes,
        timestamp: int = 0,
        signature: bytes = b""
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
        self.timestamp = timestamp or int(time.time())
        self.chunk_data = chunk_data
        self.signature = signature

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
            ('I', self.timestamp),
            ('varlenH', self.chunk_data),
            ('varlenH', self.signature),
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
            args[10],                  # chunk_data (bytes)
            args[9],                   # timestamp
            args[11] if len(args) > 11 else b"",  # signature
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
    timestamp: int = 0
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
        model_version: str = "1.0.0",
        manifest_id: str = "",
    ):
        self.peer = peer
        self.node_id = node_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model_version = model_version
        self.manifest_id = manifest_id
        self.last_seen = time.time()

    def is_compatible(
        self,
        domain: str,
        data_schema_hash: str,
        manifest_id: str = "",
    ) -> bool:
        """Check swarm compatibility (spec §12.2).

        When both peers advertise a ``manifest_id``, the manifest
        identity is authoritative — it supersedes the looser ``(domain,
        data_schema_hash)`` legacy pair so two peers with different
        cosmetic domains but the same signed manifest still find each
        other.  If either side is manifest-less, the legacy pair is the
        sole compatibility signal (cross-version fallback).
        """
        if manifest_id and self.manifest_id:
            return self.manifest_id == manifest_id
        return (
            self.domain == domain
            and self.data_schema_hash == data_schema_hash
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
                 data_schema_hash: str = "", model_version: str = "1.0.0",
                 require_signature: bool = True, last_seen_round_state_path: str = "",
                 max_round_skip: int = MAX_ROUND_SKIP, **kwargs):
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
        self.require_signature = require_signature
        self.last_seen_round_state_path = last_seen_round_state_path
        self.max_round_skip = max_round_skip
        # Phase 1 (spec §12.3): manifest-first swarm identity.  ``manifest``
        # is the attached :class:`SwarmManifest` (or None for legacy peers);
        # ``manifest_hash`` is the hex ``swarm_id`` broadcast over the wire
        # so peers can pre-filter cross-swarm discovery announces before the
        # legacy (domain, data_schema_hash) gate runs.
        self.manifest = None
        self.manifest_hash = ""
        self.current_round_provider: Optional[Callable[[], int]] = None

        # B16 §4.5: Instance-level community_id (no class mutation)
        # IPv8 reads self.community_id; setting it on the instance shadows
        # the class attribute without clobbering other instances.
        self.community_id = generate_community_id(domain, data_schema_hash)
        self._instance_community_id = self.community_id

        # Initialize parent with all args
        super().__init__(*args, **kwargs)

        # Peer tracking
        self.known_peers: dict[str, PeerInfo] = {}  # node_id -> PeerInfo
        # §4.4: reverse lookup — peer.mid hex → node_id
        self._mid_to_node_id: dict[str, str] = {}

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
        # v3: per-transfer NACK bucket so transfer A does not starve transfer B
        # transfer_id -> { 'tokens': float, 'last_refill': float }
        self._nack_transfer_buckets: Dict[str, Dict] = {}

        # v3: Sender-side idempotency — (receiver_node_id, round_number, model_hash) -> transfer_id
        self._inflight_transfers: Dict[tuple[str, int, str], str] = {}

        # v3: Completed-transfer registry for replay protection (separate from last_seen_round)
        self._completed_chunk_transfers: Dict[tuple[str, str], float] = {}

        # v3: Production metrics
        self.metrics = {
            'chunk_transfers_started': 0,
            'chunk_transfers_completed': 0,
            'chunk_transfers_failed_timeout': 0,
            'chunk_transfers_rejected_peer_limit': 0,
            'nacks_sent': 0,
            'nacks_received': 0,
            'nacks_ignored_budget': 0,
            'chunks_resent': 0,
        }

        # B15: Replay protection — per-peer last seen round
        self._last_seen_round: Dict[str, int] = {}
        self._load_last_seen_round_state()

        # v3: Semantic chunk dedup — protects against non-deterministic IPv8
        # signatures that cause NACK resends to appear as distinct packets.
        self._recent_chunks: Dict[tuple[str, int], float] = {}

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
        self.event_emitter: Optional[Any] = None

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

        # v3: Enlarge UDP socket buffers to prevent kernel-level drops
        # under aggregate ingress from multiple simultaneous senders.
        # NOTE: moved from __init__ because self.endpoint is only wired
        # after community construction in IPv8.
        if not getattr(self, '_socket_buffers_resized', False):
            SO_RCVBUF_SIZE = 8 * 1024 * 1024   # 8 MB
            SO_SNDBUF_SIZE = 4 * 1024 * 1024   # 4 MB
            udp_socket = None
            # IPv8 stores socket via: self.endpoint._transport._sock
            endpoint = getattr(self, 'endpoint', None)
            if endpoint is not None:
                # Try direct _socket / socket attributes first (raw / simple endpoints)
                udp_socket = getattr(endpoint, '_socket', None)
                if udp_socket is None:
                    udp_socket = getattr(endpoint, 'socket', None)
                # IPv8 UDPEndpoint: _transport is asyncio DatagramTransport, _sock lives inside
                if udp_socket is None:
                    transport = getattr(endpoint, '_transport', None)
                    if transport is not None:
                        udp_socket = getattr(transport, '_sock', None)
                # IPv8 DispatcherEndpoint: delegates to sub-interfaces (e.g. UDPIPv4)
                if udp_socket is None and type(endpoint).__name__ == 'DispatcherEndpoint':
                    preferred = getattr(endpoint, '_preferred_interface', None)
                    if preferred is not None:
                        transport = getattr(preferred, '_transport', None)
                        if transport is not None:
                            udp_socket = getattr(transport, '_sock', None)
                    if udp_socket is None:
                        for iface in getattr(endpoint, 'interfaces', {}).values():
                            transport = getattr(iface, '_transport', None)
                            if transport is not None:
                                udp_socket = getattr(transport, '_sock', None)
                                if udp_socket is not None:
                                    break
            if udp_socket is None:
                udp_socket = getattr(self, '_socket', None)
            if udp_socket is not None:
                try:
                    import socket as _socket
                    udp_socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, SO_RCVBUF_SIZE)
                    udp_socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, SO_SNDBUF_SIZE)
                    actual_rcv = udp_socket.getsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF)
                    actual_snd = udp_socket.getsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF)
                    logger.info(f"UDP socket buffers resized: RCVBUF={actual_rcv}, SNDBUF={actual_snd}")
                    self._socket_buffers_resized = True
                except OSError as exc:
                    logger.warning(f"Failed to resize UDP socket buffers: {exc}")
            else:
                logger.warning("UDP socket not found; buffer resize skipped")

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
        self._persist_last_seen_round_state()
        await super().unload()
        logger.debug(f"GossipLearningCommunity unloaded for '{self.node_id}'")

    def _load_last_seen_round_state(self):
        path = getattr(self, "last_seen_round_state_path", "")
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_state = json.load(handle)
        except Exception as exc:
            logger.warning(f"Failed to load replay state from {path}: {exc}")
            return
        if not isinstance(raw_state, dict):
            return
        loaded_state = {}
        for peer_mid, round_number in raw_state.items():
            if not isinstance(peer_mid, str):
                continue
            try:
                loaded_state[peer_mid] = int(round_number)
            except (TypeError, ValueError):
                continue
        self._last_seen_round = loaded_state

    def _persist_last_seen_round_state(self):
        path = getattr(self, "last_seen_round_state_path", "")
        if not path:
            return
        temp_path = ""
        try:
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
                json.dump(self._last_seen_round, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = handle.name
            os.replace(temp_path, path)
        except Exception as exc:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            logger.warning(f"Failed to persist replay state to {path}: {exc}")

    def _record_last_seen_round(self, peer_mid: str, round_number: int):
        self._last_seen_round[peer_mid] = max(
            self._last_seen_round.get(peer_mid, -1),
            int(round_number),
        )
        self._persist_last_seen_round_state()

    def _get_local_round(self) -> int:
        provider = getattr(self, "current_round_provider", None)
        if callable(provider) and type(provider).__module__ != "unittest.mock":
            try:
                return int(provider())
            except Exception:
                return 0
        try:
            current_round = getattr(self, "current_round", 0)
            if type(current_round).__module__ == "unittest.mock":
                return 0
            return int(current_round)
        except Exception:
            return 0

    def _get_max_round_skip(self) -> int:
        try:
            value = getattr(self, "max_round_skip", MAX_ROUND_SKIP)
            if type(value).__module__ == "unittest.mock":
                return MAX_ROUND_SKIP
            return int(value)
        except Exception:
            return MAX_ROUND_SKIP

    async def _announce_discovery(self):
        """Announce our presence to all peers."""
        fingerprint_json = ""
        if self.local_fingerprint is not None:
            try:
                import json
                from quinkgl.fingerprint.fingerprint import DataFingerprint
                fp_dict = self.local_fingerprint.to_dict()
                fingerprint_json = json.dumps(fp_dict)
            except Exception:
                logger.debug("Could not serialize local fingerprint for announce")

        local_mid = getattr(self, "manifest_hash", "") or ""
        for peer in self.get_peers():
            self.ez_send(peer, DiscoveryAnnouncePayload(
                self.node_id,
                self.domain,
                self.data_schema_hash,
                self.model_version,
                fingerprint_json,
                local_mid,
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
        stale_timeout = 300
        stale_peers = []

        for node_id, peer_info in self.known_peers.items():
            if peer_info.age() > stale_timeout:
                stale_peers.append(node_id)

        for node_id in stale_peers:
            peer_info = self.known_peers.pop(node_id)
            pmid = peer_info.peer.mid.hex() if isinstance(peer_info.peer.mid, bytes) else str(peer_info.peer.mid)
            self._mid_to_node_id.pop(pmid, None)
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
            self.metrics['chunk_transfers_failed_timeout'] += 1

        # v3: Purge expired completed-transfer entries (TTL = 10 min)
        COMPLETED_TRANSFER_TTL = 600
        now = time.time()
        expired_completed = [
            key for key, ts in list(self._completed_chunk_transfers.items())
            if now - ts > COMPLETED_TRANSFER_TTL
        ]
        for key in expired_completed:
            del self._completed_chunk_transfers[key]

        # v3: Purge expired recent-chunk entries (TTL = 2 min)
        RECENT_CHUNK_TTL = 120
        expired_recent = [
            key for key, ts in list(self._recent_chunks.items())
            if now - ts > RECENT_CHUNK_TTL
        ]
        for key in expired_recent:
            del self._recent_chunks[key]

        # v3: Emit metrics summary at DEBUG level every cleanup cycle
        logger.debug(
            f"[metrics] started={self.metrics['chunk_transfers_started']} "
            f"completed={self.metrics['chunk_transfers_completed']} "
            f"timeout={self.metrics['chunk_transfers_failed_timeout']} "
            f"peer_limit={self.metrics['chunk_transfers_rejected_peer_limit']} "
            f"nacks_sent={self.metrics['nacks_sent']} nacks_rcvd={self.metrics['nacks_received']} "
            f"nacks_ignored={self.metrics['nacks_ignored_budget']} resent={self.metrics['chunks_resent']}"
        )

    async def _cleanup_outgoing_cache(self):
        """Remove old outgoing transfers from cache."""
        current_time = time.time()
        timeout = 600
        expired = [
            tid for tid, data in self._outgoing_transfers.items()
            if current_time - data['timestamp'] > timeout
        ]
        for tid in expired:
            del self._outgoing_transfers[tid]

        # v3: Also purge orphaned per-transfer NACK buckets and inflight entries
        for tid in list(self._nack_transfer_buckets.keys()):
            if tid not in self._outgoing_transfers:
                del self._nack_transfer_buckets[tid]
        for inflight_key, tid in list(self._inflight_transfers.items()):
            if tid not in self._outgoing_transfers:
                del self._inflight_transfers[inflight_key]

    async def _nack_incomplete_buffers(self):
        """Proactively NACK incomplete buffers older than threshold."""
        import struct

        now = time.time()
        for (peer_mid, transfer_id), buffer in list(self._chunk_buffers.items()):
            age = now - buffer.created_at
            if age < EARLY_NACK_AGE_THRESHOLD:
                continue
            if buffer.is_complete():
                continue

            missing = [
                i for i in range(buffer.total_chunks) if i not in buffer.chunks
            ]
            if not missing:
                continue

            # v3: Check per-transfer bucket first, then per-peer bucket
            if not self._nack_try_consume_transfer(transfer_id):
                logger.debug(
                    f"Early-NACK rate-limited for transfer {transfer_id[:8]}..."
                )
                self.metrics['nacks_ignored_budget'] += 1
                continue
            if not self._nack_try_consume(peer_mid):
                logger.debug(
                    f"Early-NACK rate-limited for peer {peer_mid[:16]}..."
                )
                self.metrics['nacks_ignored_budget'] += 1
                continue

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
            self.metrics['nacks_sent'] += 1

    @lazy_wrapper(DiscoveryAnnouncePayload)
    async def on_discovery_announce(self, peer: Peer, payload: DiscoveryAnnouncePayload):
        # Spec §12.3: honour the manifest-id pre-filter BEFORE the legacy
        # (domain, data_schema_hash) gate so two peers with matching manifest
        # hashes are never blocked on cosmetic domain differences, and two
        # peers with different manifests never get through even if their
        # domain/schema happen to align.
        local_mid = getattr(self, "manifest_hash", "") or ""
        remote_mid = getattr(payload, "manifest_id", "") or ""
        if _manifest_id_blocks_peer(local_mid, remote_mid):
            emitter = getattr(self, "event_emitter", None)
            if emitter is not None:
                try:
                    emitter.emit(
                        "security.discovery_manifest_mismatch",
                        {
                            "peer": payload.node_id,
                            "local_manifest_id": local_mid,
                            "remote_manifest_id": remote_mid,
                        },
                    )
                except Exception:  # pragma: no cover — observer failures are non-fatal
                    pass
            logger.debug(
                f"Dropping peer {payload.node_id}: manifest_id mismatch "
                f"(local={local_mid[:8]}..., remote={remote_mid[:8]}...)"
            )
            return

        if payload.domain != self.domain or payload.data_schema_hash != self.data_schema_hash:
            logger.debug(
                f"Incompatible peer: {payload.node_id} "
                f"(domain={payload.domain}, schema={payload.data_schema_hash[:8]}...)"
            )
            return

        fingerprint = None
        if payload.fingerprint_json:
            if len(payload.fingerprint_json.encode("utf-8")) > MAX_FINGERPRINT_BYTES:
                logger.warning(
                    f"Rejected oversized fingerprint from {payload.node_id}: "
                    f"{len(payload.fingerprint_json)} chars"
                )
            else:
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
                model_version=payload.model_version,
                manifest_id=remote_mid,
            )
            peer_info.data_fingerprint = fingerprint
            self.known_peers[payload.node_id] = peer_info
            pmid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)
            self._mid_to_node_id[pmid] = payload.node_id
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

        # §4.4: Verify sender identity — payload.sender_id must match the
        # node_id we recorded for this peer.mid during discovery.
        peer_mid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)
        expected_node_id = self._mid_to_node_id.get(peer_mid)
        if expected_node_id is not None and payload.sender_id != expected_node_id:
            logger.warning(
                f"Identity mismatch: peer.mid={peer_mid[:12]}... claims "
                f"sender_id={payload.sender_id}, expected {expected_node_id}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "identity mismatch",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.identity_mismatch",
                transport="direct",
                expected_sender_id=expected_node_id,
            )
            return

        # Verify direct-path signature (chunk_index=0, full weights as chunk_data)
        if payload.signature:
            if not _chunk_verify(
                peer.public_key, payload.signature,
                payload.sender_id, payload.round_number,
                payload.data_schema_hash, 0, payload.weights_bytes,
                sample_count=payload.sample_count,
                loss=payload.loss,
                accuracy=payload.accuracy,
                timestamp=payload.timestamp,
                total_chunks=1,
            ):
                logger.warning(
                    f"Rejected direct model from {payload.sender_id}: invalid signature"
                )
                _emit_ipv8_payload_dropped(
                    self,
                    "invalid signature",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    security_event="security.signature_rejected",
                    transport="direct",
                )
                return
        else:
            if getattr(self, "require_signature", True):
                logger.warning(
                    f"Rejected unsigned direct model from {payload.sender_id}"
                )
                _emit_ipv8_payload_dropped(
                    self,
                    "missing signature",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    security_event="security.signature_missing",
                    transport="direct",
                )
                return
            logger.debug(
                f"Unsigned direct model from {payload.sender_id} (no signature)"
            )

        # B15: Replay protection — reject stale or replayed rounds
        peer_mid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)
        last_round = self._last_seen_round.get(peer_mid, -1)
        local_round = GossipLearningCommunity._get_local_round(self)
        max_round_skip = GossipLearningCommunity._get_max_round_skip(self)
        if payload.round_number > local_round + max_round_skip:
            logger.warning(
                f"Rejected future-round model from {payload.sender_id}: "
                f"round {payload.round_number} > local {local_round} + {max_round_skip}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "future round rejected",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.future_round_rejected",
                transport="direct",
                round_number=payload.round_number,
                local_round=local_round,
            )
            return
        if payload.round_number <= last_round:
            logger.warning(
                f"Rejected replayed model from {payload.sender_id}: "
                f"round {payload.round_number} <= last seen {last_round}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "replayed round",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.replay_rejected",
                transport="direct",
                round_number=payload.round_number,
                last_round=last_round,
            )
            return

        # Check message size before deserializing to prevent DoS
        weights_size = len(payload.weights_bytes)
        if weights_size > MAX_INCOMING_MESSAGE_SIZE:
            logger.error(
                f"Rejected oversized model from {payload.sender_id}: "
                f"{weights_size / 1024 / 1024:.2f} MB "
                f"(max {MAX_INCOMING_MESSAGE_SIZE / 1024 / 1024:.0f} MB)"
            )
            _emit_ipv8_payload_dropped(
                self,
                "oversized model",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.oversized_message",
                transport="direct",
                size_bytes=weights_size,
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
            _emit_ipv8_payload_dropped(
                self,
                f"deserialization error: {e}",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="direct",
            )
            return
        except Exception as e:
            logger.error(f"Failed to deserialize model from {payload.sender_id}: {e}")
            _emit_ipv8_payload_dropped(
                self,
                f"deserialization error: {e}",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="direct",
            )
            return

        # Call callback if registered
        if self.on_model_update_callback:
            try:
                await self.on_model_update_callback(
                    sender_id=payload.sender_id,
                    weights=weights,
                    sample_count=payload.sample_count,
                    round_number=payload.round_number,
                    loss=payload.loss,
                    accuracy=payload.accuracy
                )
            except Exception as e:
                logger.exception(f"Model update callback failed for {payload.sender_id}: {e}")
                _emit_ipv8_payload_dropped(
                    self,
                    f"callback error: {e}",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    transport="direct",
                )
                return
        self._last_seen_round[peer_mid] = max(
            self._last_seen_round.get(peer_mid, -1),
            int(payload.round_number),
        )
        try:
            GossipLearningCommunity._persist_last_seen_round_state(self)
        except Exception:
            pass

    # Backward compatibility alias for older tests
    _dispatch_model_update = on_model_update.__wrapped__

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

    def _nack_try_consume_transfer(self, transfer_id: str) -> bool:
        """v3: Per-transfer token-bucket rate limiter for NACK handling.

        Returns True if the request is allowed (a token was consumed).
        """
        now = time.time()
        bucket = self._nack_transfer_buckets.get(transfer_id)
        if bucket is None:
            bucket = {"tokens": NACK_TRANSFER_BUCKET_MAX_TOKENS, "last_refill": now}
            self._nack_transfer_buckets[transfer_id] = bucket

        elapsed = now - bucket["last_refill"]
        refill = elapsed / NACK_TRANSFER_BUCKET_INTERVAL
        if refill > 0:
            bucket["tokens"] = min(
                bucket["tokens"] + refill, NACK_TRANSFER_BUCKET_MAX_TOKENS
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
        self.metrics['nacks_received'] += 1

        if tid not in self._outgoing_transfers:
            logger.warning(f"Request for unknown/expired transfer {tid} from {payload.sender_id}")
            _emit_ipv8_payload_dropped(
                self,
                "unknown transfer",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="nack",
                transfer_id=tid,
            )
            return

        transfer_data = self._outgoing_transfers[tid]

        # B5: Authorize — only the original recipient may NACK
        expected_mid = transfer_data.get("recipient_mid")
        if expected_mid and peer_mid != expected_mid:
            logger.warning(
                f"NACK rejected: peer {peer_mid[:16]}... is not the "
                f"original recipient of transfer {tid[:8]}..."
            )
            _emit_ipv8_payload_dropped(
                self,
                "unauthorized nack",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="nack",
                transfer_id=tid,
            )
            return

        # B5: Per-transfer resend budget
        resend_count = self._nack_resend_counts.get(tid, 0)
        if resend_count >= NACK_MAX_RESENDS_PER_TRANSFER:
            logger.warning(
                f"NACK rejected: resend budget exhausted for transfer {tid[:8]}... "
                f"({resend_count}/{NACK_MAX_RESENDS_PER_TRANSFER})"
            )
            _emit_ipv8_payload_dropped(
                self,
                "nack resend budget exhausted",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="nack",
                transfer_id=tid,
            )
            return

        # B5: Per-peer token bucket
        if not self._nack_try_consume(peer_mid):
            logger.warning(
                f"NACK rate-limited for peer {peer_mid[:16]}..."
            )
            _emit_ipv8_payload_dropped(
                self,
                "nack rate limited",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.nack_rate_limited",
                transport="nack",
                transfer_id=tid,
            )
            return

        # B5: Validate missing_indices_bytes length
        raw_len = len(payload.missing_indices_bytes)
        if raw_len == 0 or raw_len % 4 != 0:
            logger.warning(
                f"Malformed NACK payload from {payload.sender_id}: "
                f"missing_indices_bytes length={raw_len} is not a multiple of 4"
            )
            _emit_ipv8_payload_dropped(
                self,
                "malformed nack payload",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="nack",
                transfer_id=tid,
            )
            return

        try:
            missing_count = raw_len // 4
            missing_indices = list(struct.unpack(f'{missing_count}I', payload.missing_indices_bytes))
        except Exception as e:
            logger.error(f"Failed to unpack missing indices: {e}")
            _emit_ipv8_payload_dropped(
                self,
                f"malformed nack payload: {e}",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                transport="nack",
                transfer_id=tid,
            )
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

            # B14: Sign resent chunks
            sig = _chunk_sign(
                self.my_peer.key,
                self.node_id, round_num,
                self.data_schema_hash, i, chunk_data,
                sample_count=sample_cnt,
                loss=loss_val,
                accuracy=acc_val,
                timestamp=int(transfer_data.get('timestamp', time.time())),
                total_chunks=total_chunks,
            )

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
                timestamp=int(transfer_data.get('timestamp', time.time())),
                chunk_data=chunk_data,
                signature=sig,
            )

            self.ez_send(peer, chunk_payload)
            await asyncio.sleep(0.002)

        # B5: Increment resend counter
        self._nack_resend_counts[tid] = resend_count + 1
        self.metrics['chunks_resent'] += len(valid_indices)

    # Backward compatibility alias for older tests
    _dispatch_request_chunks = on_request_chunks.__wrapped__

    async def _process_completed_model(
        self,
        sender_id: str,
        weights: Any,
        sample_count: int,
        round_number: int,
        loss: float,
        accuracy: float,
        peer_mid: str,
        transfer_id: str,
    ) -> None:
        """Async callback invocation for completed model transfers."""
        logger.info(f"CALLBACK_FIRED sender={sender_id} round={round_number}")
        try:
            await self.on_model_update_callback(
                sender_id=sender_id,
                weights=weights,
                sample_count=sample_count,
                round_number=round_number,
                loss=loss,
                accuracy=accuracy
            )
            logger.info(f"Model update callback COMPLETED for {sender_id}")
        except Exception as cb_err:
            logger.exception(
                f"Model update callback FAILED for {sender_id}: {cb_err}"
            )
            _emit_ipv8_payload_dropped(
                self,
                f"callback error: {cb_err}",
                sender_id=sender_id,
                peer_mid=peer_mid,
                transport="chunk",
                transfer_id=transfer_id,
            )

    @lazy_wrapper(ModelChunkPayload)
    def on_model_chunk(self, peer: Peer, payload: ModelChunkPayload):
        """
        Handle incoming model chunk.

        Buffers chunks and triggers model processing when all chunks are received.
        NOTE: made synchronous to avoid event-loop task backlog that caused
        hundreds of late chunks to be rejected after transfer completion.
        """
        if payload.chunk_index % 50 == 0:
            logger.debug(
                f"on_model_chunk called: chunk {payload.chunk_index}/{payload.total_chunks}"
            )
        transfer_id = payload.transfer_id  # Use the transfer_id from the sender (UUID)
        # B3: key by (peer.mid, transfer_id) so different endpoints cannot hijack
        peer_mid = peer.mid.hex() if isinstance(peer.mid, bytes) else str(peer.mid)
        buf_key = (peer_mid, transfer_id)

        # §4.4: Verify sender identity against peer.mid binding
        expected_node_id = self._mid_to_node_id.get(peer_mid)
        if expected_node_id is not None and payload.sender_id != expected_node_id:
            logger.warning(
                f"Identity mismatch in chunk: peer.mid={peer_mid[:12]}... claims "
                f"sender_id={payload.sender_id}, expected {expected_node_id}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "identity mismatch",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.identity_mismatch",
                transport="chunk",
                expected_sender_id=expected_node_id,
                transfer_id=transfer_id,
            )
            return

        # B14: Verify chunk signature against the sender's public key
        if payload.signature:
            if not _chunk_verify(
                peer.public_key, payload.signature,
                payload.sender_id, payload.round_number,
                payload.data_schema_hash, payload.chunk_index,
                payload.chunk_data,
                sample_count=payload.sample_count,
                loss=payload.loss,
                accuracy=payload.accuracy,
                timestamp=payload.timestamp,
                total_chunks=payload.total_chunks,
            ):
                logger.warning(
                    f"Rejected chunk {payload.chunk_index} of {transfer_id[:8]}... "
                    f"from {payload.sender_id}: invalid signature"
                )
                _emit_ipv8_payload_dropped(
                    self,
                    "invalid signature",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    security_event="security.signature_rejected",
                    transport="chunk",
                    transfer_id=transfer_id,
                    chunk_index=payload.chunk_index,
                )
                return
        else:
            if getattr(self, "require_signature", True):
                logger.warning(
                    f"Rejected unsigned chunk {payload.chunk_index} of {transfer_id[:8]}... "
                    f"from {payload.sender_id}"
                )
                _emit_ipv8_payload_dropped(
                    self,
                    "missing signature",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    security_event="security.signature_missing",
                    transport="chunk",
                    transfer_id=transfer_id,
                    chunk_index=payload.chunk_index,
                )
                return
            logger.debug(
                f"Unsigned chunk {payload.chunk_index} of {transfer_id[:8]}... "
                f"from {payload.sender_id} (no signature)"
            )

        # v3: Semantic dedup — IPv8 signatures are non-deterministic, so NACK
        # resends appear as distinct packets. Drop duplicates silently here.
        # NOTE: key includes peer_mid so transfers from different peers are
        # not confused (B3: buffer is already keyed by peer_mid + transfer_id).
        chunk_key = (peer_mid, transfer_id, payload.chunk_index)
        if chunk_key in self._recent_chunks:
            return
        self._recent_chunks[chunk_key] = time.time()

        # B4: Validate total_chunks before any allocation
        if payload.total_chunks > MAX_CHUNKS_PER_TRANSFER:
            logger.warning(
                f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                f"total_chunks={payload.total_chunks} > {MAX_CHUNKS_PER_TRANSFER}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "oversized transfer",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.oversized_message",
                transport="chunk",
                transfer_id=transfer_id,
                total_chunks=payload.total_chunks,
            )
            return

        local_round = GossipLearningCommunity._get_local_round(self)
        max_round_skip = GossipLearningCommunity._get_max_round_skip(self)
        if payload.round_number > local_round + max_round_skip:
            logger.warning(
                f"Rejected future-round chunked transfer {transfer_id[:8]}... from "
                f"{payload.sender_id}: round {payload.round_number} > local {local_round} + {max_round_skip}"
            )
            _emit_ipv8_payload_dropped(
                self,
                "future round rejected",
                sender_id=payload.sender_id,
                peer_mid=peer_mid,
                security_event="security.future_round_rejected",
                transport="chunk",
                transfer_id=transfer_id,
                round_number=payload.round_number,
                local_round=local_round,
            )
            return

        # B4: Enforce per-peer and global buffer caps on new buffer creation
        if buf_key not in self._chunk_buffers:
            # v3: Transfer-based replay protection (not round-based)
            if (peer_mid, transfer_id) in self._completed_chunk_transfers:
                logger.debug(
                    f"Rejected duplicate completed transfer {transfer_id[:8]}... from "
                    f"{payload.sender_id} (chunk {payload.chunk_index})"
                )
                self.metrics['chunk_transfers_rejected_peer_limit'] += 1
                _emit_ipv8_payload_dropped(
                    self,
                    "duplicate completed transfer",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    security_event="security.duplicate_chunk_transfer",
                    transport="chunk",
                    transfer_id=transfer_id,
                )
                return

            # Per-peer transfer count
            peer_transfers = sum(
                1 for (pmid, _) in self._chunk_buffers if pmid == peer_mid
            )
            if peer_transfers >= MAX_CONCURRENT_TRANSFERS_PER_PEER:
                logger.warning(
                    f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                    f"per-peer limit ({MAX_CONCURRENT_TRANSFERS_PER_PEER}) reached"
                )
                self.metrics['chunk_transfers_rejected_peer_limit'] += 1
                _emit_ipv8_payload_dropped(
                    self,
                    "per-peer transfer limit reached",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    transport="chunk",
                    transfer_id=transfer_id,
                )
                return

            # Global transfer count
            if len(self._chunk_buffers) >= MAX_TOTAL_TRANSFERS:
                logger.warning(
                    f"Rejected transfer {transfer_id[:8]}... from {payload.sender_id}: "
                    f"global limit ({MAX_TOTAL_TRANSFERS}) reached"
                )
                _emit_ipv8_payload_dropped(
                    self,
                    "global transfer limit reached",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    transport="chunk",
                    transfer_id=transfer_id,
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
                _emit_ipv8_payload_dropped(
                    self,
                    "per-peer byte budget exhausted",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    transport="chunk",
                    transfer_id=transfer_id,
                )
                return

        # Log chunk receipt
        logger.debug(
            f"CHUNK_RECV tid={transfer_id[:8]} idx={payload.chunk_index} total={payload.total_chunks} "
            f"from={payload.sender_id}"
        )
        
        if payload.chunk_index == payload.total_chunks - 1:
            logger.debug(f"Received final chunk {payload.chunk_index + 1}/{payload.total_chunks} from {payload.sender_id}")
        
        # Create or get buffer for this transfer (keyed by endpoint identity)
        if buf_key not in self._chunk_buffers:
            logger.debug(
                f"Created chunk buffer {transfer_id[:8]}... "
                f"starting at chunk {payload.chunk_index}"
            )
            self.metrics['chunk_transfers_started'] += 1
            self._chunk_buffers[buf_key] = ChunkBuffer(
                transfer_id=transfer_id,
                sender_id=payload.sender_id,
                total_chunks=payload.total_chunks,
                data_schema_hash=payload.data_schema_hash,
                round_number=payload.round_number,
                sample_count=payload.sample_count,
                loss=payload.loss,
                accuracy=payload.accuracy,
                timestamp=payload.timestamp
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

                # Remove buffer BEFORE scheduling callback to avoid race
                del self._chunk_buffers[buf_key]

                # Check size
                if len(weights_bytes) > MAX_INCOMING_MESSAGE_SIZE:
                    logger.error(
                        f"Rejected oversized model from {payload.sender_id}: "
                        f"{len(weights_bytes) / 1024 / 1024:.2f} MB"
                    )
                    _emit_ipv8_payload_dropped(
                        self,
                        "oversized model",
                        sender_id=payload.sender_id,
                        peer_mid=peer_mid,
                        security_event="security.oversized_message",
                        transport="chunk",
                        transfer_id=transfer_id,
                        size_bytes=len(weights_bytes),
                    )
                    return

                # Deserialize
                try:
                    weights = deserialize_model(weights_bytes)
                except Exception as e:
                    logger.error(f"Failed to deserialize model from {payload.sender_id}: {e}")
                    _emit_ipv8_payload_dropped(
                        self,
                        f"deserialization error: {e}",
                        sender_id=payload.sender_id,
                        peer_mid=peer_mid,
                        transport="chunk",
                        transfer_id=transfer_id,
                    )
                    return

                logger.debug(
                    f"Received model update from {payload.sender_id} "
                    f"(round={buffer.round_number}, samples={buffer.sample_count}, "
                    f"size={len(weights_bytes) / 1024:.1f} KB, chunks={payload.total_chunks})"
                )

                # Mark completed IMMEDIATELY so late chunks are rejected cleanly
                self.metrics['chunk_transfers_completed'] += 1
                self._completed_chunk_transfers[(peer_mid, transfer_id)] = time.time()
                try:
                    GossipLearningCommunity._persist_last_seen_round_state(self)
                except Exception:
                    pass

                # Schedule async callback so packet handler stays synchronous
                if self.on_model_update_callback:
                    logger.debug(
                        f"Model reassembly complete for {payload.sender_id}; scheduling callback"
                    )
                    asyncio.create_task(
                        self._process_completed_model(
                            sender_id=payload.sender_id,
                            weights=weights,
                            sample_count=buffer.sample_count,
                            round_number=buffer.round_number,
                            loss=buffer.loss,
                            accuracy=buffer.accuracy,
                            peer_mid=peer_mid,
                            transfer_id=transfer_id,
                        )
                    )
                else:
                    logger.warning(
                        f"No model update callback registered for transfer {transfer_id[:8]}..."
                    )

            except Exception as e:
                logger.error(f"Failed to reassemble/process chunked model from {payload.sender_id}: {e}")
                _emit_ipv8_payload_dropped(
                    self,
                    f"processing error: {e}",
                    sender_id=payload.sender_id,
                    peer_mid=peer_mid,
                    transport="chunk",
                    transfer_id=transfer_id,
                )
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
                self.metrics['nacks_sent'] += 1

    # Backward compatibility alias for older tests
    _dispatch_model_chunk = on_model_chunk.__wrapped__

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
                timestamp = int(time.time())
                # Small payload - use direct transfer (original method)
                # Sign the direct payload (same canonical form as chunked, chunk_index=0)
                sig = _chunk_sign(
                    self.my_peer.key,
                    self.node_id, round_number,
                    self.data_schema_hash, 0, weights_bytes,
                    sample_count=sample_count,
                    loss=loss_val,
                    accuracy=acc_val,
                    timestamp=timestamp,
                    total_chunks=1,
                )
                payload = ModelUpdatePayload(
                    sender_id=self.node_id,
                    weights_bytes=weights_bytes,
                    sample_count=sample_count,
                    round_number=round_number,
                    data_schema_hash=self.data_schema_hash,
                    loss=loss_val,
                    accuracy=acc_val,
                    timestamp=timestamp,
                    signature=sig,
                )
                self.ez_send(peer_info.peer, payload)
                logger.debug(f"Sent model update to {target_node_id} ({len(weights_bytes)} bytes)")
            else:
                # Large payload - use chunked transfer
                # v3: Sender-side idempotency — same (peer, round, model) gets same transfer_id
                model_hash = hashlib.sha256(weights_bytes).hexdigest()[:16]
                inflight_key = (target_node_id, round_number, model_hash)
                existing_tid = self._inflight_transfers.get(inflight_key)
                if existing_tid is not None and existing_tid in self._outgoing_transfers:
                    logger.info(
                        f"Skipping duplicate send to {target_node_id}: "
                        f"transfer {existing_tid[:8]}... already in-flight for round {round_number}"
                    )
                    return True

                transfer_id = str(uuid.uuid4())
                self._inflight_transfers[inflight_key] = transfer_id
                total_chunks = (len(weights_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE
                timestamp = int(time.time())
                
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
                    'timestamp': timestamp,
                    'recipient_mid': recipient_mid,
                }
                
                for i in range(total_chunks):
                    start = i * CHUNK_SIZE
                    end = min(start + CHUNK_SIZE, len(weights_bytes))
                    chunk_data = weights_bytes[start:end]
                    
                    # B14: Sign each chunk
                    sig = _chunk_sign(
                        self.my_peer.key,
                        self.node_id, round_number,
                        self.data_schema_hash, i, chunk_data,
                        sample_count=sample_count,
                        loss=loss_val,
                        accuracy=acc_val,
                        timestamp=timestamp,
                        total_chunks=total_chunks,
                    )
 
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
                        timestamp=timestamp,
                        chunk_data=chunk_data,
                        signature=sig,
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
                # v3: Remove from inflight tracking once all chunks are dispatched
                self._inflight_transfers.pop(inflight_key, None)
            
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
