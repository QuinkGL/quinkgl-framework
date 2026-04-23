"""Manifest Exchange Protocol (spec §13).

Three IPv8 payload classes plus a small amount of transport-agnostic
machinery — chunking, reassembly, rate-limiting, registry — that can be
unit-tested without standing up an IPv8 community.  The live ``@lazy_wrapper``
handlers on :class:`GossipLearningCommunity` (§13.5 / §13.6) are thin
adapters that delegate to these primitives.

Wire-format summary (§13.1):

====  =============================  ==========================================
id    Name                           Fields
====  =============================  ==========================================
30    ManifestRequestPayload         swarm_id_hex, request_nonce
31    ManifestResponseChunkPayload   swarm_id_hex, request_nonce,
                                     chunk_index, total_chunks, chunk_data
32    ManifestResponseNackPayload    swarm_id_hex, request_nonce, reason
====  =============================  ==========================================

Chunk payloads carry at most :data:`CHUNK_DATA_SIZE` bytes each so a full
request/response exchange fits comfortably inside UDP MTU (~1400 bytes on
most residential paths, minus IPv8 packet headers).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from ipv8.messaging.payload import Payload

from quinkgl.manifest.errors import (
    ERR_MANIFEST_HASH_MISMATCH,
    ERR_WIRE_CHUNK_INCONSISTENT,
)
from quinkgl.manifest.schema import SwarmManifest

__all__ = [
    "CHUNK_DATA_SIZE",
    "REQUEST_TIMEOUT_SECONDS",
    "NackReason",
    "ManifestRequestPayload",
    "ManifestResponseChunkPayload",
    "ManifestResponseNackPayload",
    "ManifestAssembler",
    "ManifestRegistry",
    "RateLimiter",
    "chunk_manifest_bytes",
]


# Chunk payload body cap in bytes.  Sized conservatively below the worst-case
# UDP path MTU minus IPv8 header overhead (~200 bytes) so a chunk payload
# never triggers IP fragmentation.
CHUNK_DATA_SIZE = 1200

# Total per-exchange wall-clock budget (§13.6 step 6).
REQUEST_TIMEOUT_SECONDS = 30.0

# Server-side fairness: at most 4 requests per peer per minute (§13.5 step 3).
DEFAULT_RATE_LIMIT = 4
DEFAULT_RATE_WINDOW_SECONDS = 60.0


class NackReason(str, Enum):
    """Reasons a server may refuse to serve a manifest (§13.4)."""

    UNKNOWN_SWARM = "UNKNOWN_SWARM"
    RATE_LIMITED = "RATE_LIMITED"
    TEMPORARY = "TEMPORARY"


# ---------------------------------------------------------------------------
# Payload classes — IPv8-compatible, but usable as plain dataclasses too
# ---------------------------------------------------------------------------


class ManifestRequestPayload(Payload):
    """Peer-to-peer manifest request (msg_id=30, §13.2)."""

    msg_id = 30
    format_list = ["varlenH", "Q"]

    def __init__(self, swarm_id_hex: str, request_nonce: int):
        super().__init__()
        self.swarm_id_hex = swarm_id_hex
        self.request_nonce = int(request_nonce)

    def to_pack_list(self):
        return [
            ("varlenH", self.swarm_id_hex.encode("utf-8")),
            ("Q", self.request_nonce),
        ]

    @classmethod
    def from_unpack_list(cls, swarm_id_hex, request_nonce):
        return cls(swarm_id_hex.decode("utf-8"), int(request_nonce))


class ManifestResponseChunkPayload(Payload):
    """One chunk of a manifest response (msg_id=31, §13.3)."""

    msg_id = 31
    format_list = ["varlenH", "Q", "I", "I", "varlenH"]

    def __init__(
        self,
        swarm_id_hex: str,
        request_nonce: int,
        chunk_index: int,
        total_chunks: int,
        chunk_data: bytes,
    ):
        super().__init__()
        if chunk_index < 0 or total_chunks <= 0 or chunk_index >= total_chunks:
            raise ValueError(
                f"invalid chunk indices: index={chunk_index} "
                f"total={total_chunks}"
            )
        if len(chunk_data) > CHUNK_DATA_SIZE:
            raise ValueError(
                f"chunk_data too large: {len(chunk_data)} > {CHUNK_DATA_SIZE}"
            )
        self.swarm_id_hex = swarm_id_hex
        self.request_nonce = int(request_nonce)
        self.chunk_index = int(chunk_index)
        self.total_chunks = int(total_chunks)
        self.chunk_data = chunk_data

    def to_pack_list(self):
        return [
            ("varlenH", self.swarm_id_hex.encode("utf-8")),
            ("Q", self.request_nonce),
            ("I", self.chunk_index),
            ("I", self.total_chunks),
            ("varlenH", self.chunk_data),
        ]

    @classmethod
    def from_unpack_list(
        cls, swarm_id_hex, request_nonce, chunk_index, total_chunks, chunk_data
    ):
        return cls(
            swarm_id_hex.decode("utf-8"),
            int(request_nonce),
            int(chunk_index),
            int(total_chunks),
            bytes(chunk_data),
        )


class ManifestResponseNackPayload(Payload):
    """Negative acknowledgement for an unserviceable request (msg_id=32, §13.4)."""

    msg_id = 32
    format_list = ["varlenH", "Q", "varlenH"]

    def __init__(self, swarm_id_hex: str, request_nonce: int, reason):
        super().__init__()
        # Normalise + validate the reason against the enum so a typo at the
        # call-site doesn't get silently encoded into bytes that no client
        # knows how to interpret.
        if isinstance(reason, NackReason):
            reason_value = reason
        else:
            try:
                reason_value = NackReason(reason)
            except ValueError as exc:
                raise ValueError(
                    f"invalid NACK reason {reason!r}; expected one of "
                    f"{[r.value for r in NackReason]}"
                ) from exc
        self.swarm_id_hex = swarm_id_hex
        self.request_nonce = int(request_nonce)
        self.reason = reason_value

    def to_pack_list(self):
        return [
            ("varlenH", self.swarm_id_hex.encode("utf-8")),
            ("Q", self.request_nonce),
            ("varlenH", self.reason.value.encode("utf-8")),
        ]

    @classmethod
    def from_unpack_list(cls, swarm_id_hex, request_nonce, reason):
        return cls(
            swarm_id_hex.decode("utf-8"),
            int(request_nonce),
            reason.decode("utf-8") if isinstance(reason, (bytes, bytearray)) else reason,
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_manifest_bytes(
    canonical: bytes,
    chunk_size: int = CHUNK_DATA_SIZE,
) -> List[bytes]:
    """Split canonical manifest bytes into ≤``chunk_size`` fragments.

    ``chunk_size`` defaults to the UDP-safe :data:`CHUNK_DATA_SIZE`; callers
    SHOULD only override it in tests.  An empty payload is rejected — a
    zero-chunk exchange would require teaching the client to accept
    ``total_chunks == 0`` and then there's no hash to verify against.
    """
    if not canonical:
        raise ValueError("canonical manifest bytes must be non-empty")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [canonical[i : i + chunk_size] for i in range(0, len(canonical), chunk_size)]


# ---------------------------------------------------------------------------
# Client-side assembler
# ---------------------------------------------------------------------------


@dataclass
class ManifestAssembler:
    """Collect chunks for a single manifest request and verify on finish.

    The assembler is nonce-scoped: callers SHOULD allocate one per outgoing
    ``ManifestRequestPayload`` and discard chunks that arrive under a
    different ``(swarm_id, nonce)`` pair before feeding the rest here.
    """

    swarm_id: bytes
    nonce: int
    _total: Optional[int] = field(default=None, init=False, repr=False)
    _chunks: Dict[int, bytes] = field(default_factory=dict, init=False, repr=False)

    def add_chunk(self, index: int, total: int, data: bytes) -> None:
        """Record one chunk.  ``total`` MUST stay constant across chunks."""
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        if self._total is None:
            self._total = total
        elif self._total != total:
            raise ValueError(
                ERR_WIRE_CHUNK_INCONSISTENT,
                {
                    "detail": "total_chunks drifted mid-exchange",
                    "expected": self._total,
                    "got": total,
                },
            )
        if index < 0 or index >= total:
            raise ValueError(
                ERR_WIRE_CHUNK_INCONSISTENT,
                {"detail": "chunk_index out of range", "index": index, "total": total},
            )
        # Duplicate deliveries are tolerated as long as the payload matches —
        # UDP loves to resend.  Contradictory payloads for the same index
        # are a protocol violation.
        existing = self._chunks.get(index)
        if existing is not None and existing != data:
            raise ValueError(
                ERR_WIRE_CHUNK_INCONSISTENT,
                {"detail": "duplicate chunk with differing data", "index": index},
            )
        self._chunks[index] = data

    def is_complete(self) -> bool:
        return self._total is not None and len(self._chunks) == self._total

    def assemble_and_verify(self) -> bytes:
        """Concatenate in index order and check the SHA-256 against ``swarm_id``.

        Raises ``ERR_MANIFEST_HASH_MISMATCH`` if any byte was flipped in
        flight.  This is the spec-mandated integrity gate (§13.6 step 4).
        """
        if not self.is_complete():
            raise ValueError(
                ERR_WIRE_CHUNK_INCONSISTENT,
                {
                    "detail": "assembly attempted before all chunks arrived",
                    "have": len(self._chunks),
                    "need": self._total,
                },
            )
        assembled = b"".join(self._chunks[i] for i in range(self._total or 0))
        digest = hashlib.sha256(assembled).digest()
        if digest != self.swarm_id:
            raise ValueError(
                ERR_MANIFEST_HASH_MISMATCH,
                {
                    "detail": "assembled manifest bytes do not hash to swarm_id",
                    "expected": self.swarm_id.hex(),
                    "actual": digest.hex(),
                },
            )
        return assembled


# ---------------------------------------------------------------------------
# Server-side registry + rate limiting
# ---------------------------------------------------------------------------


class ManifestRegistry:
    """In-process store of manifests this peer is willing to serve.

    Index is by hex ``swarm_id`` — the exact string that arrives on the wire
    in ``ManifestRequestPayload.swarm_id_hex`` — to avoid doing bytes/hex
    conversions on every lookup.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, bytes] = {}

    def register(self, manifest: SwarmManifest) -> str:
        canonical = manifest.canonical_bytes()
        swarm_id_hex = manifest.manifest_hash()
        self._by_id[swarm_id_hex] = canonical
        return swarm_id_hex

    def register_bytes(self, swarm_id_hex: str, canonical: bytes) -> None:
        """Bypass :class:`SwarmManifest` — useful for the loader which already
        has raw bytes in hand and hasn't yet parsed them."""
        self._by_id[swarm_id_hex] = canonical

    def get(self, swarm_id_hex: str) -> Optional[bytes]:
        return self._by_id.get(swarm_id_hex)

    def __contains__(self, swarm_id_hex: str) -> bool:
        return swarm_id_hex in self._by_id


class RateLimiter:
    """Per-peer sliding-window request rate limiter (§13.5 step 3).

    Default policy: 4 requests per 60 seconds per source peer.  Exceeding
    the limit returns ``False`` so the caller can emit
    ``NackReason.RATE_LIMITED``.
    """

    def __init__(
        self,
        *,
        limit: int = DEFAULT_RATE_LIMIT,
        window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._history: Dict[str, List[float]] = {}

    def allow(self, peer_key: str) -> bool:
        now = self._clock()
        bucket = self._history.setdefault(peer_key, [])
        cutoff = now - self.window_seconds
        # Drop stale entries in-place so unbounded memory growth is
        # impossible even under adversarial traffic.
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True
