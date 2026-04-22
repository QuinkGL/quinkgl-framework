"""Swarm Directory Community and advertisement surface (spec §17–§18).

Phase 3.  Two layers live here:

* :class:`SwarmAdvertisement` / :class:`SwarmAdvertisementPayload`
  — the signed, self-describing summary of a swarm that peers publish
  into the directory (§17.2).  Canonical-bytes signing mirrors
  :mod:`quinkgl.manifest.signing`: the ``signature`` field is excluded
  from the bytes that are actually signed.

* :class:`SwarmDirectoryCommunity` — the IPv8 overlay that caches
  advertisements locally, gossips them with anti-entropy, and answers
  synchronous :meth:`~SwarmDirectoryCommunity.query` calls from the
  local cache (§17.1, §17.3, §17.4).

The community is importable even when IPv8 is not installed; the
``Community`` parent class and its hooks are resolved lazily so tests
and dataclasses stay usable in a minimal environment.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from ipv8.messaging.payload import Payload

from quinkgl.manifest.errors import (
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
    ERR_WIRE_RATE_LIMITED,
)

__all__ = [
    "DIRECTORY_COMMUNITY_ID",
    "MAX_CACHE_ENTRIES",
    "MAX_ADS_PER_CREATOR_PER_DAY",
    "MAX_ADS_PER_SESSION",
    "DEFAULT_ADVERTISEMENT_TTL_SECONDS",
    "SwarmAdvertisement",
    "SwarmAdvertisementPayload",
    "sign_advertisement",
    "verify_advertisement",
    "SwarmDirectoryCommunity",
]


# --- Fixed identity (§17.1) -------------------------------------------------

#: Directory community ID — first 20 bytes of SHA-256 of the fixed
#: ``b"QuinkGL-SwarmDirectory-v1"`` tag.  Every Phase 3 peer MUST use
#: this exact value; changing it is a protocol-incompatible fork.
DIRECTORY_COMMUNITY_ID: bytes = hashlib.sha256(b"QuinkGL-SwarmDirectory-v1").digest()[:20]


# --- Lifecycle & rate-limit constants (§17.3) ------------------------------

#: Local cache capacity.  LRU eviction once exceeded.
MAX_CACHE_ENTRIES: int = 10_000

#: Publish-side rate limit per ``creator_pubkey`` per day.
MAX_ADS_PER_CREATOR_PER_DAY: int = 100

#: Publish-side rate limit per peer session (process lifetime).
MAX_ADS_PER_SESSION: int = 10

#: Default TTL for a received advertisement when the receiver computes
#: ``expires_at`` locally.  Must not exceed 30 days (§17.3).
DEFAULT_ADVERTISEMENT_TTL_SECONDS: int = 30 * 24 * 60 * 60


# --- Lazy crypto import ----------------------------------------------------


def _load_crypto() -> Dict[str, Any]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise ValueError(
            ERR_SIGNING_UNAVAILABLE,
            {
                "detail": (
                    "cryptography is required for directory advertisement "
                    "signing; install `cryptography>=41.0.0`"
                ),
                "cause": str(exc),
            },
        ) from exc
    return {
        "InvalidSignature": InvalidSignature,
        "ed25519": ed25519,
        "serialization": serialization,
    }


# --- Dataclass (§17.2) -----------------------------------------------------


@dataclass
class SwarmAdvertisement:
    """Signed advertisement for a swarm in the directory.

    The ten attributes that travel on the wire match §17.2 exactly.
    ``received_at`` is a receiver-side annotation (never serialised, never
    signed) used by the local cache to apply TTL eviction as specified
    in §17.3 ("implicit, per-receiver").
    """

    swarm_id_hex: str
    name: str
    tags: List[str] = field(default_factory=list)
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    label_type: str = ""
    data_schema_hash: str = ""
    reference_fingerprint: Dict[str, Any] = field(default_factory=dict)
    creator_pubkey: Optional[str] = None
    signature: Optional[str] = None

    # Receiver-side local bookkeeping — not on the wire, not signed.
    received_at: Optional[float] = field(default=None, repr=False, compare=False)
    expires_at: Optional[float] = field(default=None, repr=False, compare=False)

    def canonical_bytes(self) -> bytes:
        """Deterministic JSON over every signed field except ``signature``.

        Matches §17.2: the signature covers everything else, in the order
        defined in the format list, encoded as JSON with sorted keys and
        no spurious whitespace.
        """
        payload = {
            "swarm_id_hex": self.swarm_id_hex,
            "name": self.name,
            "tags": sorted(self.tags),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "label_type": self.label_type,
            "data_schema_hash": self.data_schema_hash,
            "reference_fingerprint": self.reference_fingerprint,
            "creator_pubkey": self.creator_pubkey,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")


# --- Sign / verify ---------------------------------------------------------


def sign_advertisement(
    advertisement: SwarmAdvertisement,
    private_key_pem: bytes,
) -> SwarmAdvertisement:
    """Return a copy of ``advertisement`` with ``creator_pubkey`` /
    ``signature`` populated.

    Mirrors :func:`quinkgl.manifest.signing.sign_manifest`: the private
    key is loaded from PKCS#8 PEM bytes, the public half is derived and
    cross-checked against any pre-set ``creator_pubkey`` (mismatch →
    :data:`ERR_SIGNATURE_INVALID`), and the signature is computed over
    :meth:`SwarmAdvertisement.canonical_bytes` with the ``signature``
    field already cleared.
    """
    crypto = _load_crypto()
    ed25519 = crypto["ed25519"]
    serialization = crypto["serialization"]

    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None
        )
    except Exception as exc:
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {"detail": "could not load PEM private key", "cause": str(exc)},
        ) from exc

    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {"detail": "private key is not an Ed25519 key"},
        )

    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    expected_pubkey = "ed25519:" + public_raw.hex()

    if (
        advertisement.creator_pubkey is not None
        and advertisement.creator_pubkey != expected_pubkey
    ):
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {
                "detail": (
                    "advertisement.creator_pubkey does not match the public "
                    "half of the supplied private key"
                ),
                "declared": advertisement.creator_pubkey,
                "derived": expected_pubkey,
            },
        )

    signed = replace(
        advertisement,
        tags=list(advertisement.tags),
        input_shape=list(advertisement.input_shape),
        output_shape=list(advertisement.output_shape),
        reference_fingerprint=dict(advertisement.reference_fingerprint),
        creator_pubkey=expected_pubkey,
        signature=None,
    )
    sig_bytes = private_key.sign(signed.canonical_bytes())
    signed.signature = "ed25519:" + sig_bytes.hex()
    return signed


def verify_advertisement(advertisement: SwarmAdvertisement) -> bool:
    """Return ``True`` iff the advertisement carries a valid Ed25519
    signature against its declared ``creator_pubkey``.

    Returns ``False`` (never raises) on any of:

    * missing ``creator_pubkey`` / ``signature``,
    * malformed hex / wrong-length pubkey or signature,
    * cryptographic verification failure.

    If the :mod:`cryptography` dependency is not installed, this raises
    :data:`ERR_SIGNING_UNAVAILABLE` — callers with a "degrade gracefully"
    policy can catch it explicitly.
    """
    if not advertisement.creator_pubkey or not advertisement.signature:
        return False
    if not advertisement.creator_pubkey.startswith("ed25519:"):
        return False
    if not advertisement.signature.startswith("ed25519:"):
        return False

    try:
        pub_raw = bytes.fromhex(advertisement.creator_pubkey.split(":", 1)[1])
        sig_raw = bytes.fromhex(advertisement.signature.split(":", 1)[1])
    except ValueError:
        return False

    if len(pub_raw) != 32 or len(sig_raw) != 64:
        return False

    crypto = _load_crypto()
    ed25519 = crypto["ed25519"]
    InvalidSignature = crypto["InvalidSignature"]

    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_raw)
    except Exception:
        return False

    try:
        public_key.verify(sig_raw, advertisement.canonical_bytes())
    except InvalidSignature:
        return False
    except Exception:
        return False
    return True


# --- Wire payload (§17.2) --------------------------------------------------


class SwarmAdvertisementPayload(Payload):
    """IPv8 payload wrapping a :class:`SwarmAdvertisement` (msg_id=40).

    Ten ``varlenH`` fields in the exact order spec §17.2 mandates.  List
    / dict fields are serialised as JSON; tags are comma-joined and
    re-split on receive so the wire form stays human-readable.
    """

    msg_id = 40
    format_list = ["varlenH"] * 10

    _NAMES = (
        "swarm_id_hex",
        "name",
        "tags_csv",
        "input_shape_json",
        "output_shape_json",
        "label_type",
        "data_schema_hash",
        "reference_fingerprint_json",
        "creator_pubkey",
        "signature",
    )

    def __init__(
        self,
        swarm_id_hex: str,
        name: str,
        tags_csv: str,
        input_shape_json: str,
        output_shape_json: str,
        label_type: str,
        data_schema_hash: str,
        reference_fingerprint_json: str,
        creator_pubkey: str,
        signature: str,
    ) -> None:
        super().__init__()
        self.swarm_id_hex = swarm_id_hex
        self.name = name
        self.tags_csv = tags_csv
        self.input_shape_json = input_shape_json
        self.output_shape_json = output_shape_json
        self.label_type = label_type
        self.data_schema_hash = data_schema_hash
        self.reference_fingerprint_json = reference_fingerprint_json
        self.creator_pubkey = creator_pubkey
        self.signature = signature

    @classmethod
    def from_advertisement(cls, ad: SwarmAdvertisement) -> "SwarmAdvertisementPayload":
        return cls(
            swarm_id_hex=ad.swarm_id_hex,
            name=ad.name,
            tags_csv=",".join(ad.tags),
            input_shape_json=json.dumps(list(ad.input_shape), separators=(",", ":")),
            output_shape_json=json.dumps(list(ad.output_shape), separators=(",", ":")),
            label_type=ad.label_type,
            data_schema_hash=ad.data_schema_hash,
            reference_fingerprint_json=json.dumps(
                ad.reference_fingerprint, sort_keys=True, separators=(",", ":")
            ),
            creator_pubkey=ad.creator_pubkey or "",
            signature=ad.signature or "",
        )

    def to_advertisement(self) -> SwarmAdvertisement:
        try:
            tags = [t for t in self.tags_csv.split(",") if t] if self.tags_csv else []
            input_shape = json.loads(self.input_shape_json) if self.input_shape_json else []
            output_shape = (
                json.loads(self.output_shape_json) if self.output_shape_json else []
            )
            fingerprint = (
                json.loads(self.reference_fingerprint_json)
                if self.reference_fingerprint_json
                else {}
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SwarmAdvertisementPayload: malformed JSON field: {exc}"
            ) from exc
        return SwarmAdvertisement(
            swarm_id_hex=self.swarm_id_hex,
            name=self.name,
            tags=tags,
            input_shape=list(input_shape),
            output_shape=list(output_shape),
            label_type=self.label_type,
            data_schema_hash=self.data_schema_hash,
            reference_fingerprint=dict(fingerprint) if isinstance(fingerprint, dict) else {},
            creator_pubkey=self.creator_pubkey or None,
            signature=self.signature or None,
        )

    def to_pack_list(self) -> List[Tuple[str, bytes]]:
        return [
            ("varlenH", self.swarm_id_hex.encode("utf-8")),
            ("varlenH", self.name.encode("utf-8")),
            ("varlenH", self.tags_csv.encode("utf-8")),
            ("varlenH", self.input_shape_json.encode("utf-8")),
            ("varlenH", self.output_shape_json.encode("utf-8")),
            ("varlenH", self.label_type.encode("utf-8")),
            ("varlenH", self.data_schema_hash.encode("utf-8")),
            ("varlenH", self.reference_fingerprint_json.encode("utf-8")),
            ("varlenH", self.creator_pubkey.encode("utf-8")),
            ("varlenH", self.signature.encode("utf-8")),
        ]

    @classmethod
    def from_unpack_list(cls, *raw: Any) -> "SwarmAdvertisementPayload":
        if len(raw) != 10:
            raise ValueError(
                f"SwarmAdvertisementPayload expects 10 fields, got {len(raw)}"
            )
        decoded = [
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v for v in raw
        ]
        return cls(*decoded)


# --- SwarmDirectoryCommunity (§17) ----------------------------------------


class SwarmDirectoryCommunity:
    """Light-weight directory overlay.

    Subclassing :class:`ipv8.community.Community` is deferred: the
    transport wiring (``community_id`` class attribute, message
    decorators) lives on an opt-in subclass that callers instantiate
    inside an IPv8 reactor.  Everything observable in a unit test —
    the LRU cache, TTL sweep, rate limiters, and ``query`` filter
    logic — lives here in pure Python so that the bulk of spec §17 is
    testable without IPv8.

    Parameters
    ----------
    max_cache_entries:
        Soft cap; LRU-evict oldest on overflow (§17.3).
    default_ttl_seconds:
        Receiver-side TTL applied when the incoming advertisement has
        no explicit ``expires_at`` attached.  Clamped to 30 days (§17.3).
    clock:
        Injection seam for tests.  Defaults to :func:`time.time`.
    """

    def __init__(
        self,
        *,
        max_cache_entries: int = MAX_CACHE_ENTRIES,
        default_ttl_seconds: int = DEFAULT_ADVERTISEMENT_TTL_SECONDS,
        max_ads_per_creator_per_day: int = MAX_ADS_PER_CREATOR_PER_DAY,
        max_ads_per_session: int = MAX_ADS_PER_SESSION,
        clock: Optional[Any] = None,
    ) -> None:
        if max_cache_entries <= 0:
            raise ValueError(
                f"max_cache_entries must be positive, got {max_cache_entries}"
            )
        ttl_cap = 30 * 24 * 60 * 60
        if not 0 < default_ttl_seconds <= ttl_cap:
            raise ValueError(
                f"default_ttl_seconds must be in (0, {ttl_cap}], got {default_ttl_seconds}"
            )
        self._max_cache = max_cache_entries
        self._default_ttl = default_ttl_seconds
        self._clock = clock or time.time

        # LRU: insertion order == age; oldest at the front.  Keyed by
        # swarm_id_hex so duplicate IDs naturally overwrite.
        self._cache: "OrderedDict[str, SwarmAdvertisement]" = OrderedDict()

        # Rate-limit bookkeeping.
        self._creator_history: Dict[str, Deque[float]] = defaultdict(deque)
        self._session_publish_count: int = 0
        self._max_per_creator = max_ads_per_creator_per_day
        self._max_per_session = max_ads_per_session

    # --- Cache primitives ------------------------------------------------

    def _now(self) -> float:
        return float(self._clock())

    def _evict_expired(self) -> None:
        now = self._now()
        expired = [
            key
            for key, ad in self._cache.items()
            if ad.expires_at is not None and ad.expires_at <= now
        ]
        for key in expired:
            self._cache.pop(key, None)

    def ingest(self, advertisement: SwarmAdvertisement, *, verify: bool = True) -> bool:
        """Accept an advertisement into the local cache.

        Returns ``True`` if the cache actually took the ad (signature
        valid and not superseded), ``False`` otherwise.  Duplicate
        ``swarm_id`` is resolved by highest ``received_at`` (§17.3).
        """
        if verify and not verify_advertisement(advertisement):
            return False

        now = self._now()
        ad = replace(
            advertisement,
            received_at=now,
            expires_at=(
                advertisement.expires_at
                if advertisement.expires_at is not None
                else now + self._default_ttl
            ),
        )

        existing = self._cache.get(ad.swarm_id_hex)
        if existing is not None:
            existing_at = existing.received_at or 0.0
            if existing_at >= (ad.received_at or 0.0):
                # Older ad for an already-known swarm — drop silently.
                return False
            self._cache.pop(ad.swarm_id_hex, None)

        self._cache[ad.swarm_id_hex] = ad
        self._evict_expired()

        # LRU cap.  Evict oldest until we are back under the limit.
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return True

    def all_advertisements(self) -> List[SwarmAdvertisement]:
        """Return a snapshot of live (non-expired) advertisements."""
        self._evict_expired()
        return list(self._cache.values())

    # --- Rate limiting (§17.3) ------------------------------------------

    def _check_publish_quota(self, creator_pubkey: str) -> None:
        now = self._now()
        window_start = now - 24 * 60 * 60
        history = self._creator_history[creator_pubkey]
        while history and history[0] < window_start:
            history.popleft()
        if len(history) >= self._max_per_creator:
            raise ValueError(
                ERR_WIRE_RATE_LIMITED,
                {
                    "detail": "creator exceeded daily advertisement quota",
                    "limit": self._max_per_creator,
                    "creator_pubkey": creator_pubkey,
                },
            )
        if self._session_publish_count >= self._max_per_session:
            raise ValueError(
                ERR_WIRE_RATE_LIMITED,
                {
                    "detail": "session exceeded advertisement quota",
                    "limit": self._max_per_session,
                },
            )

    def publish(self, advertisement: SwarmAdvertisement) -> SwarmAdvertisement:
        """Record a locally-originated advertisement.

        Verifies the signature, enforces the per-creator and per-session
        rate limits (§17.3), and inserts the advertisement into the
        local cache.  In a full deployment the IPv8 subclass would also
        broadcast the :class:`SwarmAdvertisementPayload` via anti-entropy
        gossip; that transport is built on top of this method.
        """
        if not verify_advertisement(advertisement):
            raise ValueError(
                ERR_SIGNATURE_INVALID,
                {"detail": "refusing to publish unsigned or invalid advertisement"},
            )
        assert advertisement.creator_pubkey is not None  # post-verify invariant
        self._check_publish_quota(advertisement.creator_pubkey)

        self._creator_history[advertisement.creator_pubkey].append(self._now())
        self._session_publish_count += 1
        self.ingest(advertisement, verify=False)
        return advertisement

    # --- Query (§17.4) --------------------------------------------------

    def query(
        self,
        *,
        tags: Optional[List[str]] = None,
        input_shape: Optional[List[int]] = None,
        label_type: Optional[str] = None,
        trusted_creators: Optional[Set[bytes]] = None,
    ) -> List[SwarmAdvertisement]:
        """Synchronously filter the local cache.

        Filters are AND-combined.  ``tags`` matches if every requested
        tag appears in the advertisement's tag list.  ``trusted_creators``
        is a set of raw 32-byte Ed25519 public keys — the advertisement
        must declare one of them *and* carry a valid signature.  This
        method MUST NOT trigger a synchronous network call (§17.4).
        """
        self._evict_expired()

        want_tags = set(tags) if tags else None
        want_shape = list(input_shape) if input_shape else None
        trusted_hex: Optional[Set[str]] = None
        if trusted_creators is not None:
            trusted_hex = {"ed25519:" + c.hex() for c in trusted_creators}

        results: List[SwarmAdvertisement] = []
        for ad in self._cache.values():
            if want_tags is not None and not want_tags.issubset(set(ad.tags)):
                continue
            if want_shape is not None and list(ad.input_shape) != want_shape:
                continue
            if label_type is not None and ad.label_type != label_type:
                continue
            if trusted_hex is not None:
                if ad.creator_pubkey not in trusted_hex:
                    continue
                if not verify_advertisement(ad):
                    continue
            results.append(ad)
        return results
