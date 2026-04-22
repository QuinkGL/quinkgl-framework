"""Auto-discovery orchestrator and multi-swarm manifest registry (spec §18).

Spec §18.1 flow:

1. Compute the caller's :class:`DataFingerprint`.
2. Query the local :class:`SwarmDirectoryCommunity` cache.
3. Verify each ad's signature and score affinity.
4. Sort descending, truncate to top-K.
5. Load each winner's manifest and join.

This module is intentionally transport-free: step 5's manifest loader is
an injected async callable, so the orchestrator is unit-testable without
an IPv8 reactor.  The live :class:`GossipNode.discover_and_join` method
is a thin wrapper that delegates here.

:class:`ManifestRegistry` is the §18.3 multi-swarm routing scaffold.
The API is deliberately minimal — "register a 32-byte swarm id, look it
up, route a packet" — since the full per-packet demux design lives in a
Phase 3 follow-up ticket.  What we ship today lets a host process wire
several communities onto the same IPv8 endpoint without colliding.
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from quinkgl.fingerprint import DataFingerprint
from quinkgl.manifest.errors import ERR_WIRE_UNKNOWN_SWARM
from quinkgl.network.directory import (
    SwarmAdvertisement,
    SwarmDirectoryCommunity,
    verify_advertisement,
)

__all__ = [
    "rank_candidates",
    "discover_and_join",
    "ManifestRegistry",
]


# --- Affinity ranking (§18.1 steps 2–4) -----------------------------------


def _score_ad(
    fingerprint: DataFingerprint,
    ad: SwarmAdvertisement,
) -> Optional[float]:
    """Return the affinity score for ``ad`` against ``fingerprint``.

    Returns ``None`` when the advertisement's reference fingerprint is
    unusable (unknown schema, malformed payload, etc.) so the caller can
    skip rather than treating a decode error as score=0.
    """
    try:
        other = DataFingerprint.from_dict(ad.reference_fingerprint)
    except Exception:
        return None
    try:
        return float(fingerprint.affinity_score(other))
    except Exception:
        return None


def rank_candidates(
    *,
    directory: SwarmDirectoryCommunity,
    fingerprint: DataFingerprint,
    tags: Optional[List[str]] = None,
    input_shape: Optional[List[int]] = None,
    label_type: Optional[str] = None,
    min_affinity: float = 0.5,
    max_swarms: Optional[int] = None,
    trust_policy: str = "open",
    trusted_creator_pubkeys: Optional[Set[bytes]] = None,
) -> List[Tuple[float, SwarmAdvertisement]]:
    """Filter + rank directory advertisements against a local fingerprint.

    Implements §18.1 steps 2–4.  Trust policies:

    * ``"open"``: accept any signed advertisement (signature is still
      verified — the "open" relaxes *who* may sign, not *whether*).
    * ``"pinned"``: the ad's ``creator_pubkey`` MUST be in
      ``trusted_creator_pubkeys`` AND the signature must verify.

    ``tags`` / ``input_shape`` / ``label_type`` are AND-combined
    pre-filters matching :meth:`SwarmDirectoryCommunity.query`.  They
    run against the directory before any fingerprint math so we only
    score a manageable handful of ads even when the cache is full.
    """
    if trust_policy not in {"open", "pinned"}:
        raise ValueError(
            f"trust_policy must be 'open' or 'pinned', got {trust_policy!r}"
        )
    if trust_policy == "pinned" and not trusted_creator_pubkeys:
        raise ValueError(
            "trust_policy='pinned' requires a non-empty trusted_creator_pubkeys set"
        )

    # Honour the directory's own trust filter when the caller pins creators
    # so the community-level signature check runs once per ad instead of
    # once per fingerprint scoring loop.
    trusted_for_query = (
        trusted_creator_pubkeys if trust_policy == "pinned" else None
    )
    prefiltered = directory.query(
        tags=tags,
        input_shape=input_shape,
        label_type=label_type,
        trusted_creators=trusted_for_query,
    )

    scored: List[Tuple[float, SwarmAdvertisement]] = []
    for ad in prefiltered:
        # Defence in depth: directory.query already verified signatures
        # under ``pinned``, but the ``open`` path hands us the raw cache
        # so we re-check here instead of trusting whatever happened to
        # be ingested earlier.
        if not verify_advertisement(ad):
            continue
        score = _score_ad(fingerprint, ad)
        if score is None:
            continue
        if score < min_affinity:
            continue
        scored.append((score, ad))

    # Stable descending sort: highest affinity first, ties keep
    # insertion order so tests stay deterministic on equal-score ads.
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if max_swarms is not None:
        if max_swarms < 0:
            raise ValueError(f"max_swarms must be >= 0, got {max_swarms}")
        scored = scored[:max_swarms]
    return scored


# --- Orchestrator (§18.1 step 5) ------------------------------------------


ManifestLoader = Callable[[SwarmAdvertisement], Awaitable[Any]]


async def discover_and_join(
    *,
    directory: SwarmDirectoryCommunity,
    fingerprint: DataFingerprint,
    manifest_loader: ManifestLoader,
    tags: Optional[List[str]] = None,
    input_shape: Optional[List[int]] = None,
    label_type: Optional[str] = None,
    min_affinity: float = 0.5,
    max_swarms: int = 1,
    trust_policy: str = "open",
    trusted_creator_pubkeys: Optional[Set[bytes]] = None,
) -> List[Any]:
    """Run the full §18.1 auto-discovery flow.

    ``manifest_loader`` is an injection seam: an async callable that
    takes the winning :class:`SwarmAdvertisement` and returns whatever
    object the caller wants to treat as "joined swarm state" —
    typically a :class:`SwarmManifest`.  In production, a GossipNode's
    implementation will issue a manifest exchange (§13) against
    ``ad.swarm_id_hex`` and materialise the manifest locally.

    Loader exceptions are tolerated per-candidate: a transient fetch
    failure for one ad MUST NOT block the rest of the top-K list.
    Exceptions are logged-and-swallowed via the standard library's
    ``logging`` module so callers get visibility without the
    orchestrator deciding on a retry policy of its own.
    """
    import logging

    candidates = rank_candidates(
        directory=directory,
        fingerprint=fingerprint,
        tags=tags,
        input_shape=input_shape,
        label_type=label_type,
        min_affinity=min_affinity,
        max_swarms=max_swarms,
        trust_policy=trust_policy,
        trusted_creator_pubkeys=trusted_creator_pubkeys,
    )

    logger = logging.getLogger(__name__)
    joined: List[Any] = []
    for score, ad in candidates:
        try:
            manifest = await manifest_loader(ad)
        except Exception as exc:
            logger.warning(
                "discover_and_join: manifest loader failed for swarm %s "
                "(score=%.3f): %s",
                ad.swarm_id_hex,
                score,
                exc,
            )
            continue
        joined.append(manifest)
    return joined


# --- ManifestRegistry (§18.3) ---------------------------------------------


class ManifestRegistry:
    """Per-swarm-id routing table for multi-swarm hosts.

    A single process that participates in several swarms needs a way to
    demultiplex incoming packets by ``swarm_id``.  The full per-packet
    design is a Phase 3 follow-up ticket; this class is the thin
    scaffolding everyone else can rely on today — it accepts 32-byte
    swarm ids (the raw form of ``swarm_id_hex``), maps them to
    community objects, and exposes a ``route()`` helper that invokes a
    named handler method on the registered community.

    The registry is deliberately not thread-safe: ownership lives on a
    single event loop, same as the IPv8 community it fronts.
    """

    def __init__(self) -> None:
        self._entries: Dict[bytes, Any] = {}

    @staticmethod
    def _normalise_swarm_id(swarm_id: bytes) -> bytes:
        if not isinstance(swarm_id, (bytes, bytearray)):
            raise TypeError(
                f"swarm_id must be bytes, got {type(swarm_id).__name__}"
            )
        if len(swarm_id) != 32:
            raise ValueError(
                f"swarm_id must be 32 bytes (SHA-256 of canonical manifest), "
                f"got {len(swarm_id)}"
            )
        return bytes(swarm_id)

    def register(self, swarm_id: bytes, community: Any) -> None:
        """Register ``community`` under ``swarm_id``.  Replaces any prior
        registration for the same id so a node can hot-swap a stale
        community instance without first unregistering."""
        key = self._normalise_swarm_id(swarm_id)
        self._entries[key] = community

    def unregister(self, swarm_id: bytes) -> Optional[Any]:
        """Remove ``swarm_id`` from the registry and return the prior
        community, or ``None`` if nothing was registered."""
        key = self._normalise_swarm_id(swarm_id)
        return self._entries.pop(key, None)

    def get(self, swarm_id: bytes) -> Optional[Any]:
        key = self._normalise_swarm_id(swarm_id)
        return self._entries.get(key)

    def __contains__(self, swarm_id: Any) -> bool:
        try:
            key = self._normalise_swarm_id(swarm_id)
        except (TypeError, ValueError):
            return False
        return key in self._entries

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def swarm_ids(self) -> List[bytes]:
        """Return a snapshot of the registered swarm ids."""
        return list(self._entries.keys())

    def route(
        self,
        swarm_id: bytes,
        *,
        packet: Any,
        handler: str = "on_packet",
    ) -> Any:
        """Dispatch ``packet`` to the community registered for ``swarm_id``.

        Raises ``ValueError(ERR_WIRE_UNKNOWN_SWARM, …)`` when no
        community is registered — matches the wire-level error taxonomy
        so upstream transports can surface the same code regardless of
        whether the miss happened at the registry or mid-handshake.
        """
        key = self._normalise_swarm_id(swarm_id)
        community = self._entries.get(key)
        if community is None:
            raise ValueError(
                ERR_WIRE_UNKNOWN_SWARM,
                {
                    "detail": "no community registered for swarm_id",
                    "swarm_id_hex": key.hex(),
                },
            )
        method = getattr(community, handler, None)
        if method is None or not callable(method):
            raise AttributeError(
                f"registered community for swarm {key.hex()!r} does not "
                f"implement {handler!r}"
            )
        return method(packet)
