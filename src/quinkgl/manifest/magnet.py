"""Magnet URI grammar for Quinkgl swarms (spec §8).

A ``quinkgl:?…`` magnet URI encodes the minimum information required for a
peer to locate a swarm's manifest without having the manifest itself in
hand:

* ``xt=urn:qgl:<64 lowercase hex>`` — REQUIRED, exactly once.  This is
  ``SHA-256(canonical_bytes(manifest))`` and doubles as the ``swarm_id``.
* ``dn=<UTF-8 display name>`` — OPTIONAL, ≤ 1.
* ``kw=<tag>(,<tag>)*`` — OPTIONAL, ≤ 1 (a single comma-joined list).
* ``v=<protocol version integer>`` — OPTIONAL, ≤ 1 (defaults to ``1``).
* ``tr=<tracker URL>`` — OPTIONAL, repeatable.
* ``bs=<bootstrap host:port>`` — OPTIONAL, repeatable.

Unknown parameters are silently ignored (forward compatibility).  Error
cases map to the ``ERR_MAGNET_*`` codes in §19.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote, unquote

from quinkgl.manifest.errors import (
    ERR_MAGNET_DUPLICATE,
    ERR_MAGNET_SCHEME,
    ERR_MAGNET_XT,
)

__all__ = ["MagnetLink", "parse_magnet", "format_magnet"]


_SCHEME = "quinkgl:?"
_XT_URN_PREFIX = "urn:qgl:"
# Characters safe in percent-encoded values — we keep the URL-path-safe set
# so that tracker URLs containing ``?``, ``=``, ``/`` get escaped predictably
# and the resulting magnet URI is unambiguous to parse.
_QUOTE_SAFE = ""


def _raise(code: str, **ctx: object) -> None:
    raise ValueError(code, ctx)


@dataclass
class MagnetLink:
    """Parsed representation of a ``quinkgl:?…`` magnet URI (§8.2)."""

    swarm_id: bytes = b""  # 32 bytes (SHA-256)
    display_name: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    trackers: List[str] = field(default_factory=list)
    bootstrap_peers: List[str] = field(default_factory=list)
    protocol_version: int = 1


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_magnet(uri: str) -> MagnetLink:
    """Parse a ``quinkgl:?…`` URI into a :class:`MagnetLink`.

    Raises ``ValueError(ERR_MAGNET_SCHEME, …)`` for non-``quinkgl:?`` inputs
    and ``ERR_MAGNET_XT`` / ``ERR_MAGNET_DUPLICATE`` for structural errors.
    """
    if not isinstance(uri, str) or not uri.startswith(_SCHEME):
        _raise(ERR_MAGNET_SCHEME, detail="URI must start with 'quinkgl:?'", uri=uri)
    query = uri[len(_SCHEME) :]

    # Manually split on '&' (``urllib.parse.parse_qsl`` would silently
    # collapse duplicates and happily accept ``?dn=a&dn=b``; we MUST detect
    # duplicates of ≤1 params to emit ``ERR_MAGNET_DUPLICATE``).
    pairs: List[tuple[str, str]] = []
    if query:
        for raw in query.split("&"):
            if not raw:
                continue
            if "=" not in raw:
                # Tolerate bare keys (e.g. trailing ``&``) — skip silently.
                continue
            key, _, value = raw.partition("=")
            pairs.append((key, value))

    xt_values: List[str] = []
    dn_values: List[str] = []
    kw_values: List[str] = []
    v_values: List[str] = []
    tr_values: List[str] = []
    bs_values: List[str] = []
    for key, value in pairs:
        if key == "xt":
            xt_values.append(value)
        elif key == "dn":
            dn_values.append(value)
        elif key == "kw":
            kw_values.append(value)
        elif key == "v":
            v_values.append(value)
        elif key == "tr":
            tr_values.append(value)
        elif key == "bs":
            bs_values.append(value)
        # Unknown params: ignore (§8.1 forward-compat).

    # xt: REQUIRED, exactly 1.
    if len(xt_values) == 0:
        _raise(ERR_MAGNET_XT, detail="magnet is missing required `xt` parameter")
    if len(xt_values) > 1:
        _raise(
            ERR_MAGNET_XT,
            detail="`xt` must appear exactly once",
            count=len(xt_values),
        )
    swarm_id = _parse_xt(xt_values[0])

    # dn / kw / v: OPTIONAL, ≤ 1 each.
    if len(dn_values) > 1:
        _raise(ERR_MAGNET_DUPLICATE, param="dn", count=len(dn_values))
    if len(kw_values) > 1:
        _raise(ERR_MAGNET_DUPLICATE, param="kw", count=len(kw_values))
    if len(v_values) > 1:
        _raise(ERR_MAGNET_DUPLICATE, param="v", count=len(v_values))

    display_name = unquote(dn_values[0]) if dn_values else None
    keywords = (
        [t for t in unquote(kw_values[0]).split(",") if t]
        if kw_values
        else []
    )
    protocol_version = int(v_values[0]) if v_values else 1
    trackers = [unquote(v) for v in tr_values]
    bootstrap_peers = [unquote(v) for v in bs_values]

    return MagnetLink(
        swarm_id=swarm_id,
        display_name=display_name,
        keywords=keywords,
        trackers=trackers,
        bootstrap_peers=bootstrap_peers,
        protocol_version=protocol_version,
    )


def _parse_xt(value: str) -> bytes:
    if not value.startswith(_XT_URN_PREFIX):
        _raise(
            ERR_MAGNET_XT,
            detail="`xt` must start with 'urn:qgl:'",
            value=value,
        )
    hex_part = value[len(_XT_URN_PREFIX) :]
    if len(hex_part) != 64 or any(c not in "0123456789abcdef" for c in hex_part):
        _raise(
            ERR_MAGNET_XT,
            detail="`xt` payload must be 64 lowercase hex characters",
            value=value,
        )
    return bytes.fromhex(hex_part)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_magnet(link: MagnetLink) -> str:
    """Render a :class:`MagnetLink` in canonical form.

    Canonical parameter order (MUST match round-trip tests): ``xt``,
    ``dn``, ``kw``, ``v``, ``tr*``, ``bs*``.  Each value is percent-encoded
    per RFC 3986.
    """
    if not isinstance(link.swarm_id, (bytes, bytearray)) or len(link.swarm_id) != 32:
        _raise(
            ERR_MAGNET_XT,
            detail="MagnetLink.swarm_id must be exactly 32 bytes",
            got=len(link.swarm_id) if isinstance(link.swarm_id, (bytes, bytearray)) else None,
        )

    parts: List[str] = [f"xt={_XT_URN_PREFIX}{link.swarm_id.hex()}"]
    if link.display_name is not None:
        parts.append(f"dn={quote(link.display_name, safe=_QUOTE_SAFE)}")
    if link.keywords:
        # Commas are preserved literally — they separate tags inside a
        # single ``kw`` value per §8.1.
        encoded_tags = ",".join(quote(tag, safe=_QUOTE_SAFE) for tag in link.keywords)
        parts.append(f"kw={encoded_tags}")
    # Spec default is 1; still emit it if the user explicitly set a
    # non-default (or any value) so round-trip is lossless.
    if link.protocol_version != 1:
        parts.append(f"v={int(link.protocol_version)}")
    for tracker in link.trackers:
        parts.append(f"tr={quote(tracker, safe=_QUOTE_SAFE)}")
    for peer in link.bootstrap_peers:
        parts.append(f"bs={quote(peer, safe=_QUOTE_SAFE)}")
    return _SCHEME + "&".join(parts)
