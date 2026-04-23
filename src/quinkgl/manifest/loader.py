"""Unified manifest loader (spec §10.3).

:func:`load_manifest` is the single entry point callers SHOULD use instead
of juggling :meth:`SwarmManifest.from_file`, HTTP clients, and magnet
parsers themselves.  It dispatches on the shape of ``source``:

``quinkgl:?…``
    Magnet URI.  Parses with :func:`parse_magnet`, delegates the actual
    bytes retrieval to the caller-supplied ``peer_fetcher`` (the loader
    has no opinion on how peers are contacted — that's a network concern).
    Verifies the returned bytes hash to the magnet's ``swarm_id``.
``http://`` / ``https://``
    Stdlib ``urllib.request`` GET.  No third-party HTTP dependency.
Anything else
    Filesystem path → :meth:`SwarmManifest.from_file`.

When ``expected_swarm_id`` is supplied, the loader re-hashes the parsed
manifest's canonical bytes and raises ``ERR_MANIFEST_HASH_MISMATCH`` on
disagreement.  This is the integrity gate peers use when fetching a
manifest they already know the ID of (e.g. from a signed announcement).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional
from urllib.request import urlopen as _urlopen_default

from quinkgl.manifest.errors import (
    ERR_MANIFEST_FETCH_REQUIRED,
    ERR_MANIFEST_HASH_MISMATCH,
    ERR_MANIFEST_INVALID_JSON,
)
from quinkgl.manifest.magnet import parse_magnet
from quinkgl.manifest.schema import SwarmManifest

__all__ = ["load_manifest"]


# Indirected so tests can monkey-patch without touching ``urllib``.
_urlopen = _urlopen_default


_PeerFetcher = Callable[[bytes], bytes]


def load_manifest(
    source: Any,
    *,
    peer_fetcher: Optional[_PeerFetcher] = None,
    strict: bool = True,
    expected_swarm_id: Optional[bytes] = None,
    timeout: float = 10.0,
) -> SwarmManifest:
    """Load a :class:`SwarmManifest` from a path, URL, or magnet URI.

    See module docstring for dispatch rules.  ``timeout`` applies only to
    the HTTP fetch path.
    """
    src = str(source) if not isinstance(source, str) else source

    if src.startswith("quinkgl:?"):
        manifest, parsed_id = _load_from_magnet(
            src, peer_fetcher=peer_fetcher, strict=strict
        )
    elif src.startswith("http://") or src.startswith("https://"):
        manifest = _load_from_http(src, strict=strict, timeout=timeout)
        parsed_id = hashlib.sha256(manifest.canonical_bytes()).digest()
    else:
        manifest = SwarmManifest.from_file(source, strict=strict)
        parsed_id = hashlib.sha256(manifest.canonical_bytes()).digest()

    if expected_swarm_id is not None and parsed_id != expected_swarm_id:
        raise ValueError(
            ERR_MANIFEST_HASH_MISMATCH,
            {
                "detail": "loaded manifest hash != expected_swarm_id",
                "expected": expected_swarm_id.hex(),
                "actual": parsed_id.hex(),
            },
        )

    return manifest


# ---------------------------------------------------------------------------
# Magnet path
# ---------------------------------------------------------------------------


def _load_from_magnet(
    uri: str,
    *,
    peer_fetcher: Optional[_PeerFetcher],
    strict: bool,
) -> tuple[SwarmManifest, bytes]:
    link = parse_magnet(uri)
    if peer_fetcher is None:
        raise ValueError(
            ERR_MANIFEST_FETCH_REQUIRED,
            {
                "detail": (
                    "magnet source requires a `peer_fetcher` callable; "
                    "supply one that resolves swarm_id → canonical bytes"
                ),
                "swarm_id": link.swarm_id.hex(),
            },
        )
    raw = peer_fetcher(link.swarm_id)
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(
            ERR_MANIFEST_INVALID_JSON,
            {
                "detail": "peer_fetcher must return bytes",
                "got_type": type(raw).__name__,
            },
        )
    fetched_id = hashlib.sha256(bytes(raw)).digest()
    if fetched_id != link.swarm_id:
        raise ValueError(
            ERR_MANIFEST_HASH_MISMATCH,
            {
                "detail": (
                    "peer_fetcher returned bytes whose SHA-256 does not "
                    "match the magnet `xt` value — integrity check failed"
                ),
                "expected": link.swarm_id.hex(),
                "actual": fetched_id.hex(),
            },
        )
    manifest = _parse_bytes(bytes(raw), strict=strict)
    return manifest, fetched_id


# ---------------------------------------------------------------------------
# HTTP path
# ---------------------------------------------------------------------------


def _load_from_http(url: str, *, strict: bool, timeout: float) -> SwarmManifest:
    try:
        response = _urlopen(url, timeout=timeout)
    except Exception as exc:
        # Any transport-layer failure collapses into INVALID_JSON from the
        # caller's POV: they asked for a manifest and got nothing usable.
        raise ValueError(
            ERR_MANIFEST_INVALID_JSON,
            {"detail": f"HTTP fetch failed: {exc}", "url": url},
        ) from exc
    try:
        body = response.read()
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
    return _parse_bytes(body, strict=strict)


# ---------------------------------------------------------------------------
# Shared parsing
# ---------------------------------------------------------------------------


def _parse_bytes(raw: bytes, *, strict: bool) -> SwarmManifest:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(
            ERR_MANIFEST_INVALID_JSON,
            {"detail": "UTF-8 BOM not permitted"},
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            ERR_MANIFEST_INVALID_JSON,
            {"detail": f"not valid UTF-8: {exc}"},
        ) from exc
    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            ERR_MANIFEST_INVALID_JSON,
            {"detail": str(exc), "line": exc.lineno, "col": exc.colno},
        ) from exc
    manifest = SwarmManifest.from_dict(data, strict=strict)
    if strict:
        manifest.validate()
    return manifest
