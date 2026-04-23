# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Trust-on-first-use (TOFU) creator-key cache (spec §15.1).

Responsibilities
----------------
* Persist the ``swarm_id → creator_pubkey`` mapping seen the first time
  a swarm is joined with ``trust_policy="tofu"``.
* On every subsequent join, compare the incoming ``creator_pubkey``
  against the cached entry and raise :data:`ERR_TRUST_TOFU_CONFLICT`
  when they disagree — the swarm has been re-published under a new
  (potentially attacker-controlled) key.
* Commit each update atomically so a crash mid-write cannot leave the
  cache in a partially-serialised, unparseable state.

The cache is a single JSON document; the file lives at
``$IPV8_WORK_DIR/tofu_creators.json`` per spec §15.1.  Multi-process
safety is NOT a goal — QuinkGL peers do not share a work dir — but
coarse threading safety is, because a single node can receive manifest
responses on the IPv8 reactor thread while the CLI driver updates the
cache from the main thread.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from quinkgl.manifest.errors import ERR_TRUST_TOFU_CONFLICT

__all__ = ["TofuCache", "TofuEntry", "default_tofu_cache_path"]

logger = logging.getLogger("quinkgl.network.tofu")


class TofuEntry:
    """A single cached ``swarm_id → creator_pubkey`` binding.

    Kept as a small typed class rather than a dataclass so we can JSON
    round-trip without importing dataclasses at the serialisation layer.
    """

    __slots__ = ("creator_pubkey", "first_seen")

    def __init__(self, creator_pubkey: str, first_seen: str):
        self.creator_pubkey = creator_pubkey
        self.first_seen = first_seen

    def to_dict(self) -> Dict[str, str]:
        return {
            "creator_pubkey": self.creator_pubkey,
            "first_seen": self.first_seen,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TofuEntry":
        pk = data.get("creator_pubkey")
        first = data.get("first_seen")
        if not isinstance(pk, str) or not isinstance(first, str):
            raise ValueError(
                "TOFU cache entry missing required str fields 'creator_pubkey' / 'first_seen'"
            )
        return cls(creator_pubkey=pk, first_seen=first)


def default_tofu_cache_path(work_dir: Optional[str] = None) -> Path:
    """Return the canonical TOFU cache location.

    Resolution order (spec §15.1):
    1. Explicit ``work_dir`` passed by the caller.
    2. ``$IPV8_WORK_DIR`` environment variable.
    3. ``./.quinkgl`` relative to the current working directory (matches
       the default ``--work-dir`` used by ``quinkgl run``).
    """
    base = work_dir or os.environ.get("IPV8_WORK_DIR") or ".quinkgl"
    return Path(base) / "tofu_creators.json"


class TofuCache:
    """Persistent, thread-safe TOFU creator-key cache.

    The class is a thin wrapper over a JSON document on disk.  Reads
    lazy-load the document into memory; writes serialise the entire
    document through a tmp+fsync+rename pair so readers either see the
    previous fully-valid snapshot or the new one — never a torn byte
    stream.

    Instances are safe to share between coroutines and threads (a
    ``threading.Lock`` serialises mutations), but an instance per
    process is the normal pattern.
    """

    def __init__(self, path: str | os.PathLike):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, TofuEntry]] = None

    @property
    def path(self) -> Path:
        return self._path

    # --- low-level persistence -------------------------------------------------

    def _load_if_needed(self) -> Dict[str, TofuEntry]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Corrupt cache = degrade to empty rather than crash the
            # node.  The operator will see the warning and decide
            # whether to delete / back up the file.
            logger.warning(
                "TOFU cache %s unreadable (%s); starting from empty state",
                self._path,
                exc,
            )
            self._cache = {}
            return self._cache
        if not isinstance(raw, dict):
            logger.warning(
                "TOFU cache %s has non-object root; starting from empty state",
                self._path,
            )
            self._cache = {}
            return self._cache

        out: Dict[str, TofuEntry] = {}
        for swarm_id, entry in raw.items():
            if not isinstance(swarm_id, str) or not isinstance(entry, dict):
                continue
            try:
                out[swarm_id] = TofuEntry.from_dict(entry)
            except ValueError as exc:
                logger.warning(
                    "TOFU cache %s entry %s ignored: %s",
                    self._path,
                    swarm_id,
                    exc,
                )
        self._cache = out
        return self._cache

    def _flush(self) -> None:
        assert self._cache is not None
        data = {sid: entry.to_dict() for sid, entry in sorted(self._cache.items())}
        payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")

        # tmp+fsync+rename: any interrupt before ``os.replace`` leaves
        # the old file exactly where it was.  We stage the tmp file
        # inside the same directory so the rename is guaranteed atomic
        # on POSIX (same filesystem).
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tofu_creators.", suffix=".tmp", dir=self._path.parent
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:  # pragma: no cover — non-POSIX
                    pass
            os.replace(tmp_path, self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:  # pragma: no cover — non-POSIX
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # --- public API ----------------------------------------------------------

    def get(self, swarm_id: str) -> Optional[TofuEntry]:
        """Return the cached entry for ``swarm_id`` or ``None`` when absent."""
        with self._lock:
            return self._load_if_needed().get(swarm_id)

    def record_or_validate(
        self,
        swarm_id: str,
        creator_pubkey: str,
        *,
        now: Optional[datetime] = None,
    ) -> TofuEntry:
        """Cache a first-sight creator key, or enforce the bound one.

        * First encounter of ``swarm_id``: store the mapping and return
          the freshly-inserted :class:`TofuEntry`.
        * Subsequent encounter with the same ``creator_pubkey``: return
          the existing entry unchanged.
        * Subsequent encounter with a *different* ``creator_pubkey``:
          raise :class:`ValueError` carrying
          :data:`ERR_TRUST_TOFU_CONFLICT` and a diagnostic payload.

        Raises ``ValueError`` when either ``swarm_id`` or
        ``creator_pubkey`` is not a non-empty string, to catch
        serialisation bugs loudly rather than letting them corrupt the
        cache.
        """
        if not isinstance(swarm_id, str) or not swarm_id:
            raise ValueError("swarm_id must be a non-empty string")
        if not isinstance(creator_pubkey, str) or not creator_pubkey:
            raise ValueError("creator_pubkey must be a non-empty string")

        with self._lock:
            cache = self._load_if_needed()
            existing = cache.get(swarm_id)
            if existing is None:
                ts = (now or datetime.now(timezone.utc)).replace(microsecond=0)
                entry = TofuEntry(
                    creator_pubkey=creator_pubkey,
                    first_seen=ts.isoformat().replace("+00:00", "Z"),
                )
                cache[swarm_id] = entry
                self._flush()
                return entry
            if existing.creator_pubkey == creator_pubkey:
                return existing
            # Conflict — do NOT overwrite; surface to the caller.
            raise ValueError(
                ERR_TRUST_TOFU_CONFLICT,
                {
                    "detail": (
                        "creator_pubkey for this swarm does not match the "
                        "TOFU-cached binding from first encounter"
                    ),
                    "swarm_id": swarm_id,
                    "expected": existing.creator_pubkey,
                    "actual": creator_pubkey,
                    "first_seen": existing.first_seen,
                },
            )

    def forget(self, swarm_id: str) -> bool:
        """Drop ``swarm_id`` from the cache.

        Intended for operator use (``quinkgl trust forget <swarm_id>``)
        when a legitimate rekey is known to be safe.  Returns ``True``
        when an entry was removed, ``False`` when the id was not cached.
        """
        with self._lock:
            cache = self._load_if_needed()
            if swarm_id not in cache:
                return False
            del cache[swarm_id]
            self._flush()
            return True

    def clear(self) -> None:
        """Drop every entry (primarily for tests)."""
        with self._lock:
            self._cache = {}
            if self._path.exists():
                self._flush()

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        """Return a JSON-compatible snapshot (defensive copy)."""
        with self._lock:
            cache = self._load_if_needed()
            return {sid: entry.to_dict() for sid, entry in cache.items()}
