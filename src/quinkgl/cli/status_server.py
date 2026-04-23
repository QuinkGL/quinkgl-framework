# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Unix-socket introspection server for ``quinkgl status`` (spec §11.8).

A running peer binds a one-shot JSON server at
``<work-dir>/running/<node-id>.sock``.  ``quinkgl status`` connects,
reads a single newline-terminated JSON blob, and exits — no request
framing, no authentication, no long-lived session.  The socket is a
pure *local* loopback mechanism: file-system permissions (mode 0600)
are the only access control, which matches the spec's "local
introspection only" intent and the fact that the same process also
writes a sibling ``.json`` file for tools that cannot speak unix
sockets (e.g. Windows).

The server is intentionally decoupled from :class:`GossipNode`: it
takes a zero-arg ``state_provider`` callable that returns the state
dict to serialise.  Tests can drive it with a stub provider;
production callers wire it to :func:`build_status_snapshot`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

__all__ = [
    "StatusServer",
    "build_status_snapshot",
    "read_status_from_socket",
]


logger = logging.getLogger("quinkgl.cli.status_server")

# Hard cap on a single state payload.  Status dicts are small in practice
# (<1 KiB) so 64 KiB is pessimistic; it protects both sides against a
# runaway producer trying to OOM the reader.
_MAX_PAYLOAD_BYTES = 64 * 1024


class StatusServer:
    """Async unix-socket server that serves a JSON state snapshot per connect.

    Parameters
    ----------
    socket_path:
        Filesystem path to bind.  The parent directory is created if it
        does not exist; a pre-existing socket file at this path is
        unlinked before ``bind()`` so a previous crashed process leaves
        no stale residue.
    state_provider:
        Zero-arg callable returning the state dict to serialise.  It
        runs inside the asyncio event loop — callers SHOULD keep it
        lightweight (no blocking I/O, no DB calls).  Exceptions raised
        by the provider are caught and surfaced to the client as a
        best-effort error JSON instead of crashing the server.
    """

    def __init__(
        self,
        socket_path: str,
        state_provider: Callable[[], Dict[str, Any]],
    ) -> None:
        self._socket_path = str(socket_path)
        self._state_provider = state_provider
        self._server: Optional[asyncio.base_events.Server] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    async def start(self) -> None:
        """Bind the unix socket and start accepting connections.

        Safe to call once per instance; a second call raises
        ``RuntimeError``.
        """
        if self._server is not None:
            raise RuntimeError("StatusServer already started")

        parent = Path(self._socket_path).parent
        parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError as exc:  # pragma: no cover — rare
                logger.warning(
                    "StatusServer: could not remove stale socket %s: %s",
                    self._socket_path,
                    exc,
                )

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self._socket_path
        )

        # 0600: owner-only read/write.  Treat the socket as a secret
        # handle; anyone who can read it can see the node's live state.
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError:  # pragma: no cover — non-POSIX
            pass

        logger.info("StatusServer listening at %s", self._socket_path)

    async def stop(self) -> None:
        """Close the server and unlink the socket file."""
        if self._server is None:
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        except Exception:  # pragma: no cover — defensive
            pass
        self._server = None
        try:
            if os.path.exists(self._socket_path):
                os.unlink(self._socket_path)
        except OSError:  # pragma: no cover — rare
            pass

    async def __aenter__(self) -> "StatusServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            try:
                state = dict(self._state_provider() or {})
            except Exception as exc:  # pragma: no cover — provider bugs
                logger.exception("StatusServer state_provider raised")
                state = {"error": "state_provider failed", "detail": str(exc)}

            payload = (
                json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            ).encode("utf-8")

            if len(payload) > _MAX_PAYLOAD_BYTES:
                # Truncate rather than refuse: better a partial answer than
                # a silent disconnect that leaves the CLI with no state at all.
                payload = payload[:_MAX_PAYLOAD_BYTES]

            writer.write(payload)
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                # Client hung up mid-write.  Nothing to do.
                return
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass


def read_status_from_socket(socket_path: str, *, timeout: float = 2.0) -> Dict[str, Any]:
    """Client-side reader.

    Synchronous and self-contained so the ``quinkgl status`` command —
    which is not itself an asyncio program — can stay blocking.  Connects
    to the unix socket at ``socket_path``, reads one newline-terminated
    JSON blob, and returns the decoded dict.

    Raises ``FileNotFoundError`` when the socket is missing,
    ``ConnectionRefusedError`` when nothing is listening on the path,
    ``TimeoutError`` when the server does not respond within
    ``timeout`` seconds, and ``ValueError`` when the payload is not
    valid JSON.
    """
    import socket

    if not os.path.exists(socket_path):
        raise FileNotFoundError(f"status socket does not exist: {socket_path}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        chunks = []
        total = 0
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_PAYLOAD_BYTES:
                break
    finally:
        sock.close()

    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not raw:
        raise ValueError(f"status socket at {socket_path} returned empty payload")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"status socket at {socket_path} returned invalid JSON: {exc}"
        ) from exc


def build_status_snapshot(node: Any, *, since: Optional[str] = None) -> Dict[str, Any]:
    """Assemble a §11.8 state dict from a live :class:`GossipNode`.

    Best-effort: fields the node does not expose (e.g. last-exchange age
    while the live gossip stats are behind a lock the caller should not
    hold) fall back to sensible defaults so the CLI output stays
    printable.  ``since`` is threaded in from the caller so the timestamp
    matches the moment the server was started rather than the moment the
    snapshot is taken (which can skew by tens of seconds under load).
    """
    state_enum = getattr(node, "state", None)
    status = getattr(state_enum, "name", None) or str(state_enum or "UNKNOWN")

    manifest = getattr(node, "manifest", None)
    swarm_name = getattr(manifest, "name", "") or ""
    swarm_id_short = ""
    try:
        if manifest is not None:
            swarm_id_short = manifest.manifest_hash()[:12]
    except Exception:  # pragma: no cover — defensive
        swarm_id_short = ""

    gl_node = getattr(node, "gl_node", None)
    current_round = 0
    last_loss = None
    if gl_node is not None:
        current_round = int(getattr(gl_node, "current_round", 0) or 0)
        aggregator = getattr(gl_node, "aggregator", None)
        if aggregator is not None:
            last_loss = getattr(aggregator, "last_loss", None)

    ipv8_manager = getattr(node, "ipv8_manager", None)
    ipv8_port = int(getattr(ipv8_manager, "port", 0) or 0) if ipv8_manager else 0

    community = getattr(node, "community", None)
    peers_connected = 0
    peers_discovered = 0
    if community is not None:
        peers = getattr(community, "peers", None) or []
        try:
            peers_connected = len(peers)
        except Exception:
            peers_connected = 0
        peers_discovered = int(
            getattr(community, "peers_discovered_count", peers_connected) or 0
        )

    return {
        "node_id": getattr(node, "node_id", ""),
        "status": status,
        "since": since or "",
        "swarm_name": swarm_name,
        "swarm_id_short": swarm_id_short,
        "ipv8_port": ipv8_port,
        "peers_connected": peers_connected,
        "peers_discovered": peers_discovered,
        "current_round": current_round,
        "last_loss": last_loss,
    }
