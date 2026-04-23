"""Unix-socket status introspection (spec §11.8).

Covers both sides of the loop: the :class:`StatusServer` writes the
provider's snapshot as one newline-terminated JSON blob per connect,
and the client-side :func:`read_status_from_socket` used by
``quinkgl status`` consumes it.  The CLI dispatch is exercised by
driving ``quinkgl status --node-id …`` against a running server in a
temp work-dir.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def short_tmp_dir():
    # macOS caps AF_UNIX paths at ~104 bytes; pytest's short_tmp_dir often
    # exceeds this.  Use a short /tmp-rooted directory instead.
    with tempfile.TemporaryDirectory(prefix="qgl_") as d:
        yield Path(d)

from quinkgl.cli.__main__ import main as cli_main
from quinkgl.cli.status_server import (
    StatusServer,
    build_status_snapshot,
    read_status_from_socket,
)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="unix sockets require POSIX"
)
class TestStatusServerRoundTrip:
    @pytest.mark.asyncio
    async def test_serves_provider_snapshot_as_json(self, short_tmp_dir: Path):
        socket_path = short_tmp_dir / "running" / "peer-abc.sock"
        snapshot = {"node_id": "peer-abc", "status": "TRAINING", "current_round": 7}

        async with StatusServer(str(socket_path), lambda: snapshot):
            # read_status_from_socket is sync by design; run it in the
            # default executor so we don't block the loop.
            state = await asyncio.get_event_loop().run_in_executor(
                None, read_status_from_socket, str(socket_path)
            )
        assert state == snapshot
        assert not socket_path.exists()  # stop() unlinks the socket

    @pytest.mark.asyncio
    async def test_socket_mode_is_0600(self, short_tmp_dir: Path):
        socket_path = short_tmp_dir / "peer.sock"
        async with StatusServer(str(socket_path), lambda: {"ok": True}):
            mode = os.stat(socket_path).st_mode & 0o777
            assert mode == 0o600

    @pytest.mark.asyncio
    async def test_stale_socket_is_overwritten_on_start(self, short_tmp_dir: Path):
        socket_path = short_tmp_dir / "peer.sock"
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.write_text("stale bytes")  # not actually a socket

        async with StatusServer(str(socket_path), lambda: {"ok": True}):
            state = await asyncio.get_event_loop().run_in_executor(
                None, read_status_from_socket, str(socket_path)
            )
        assert state == {"ok": True}

    @pytest.mark.asyncio
    async def test_provider_exception_surfaces_as_error_field(self, short_tmp_dir: Path):
        socket_path = short_tmp_dir / "peer.sock"

        def _boom():
            raise RuntimeError("provider explosion")

        async with StatusServer(str(socket_path), _boom):
            state = await asyncio.get_event_loop().run_in_executor(
                None, read_status_from_socket, str(socket_path)
            )
        assert state["error"] == "state_provider failed"
        assert "provider explosion" in state["detail"]


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="unix sockets require POSIX"
)
class TestReadStatusFromSocket:
    def test_missing_socket_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_status_from_socket(str(tmp_path / "nope.sock"))

    @pytest.mark.asyncio
    async def test_invalid_json_raises_value_error(self, short_tmp_dir: Path):
        socket_path = short_tmp_dir / "bad.sock"

        async def _handle(reader, writer):
            writer.write(b"not-json\n")
            await writer.drain()
            writer.close()

        server = await asyncio.start_unix_server(_handle, path=str(socket_path))
        try:
            with pytest.raises(ValueError):
                await asyncio.get_event_loop().run_in_executor(
                    None, read_status_from_socket, str(socket_path)
                )
        finally:
            server.close()
            await server.wait_closed()
            if socket_path.exists():
                socket_path.unlink()


# --- build_status_snapshot -------------------------------------------------


class _FakeNode:
    class _State:
        name = "TRAINING"

    class _GLNode:
        current_round = 42

        class aggregator:
            last_loss = 0.123

    class _Community:
        peers = ["p1", "p2"]
        peers_discovered_count = 5

    class _Ipv8:
        port = 7000

    def __init__(self):
        self.node_id = "peer-xyz"
        self.state = _FakeNode._State()
        self.gl_node = _FakeNode._GLNode()
        self.community = _FakeNode._Community()
        self.ipv8_manager = _FakeNode._Ipv8()
        self.manifest = None


def test_build_status_snapshot_from_fake_node():
    snap = build_status_snapshot(_FakeNode(), since="2026-04-22T10:00:00Z")
    assert snap["node_id"] == "peer-xyz"
    assert snap["status"] == "TRAINING"
    assert snap["ipv8_port"] == 7000
    assert snap["peers_connected"] == 2
    assert snap["peers_discovered"] == 5
    assert snap["current_round"] == 42
    assert snap["since"] == "2026-04-22T10:00:00Z"
    assert snap["last_loss"] == 0.123


# --- CLI dispatch ---------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="unix sockets require POSIX"
)
class TestStatusCLIAgainstSocket:
    @pytest.mark.asyncio
    async def test_status_reads_from_running_socket(
        self, short_tmp_dir: Path, capsys, monkeypatch
    ):
        socket_path = short_tmp_dir / "running" / "peer-xyz.sock"
        snapshot = {
            "node_id": "peer-xyz",
            "status": "TRAINING",
            "since": "2026-04-22T10:00:00Z",
            "swarm_name": "demo",
            "swarm_id_short": "abc123def456",
            "ipv8_port": 7000,
            "peers_connected": 2,
            "peers_discovered": 5,
            "current_round": 42,
            "last_loss": 0.5,
        }

        async with StatusServer(str(socket_path), lambda: snapshot):
            # Run the CLI off-loop so argparse's blocking style is fine.
            rc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cli_main(
                    [
                        "--json",
                        "--work-dir",
                        str(short_tmp_dir),
                        "status",
                        "--node-id",
                        "peer-xyz",
                    ]
                ),
            )
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["node_id"] == "peer-xyz"
        assert payload["current_round"] == 42

    def test_status_missing_socket_returns_trust_error(self, short_tmp_dir: Path):
        # Empty work-dir → no running node → exit 4 per §11.11.
        rc = cli_main(
            [
                "--work-dir",
                str(short_tmp_dir),
                "status",
                "--node-id",
                "peer-missing",
            ]
        )
        assert rc == 4
