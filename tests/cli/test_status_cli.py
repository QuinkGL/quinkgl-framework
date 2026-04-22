# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Status CLI tests (B-5 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import IO_ERROR, SUCCESS, TRUST_ERROR


class TestStatus:
    def test_status_no_node(self, tmp_path: Path) -> None:
        assert main(["--work-dir", str(tmp_path), "status"]) == TRUST_ERROR

    def test_status_found(self, tmp_path: Path) -> None:
        running = tmp_path / "running"
        running.mkdir()
        state = {
            "node_id": "alice",
            "status": "TRAINING",
            "since": "2026-04-23T12:00:00Z",
            "swarm_name": "test",
            "swarm_id_short": "abc123",
            "ipv8_port": 7759,
            "peers_connected": 2,
            "peers_discovered": 5,
            "current_round": 7,
        }
        (running / "alice.json").write_text(json.dumps(state))
        assert main(["--work-dir", str(tmp_path), "status", "--node-id", "alice"]) == SUCCESS

    def status_multiple_nodes_no_id(self, tmp_path: Path) -> None:
        running = tmp_path / "running"
        running.mkdir()
        for name in ("alice", "bob"):
            (running / f"{name}.json").write_text(json.dumps({"node_id": name}))
        assert main(["--work-dir", str(tmp_path), "status"]) == TRUST_ERROR
