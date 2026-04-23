# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl status — local peer introspection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import _SubParsersAction

from .exit_codes import IO_ERROR, SUCCESS, TRUST_ERROR


def build_parser(sub: _SubParsersAction) -> None:
    parser = sub.add_parser("status", help="Show state of a running node")
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--watch", action="store_true")


def _discover_nodes(work_dir: Path) -> list[tuple[str, Path]]:
    running_dir = work_dir / "running"
    if not running_dir.exists():
        return []
    nodes = []
    seen: set[str] = set()
    # Prefer .sock over .json for the same node_id so a single running
    # peer does not appear as two entries (spec §11.8).
    for p in sorted(
        running_dir.iterdir(),
        key=lambda x: (x.stem, 0 if x.suffix == ".sock" else 1),
    ):
        if p.suffix in {".sock", ".json"}:
            nid = p.stem
            if nid not in seen:
                nodes.append((nid, p))
                seen.add(nid)
    return nodes


def _read_state(path: Path) -> dict | None:
    """Resolve a running peer's state from its discovery artefact.

    Handles both transports from spec §11.8:

    * ``.sock`` — connect to the unix socket, read one newline-terminated
      JSON payload (see :class:`quinkgl.cli.status_server.StatusServer`).
    * ``.json`` — read the sibling on-disk state file.

    Returns ``None`` on any decode, connect, or timeout failure so the
    caller can differentiate "no running node" from "node misbehaving"
    at the exit-code layer.
    """
    if path.suffix == ".json" and path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    if path.suffix == ".sock":
        from .status_server import read_status_from_socket

        try:
            return read_status_from_socket(str(path))
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError, OSError):
            # Server may have crashed; fall back to the sibling .json
            # snapshot if it exists (spec §11.8 graceful degradation).
            json_path = path.with_suffix(".json")
            if json_path.exists():
                try:
                    return json.loads(json_path.read_text())
                except Exception:
                    return None
            return None
        except ValueError:
            # Server returned garbage — surface as "cannot read" rather
            # than crash the CLI; the status command will already show
            # "Cannot read state" with exit 2.
            return None
    return None


def _print_status(state: dict, args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(state, indent=2))
        return
    print(f"Node:          {state.get('node_id', 'unknown')}")
    print(f"Status:        {state.get('status', 'UNKNOWN')}")
    print(f"Since:         {state.get('since', '')}")
    print(f"Swarm:         {state.get('swarm_name', '')} ({state.get('swarm_id_short', '')})")
    print(f"IPv8 port:     {state.get('ipv8_port', 0)}")
    print(f"Peers:         {state.get('peers_connected', 0)}/{state.get('peers_discovered', 0)}")
    print(f"Current round: {state.get('current_round', 0)}")


def _show_once(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    nodes = _discover_nodes(work_dir)

    if not nodes:
        print("No running node found.", file=sys.stderr)
        return TRUST_ERROR

    if args.node_id:
        matches = [(nid, path) for nid, path in nodes if nid == args.node_id]
    else:
        matches = nodes

    if len(matches) == 0:
        print(f"No running node matches node-id={args.node_id}", file=sys.stderr)
        return TRUST_ERROR

    if len(matches) > 1 and args.node_id is None:
        print("Multiple nodes running. Use --node-id to select one:", file=sys.stderr)
        for nid, _ in matches:
            print(f"  {nid}", file=sys.stderr)
        return TRUST_ERROR

    node_id, path = matches[0]
    state = _read_state(path)
    if state is None:
        print(f"Cannot read state for node {node_id}", file=sys.stderr)
        return IO_ERROR

    _print_status(state, args)
    return SUCCESS


def run(args: argparse.Namespace) -> int:
    if not args.watch:
        return _show_once(args)

    # --watch mode: refresh every 2s until Ctrl-C (§11.8)
    try:
        while True:
            # Clear screen for clean refresh (optional)
            # os.system('clear')  # too invasive; skip
            rc = _show_once(args)
            if rc != SUCCESS:
                return rc
            time.sleep(2.0)
    except KeyboardInterrupt:
        return SUCCESS
