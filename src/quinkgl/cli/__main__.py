# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""QuinkGL CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import quinkgl

from .exit_codes import SUCCESS, VALIDATION_ERROR


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quinkgl",
        description="QuinkGL: Decentralized Gossip Learning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"quinkgl {quinkgl.__version__}"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON to stdout"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warn", "error"],
        default="info",
        help="Controls stderr log verbosity",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "quinkgl",
        help="IPv8 state, TOFU cache, running-node socket",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="TOML config file"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colors"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error stderr output"
    )

    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # manifest
    from . import manifest_cmd

    manifest_cmd.build_parser(sub)

    # keygen
    from . import keygen_cmd

    keygen_cmd.build_parser(sub)

    # run
    from . import run_cmd

    run_cmd.build_parser(sub)

    # status
    from . import status_cmd

    status_cmd.build_parser(sub)

    # info
    from . import info_cmd

    info_cmd.build_parser(sub)

    # init
    from . import init_cmd

    init_cmd.build_parser(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        _setup_logging("error")
    else:
        _setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return SUCCESS

    dispatch = {
        "manifest": "manifest_cmd",
        "keygen": "keygen_cmd",
        "run": "run_cmd",
        "status": "status_cmd",
        "info": "info_cmd",
        "init": "init_cmd",
    }

    mod_name = dispatch.get(args.command)
    if not mod_name:
        parser.print_help()
        return SUCCESS

    mod = sys.modules[f"quinkgl.cli.{mod_name}"]
    return mod.run(args)


if __name__ == "__main__":
    sys.exit(main())
