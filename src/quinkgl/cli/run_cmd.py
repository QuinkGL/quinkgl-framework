# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl run — start a peer node (Modes A and B)."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

from quinkgl import GossipNode
from quinkgl.manifest import SwarmManifest, load_manifest
from quinkgl.manifest.errors import ERR_RUN_NO_STANDARD_MODEL, ERR_SCRIPT_CALLABLES_MISSING
from quinkgl.models import PyTorchModel

from .exit_codes import (
    NODE_CONFIG_ERROR,
    SUCCESS,
    VALIDATION_ERROR,
)

if TYPE_CHECKING:
    from argparse import _SubParsersAction

log = logging.getLogger("quinkgl.cli.run")


def build_parser(sub: _SubParsersAction) -> None:
    parser = sub.add_parser("run", help="Start a QuinkGL peer node")
    parser.add_argument("--manifest", required=True, help="Path, URL, or magnet URI")
    parser.add_argument("--data", default=None, help="Data directory (Mode A)")
    parser.add_argument("--script", default=None, help="User script path (Mode B)")
    parser.add_argument("--script-arg", action="append", default=[], help="key=value")
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--trust-policy", choices=["open", "tofu", "pinned"], default="open")
    parser.add_argument("--trusted-pubkey", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--telemetry-url", default=None)
    parser.add_argument("--telemetry-secret", default=None)
    parser.add_argument("--telemetry-heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify manifest and exit")


def _load_script_module(script_path: str):
    """Load a user script via importlib without installing it as a package."""
    p = Path(script_path).resolve()
    script_dir = str(p.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("_quinkgl_user_script", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_script_args(pairs: list[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    reserved = {"node_id", "manifest", "trust_policy", "trusted_creator_pubkeys"}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--script-arg must be key=value, got {pair}")
        k, v = pair.split("=", 1)
        if k in reserved:
            raise ValueError(f"--script-arg key {k!r} is reserved")
        out[k] = v
    return out


def _build_standard_model(manifest: SwarmManifest):
    """Attempt to build a standard model from manifest spec.

    Returns (model, train_loader, val_loader) or raises ERR_RUN_NO_STANDARD_MODEL.
    """
    # Standard model loading is not yet implemented for arbitrary arch_spec.
    # Only very simple specs could be auto-built; everything else needs Mode B.
    raise ValueError(
        ERR_RUN_NO_STANDARD_MODEL,
        {
            "detail": (
                "No standard model loader available for this manifest. "
                "Use --script <path.py> (Mode B) or write a custom peer script (Mode C)."
            ),
        },
    )


def _run_training(node: GossipNode, rounds: int) -> None:
    """Run the training loop and handle graceful shutdown on SIGINT."""
    loop = asyncio.get_event_loop()

    def _signal_handler(sig):
        log.info("Received signal %s, shutting down gracefully...", sig)
        asyncio.create_task(node.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))

    try:
        loop.run_until_complete(node.train(rounds=rounds))
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


async def _async_run(args: argparse.Namespace) -> int:
    manifest_source = args.manifest

    # 1. Load manifest
    try:
        if Path(manifest_source).exists():
            manifest = SwarmManifest.from_file(manifest_source)
        else:
            manifest = load_manifest(manifest_source)
    except Exception as exc:
        log.error("Failed to load manifest: %s", exc)
        return VALIDATION_ERROR

    if args.dry_run:
        log.info("Manifest loaded successfully (dry-run).")
        if args.json:
            print(json.dumps({"swarm_id": manifest.manifest_hash(), "valid": True}))
        return SUCCESS

    # Determine mode
    mode = "A" if args.data else ("B" if args.script else None)
    if mode is None:
        log.error("Either --data (Mode A) or --script (Mode B) must be provided.")
        return VALIDATION_ERROR

    node_id = args.node_id or f"peer-{os.urandom(4).hex()}"

    # 2. Build model and loaders
    model = None
    train_loader = None
    val_loader = None

    if mode == "A":
        try:
            model, train_loader, val_loader = _build_standard_model(manifest)
        except ValueError as exc:
            if exc.args and exc.args[0] == ERR_RUN_NO_STANDARD_MODEL:
                log.error("%s", exc.args[1].get("detail", str(exc)))
                return NODE_CONFIG_ERROR
            raise
    else:
        # Mode B
        try:
            mod = _load_script_module(args.script)
        except Exception as exc:
            log.error("Failed to load script: %s", exc)
            return VALIDATION_ERROR

        if not callable(getattr(mod, "build_model", None)):
            log.error("Script does not export callable 'build_model'")
            return NODE_CONFIG_ERROR
        if not callable(getattr(mod, "build_loaders", None)):
            log.error("Script does not export callable 'build_loaders'")
            return NODE_CONFIG_ERROR

        script_args = _collect_script_args(args.script_arg)
        try:
            model = mod.build_model(manifest, **script_args)
        except Exception as exc:
            log.error("build_model failed: %s", exc, exc_info=True)
            return NODE_CONFIG_ERROR

        try:
            loaders = mod.build_loaders(manifest, **script_args)
        except Exception as exc:
            log.error("build_loaders failed: %s", exc, exc_info=True)
            return NODE_CONFIG_ERROR

        if isinstance(loaders, tuple):
            train_loader, val_loader = loaders[0], loaders[1] if len(loaders) > 1 else None
        else:
            train_loader = loaders

    if model is None:
        log.error("Model is None after loading.")
        return NODE_CONFIG_ERROR

    # 3. Construct GossipNode
    try:
        node = GossipNode(
            node_id=node_id,
            manifest=manifest,
            model=model,
            port=args.port,
            trust_policy=args.trust_policy,
            quiet=args.quiet,
        )
    except Exception as exc:
        log.error("Failed to construct GossipNode: %s", exc)
        return NODE_CONFIG_ERROR

    # 4. Training loop
    rounds = args.rounds or manifest.round_limit or 1000
    try:
        async with node:
            await node.train(rounds=rounds)
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        return NODE_CONFIG_ERROR

    return SUCCESS


def run(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_async_run(args))
    except KeyboardInterrupt:
        return SUCCESS
