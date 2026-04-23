# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl run — start a peer node (Modes A and B)."""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import importlib.util
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from quinkgl import GossipNode
from quinkgl.manifest import SwarmManifest, load_manifest
from quinkgl.manifest.errors import ERR_RUN_NO_STANDARD_MODEL
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
    raise ValueError(
        ERR_RUN_NO_STANDARD_MODEL,
        {
            "detail": (
                "No standard model loader available for this manifest. "
                "Use --script <path.py> (Mode B) or write a custom peer script (Mode C)."
            ),
        },
    )


def _resolve_rounds(args_rounds: int | None, manifest_limit: int | None) -> int:
    """Return effective round count capped by manifest.round_limit (§10.5.4)."""
    requested = args_rounds
    limit = manifest_limit
    if requested is None and limit is None:
        return 1000
    if requested is None:
        return limit  # type: ignore[return-value]
    if limit is None:
        return requested
    return min(requested, limit)


def _attach_telemetry(node: GossipNode, args: argparse.Namespace) -> None:
    """Wire TelemetryClient to node EventEmitter when --telemetry-url is set (§10.8)."""
    if not args.telemetry_url:
        return
    try:
        from quinkgl.telemetry import TelemetryClient
    except ImportError:
        log.warning("TelemetryClient not available; skipping telemetry wiring")
        return

    client = TelemetryClient(
        base_url=args.telemetry_url,
        secret=args.telemetry_secret or os.environ.get("QUINKGL_TELEMETRY_SECRET"),
        heartbeat_interval=args.telemetry_heartbeat_interval,
    )
    node.event_emitter.subscribe(client.handle)
    log.info("TelemetryClient wired to %s", args.telemetry_url)


def _attach_hooks(node: GossipNode, mod: Any) -> None:
    """Attach optional user-script callbacks to GossipNode hooks (§10.5.5)."""
    # ``on_round_end`` is handled directly by _build_on_round_end so the
    # CLI layer can interleave checkpointing, scheduler-stepping, and
    # status snapshot refresh around the user callback.
    for hook_name in (
        "on_model_received",
        "on_aggregation_done",
        "on_peer_discovered",
        "on_fingerprint_ready",
    ):
        cb = getattr(mod, hook_name, None)
        if callable(cb):
            setattr(node, hook_name, cb)
            log.debug("Attached user hook %s", hook_name)


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
    script_mod = None

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
            script_mod = _load_script_module(args.script)
        except Exception as exc:
            log.error("Failed to load script: %s", exc)
            return VALIDATION_ERROR

        if not callable(getattr(script_mod, "build_model", None)):
            log.error("Script does not export callable 'build_model'")
            return NODE_CONFIG_ERROR
        if not callable(getattr(script_mod, "build_loaders", None)):
            log.error("Script does not export callable 'build_loaders'")
            return NODE_CONFIG_ERROR

        script_args = _collect_script_args(args.script_arg)
        try:
            model = script_mod.build_model(manifest, **script_args)
        except Exception as exc:
            log.error("build_model failed: %s", exc, exc_info=True)
            return NODE_CONFIG_ERROR

        try:
            loaders = script_mod.build_loaders(manifest, **script_args)
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

    # 3. Build trusted_creator_pubkeys set for pinned policy
    trusted_pubkeys: set[bytes] | None = None
    if args.trusted_pubkey:
        trusted_pubkeys = set()
        for pk in args.trusted_pubkey:
            pk = pk.strip()
            if pk.startswith("ed25519:"):
                pk = pk[8:]
            try:
                trusted_pubkeys.add(bytes.fromhex(pk))
            except ValueError:
                log.error("Invalid --trusted-pubkey format: %s", pk)
                return VALIDATION_ERROR

    # 3b. Optional user-script optimizer / scheduler hooks (§10.7).
    # ``build_optimizer(manifest, model)`` returns a concrete optimizer
    # instance (e.g. ``torch.optim.SGD(...)``) that overrides the
    # framework default; ``build_scheduler(optimizer, manifest)`` returns
    # a LR scheduler whose ``.step()`` we call at the end of each round.
    training_config = None
    user_scheduler = None
    if script_mod is not None:
        build_opt = getattr(script_mod, "build_optimizer", None)
        build_sched = getattr(script_mod, "build_scheduler", None)
        optimizer_instance = None
        if callable(build_opt):
            try:
                optimizer_instance = build_opt(manifest, model)
            except Exception as exc:
                log.error("build_optimizer failed: %s", exc, exc_info=True)
                return NODE_CONFIG_ERROR
        if callable(build_sched):
            if optimizer_instance is None:
                log.error(
                    "build_scheduler was provided without build_optimizer; "
                    "a scheduler must wrap a concrete optimizer instance."
                )
                return NODE_CONFIG_ERROR
            try:
                user_scheduler = build_sched(optimizer_instance, manifest)
            except Exception as exc:
                log.error("build_scheduler failed: %s", exc, exc_info=True)
                return NODE_CONFIG_ERROR
        if optimizer_instance is not None:
            training_config = _make_training_config_with_optimizer(
                manifest, optimizer_instance
            )
        # Stash the LR scheduler on the script module so the
        # per-round callback (_build_on_round_end) can reach it without
        # threading yet another argument through the call chain.
        if user_scheduler is not None:
            setattr(script_mod, "_quinkgl_user_scheduler", user_scheduler)

    # 4. Construct GossipNode
    try:
        node = GossipNode(
            node_id=node_id,
            manifest=manifest,
            model=model,
            port=args.port,
            trust_policy=args.trust_policy,
            trusted_creator_pubkeys=trusted_pubkeys,
            quiet=args.quiet,
            training_config=training_config,
        )
    except Exception as exc:
        log.error("Failed to construct GossipNode: %s", exc)
        return NODE_CONFIG_ERROR

    # 5. Attach optional hooks from user script
    if script_mod is not None:
        _attach_hooks(node, script_mod)

    # 6. Telemetry wiring
    _attach_telemetry(node, args)

    # 7. Checkpoint / resume (spec §11.4, §11.8 — persistent state lives
    # under --checkpoint-dir; --resume re-seeds the model from the most
    # recent checkpoint before the first gossip round).
    ckpt_store = _open_checkpoint_store(args.checkpoint_dir)
    start_round = _maybe_resume(ckpt_store, model, args.resume)

    # 8. Status socket (spec §11.8): bind a local introspection server so
    # `quinkgl status` can read live state.  We also drop a sibling
    # ``.json`` snapshot so Windows tools that cannot speak AF_UNIX still
    # see *something*; the snapshot is refreshed from the on_round_end
    # hook below.
    status_server, status_json_path, since_ts = await _start_status_artefacts(
        args, node
    )

    # 9. Training loop + periodic checkpoint / status refresh
    rounds = _resolve_rounds(args.rounds, manifest.round_limit)
    per_round = _build_on_round_end(
        node,
        ckpt_store=ckpt_store,
        script_mod=script_mod,
        status_json_path=status_json_path,
        since_ts=since_ts,
    )

    try:
        async with node:
            await node.train(rounds=rounds, on_round_end=per_round)
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        return NODE_CONFIG_ERROR
    finally:
        if status_server is not None:
            try:
                await status_server.stop()
            except Exception:  # pragma: no cover — teardown best-effort
                pass
        if status_json_path is not None:
            try:
                status_json_path.unlink(missing_ok=True)
            except Exception:
                pass

    # 10. Final checkpoint — best-effort; failures must not mask a
    # successful training run.
    if ckpt_store is not None:
        try:
            _save_checkpoint(ckpt_store, model, round_number=start_round + rounds - 1)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Final checkpoint save failed: %s", exc)

    return SUCCESS


# ---------------------------------------------------------------------------
# Helpers: checkpoint / resume, status artefacts, user optimizer/scheduler.
# ---------------------------------------------------------------------------


def _open_checkpoint_store(checkpoint_dir: Optional[str]):
    """Return a :class:`ModelStore` rooted at ``checkpoint_dir`` or ``None``.

    We import lazily so a run without ``--checkpoint-dir`` never pulls in
    msgpack / numpy just to dispatch the CLI.
    """
    if not checkpoint_dir:
        return None
    try:
        from quinkgl.storage import ModelStore

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return ModelStore(storage_dir=checkpoint_dir)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "Failed to open checkpoint store at %s: %s", checkpoint_dir, exc
        )
        return None


def _maybe_resume(store, model, resume_flag: bool) -> int:
    """Seed ``model`` from the latest checkpoint if ``--resume`` is set.

    Returns the resumed round number (0 when nothing was loaded) so the
    caller can emit correctly-numbered final checkpoints.
    """
    if store is None or not resume_flag:
        return 0
    try:
        latest = store.get_latest_checkpoint()
    except Exception as exc:
        log.warning("Could not list checkpoints: %s", exc)
        return 0
    if latest is None:
        log.info("--resume requested but no checkpoint found; starting fresh.")
        return 0
    try:
        model.set_weights(latest.weights)
    except Exception as exc:
        log.warning(
            "Loaded checkpoint %s but set_weights failed: %s",
            latest.checkpoint_id,
            exc,
        )
        return 0
    log.info(
        "Resumed from checkpoint %s (round %d)",
        latest.checkpoint_id,
        latest.round_number,
    )
    return int(latest.round_number)


def _save_checkpoint(store, model, *, round_number: int) -> None:
    """Save the model's current weights as a new checkpoint.

    Exceptions are propagated; the caller decides whether a save
    failure is fatal or merely logged.
    """
    weights = model.get_weights()
    store.save_checkpoint(
        round_number=max(0, int(round_number)),
        weights=weights,
    )


async def _start_status_artefacts(
    args: argparse.Namespace, node: GossipNode
) -> tuple[Any, Optional[Path], str]:
    """Bind the status socket + sibling JSON snapshot (spec §11.8).

    Returns ``(StatusServer_or_None, json_path_or_None, since_iso)``.
    A failure to bind the socket degrades gracefully to the JSON-only
    transport so non-POSIX hosts still get observability.
    """
    work_dir = Path(getattr(args, "work_dir", ".quinkgl"))
    running_dir = work_dir / "running"
    running_dir.mkdir(parents=True, exist_ok=True)

    node_id = node.node_id
    since_ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

    from .status_server import StatusServer, build_status_snapshot

    def _provider() -> Dict[str, Any]:
        return build_status_snapshot(node, since=since_ts)

    json_path = running_dir / f"{node_id}.json"
    try:
        json_path.write_text(json.dumps(_provider(), default=str))
    except Exception as exc:  # pragma: no cover — non-fatal
        log.warning("Could not write status JSON at %s: %s", json_path, exc)
        json_path = None  # type: ignore[assignment]

    server: Any = None
    sock_path = running_dir / f"{node_id}.sock"
    try:
        server = StatusServer(str(sock_path), _provider)
        await server.start()
    except Exception as exc:
        log.warning("StatusServer at %s unavailable: %s", sock_path, exc)
        server = None

    return server, json_path, since_ts


def _build_on_round_end(
    node: GossipNode,
    *,
    ckpt_store,
    script_mod: Any,
    status_json_path: Optional[Path],
    since_ts: str,
) -> Callable[[int, Dict[str, float]], "Any"]:
    """Compose the per-round callback that the CLI layer owns.

    Responsibilities, in order:
      1. refresh the sibling ``.json`` status snapshot,
      2. step a user-provided LR scheduler (from ``build_scheduler``),
      3. save a checkpoint every 10 rounds,
      4. invoke the script-level ``on_round_end`` hook.

    Every stage is wrapped in its own try/except so one failing
    responsibility does not mask the others (§10.5.5).
    """
    from .status_server import build_status_snapshot

    user_scheduler = (
        getattr(script_mod, "_quinkgl_user_scheduler", None) if script_mod else None
    )
    user_on_round_end = (
        getattr(script_mod, "on_round_end", None) if script_mod else None
    )

    async def _cb(round_idx: int, metrics: Dict[str, float]) -> None:
        if status_json_path is not None:
            try:
                status_json_path.write_text(
                    json.dumps(
                        build_status_snapshot(node, since=since_ts),
                        default=str,
                    )
                )
            except Exception:  # pragma: no cover — non-fatal
                pass

        if user_scheduler is not None:
            step_fn = getattr(user_scheduler, "step", None)
            if callable(step_fn):
                try:
                    step_fn()
                except Exception as exc:
                    log.warning("user scheduler.step() raised: %s", exc)

        if ckpt_store is not None and (round_idx + 1) % 10 == 0:
            try:
                _save_checkpoint(ckpt_store, node.model, round_number=round_idx)
            except Exception as exc:
                log.warning(
                    "periodic checkpoint save at round %d failed: %s",
                    round_idx,
                    exc,
                )

        if callable(user_on_round_end):
            try:
                result = user_on_round_end(round_idx, metrics)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning("user on_round_end raised: %s", exc)

    return _cb


def _make_training_config_with_optimizer(manifest: SwarmManifest, optimizer: Any):
    """Build a :class:`TrainingConfig` that injects a user optimizer.

    Also propagates ``manifest.task.learning_rate`` / ``.batch_size`` when
    present, so users who override the optimizer but not the rest of the
    training schedule still get the manifest-declared values.
    """
    from quinkgl.models.base import TrainingConfig

    task = getattr(manifest, "task", None)
    if isinstance(task, dict):
        lr = task.get("learning_rate", 0.001)
        bs = task.get("batch_size", 32)
    else:
        lr = float(getattr(task, "learning_rate", 0.001) or 0.001)
        bs = int(getattr(task, "batch_size", 32) or 32)
    return TrainingConfig(
        epochs=1,
        batch_size=int(bs),
        learning_rate=float(lr),
        optimizer=optimizer,
    )


def run(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_async_run(args))
    except KeyboardInterrupt:
        return SUCCESS
