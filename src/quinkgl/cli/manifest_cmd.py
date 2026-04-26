# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl manifest subcommands: create, show, verify, magnet."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import quinkgl
from quinkgl.manifest import (
    ByzantineSpec,
    DataPolicy,
    MagnetLink,
    ModelSpec,
    SwarmManifest,
    TaskSpec,
    TelemetryConfig,
    format_magnet,
    load_manifest,
    parse_magnet,
)
from quinkgl.telemetry.api import DEFAULT_TELEMETRY_BASE_URL
from quinkgl.manifest.errors import (
    ERR_CREATOR_NOT_TRUSTED,
    ERR_MANIFEST_DATA_POLICY,
    ERR_MANIFEST_EXPIRED,
    ERR_MANIFEST_FIELD_INVALID,
    ERR_MANIFEST_HASH_MISMATCH,
    ERR_MANIFEST_INVALID_JSON,
    ERR_MANIFEST_MISSING_KEYS,
    ERR_MANIFEST_NOT_OBJECT,
    ERR_MANIFEST_SCHEMA_VERSION,
    ERR_MANIFEST_UNKNOWN_KEYS,
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
)

from .exit_codes import (
    CRYPTO_ERROR,
    HASH_MISMATCH,
    IO_ERROR,
    SUCCESS,
    TRUST_ERROR,
    VALIDATION_ERROR,
)

if TYPE_CHECKING:
    from argparse import _SubParsersAction


def build_parser(sub: _SubParsersAction) -> None:
    parser = sub.add_parser("manifest", help="Manifest operations")
    manifest_sub = parser.add_subparsers(dest="manifest_command")

    # create
    create = manifest_sub.add_parser("create", help="Build a .qgl file")
    create.add_argument("--name", required=True)
    create.add_argument("--task-type", required=True, choices=["class", "regr", "seg", "det"])
    create.add_argument("--input-shape", required=True, help="Comma-separated positive ints")
    create.add_argument("--output-shape", required=True, help="Comma-separated positive ints")
    create.add_argument("--label-type", required=True)
    create.add_argument("--tags", default="", help="Comma-separated tags")
    create.add_argument("--model-framework", required=True)
    create.add_argument("--model-arch-hash", required=True)
    create.add_argument("--model-arch-file", default=None)
    create.add_argument("--aggregation", required=True)
    create.add_argument("--aggregation-param", action="append", default=[], help="k=v")
    create.add_argument("--topology", required=True)
    create.add_argument("--topology-param", action="append", default=[], help="k=v")
    create.add_argument("--data-policy", default=None, help="JSON path")
    create.add_argument("--byzantine-f", type=int, default=0)
    create.add_argument("--round-limit", type=int, default=None)
    create.add_argument("--expires-at", default=None)
    create.add_argument("--bootstrap-peer", action="append", default=[], help="host:port")
    create.add_argument("--tracker-tier", action="append", default=[], help="url1,url2")
    create.add_argument(
        "--telemetry-dashboard-url",
        default=DEFAULT_TELEMETRY_BASE_URL,
        help="Secret-free dashboard origin to place in the manifest.",
    )
    create.add_argument(
        "--telemetry-enrollment",
        choices=["invite-required", "none"],
        default="invite-required",
        help="Telemetry enrollment mode declared by the manifest.",
    )
    create.add_argument("--sign-with", default=None)
    create.add_argument("--output", required=True)

    # show
    show = manifest_sub.add_parser("show", help="Pretty-print a .qgl file")
    show.add_argument("path", help="Path to .qgl file")
    show.add_argument("--show-signature-bytes", action="store_true")

    # verify
    verify = manifest_sub.add_parser("verify", help="Validate schema + hash")
    verify.add_argument("path", help="Path to .qgl file")
    verify.add_argument("--trusted-pubkey", action="append", default=[])
    verify.add_argument("--expected-swarm-id", default=None)

    # magnet
    magnet = manifest_sub.add_parser("magnet", help="Derive magnet URI from a .qgl")
    magnet.add_argument("path", help="Path to .qgl file")
    magnet.add_argument("--tracker", action="append", default=[])
    magnet.add_argument("--bootstrap", action="append", default=[])


def _parse_shape(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"param must be k=v, got {pair}")
        k, v = pair.split("=", 1)
        # Try int/float/bool, else keep string
        try:
            v_parsed = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                if v.lower() == "true":
                    v_parsed = True
                elif v.lower() == "false":
                    v_parsed = False
                else:
                    v_parsed = v
        out[k] = v_parsed
    return out


def _load_data_policy(path: str | None) -> DataPolicy:
    if path is None:
        return DataPolicy()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data-policy file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return DataPolicy.from_dict(data, strict=False)


def _handle_manifest_error(exc: Exception) -> int:
    """Map manifest exceptions to CLI exit codes (§11.11)."""
    if isinstance(exc, ValueError) and exc.args:
        code = exc.args[0]
        if isinstance(code, str) and code.startswith("ERR_"):
            if code in {
                ERR_MANIFEST_INVALID_JSON,
                ERR_MANIFEST_NOT_OBJECT,
                ERR_MANIFEST_SCHEMA_VERSION,
                ERR_MANIFEST_UNKNOWN_KEYS,
                ERR_MANIFEST_MISSING_KEYS,
                ERR_MANIFEST_FIELD_INVALID,
                ERR_MANIFEST_EXPIRED,
                ERR_MANIFEST_DATA_POLICY,
            }:
                return VALIDATION_ERROR
            if code == ERR_MANIFEST_HASH_MISMATCH:
                return HASH_MISMATCH
    if isinstance(exc, FileNotFoundError):
        return IO_ERROR
    if isinstance(exc, json.JSONDecodeError):
        return VALIDATION_ERROR
    return VALIDATION_ERROR


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        task = TaskSpec(
            type=_task_type_map(args.task_type),
            input_shape=_parse_shape(args.input_shape),
            output_shape=_parse_shape(args.output_shape),
            label_type=args.label_type,
            tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        )
        model = ModelSpec(
            framework=args.model_framework,
            arch_hash=args.model_arch_hash,
        )
        byzantine = ByzantineSpec(f=args.byzantine_f)
        data_policy = _load_data_policy(args.data_policy)

        bootstrap_peers = [
            {"kind": "ipv8", "peer_id": "", "address": addr}
            for addr in args.bootstrap_peer
        ]
        tracker_urls = [
            [url.strip() for url in tier.split(",") if url.strip()]
            for tier in args.tracker_tier
        ]

        manifest = SwarmManifest(
            name=args.name,
            task=task,
            model=model,
            byzantine=byzantine,
            data_policy=data_policy,
            aggregation_name=args.aggregation,
            aggregation_params=_parse_params(args.aggregation_param),
            topology_name=args.topology,
            topology_params=_parse_params(args.topology_param),
            model_arch_fingerprint=args.model_arch_hash,
            data_schema_hash="sha256:" + "0" * 64,  # placeholder
            round_limit=args.round_limit,
            expires_at=args.expires_at,
            bootstrap_peers=bootstrap_peers,
            tracker_urls=tracker_urls,
            telemetry=TelemetryConfig(
                dashboard_url=args.telemetry_dashboard_url,
                enrollment=args.telemetry_enrollment,
            ),
        )
        manifest.validate()

        if args.sign_with:
            try:
                manifest = _sign_with_key_file(manifest, args.sign_with)
            except FileNotFoundError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return IO_ERROR
            except ValueError as exc:
                code = exc.args[0] if exc.args else ""
                if code == ERR_SIGNING_UNAVAILABLE:
                    print(
                        "Error: cryptography package is not installed; cannot "
                        "sign manifest. Install `cryptography>=41.0.0`.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Error: signing failed: {exc}", file=sys.stderr)
                return CRYPTO_ERROR

        manifest.to_file(args.output, pretty=True)

        swarm_id = manifest.manifest_hash()
        magnet = manifest.to_magnet()

        if args.json:
            payload = {"swarm_id": swarm_id, "magnet": magnet}
            if manifest.creator_pubkey is not None:
                payload["creator_pubkey"] = manifest.creator_pubkey
                payload["signed"] = manifest.signature is not None
            print(json.dumps(payload, indent=2))
        else:
            print(f"swarm_id: {swarm_id}")
            print(f"magnet: {magnet}")
            if manifest.creator_pubkey is not None:
                sig_status = "signed" if manifest.signature else "unsigned"
                print(f"creator: {manifest.creator_pubkey} ({sig_status})")
        return SUCCESS
    except Exception as exc:
        code = _handle_manifest_error(exc)
        print(f"Error: {exc}", file=sys.stderr)
        return code


def _sign_with_key_file(manifest: SwarmManifest, path: str) -> SwarmManifest:
    """Load a PKCS#8 PEM from ``path`` and return a signed copy of ``manifest``."""
    from quinkgl.manifest import sign_manifest

    key_bytes = Path(path).read_bytes()
    return sign_manifest(manifest, key_bytes)


def _task_type_map(short: str) -> str:
    mapping = {
        "class": "classification",
        "regr": "regression",
        "seg": "segmentation",
        "det": "detection",
    }
    return mapping.get(short, short)


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        manifest = SwarmManifest.from_file(args.path)
    except Exception as exc:
        code = _handle_manifest_error(exc)
        print(f"Error: {exc}", file=sys.stderr)
        return code

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2))
        return SUCCESS

    print(f"Swarm:        {manifest.name}")
    print(f"Swarm ID:     {manifest.manifest_hash()}")
    creator = manifest.creator_pubkey or "unsigned"
    sig_status = _signature_status_label(manifest)
    print(f"Creator:      {creator} (signature: {sig_status})")
    if args.show_signature_bytes and manifest.signature:
        print(f"Signature:    {manifest.signature}")
    print(f"Created:      {manifest.created_at}")
    print(f"Expires:      {manifest.expires_at or 'never'}")
    print()
    print(f"Task:         {manifest.task.type}, input={manifest.task.input_shape}, output={manifest.task.output_shape}, {manifest.task.label_type}")
    print(f"Tags:         {', '.join(manifest.task.tags)}")
    print()
    print(f"Model:        {manifest.model.framework}, arch={manifest.model.arch_hash}")
    print(f"Aggregation:  {manifest.aggregation_name} ({json.dumps(manifest.aggregation_params)})")
    print(f"Topology:     {manifest.topology_name} ({json.dumps(manifest.topology_params)})")
    print(f"Byzantine:    f={manifest.byzantine.f}, enforce={manifest.byzantine.enforce_n_gt_2f_plus_2}")
    print(f"Round limit:  {manifest.round_limit or 'unbounded'}")
    print()
    print("Data Policy:")
    dp = manifest.data_policy.to_dict()
    for k, v in list(dp.items())[:6]:
        print(f"  {k}: {v}")
    print()
    print(f"Bootstrap peers: {len(manifest.bootstrap_peers)} configured")
    total_trackers = sum(len(tier) for tier in manifest.tracker_urls)
    print(f"Tracker tiers:   {len(manifest.tracker_urls)} tier(s), {total_trackers} URL(s)")
    return SUCCESS


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        manifest = SwarmManifest.from_file(args.path)
    except Exception as exc:
        code = _handle_manifest_error(exc)
        print(f"Error: {exc}", file=sys.stderr)
        return code

    if args.expected_swarm_id:
        expected = args.expected_swarm_id.lower().strip()
        actual = manifest.manifest_hash()
        if expected != actual:
            print(
                f"Hash mismatch: expected {expected}, got {actual}", file=sys.stderr
            )
            return HASH_MISMATCH

    # Exit-code precedence (§11.3): signature errors (3) supersede trust
    # errors (4).  Users who don't care about signatures can still run
    # `verify` without a signed manifest and get exit 0.
    if manifest.signature is not None or manifest.creator_pubkey is not None:
        try:
            from quinkgl.manifest import verify_manifest
        except ImportError as exc:  # pragma: no cover — always importable
            print(f"Error: cannot load signing subsystem: {exc}", file=sys.stderr)
            return CRYPTO_ERROR

        try:
            valid = verify_manifest(manifest)
        except ValueError as exc:
            code = exc.args[0] if exc.args else ""
            if code == ERR_SIGNING_UNAVAILABLE:
                print(
                    "Error: cryptography package is not installed; cannot "
                    "verify signature.",
                    file=sys.stderr,
                )
                return CRYPTO_ERROR
            print(f"Error: signature verification failed: {exc}", file=sys.stderr)
            return CRYPTO_ERROR

        if not valid:
            print(
                "Signature check failed: manifest tampered or creator_pubkey/"
                "signature mismatch.",
                file=sys.stderr,
            )
            return CRYPTO_ERROR

    if args.trusted_pubkey:
        trusted = {pk.strip().lower() for pk in args.trusted_pubkey if pk.strip()}
        creator = (manifest.creator_pubkey or "").lower()
        if not creator or creator not in trusted:
            print(
                f"Creator '{creator or 'unsigned'}' is not in the "
                f"--trusted-pubkey set.",
                file=sys.stderr,
            )
            return TRUST_ERROR

    print("Manifest is valid.")
    return SUCCESS


def _signature_status_label(manifest: SwarmManifest) -> str:
    """Return a human-readable signature status for ``manifest show``."""
    if manifest.signature is None:
        return "ABSENT"
    try:
        from quinkgl.manifest import verify_manifest
    except ImportError:  # pragma: no cover
        return "UNVERIFIED"
    try:
        valid = verify_manifest(manifest)
    except ValueError as exc:
        if exc.args and exc.args[0] == ERR_SIGNING_UNAVAILABLE:
            return "UNVERIFIED (cryptography not installed)"
        return "UNVERIFIED"
    return "VALID" if valid else "INVALID"


def _cmd_magnet(args: argparse.Namespace) -> int:
    try:
        manifest = SwarmManifest.from_file(args.path)
    except Exception as exc:
        code = _handle_manifest_error(exc)
        print(f"Error: {exc}", file=sys.stderr)
        return code

    trackers = args.tracker or None
    bootstrap = args.bootstrap or None
    uri = manifest.to_magnet(trackers=trackers, bootstrap_peers=bootstrap)
    swarm_id = manifest.manifest_hash()

    if args.json:
        print(json.dumps({"magnet": uri, "swarm_id": swarm_id}, indent=2))
    else:
        print(uri)
    return SUCCESS


def run(args: argparse.Namespace) -> int:
    cmd = getattr(args, "manifest_command", None)
    if cmd == "create":
        return _cmd_create(args)
    if cmd == "show":
        return _cmd_show(args)
    if cmd == "verify":
        return _cmd_verify(args)
    if cmd == "magnet":
        return _cmd_magnet(args)
    print("Usage: quinkgl manifest {create|show|verify|magnet} ...", file=sys.stderr)
    return VALIDATION_ERROR
