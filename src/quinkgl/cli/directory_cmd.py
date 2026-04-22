# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""``quinkgl publish`` / ``query`` / ``discover`` — Phase 3 directory CLI.

These three commands stay intentionally transport-free so they work
without a running IPv8 reactor: ads travel as JSON files on disk,
so an operator can mint a signed advertisement with ``publish``, hand
the file off to a running node out of band, and later use ``query`` or
``discover`` against a cache snapshot exported by that node.  Once the
in-reactor :class:`SwarmDirectoryCommunity` grows a public "dump cache
to JSON / load cache from JSON" surface, these commands can be wired
straight onto the live overlay without changing their UX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from quinkgl.manifest.errors import (
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
)

from .exit_codes import (
    CRYPTO_ERROR,
    IO_ERROR,
    SUCCESS,
    TRUST_ERROR,
    VALIDATION_ERROR,
)

if TYPE_CHECKING:
    from argparse import _SubParsersAction


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def build_parser(sub: "_SubParsersAction") -> None:
    _build_publish_parser(sub)
    _build_query_parser(sub)
    _build_discover_parser(sub)


def _build_publish_parser(sub: "_SubParsersAction") -> None:
    parser = sub.add_parser(
        "publish",
        help="Mint a signed SwarmAdvertisement for a manifest "
        "(writes JSON; no live broadcast).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the .qgl manifest that this advertisement describes.",
    )
    parser.add_argument(
        "--sign-with",
        required=True,
        help="Path to the PKCS#8 PEM Ed25519 private key used to sign the ad.",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tag list (e.g. 'vision,pytorch').",
    )
    parser.add_argument(
        "--reference-fingerprint",
        default=None,
        help="Path to a JSON dump of a DataFingerprint to embed in the ad.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSON file for the signed advertisement.",
    )


def _build_query_parser(sub: "_SubParsersAction") -> None:
    parser = sub.add_parser(
        "query",
        help="Filter a local directory cache (JSON file of advertisements).",
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to a JSON file holding a list of advertisement objects.",
    )
    parser.add_argument("--tags", default=None, help="Comma-separated tag filter.")
    parser.add_argument(
        "--input-shape",
        default=None,
        help="Comma-separated integer shape (e.g. '3,32,32').",
    )
    parser.add_argument("--label-type", default=None)
    parser.add_argument(
        "--trusted-pubkey",
        action="append",
        default=[],
        help=(
            "Hex-encoded raw 32-byte Ed25519 public key.  Repeatable.  "
            "When set, only ads signed by one of the listed creators pass."
        ),
    )


def _build_discover_parser(sub: "_SubParsersAction") -> None:
    parser = sub.add_parser(
        "discover",
        help="Rank directory ads by affinity against a local fingerprint.",
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument(
        "--fingerprint",
        required=True,
        help="Path to a JSON dump of the caller's DataFingerprint.",
    )
    parser.add_argument("--tags", default=None)
    parser.add_argument("--input-shape", default=None)
    parser.add_argument("--label-type", default=None)
    parser.add_argument("--min-affinity", type=float, default=0.5)
    parser.add_argument("--max-swarms", type=int, default=None)
    parser.add_argument(
        "--trust-policy",
        default="open",
        choices=["open", "pinned"],
    )
    parser.add_argument(
        "--trusted-pubkey",
        action="append",
        default=[],
        help="Hex-encoded raw Ed25519 public key; repeatable.",
    )


# ---------------------------------------------------------------------------
# Dispatch (__main__ routes "publish" / "query" / "discover" here)
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if args.command == "publish":
        return _cmd_publish(args)
    if args.command == "query":
        return _cmd_query(args)
    if args.command == "discover":
        return _cmd_discover(args)
    print(f"Error: unknown directory subcommand {args.command!r}", file=sys.stderr)
    return VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_input_shape(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw == "":
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"--input-shape must be integers, got {raw!r}: {exc}") from exc


def _parse_trusted_pubkeys(raw_list: List[str]) -> Optional[Set[bytes]]:
    if not raw_list:
        return None
    keys: Set[bytes] = set()
    for raw in raw_list:
        hex_part = raw.split(":", 1)[1] if raw.startswith("ed25519:") else raw
        try:
            pub = bytes.fromhex(hex_part)
        except ValueError as exc:
            raise ValueError(
                f"invalid --trusted-pubkey {raw!r}: not hex"
            ) from exc
        if len(pub) != 32:
            raise ValueError(
                f"invalid --trusted-pubkey {raw!r}: expected 32 raw bytes, "
                f"got {len(pub)}"
            )
        keys.add(pub)
    return keys


def _ad_to_dict(ad) -> Dict[str, Any]:
    return {
        "swarm_id_hex": ad.swarm_id_hex,
        "name": ad.name,
        "tags": list(ad.tags),
        "input_shape": list(ad.input_shape),
        "output_shape": list(ad.output_shape),
        "label_type": ad.label_type,
        "data_schema_hash": ad.data_schema_hash,
        "reference_fingerprint": dict(ad.reference_fingerprint),
        "creator_pubkey": ad.creator_pubkey,
        "signature": ad.signature,
    }


def _ad_from_dict(data: Dict[str, Any]):
    from quinkgl.network.directory import SwarmAdvertisement

    return SwarmAdvertisement(
        swarm_id_hex=data.get("swarm_id_hex", ""),
        name=data.get("name", ""),
        tags=list(data.get("tags", [])),
        input_shape=list(data.get("input_shape", [])),
        output_shape=list(data.get("output_shape", [])),
        label_type=data.get("label_type", ""),
        data_schema_hash=data.get("data_schema_hash", ""),
        reference_fingerprint=dict(data.get("reference_fingerprint", {})),
        creator_pubkey=data.get("creator_pubkey"),
        signature=data.get("signature"),
    )


def _load_cache(cache_path: str):
    """Read an ads-list JSON file and return a pre-seeded directory + list."""
    from quinkgl.network.directory import SwarmDirectoryCommunity

    try:
        raw = Path(cache_path).read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cache file not found: {cache_path}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cache file is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"cache file must be a JSON list of ads, got {type(payload).__name__}"
        )

    ads = [_ad_from_dict(entry) for entry in payload]
    community = SwarmDirectoryCommunity(
        # The on-disk cache is already the operator-curated list — skip
        # the local rate limiter entirely so ingest doesn't randomly drop
        # ads based on wall-clock state the CLI has no say in.
        max_ads_per_creator_per_day=10**9,
        max_ads_per_session=10**9,
    )
    for ad in ads:
        community.ingest(ad, verify=True)
    return community, ads


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _cmd_publish(args: argparse.Namespace) -> int:
    from quinkgl.manifest.schema import SwarmManifest
    from quinkgl.network.directory import (
        SwarmAdvertisement,
        sign_advertisement,
    )

    try:
        manifest = SwarmManifest.from_file(args.manifest)
    except FileNotFoundError as exc:
        print(f"Error: manifest not found: {exc}", file=sys.stderr)
        return IO_ERROR
    except ValueError as exc:
        print(f"Error: manifest is invalid: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    try:
        private_pem = Path(args.sign_with).read_bytes()
    except FileNotFoundError as exc:
        print(f"Error: private key not found: {exc}", file=sys.stderr)
        return IO_ERROR
    except OSError as exc:
        print(f"Error: cannot read private key: {exc}", file=sys.stderr)
        return IO_ERROR

    reference_fingerprint: Dict[str, Any] = {}
    if args.reference_fingerprint:
        try:
            reference_fingerprint = json.loads(
                Path(args.reference_fingerprint).read_text()
            )
        except FileNotFoundError as exc:
            print(f"Error: reference fingerprint not found: {exc}", file=sys.stderr)
            return IO_ERROR
        except json.JSONDecodeError as exc:
            print(
                f"Error: reference fingerprint is not valid JSON: {exc}",
                file=sys.stderr,
            )
            return VALIDATION_ERROR

    # ``task`` is a TaskSpec dataclass on modern manifests, but older test
    # fixtures may pass a raw dict.  Normalise into a dict either way.
    task = manifest.task
    task_dict = task.to_dict() if hasattr(task, "to_dict") else dict(task or {})
    ad = SwarmAdvertisement(
        swarm_id_hex=manifest.manifest_hash(),
        name=manifest.name,
        tags=_parse_tags(args.tags),
        input_shape=list(task_dict.get("input_shape", [])),
        output_shape=list(task_dict.get("output_shape", [])),
        label_type=task_dict.get("label_type", ""),
        data_schema_hash=getattr(manifest, "data_schema_hash", "") or "",
        reference_fingerprint=reference_fingerprint,
    )

    try:
        signed = sign_advertisement(ad, private_pem)
    except ValueError as exc:
        code = exc.args[0] if exc.args else ""
        if code == ERR_SIGNING_UNAVAILABLE:
            print(
                "Error: cryptography package is not installed. Install "
                "`cryptography>=41.0.0` to publish signed advertisements.",
                file=sys.stderr,
            )
        elif code == ERR_SIGNATURE_INVALID:
            print(f"Error: signing rejected: {exc}", file=sys.stderr)
        else:
            print(f"Error: signing failed: {exc}", file=sys.stderr)
        return CRYPTO_ERROR

    try:
        Path(args.output).write_text(
            json.dumps(_ad_to_dict(signed), indent=2, sort_keys=False)
        )
    except OSError as exc:
        print(f"Error: cannot write advertisement: {exc}", file=sys.stderr)
        return IO_ERROR

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "swarm_id": signed.swarm_id_hex,
                    "creator_pubkey": signed.creator_pubkey,
                    "signed": True,
                    "output": args.output,
                }
            )
        )
    else:
        print(f"swarm_id: {signed.swarm_id_hex}")
        print(f"creator: {signed.creator_pubkey}")
        print(f"wrote:   {args.output}")
    return SUCCESS


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def _cmd_query(args: argparse.Namespace) -> int:
    try:
        community, _ = _load_cache(args.cache)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return IO_ERROR
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    try:
        trusted = _parse_trusted_pubkeys(list(args.trusted_pubkey or []))
        shape = _parse_input_shape(args.input_shape)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    try:
        results = community.query(
            tags=_parse_tags(args.tags) or None,
            input_shape=shape,
            label_type=args.label_type,
            trusted_creators=trusted,
        )
    except ValueError as exc:
        # Catch the lazy-crypto ERR_SIGNING_UNAVAILABLE raised by
        # verify_advertisement when a trusted-pubkey filter is active.
        code = exc.args[0] if exc.args else ""
        if code == ERR_SIGNING_UNAVAILABLE:
            print(
                "Error: cryptography package is required for --trusted-pubkey.",
                file=sys.stderr,
            )
            return CRYPTO_ERROR
        raise

    payload = {"results": [_ad_to_dict(ad) for ad in results]}
    if getattr(args, "json", False):
        print(json.dumps(payload))
    else:
        if not results:
            print("No matching advertisements.")
            return SUCCESS
        for ad in results:
            print(
                f"{ad.swarm_id_hex}  {ad.name}  tags={','.join(ad.tags)}  "
                f"creator={ad.creator_pubkey}"
            )
    return SUCCESS


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


def _cmd_discover(args: argparse.Namespace) -> int:
    from quinkgl.fingerprint import DataFingerprint
    from quinkgl.network.auto_discovery import rank_candidates

    try:
        community, _ = _load_cache(args.cache)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return IO_ERROR
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    try:
        fingerprint_data = json.loads(Path(args.fingerprint).read_text())
        fingerprint = DataFingerprint.from_dict(fingerprint_data)
    except FileNotFoundError as exc:
        print(f"Error: fingerprint file not found: {exc}", file=sys.stderr)
        return IO_ERROR
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error: fingerprint is not a valid DataFingerprint: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    try:
        trusted = _parse_trusted_pubkeys(list(args.trusted_pubkey or []))
        shape = _parse_input_shape(args.input_shape)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    if args.trust_policy == "pinned" and not trusted:
        print(
            "Error: --trust-policy=pinned requires at least one --trusted-pubkey.",
            file=sys.stderr,
        )
        return TRUST_ERROR

    try:
        ranked = rank_candidates(
            directory=community,
            fingerprint=fingerprint,
            tags=_parse_tags(args.tags) or None,
            input_shape=shape,
            label_type=args.label_type,
            min_affinity=args.min_affinity,
            max_swarms=args.max_swarms,
            trust_policy=args.trust_policy,
            trusted_creator_pubkeys=trusted,
        )
    except ValueError as exc:
        code = exc.args[0] if exc.args else ""
        if code == ERR_SIGNING_UNAVAILABLE:
            print(
                "Error: cryptography package is required for signature "
                "verification in discover.",
                file=sys.stderr,
            )
            return CRYPTO_ERROR
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR

    candidates = [
        {"score": round(float(score), 6), **_ad_to_dict(ad)} for score, ad in ranked
    ]
    if getattr(args, "json", False):
        print(json.dumps({"candidates": candidates}))
    else:
        if not candidates:
            print("No candidates passed the affinity filter.")
            return SUCCESS
        for entry in candidates:
            print(
                f"{entry['score']:.3f}  {entry['swarm_id_hex']}  {entry['name']}  "
                f"tags={','.join(entry['tags'])}"
            )
    return SUCCESS
