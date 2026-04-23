# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Phase 3 directory commands: publish, query, discover (spec §17–18)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from quinkgl.fingerprint import DataFingerprint
from quinkgl.manifest import SwarmManifest
from quinkgl.manifest.errors import ERR_SIGNING_UNAVAILABLE
from quinkgl.network.directory import (
    DIRECTORY_COMMUNITY_ID,
    SwarmAdvertisement,
    sign_advertisement,
)

from .exit_codes import (
    CRYPTO_ERROR,
    IO_ERROR,
    SUCCESS,
    VALIDATION_ERROR,
)

if TYPE_CHECKING:
    from argparse import _SubParsersAction


def build_parser(sub: _SubParsersAction) -> None:
    # --- publish -----------------------------------------------------------
    pub = sub.add_parser(
        "publish",
        help="Sign and serialize a SwarmAdvertisement JSON",
    )
    pub.add_argument("--manifest", required=True)
    pub.add_argument("--sign-with", required=True)
    pub.add_argument("--reference-fingerprint", default=None)
    pub.add_argument("--tags", default=None)
    pub.add_argument("--output", required=True)

    # --- query -------------------------------------------------------------
    qry = sub.add_parser(
        "query",
        help="Filter a local SwarmAdvertisement cache",
    )
    qry.add_argument("--cache", required=True)
    qry.add_argument("--tag", action="append", default=[])
    qry.add_argument("--tags", default=None, help="Comma-separated tags (alternative to repeated --tag)")
    qry.add_argument("--input-shape", default=None)
    qry.add_argument("--label-type", default=None)
    qry.add_argument("--trusted-pubkey", action="append", default=[])

    # --- discover ----------------------------------------------------------
    disc = sub.add_parser(
        "discover",
        help="Rank cached ads by affinity against a local fingerprint",
    )
    disc.add_argument("--cache", required=True)
    disc.add_argument("--fingerprint", required=True)
    disc.add_argument("--min-affinity", type=float, default=0.5)
    disc.add_argument("--max-swarms", type=int, default=1)


def _affinity_score(local_fp: DataFingerprint, ref_fp: DataFingerprint) -> float:
    """Delegate to DataFingerprint.affinity_score (Track A API)."""
    return local_fp.affinity_score(ref_fp)


def _load_manifest(path: str) -> SwarmManifest:
    return SwarmManifest.from_file(path)


def _load_fingerprint(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _load_cache(path: str) -> list[SwarmAdvertisement]:
    raw = json.loads(Path(path).read_text())
    out = []
    for item in raw:
        ad = SwarmAdvertisement(
            swarm_id_hex=item["swarm_id_hex"],
            name=item.get("name", ""),
            tags=item.get("tags", []),
            input_shape=item.get("input_shape", []),
            output_shape=item.get("output_shape", []),
            label_type=item.get("label_type", ""),
            data_schema_hash=item.get("data_schema_hash", ""),
            reference_fingerprint=item.get("reference_fingerprint"),
            creator_pubkey=item.get("creator_pubkey"),
            signature=item.get("signature"),
        )
        out.append(ad)
    return out


def _hex_pubkey(pk: str) -> bytes:
    pk = pk.strip()
    if pk.startswith("ed25519:"):
        pk = pk[8:]
    return bytes.fromhex(pk)


def _cmd_publish(args: argparse.Namespace) -> int:
    try:
        manifest = _load_manifest(args.manifest)
    except Exception as exc:
        print(f"Error: failed to load manifest: {exc}", file=sys.stderr)
        return IO_ERROR

    try:
        key_bytes = Path(args.sign_with).read_bytes()
    except FileNotFoundError:
        print(f"Error: key file not found: {args.sign_with}", file=sys.stderr)
        return IO_ERROR

    ref_fp = None
    if args.reference_fingerprint:
        try:
            ref_fp = _load_fingerprint(args.reference_fingerprint)
        except Exception as exc:
            print(f"Error: failed to load fingerprint: {exc}", file=sys.stderr)
            return IO_ERROR

    tags = []
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    ad = SwarmAdvertisement(
        swarm_id_hex=manifest.manifest_hash(),
        name=manifest.name,
        tags=tags,
        input_shape=manifest.task.input_shape,
        output_shape=manifest.task.output_shape,
        label_type=manifest.task.label_type,
        data_schema_hash=manifest.data_schema_hash,
        reference_fingerprint=ref_fp,
    )

    try:
        signed = sign_advertisement(ad, key_bytes)
    except ValueError as exc:
        code = exc.args[0] if exc.args else ""
        if code == ERR_SIGNING_UNAVAILABLE:
            print("Error: cryptography package is not installed.", file=sys.stderr)
            return CRYPTO_ERROR
        print(f"Error: signing failed: {exc}", file=sys.stderr)
        return CRYPTO_ERROR

    payload = {
        "swarm_id_hex": signed.swarm_id_hex,
        "name": signed.name,
        "tags": signed.tags,
        "input_shape": signed.input_shape,
        "output_shape": signed.output_shape,
        "label_type": signed.label_type,
        "data_schema_hash": signed.data_schema_hash,
        "reference_fingerprint": signed.reference_fingerprint,
        "creator_pubkey": signed.creator_pubkey,
        "signature": signed.signature,
    }

    Path(args.output).write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps({"output": args.output, "creator_pubkey": signed.creator_pubkey}))
    else:
        print(f"Advertisement written to {args.output}")
    return SUCCESS


def _cmd_query(args: argparse.Namespace) -> int:
    try:
        ads = _load_cache(args.cache)
    except Exception as exc:
        print(f"Error: failed to load cache: {exc}", file=sys.stderr)
        return IO_ERROR

    results = ads

    tags = list(args.tag)
    if args.tags:
        tags.extend(t.strip() for t in args.tags.split(",") if t.strip())
    if tags:
        wanted = {t.strip().lower() for t in tags}
        results = [a for a in results if wanted.issubset({t.lower() for t in a.tags})]

    if args.input_shape:
        try:
            shape = [int(x) for x in args.input_shape.split(",")]
            results = [a for a in results if a.input_shape == shape]
        except ValueError:
            print("Error: --input-shape must be comma-separated ints", file=sys.stderr)
            return VALIDATION_ERROR

    if args.label_type:
        results = [a for a in results if a.label_type == args.label_type]

    if args.trusted_pubkey:
        trusted = {_hex_pubkey(pk) for pk in args.trusted_pubkey}
        def _pk_bytes(ad):
            pk = ad.creator_pubkey or ""
            if pk.startswith("ed25519:"):
                pk = pk[8:]
            try:
                return bytes.fromhex(pk)
            except ValueError:
                return b""
        results = [a for a in results if _pk_bytes(a) in trusted]

    if args.json:
        print(json.dumps({"results": [_ad_dict(a) for a in results]}))
    else:
        print(f"{len(results)} result(s)")
        for a in results:
            print(f"  {a.swarm_id_hex[:16]}…  {a.name}  tags={','.join(a.tags)}")
    return SUCCESS


def _cmd_discover(args: argparse.Namespace) -> int:
    try:
        ads = _load_cache(args.cache)
    except Exception as exc:
        print(f"Error: failed to load cache: {exc}", file=sys.stderr)
        return IO_ERROR

    try:
        local_fp = DataFingerprint.from_dict(_load_fingerprint(args.fingerprint))
    except Exception as exc:
        print(f"Error: failed to load fingerprint: {exc}", file=sys.stderr)
        return IO_ERROR

    candidates = []
    for ad in ads:
        if ad.reference_fingerprint is None:
            continue
        try:
            ref_fp = DataFingerprint.from_dict(ad.reference_fingerprint)
            score = _affinity_score(local_fp, ref_fp)
        except Exception:
            continue
        if score >= args.min_affinity:
            candidates.append((score, ad))

    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[: args.max_swarms]

    payload = {
        "candidates": [
            {
                "swarm_id_hex": ad.swarm_id_hex,
                "name": ad.name,
                "score": round(score, 4),
                "tags": ad.tags,
            }
            for score, ad in candidates
        ]
    }

    if args.json:
        print(json.dumps(payload))
    else:
        print(f"{len(candidates)} candidate(s)")
        for score, ad in candidates:
            print(f"  {ad.swarm_id_hex[:16]}…  {ad.name}  score={score:.4f}")
    return SUCCESS


def _ad_dict(ad: SwarmAdvertisement) -> dict:
    return {
        "swarm_id_hex": ad.swarm_id_hex,
        "name": ad.name,
        "tags": ad.tags,
        "input_shape": ad.input_shape,
        "output_shape": ad.output_shape,
        "label_type": ad.label_type,
        "data_schema_hash": ad.data_schema_hash,
        "reference_fingerprint": ad.reference_fingerprint,
        "creator_pubkey": ad.creator_pubkey,
        "signature": ad.signature,
    }


def run(args: argparse.Namespace) -> int:
    cmd = getattr(args, "command", None)
    if cmd == "publish":
        return _cmd_publish(args)
    if cmd == "query":
        return _cmd_query(args)
    if cmd == "discover":
        return _cmd_discover(args)
    print("Usage: quinkgl {publish|query|discover} ...", file=sys.stderr)
    return VALIDATION_ERROR
