# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""``quinkgl keygen`` — Ed25519 keypair generation (spec §11.7, §16)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from quinkgl.manifest.errors import (
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
)

from .exit_codes import (
    CRYPTO_ERROR,
    IO_ERROR,
    SUCCESS,
)

if TYPE_CHECKING:
    from argparse import _SubParsersAction


def build_parser(sub: _SubParsersAction) -> None:
    parser = sub.add_parser(
        "keygen",
        help="Generate an Ed25519 keypair (writes PKCS#8 PEM, prints pubkey)",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Destination for the PKCS#8 PEM private key. "
        "Required unless --print-public-only is passed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing key file at --output",
    )
    parser.add_argument(
        "--print-public-only",
        action="store_true",
        help="Do not write any file; print the generated pubkey and exit.",
    )


def run(args: argparse.Namespace) -> int:
    # §11.7: --output is required unless the user explicitly asked for the
    # stateless "print the pubkey" mode.  Mirrors the ``required=True``
    # contract in spec §11.7 while still supporting the documented variant.
    if not args.print_public_only and not args.output:
        print(
            "Error: --output is required unless --print-public-only is set.",
            file=sys.stderr,
        )
        return IO_ERROR

    try:
        from quinkgl.manifest import keygen
    except ImportError as exc:  # pragma: no cover — quinkgl itself must import
        print(f"Error: failed to import quinkgl.manifest: {exc}", file=sys.stderr)
        return CRYPTO_ERROR

    try:
        if args.print_public_only:
            _, public_raw = keygen(None)
        else:
            _, public_raw = keygen(args.output, overwrite=args.overwrite)
    except FileExistsError as exc:
        # §11.7: "3 file-exists-without-overwrite" maps onto CRYPTO_ERROR
        # (exit 3) per §11.11.
        print(f"Error: {exc}", file=sys.stderr)
        return CRYPTO_ERROR
    except ValueError as exc:
        # ERR_SIGNING_UNAVAILABLE surfaces as crypto error (exit 3).  Any
        # other ValueError from the signing stack is also a crypto issue.
        code = exc.args[0] if exc.args else ""
        if code == ERR_SIGNING_UNAVAILABLE:
            print(
                "Error: cryptography package is not installed. Install it "
                "with `pip install cryptography>=41.0.0` to enable "
                "manifest signing.",
                file=sys.stderr,
            )
            return CRYPTO_ERROR
        if code == ERR_SIGNATURE_INVALID:
            print(f"Error: signature subsystem rejected request: {exc}", file=sys.stderr)
            return CRYPTO_ERROR
        print(f"Error: {exc}", file=sys.stderr)
        return CRYPTO_ERROR
    except OSError as exc:
        print(f"Error: unable to write private key: {exc}", file=sys.stderr)
        return IO_ERROR

    pubkey = "ed25519:" + public_raw.hex()

    if not args.print_public_only:
        # §11.7 mandates a stderr security notice before printing the pubkey.
        print(
            "Private key written with 0600 permissions. Treat this file as "
            "a secret: anyone with read access can impersonate this creator "
            "when signing manifests.",
            file=sys.stderr,
        )

    if getattr(args, "json", False):
        payload = {"public_key": pubkey}
        if not args.print_public_only:
            payload["output"] = args.output
        print(json.dumps(payload))
    else:
        # Pubkey on stdout's last line so operators can pipe it directly
        # into `--trusted-pubkey` flags.
        print(pubkey)
    return SUCCESS
