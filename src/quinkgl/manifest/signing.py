"""Ed25519 manifest signing (spec §16, Phase 2).

Implements the public signing surface documented in §16.2:

* :func:`keygen` — produce an Ed25519 keypair, optionally writing the private
  half as PKCS#8 PEM with POSIX ``0600`` permissions (§16.5, §11.7).
* :func:`sign_manifest` — return a copy of ``manifest`` with the signature
  field populated; ``creator_pubkey`` is derived from the private key if the
  caller did not already set it (§16.3).
* :func:`verify_manifest` — ``True`` iff the signature on ``manifest``
  validates against its ``creator_pubkey``.  Matches the exact contract in
  §16.4, including the early-exit when either field is absent.

The ``cryptography`` package is a **Phase 2 requirement** but is imported
lazily so that Phase-1 deployments that have no need for manifest signing
can install QuinkGL without it.  Any of the three functions above will raise
``ValueError(ERR_SIGNING_UNAVAILABLE, ...)`` when the dependency is missing,
per §16.1.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

from quinkgl.manifest.errors import (
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
)
from quinkgl.manifest.schema import SwarmManifest

if TYPE_CHECKING:  # pragma: no cover — type-only
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _ed25519


# Cached handle to the ``cryptography`` symbols we use so that successful
# imports happen exactly once per process.  ``_load_crypto`` is responsible
# for both populating this and raising the ``ERR_SIGNING_UNAVAILABLE`` the
# spec demands.
_CRYPTO: Optional[dict] = None


def _load_crypto() -> dict:
    """Return the handful of ``cryptography`` entry points we need.

    Raised as ``ERR_SIGNING_UNAVAILABLE`` (not ``ImportError``) so callers
    can use a single ``except ValueError`` branch across schema / crypto /
    trust errors; see §19.4.
    """
    global _CRYPTO
    if _CRYPTO is not None:
        return _CRYPTO
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise ValueError(
            ERR_SIGNING_UNAVAILABLE,
            {
                "detail": (
                    "The 'cryptography' package is required for Phase 2 "
                    "manifest signing. Install it with "
                    "`pip install cryptography>=41.0.0`."
                ),
                "underlying": str(exc),
            },
        ) from exc
    _CRYPTO = {
        "ed25519": ed25519,
        "serialization": serialization,
        "InvalidSignature": InvalidSignature,
    }
    return _CRYPTO


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def keygen(
    out_path: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> Tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Parameters
    ----------
    out_path
        Destination for the PKCS#8-encoded PEM private key.  If ``None``,
        no file is written — useful for in-memory tests and for piping into
        a hardware-backed store.
    overwrite
        When ``False`` (default), refuse to clobber an existing file at
        ``out_path`` and raise :class:`FileExistsError`.  The CLI surfaces
        this as exit code ``3`` (§11.7).

    Returns
    -------
    (private_pem, public_raw)
        ``private_pem`` is the PKCS#8 PEM-encoded private key (bytes);
        ``public_raw`` is the 32-byte raw public key.  The caller is
        responsible for treating the former as a secret.

    Security
    --------
    On POSIX systems the output file is created with ``0o600`` permissions
    (umask-independent) to match §16.5.  On platforms without POSIX
    permissions we rely on the default filesystem ACL — callers deploying
    to those environments SHOULD further constrain access.
    """
    crypto = _load_crypto()
    ed25519 = crypto["ed25519"]
    serialization = crypto["serialization"]

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    if out_path is not None:
        _write_private_pem(out_path, private_pem, overwrite=overwrite)

    return private_pem, public_raw


def _write_private_pem(path: str, data: bytes, *, overwrite: bool) -> None:
    """Atomically write ``data`` to ``path`` with POSIX ``0o600`` perms."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing key file: {path} "
            "(pass overwrite=True or remove the file first)"
        )

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ``os.open`` with explicit mode 0o600 beats chmod-after because it
    # avoids a window where the file is world-readable.  On Windows the
    # mode bits are ignored but the default ACL is already restrictive.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if not sys.platform.startswith("win"):
        flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # pragma: no cover — best-effort on exotic FS
                pass
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise

    # Enforce 0o600 again in case an umask-less environment silently
    # dropped the bits (some CI runners mask ``O_NOFOLLOW``-gated opens).
    if not sys.platform.startswith("win"):
        try:
            os.chmod(path, 0o600)
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Signing / verification
# ---------------------------------------------------------------------------


def sign_manifest(
    manifest: SwarmManifest,
    private_key_pem: bytes,
) -> SwarmManifest:
    """Return a copy of ``manifest`` with a fresh Ed25519 signature attached.

    The input manifest is **not** mutated.  Per §16.3 the signature is
    computed over ``manifest.canonical_bytes()``, which already excludes
    the ``signature`` field itself.

    If the caller pre-populated ``creator_pubkey`` we require that it match
    the public half of ``private_key_pem`` — silently overwriting it would
    let sloppy callers change identities without noticing, breaking TOFU
    caches and advertisement-signing (§17.2).  A mismatch raises
    ``ERR_SIGNATURE_INVALID``.
    """
    crypto = _load_crypto()
    ed25519 = crypto["ed25519"]
    serialization = crypto["serialization"]

    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None
        )
    except Exception as exc:
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {"detail": "private key could not be parsed", "underlying": str(exc)},
        ) from exc

    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {
                "detail": "private key is not Ed25519",
                "got": type(private_key).__name__,
            },
        )

    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    expected_pubkey = "ed25519:" + public_raw.hex()

    if manifest.creator_pubkey is not None and manifest.creator_pubkey != expected_pubkey:
        raise ValueError(
            ERR_SIGNATURE_INVALID,
            {
                "detail": (
                    "manifest.creator_pubkey does not match the public half "
                    "of the supplied private key"
                ),
                "declared": manifest.creator_pubkey,
                "derived": expected_pubkey,
            },
        )

    # Build the to-be-signed manifest with the correct creator_pubkey in
    # place; canonical_bytes() excludes ``signature`` so it is safe to
    # compute the digest *before* we know the signature value (§5.3).
    signed = replace(
        manifest,
        data_policy=deepcopy(manifest.data_policy),
        task=deepcopy(manifest.task),
        model=deepcopy(manifest.model),
        byzantine=deepcopy(manifest.byzantine),
        creator_pubkey=expected_pubkey,
        signature=None,
    )
    sig_bytes = private_key.sign(signed.canonical_bytes())
    signed.signature = "ed25519:" + sig_bytes.hex()
    return signed


def verify_manifest(manifest: SwarmManifest) -> bool:
    """Return ``True`` iff ``manifest`` carries a valid signature.

    Matches §16.4 verbatim: missing signature or missing ``creator_pubkey``
    short-circuits to ``False`` rather than raising, so callers enforcing a
    trust policy can decide independently whether to treat "unsigned" as a
    fatal error (``trust_policy="pinned"``) or an acceptable state
    (``trust_policy="open"``).
    """
    if manifest.signature is None or manifest.creator_pubkey is None:
        return False

    crypto = _load_crypto()
    ed25519 = crypto["ed25519"]
    InvalidSignature = crypto["InvalidSignature"]

    try:
        pubkey_hex = manifest.creator_pubkey.removeprefix("ed25519:")
        sig_hex = manifest.signature.removeprefix("ed25519:")
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        signature = bytes.fromhex(sig_hex)
    except (ValueError, TypeError):
        # Malformed hex or wrong length.  Signature invalid rather than
        # fatal: the caller has already accepted the manifest into memory
        # by the time we run here, so raising would just cascade badly.
        return False

    try:
        public_key.verify(signature, manifest.canonical_bytes())
        return True
    except InvalidSignature:
        return False


__all__ = [
    "keygen",
    "sign_manifest",
    "verify_manifest",
]
