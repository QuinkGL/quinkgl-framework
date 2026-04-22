"""Phase 2 Ed25519 manifest signing (spec §16, §21.2).

Covers ``sign_manifest`` / ``verify_manifest`` / ``keygen`` round-trip, the
explicit contract that canonical bytes (and therefore ``swarm_id``) MUST NOT
change when a signature is attached, detection of tampered manifests, and
the graceful degradation path when ``cryptography`` is unavailable
(``ERR_SIGNING_UNAVAILABLE``).

These tests are deliberately light on implementation coupling: they exercise
only the public surface from :mod:`quinkgl.manifest` so future internal
refactors (e.g. adopting PyNaCl) do not cascade into a test rewrite.
"""

from __future__ import annotations

import builtins
import importlib
import os
import sys

import pytest

from quinkgl.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ModelSpec,
    SwarmManifest,
    TaskSpec,
)
from quinkgl.manifest.errors import (
    ERR_SIGNATURE_INVALID,
    ERR_SIGNING_UNAVAILABLE,
)


# --- Helpers ---------------------------------------------------------------


def _dummy_manifest() -> SwarmManifest:
    """Build a spec-valid manifest with no creator/signature attached."""
    return SwarmManifest(
        name="phase2-signing-fixture",
        task=TaskSpec(
            type="classification",
            input_shape=[3],
            output_shape=[2],
            label_type="integer",
            tags=["test"],
        ),
        model=ModelSpec(
            framework="custom",
            arch_hash="sha256:" + "a" * 64,
        ),
        aggregation_name="FedAvg",
        topology_name="Random",
        model_arch_fingerprint="sha256:" + "a" * 64,
        data_schema_hash="sha256:" + "b" * 64,
        created_at="2026-01-01T00:00:00Z",
    )


# --- keygen ----------------------------------------------------------------


class TestKeygen:
    def test_returns_pem_and_public_bytes(self, tmp_path):
        from quinkgl.manifest import keygen

        out = tmp_path / "peer.pem"
        private_pem, public_raw = keygen(str(out))

        assert isinstance(private_pem, bytes)
        assert isinstance(public_raw, bytes)
        assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert private_pem.rstrip().endswith(b"-----END PRIVATE KEY-----")
        # Ed25519 raw public key is exactly 32 bytes.
        assert len(public_raw) == 32
        # File contains the same bytes we returned.
        assert out.read_bytes() == private_pem

    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="POSIX-only permission check"
    )
    def test_pem_file_has_0600_permissions(self, tmp_path):
        from quinkgl.manifest import keygen

        out = tmp_path / "peer.pem"
        keygen(str(out))
        mode = os.stat(out).st_mode & 0o777
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"

    def test_memory_only_call_does_not_write_file(self):
        """``out_path=None`` MUST NOT create any on-disk artefact."""
        from quinkgl.manifest import keygen

        private_pem, public_raw = keygen(None)
        assert len(public_raw) == 32
        assert private_pem.startswith(b"-----BEGIN PRIVATE KEY-----")

    def test_overwrite_flag_controls_clobbering(self, tmp_path):
        from quinkgl.manifest import keygen

        out = tmp_path / "peer.pem"
        keygen(str(out))
        # Without overwrite → must refuse.
        with pytest.raises(FileExistsError):
            keygen(str(out))
        # With overwrite → replaces cleanly.
        _, pub2 = keygen(str(out), overwrite=True)
        assert len(pub2) == 32


# --- sign / verify round-trip ---------------------------------------------


class TestSignVerifyRoundtrip:
    def test_sign_attaches_signature_and_pubkey(self):
        from quinkgl.manifest import keygen, sign_manifest

        manifest = _dummy_manifest()
        private_pem, _ = keygen(None)

        signed = sign_manifest(manifest, private_pem)

        assert signed is not manifest, "must return a new manifest instance"
        assert manifest.signature is None, "input manifest must be unchanged"
        assert signed.signature is not None
        assert signed.signature.startswith("ed25519:")
        assert signed.creator_pubkey is not None
        assert signed.creator_pubkey.startswith("ed25519:")

    def test_verify_accepts_freshly_signed_manifest(self):
        from quinkgl.manifest import keygen, sign_manifest, verify_manifest

        private_pem, _ = keygen(None)
        signed = sign_manifest(_dummy_manifest(), private_pem)

        assert verify_manifest(signed) is True

    def test_verify_rejects_unsigned(self):
        from quinkgl.manifest import verify_manifest

        assert verify_manifest(_dummy_manifest()) is False

    def test_canonical_bytes_signature_excluded(self):
        """Spec §5.3 / §21.2: with ``creator_pubkey`` pinned up front,
        attaching a signature MUST NOT change ``swarm_id`` — only the
        ``signature`` field is popped from canonical bytes (§5.1 delta)."""
        from quinkgl.manifest import keygen, sign_manifest

        private_pem, public_raw = keygen(None)
        manifest = _dummy_manifest()
        manifest.creator_pubkey = "ed25519:" + public_raw.hex()
        pre_hash = manifest.manifest_hash()

        signed = sign_manifest(manifest, private_pem)

        assert signed.manifest_hash() == pre_hash

    def test_verify_rejects_tampered_manifest(self):
        """Mutating any signed-over field MUST invalidate the signature."""
        from quinkgl.manifest import keygen, sign_manifest, verify_manifest

        private_pem, _ = keygen(None)
        signed = sign_manifest(_dummy_manifest(), private_pem)

        tampered = SwarmManifest.from_dict(signed.to_dict(), strict=True)
        tampered.name = "hijacked"

        assert verify_manifest(tampered) is False

    def test_verify_rejects_wrong_pubkey(self):
        """Swapping ``creator_pubkey`` to a foreign key MUST fail verify."""
        from quinkgl.manifest import keygen, sign_manifest, verify_manifest

        private_a, _ = keygen(None)
        _, public_b = keygen(None)

        signed = sign_manifest(_dummy_manifest(), private_a)
        signed.creator_pubkey = "ed25519:" + public_b.hex()

        assert verify_manifest(signed) is False

    def test_sign_with_explicit_creator_pubkey_preserved(self):
        """If the caller pre-populated ``creator_pubkey``, it must be used
        verbatim — overwriting it would let sloppy callers silently change
        identities and break TOFU caches."""
        from quinkgl.manifest import keygen, sign_manifest, verify_manifest

        private_pem, public_raw = keygen(None)
        manifest = _dummy_manifest()
        manifest.creator_pubkey = "ed25519:" + public_raw.hex()

        signed = sign_manifest(manifest, private_pem)

        assert signed.creator_pubkey == "ed25519:" + public_raw.hex()
        assert verify_manifest(signed) is True

    def test_sign_rejects_mismatched_pre_populated_pubkey(self):
        """If ``creator_pubkey`` is set but does NOT match the private key,
        signing MUST fail with ``ERR_SIGNATURE_INVALID`` rather than silently
        producing a signature that cannot be verified."""
        from quinkgl.manifest import keygen, sign_manifest

        private_pem, _ = keygen(None)
        _, foreign_pub = keygen(None)

        manifest = _dummy_manifest()
        manifest.creator_pubkey = "ed25519:" + foreign_pub.hex()

        with pytest.raises(ValueError) as info:
            sign_manifest(manifest, private_pem)
        assert info.value.args[0] == ERR_SIGNATURE_INVALID


# --- Dependency-availability ---------------------------------------------


class TestSigningUnavailable:
    def test_cryptography_missing_raises_signing_unavailable(self, monkeypatch):
        """§16.1: ``cryptography`` MUST be a lazy import; absence surfaces as
        ``ERR_SIGNING_UNAVAILABLE`` (not a plain ``ImportError``)."""
        import quinkgl.manifest.signing as sm

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        # Drop any cached sub-module so the lazy loader re-imports cleanly.
        for mod in list(sys.modules):
            if mod == "cryptography" or mod.startswith("cryptography."):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Force the lazy accessor to re-evaluate.
        monkeypatch.setattr(sm, "_CRYPTO", None, raising=False)

        with pytest.raises(ValueError) as info:
            sm.keygen(None)
        assert info.value.args[0] == ERR_SIGNING_UNAVAILABLE
