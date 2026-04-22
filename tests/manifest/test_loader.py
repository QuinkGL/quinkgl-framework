"""Unified manifest loader (spec §10.3).

``load_manifest`` dispatches on ``source`` prefix:

* ``quinkgl:?…`` → parse magnet, call ``peer_fetcher(swarm_id)`` to obtain
  canonical bytes, verify hash, parse.  No ``peer_fetcher`` ⇒
  ``ERR_MANIFEST_FETCH_REQUIRED``.
* ``http(s)://…`` → HTTP GET, parse response body.
* otherwise → filesystem path, delegates to
  :meth:`SwarmManifest.from_file`.

Post-load invariant: when ``expected_swarm_id`` is provided,
``SHA-256(canonical_bytes) == expected_swarm_id``; mismatch raises
``ERR_MANIFEST_HASH_MISMATCH``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from quinkgl.manifest import SwarmManifest, load_manifest
from quinkgl.manifest import errors as E


def _sample() -> SwarmManifest:
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="LoaderTest",
    )


# --- Filesystem dispatch ----------------------------------------------------


class TestFilesystemSource:
    def test_load_from_file(self, tmp_path: Path):
        p = tmp_path / "swarm.qgl"
        m = _sample()
        m.to_file(p, pretty=False)
        restored = load_manifest(str(p))
        assert restored.manifest_hash() == m.manifest_hash()

    def test_pathlike_source_accepted(self, tmp_path: Path):
        p = tmp_path / "swarm.qgl"
        _sample().to_file(p)
        assert load_manifest(p) is not None  # type: ignore[arg-type]

    def test_expected_swarm_id_match(self, tmp_path: Path):
        p = tmp_path / "swarm.qgl"
        m = _sample()
        m.to_file(p, pretty=False)
        expected = bytes.fromhex(m.manifest_hash())
        load_manifest(str(p), expected_swarm_id=expected)  # no raise

    def test_expected_swarm_id_mismatch(self, tmp_path: Path):
        p = tmp_path / "swarm.qgl"
        _sample().to_file(p, pretty=False)
        with pytest.raises(ValueError) as exc:
            load_manifest(str(p), expected_swarm_id=b"\x00" * 32)
        assert exc.value.args[0] == E.ERR_MANIFEST_HASH_MISMATCH


# --- Magnet dispatch --------------------------------------------------------


class TestMagnetSource:
    def test_magnet_without_peer_fetcher_raises(self):
        m = _sample()
        uri = m.to_magnet()
        with pytest.raises(ValueError) as exc:
            load_manifest(uri)
        assert exc.value.args[0] == E.ERR_MANIFEST_FETCH_REQUIRED

    def test_magnet_with_peer_fetcher(self):
        m = _sample()
        canonical = m.canonical_bytes()
        expected_id = hashlib.sha256(canonical).digest()

        calls: list[bytes] = []

        def fetcher(swarm_id: bytes) -> bytes:
            calls.append(swarm_id)
            return canonical

        restored = load_manifest(m.to_magnet(), peer_fetcher=fetcher)
        assert calls == [expected_id]
        assert restored.manifest_hash() == m.manifest_hash()

    def test_magnet_peer_returns_wrong_bytes(self):
        """Fetcher that returns bytes whose SHA-256 ≠ magnet xt must fail."""
        m = _sample()
        uri = m.to_magnet()

        def bad_fetcher(swarm_id: bytes) -> bytes:
            # Return a *valid* manifest, but not the one the magnet points to.
            other = SwarmManifest(
                model_arch_fingerprint="xyz",
                data_schema_hash="qqq",
                name="WrongOne",
            )
            return other.canonical_bytes()

        with pytest.raises(ValueError) as exc:
            load_manifest(uri, peer_fetcher=bad_fetcher)
        assert exc.value.args[0] == E.ERR_MANIFEST_HASH_MISMATCH

    def test_expected_swarm_id_mismatch_overrides_magnet(self):
        m = _sample()

        def fetcher(_sid: bytes) -> bytes:
            return m.canonical_bytes()

        with pytest.raises(ValueError) as exc:
            load_manifest(
                m.to_magnet(),
                peer_fetcher=fetcher,
                expected_swarm_id=b"\x00" * 32,
            )
        assert exc.value.args[0] == E.ERR_MANIFEST_HASH_MISMATCH


# --- HTTP dispatch (stubbed) ------------------------------------------------


class TestHttpSource:
    """We stub ``urllib.request.urlopen`` rather than hit the network.

    The loader's contract we need to verify is: it calls urlopen with the
    right URL, reads the body, and feeds it to ``from_dict``.
    """

    def test_https_source_parses_body(self, monkeypatch: pytest.MonkeyPatch):
        import quinkgl.manifest.loader as loader_mod

        m = _sample()
        canonical = m.canonical_bytes()

        class FakeResponse:
            headers = {"Content-Type": "application/vnd.quinkgl+json"}

            def read(self, _n=None):
                return canonical

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(url, timeout=None):
            assert url == "https://example/swarm.qgl"
            return FakeResponse()

        monkeypatch.setattr(loader_mod, "_urlopen", fake_urlopen)
        restored = load_manifest("https://example/swarm.qgl")
        assert restored.manifest_hash() == m.manifest_hash()
