"""`.qgl` file-format round-trip (spec §7, §10.2).

`SwarmManifest.from_file` / `to_file` MUST:

* Read and write UTF-8 JSON with no BOM.
* Emit either pretty (2-space indent) or canonical form per the ``pretty``
  kwarg.  Both forms MUST round-trip to a byte-identical canonical hash.
* Reject files larger than 1 MiB (defensive DOS guard, TASKS_SPLIT A-4
  acceptance).
* Raise ``ValueError(ERR_MANIFEST_INVALID_JSON, …)`` on malformed input.
* Write atomically (``tmp + rename``) so that an interrupted write does not
  corrupt the destination file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quinkgl.manifest import SwarmManifest
from quinkgl.manifest import errors as E


def _sample() -> SwarmManifest:
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="FileRoundtrip",
    )


# --- Round-trip -------------------------------------------------------------


class TestRoundTrip:
    def test_pretty_roundtrip(self, tmp_path: Path):
        path = tmp_path / "swarm.qgl"
        m = _sample()
        m.to_file(path, pretty=True)
        data = path.read_bytes()
        # Pretty form uses newlines/indentation.
        assert b"\n" in data
        assert data.startswith(b"{")
        restored = SwarmManifest.from_file(path)
        assert restored.manifest_hash() == m.manifest_hash()

    def test_canonical_roundtrip(self, tmp_path: Path):
        path = tmp_path / "swarm.qgl"
        m = _sample()
        m.to_file(path, pretty=False)
        data = path.read_bytes()
        # Canonical form uses no insignificant whitespace.
        assert b"\n" not in data
        # Must equal canonical_bytes() exactly (§7: canonical-on-disk form).
        assert data == m.canonical_bytes()
        restored = SwarmManifest.from_file(path)
        assert restored.manifest_hash() == m.manifest_hash()

    def test_to_file_accepts_str_or_path(self, tmp_path: Path):
        p = tmp_path / "a.qgl"
        m = _sample()
        m.to_file(str(p))
        assert p.exists()
        SwarmManifest.from_file(str(p))  # should not raise


# --- Invalid inputs ---------------------------------------------------------


class TestInvalidInput:
    def test_invalid_json_raises(self, tmp_path: Path):
        p = tmp_path / "bad.qgl"
        p.write_bytes(b"{not: json")
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_file(p)
        assert exc.value.args[0] == E.ERR_MANIFEST_INVALID_JSON

    def test_bom_rejected(self, tmp_path: Path):
        p = tmp_path / "bom.qgl"
        p.write_bytes(b"\xef\xbb\xbf" + _sample().canonical_bytes())
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_file(p)
        # Either invalid_json or field_invalid is acceptable — the spec is
        # that UTF-8 BOMs are not allowed.  We pin on invalid_json because
        # a BOM corrupts the JSON-syntax layer before field validation.
        assert exc.value.args[0] == E.ERR_MANIFEST_INVALID_JSON

    def test_oversized_file_rejected(self, tmp_path: Path):
        p = tmp_path / "big.qgl"
        # Just over the 1 MiB guard; content doesn't have to be valid JSON.
        p.write_bytes(b"{" + b" " * (1024 * 1024))
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_file(p)
        assert exc.value.args[0] == E.ERR_MANIFEST_INVALID_JSON

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path):
        p = tmp_path / "missing.qgl"
        with pytest.raises(FileNotFoundError):
            SwarmManifest.from_file(p)


# --- Atomicity --------------------------------------------------------------


class TestAtomicWrite:
    def test_rewrite_does_not_leave_tmp_files(self, tmp_path: Path):
        p = tmp_path / "swarm.qgl"
        m = _sample()
        for _ in range(3):
            m.to_file(p)
        siblings = sorted(os.listdir(tmp_path))
        assert siblings == ["swarm.qgl"], siblings

    def test_existing_file_preserved_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        p = tmp_path / "swarm.qgl"
        _sample().to_file(p)
        original = p.read_bytes()

        import os as _os

        real_replace = _os.replace

        def boom(src, dst):  # simulate crash after tmp write, before rename
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(_os, "replace", boom)
        with pytest.raises(RuntimeError):
            _sample().to_file(p)
        # After the crash the original file must still be there and intact.
        assert p.read_bytes() == original
        monkeypatch.setattr(_os, "replace", real_replace)
