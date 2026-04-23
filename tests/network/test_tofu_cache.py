"""TOFU creator-key cache persistence + conflict semantics (spec §15.1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quinkgl.manifest.errors import ERR_TRUST_TOFU_CONFLICT
from quinkgl.network.tofu import (
    TofuCache,
    TofuEntry,
    default_tofu_cache_path,
)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "ipv8work" / "tofu_creators.json"


class TestTofuEntry:
    def test_round_trip(self):
        e = TofuEntry("ed25519:deadbeef", "2026-04-22T10:00:00Z")
        assert TofuEntry.from_dict(e.to_dict()).creator_pubkey == "ed25519:deadbeef"

    def test_from_dict_rejects_bad_payload(self):
        with pytest.raises(ValueError):
            TofuEntry.from_dict({"creator_pubkey": "x"})
        with pytest.raises(ValueError):
            TofuEntry.from_dict({"first_seen": "2026-04-22T10:00:00Z"})


class TestFirstEncounter:
    def test_record_creates_file_with_entry(self, cache_path: Path):
        cache = TofuCache(cache_path)
        entry = cache.record_or_validate("swarm-abc", "ed25519:aa")

        assert entry.creator_pubkey == "ed25519:aa"
        assert entry.first_seen.endswith("Z")
        # File must actually exist on disk — the whole point of TOFU
        # is that it survives a restart.
        assert cache_path.exists()
        payload = json.loads(cache_path.read_text())
        assert payload["swarm-abc"]["creator_pubkey"] == "ed25519:aa"

    def test_file_mode_is_0600_on_posix(self, cache_path: Path):
        if os.name != "posix":
            pytest.skip("POSIX-only permission check")
        cache = TofuCache(cache_path)
        cache.record_or_validate("swarm-abc", "ed25519:aa")
        mode = cache_path.stat().st_mode & 0o777
        assert mode == 0o600


class TestSecondEncounter:
    def test_same_pubkey_is_noop(self, cache_path: Path):
        cache = TofuCache(cache_path)
        first = cache.record_or_validate("swarm-abc", "ed25519:aa")
        second = cache.record_or_validate("swarm-abc", "ed25519:aa")
        assert first.first_seen == second.first_seen

    def test_different_pubkey_raises_tofu_conflict(self, cache_path: Path):
        cache = TofuCache(cache_path)
        cache.record_or_validate("swarm-abc", "ed25519:aa")

        with pytest.raises(ValueError) as excinfo:
            cache.record_or_validate("swarm-abc", "ed25519:bb")

        err_code, payload = excinfo.value.args
        assert err_code == ERR_TRUST_TOFU_CONFLICT
        assert payload["expected"] == "ed25519:aa"
        assert payload["actual"] == "ed25519:bb"
        assert payload["swarm_id"] == "swarm-abc"

    def test_conflict_does_not_mutate_cache(self, cache_path: Path):
        cache = TofuCache(cache_path)
        cache.record_or_validate("swarm-abc", "ed25519:aa")
        with pytest.raises(ValueError):
            cache.record_or_validate("swarm-abc", "ed25519:bb")

        # Re-open from disk to be sure the on-disk state is untouched.
        reloaded = TofuCache(cache_path)
        entry = reloaded.get("swarm-abc")
        assert entry is not None
        assert entry.creator_pubkey == "ed25519:aa"


class TestPersistence:
    def test_survives_fresh_instance(self, cache_path: Path):
        TofuCache(cache_path).record_or_validate("s1", "ed25519:aa")
        TofuCache(cache_path).record_or_validate("s2", "ed25519:bb")

        snap = TofuCache(cache_path).as_dict()
        assert set(snap.keys()) == {"s1", "s2"}
        assert snap["s1"]["creator_pubkey"] == "ed25519:aa"

    def test_corrupt_json_degrades_to_empty(
        self, cache_path: Path, caplog: pytest.LogCaptureFixture
    ):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("{not json")
        cache = TofuCache(cache_path)
        # Should NOT raise and should accept a fresh recording.
        entry = cache.record_or_validate("s1", "ed25519:aa")
        assert entry.creator_pubkey == "ed25519:aa"

    def test_non_object_root_degrades_to_empty(self, cache_path: Path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(["not", "an", "object"]))
        cache = TofuCache(cache_path)
        cache.record_or_validate("s1", "ed25519:aa")  # must not raise
        assert cache.get("s1") is not None


class TestMaintenance:
    def test_forget_removes_and_returns_true(self, cache_path: Path):
        cache = TofuCache(cache_path)
        cache.record_or_validate("s1", "ed25519:aa")
        assert cache.forget("s1") is True
        assert cache.get("s1") is None

    def test_forget_missing_returns_false(self, cache_path: Path):
        cache = TofuCache(cache_path)
        assert cache.forget("unknown") is False

    def test_clear_wipes_everything(self, cache_path: Path):
        cache = TofuCache(cache_path)
        cache.record_or_validate("s1", "ed25519:aa")
        cache.record_or_validate("s2", "ed25519:bb")
        cache.clear()
        assert cache.as_dict() == {}


class TestValidation:
    def test_rejects_empty_swarm_id(self, cache_path: Path):
        cache = TofuCache(cache_path)
        with pytest.raises(ValueError):
            cache.record_or_validate("", "ed25519:aa")

    def test_rejects_empty_pubkey(self, cache_path: Path):
        cache = TofuCache(cache_path)
        with pytest.raises(ValueError):
            cache.record_or_validate("s1", "")

    def test_rejects_non_string_types(self, cache_path: Path):
        cache = TofuCache(cache_path)
        with pytest.raises(ValueError):
            cache.record_or_validate(None, "ed25519:aa")  # type: ignore[arg-type]


class TestDefaultPath:
    def test_uses_ipv8_work_dir_env_var(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("IPV8_WORK_DIR", str(tmp_path / "ipv8"))
        assert default_tofu_cache_path() == tmp_path / "ipv8" / "tofu_creators.json"

    def test_explicit_work_dir_wins_over_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("IPV8_WORK_DIR", str(tmp_path / "from_env"))
        result = default_tofu_cache_path(work_dir=str(tmp_path / "explicit"))
        assert result == tmp_path / "explicit" / "tofu_creators.json"
