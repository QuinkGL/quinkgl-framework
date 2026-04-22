"""CLI tests for the Phase 3 directory commands
(``quinkgl publish`` / ``quinkgl query`` / ``quinkgl discover``).

These commands operate entirely on on-disk state so they stay useful
without a running IPv8 reactor: ``publish`` mints a signed
:class:`SwarmAdvertisement` into a JSON file, ``query`` filters a JSON
cache of ads, and ``discover`` ranks them by affinity against a local
fingerprint file.  A full in-reactor deployment will pipe the same
files into live community state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main as cli_main
from quinkgl.fingerprint import DataFingerprint
from quinkgl.manifest import keygen
from quinkgl.manifest import ModelSpec, SwarmManifest, TaskSpec
from quinkgl.network.directory import (
    SwarmAdvertisement,
    sign_advertisement,
)


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def tmp_priv_path(tmp_path: Path) -> Path:
    pem_path = tmp_path / "signer.pem"
    keygen(str(pem_path))
    return pem_path


@pytest.fixture
def tmp_manifest_path(tmp_path: Path) -> Path:
    manifest = SwarmManifest(
        name="phase3-cli-swarm",
        task=TaskSpec(
            type="classification",
            input_shape=[3, 32, 32],
            output_shape=[10],
            label_type="integer",
            tags=["test"],
        ),
        model=ModelSpec(
            framework="pytorch",
            arch_hash="sha256:" + "0" * 64,
        ),
        aggregation_name="FedAvg",
        topology_name="Random",
        model_arch_fingerprint="sha256:" + "0" * 64,
        data_schema_hash="sha256:" + "b" * 64,
        created_at="2026-01-01T00:00:00Z",
    )
    out = tmp_path / "swarm.qgl"
    manifest.to_file(str(out), pretty=True)
    return out


def _write_fingerprint(path: Path, *, bucket: str = "medium") -> Path:
    fp = DataFingerprint(
        label_buckets={"cls": bucket},
        noised_moments={"f": (0.1, 0.2)},
        sample_bucket="medium",
        num_classes=3,
    )
    path.write_text(json.dumps(fp.to_dict()))
    return path


def _signed_ad(swarm_id: str, priv_pem: bytes, *, tags=None, bucket="medium"):
    fp = DataFingerprint(
        label_buckets={"cls": bucket},
        noised_moments={"f": (0.1, 0.2)},
        sample_bucket="medium",
        num_classes=3,
    )
    ad = SwarmAdvertisement(
        swarm_id_hex=swarm_id,
        name=f"swarm-{swarm_id[:6]}",
        tags=list(tags or ["vision"]),
        input_shape=[3, 32, 32],
        output_shape=[10],
        label_type="integer",
        data_schema_hash="sha256:" + "0" * 64,
        reference_fingerprint=fp.to_dict(),
    )
    return sign_advertisement(ad, priv_pem)


def _dump_cache(path: Path, ads) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "swarm_id_hex": a.swarm_id_hex,
                    "name": a.name,
                    "tags": a.tags,
                    "input_shape": a.input_shape,
                    "output_shape": a.output_shape,
                    "label_type": a.label_type,
                    "data_schema_hash": a.data_schema_hash,
                    "reference_fingerprint": a.reference_fingerprint,
                    "creator_pubkey": a.creator_pubkey,
                    "signature": a.signature,
                }
                for a in ads
            ]
        )
    )
    return path


# --- publish ---------------------------------------------------------------


class TestPublishCLI:
    def test_publish_writes_signed_ad_json(
        self, tmp_path: Path, tmp_manifest_path: Path, tmp_priv_path: Path, capsys
    ):
        fp_path = _write_fingerprint(tmp_path / "fp.json")
        out_path = tmp_path / "ad.json"

        rc = cli_main(
            [
                "publish",
                "--manifest",
                str(tmp_manifest_path),
                "--sign-with",
                str(tmp_priv_path),
                "--reference-fingerprint",
                str(fp_path),
                "--tags",
                "vision,pytorch",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["creator_pubkey"].startswith("ed25519:")
        assert data["signature"].startswith("ed25519:")
        assert set(data["tags"]) == {"vision", "pytorch"}

    def test_publish_missing_key_is_io_error(
        self, tmp_path: Path, tmp_manifest_path: Path, capsys
    ):
        out_path = tmp_path / "ad.json"
        rc = cli_main(
            [
                "publish",
                "--manifest",
                str(tmp_manifest_path),
                "--sign-with",
                str(tmp_path / "nonexistent.pem"),
                "--output",
                str(out_path),
            ]
        )
        assert rc == 2  # IO_ERROR
        assert not out_path.exists()


# --- query -----------------------------------------------------------------


class TestQueryCLI:
    def test_query_filters_by_tags_and_input_shape(
        self, tmp_path: Path, tmp_priv_path: Path, capsys
    ):
        priv_pem = tmp_priv_path.read_bytes()
        ads = [
            _signed_ad("a" * 64, priv_pem, tags=["vision", "pytorch"]),
            _signed_ad("b" * 64, priv_pem, tags=["audio"]),
        ]
        cache_path = _dump_cache(tmp_path / "cache.json", ads)

        rc = cli_main(
            [
                "--json",
                "query",
                "--cache",
                str(cache_path),
                "--tags",
                "vision",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert [a["swarm_id_hex"] for a in payload["results"]] == ["a" * 64]

    def test_query_trusted_pubkey_filter(
        self, tmp_path: Path, tmp_priv_path: Path, capsys
    ):
        priv_pem = tmp_priv_path.read_bytes()
        # Different signer for the second ad.
        other_pem_path = tmp_path / "other.pem"
        keygen(str(other_pem_path))
        other_pem = other_pem_path.read_bytes()

        ads = [
            _signed_ad("a" * 64, priv_pem),
            _signed_ad("b" * 64, other_pem),
        ]
        cache_path = _dump_cache(tmp_path / "cache.json", ads)

        trusted_hex = ads[0].creator_pubkey.split(":", 1)[1]
        rc = cli_main(
            [
                "--json",
                "query",
                "--cache",
                str(cache_path),
                "--trusted-pubkey",
                trusted_hex,
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert [a["swarm_id_hex"] for a in payload["results"]] == ["a" * 64]


# --- discover --------------------------------------------------------------


class TestDiscoverCLI:
    def test_discover_ranks_by_affinity_and_applies_max_swarms(
        self, tmp_path: Path, tmp_priv_path: Path, capsys
    ):
        priv_pem = tmp_priv_path.read_bytes()
        ads = [
            _signed_ad("a" * 64, priv_pem, bucket="medium"),
            _signed_ad("b" * 64, priv_pem, bucket="high"),
            _signed_ad("c" * 64, priv_pem, bucket="low"),
        ]
        cache_path = _dump_cache(tmp_path / "cache.json", ads)
        fp_path = _write_fingerprint(tmp_path / "fp.json", bucket="medium")

        rc = cli_main(
            [
                "--json",
                "discover",
                "--cache",
                str(cache_path),
                "--fingerprint",
                str(fp_path),
                "--min-affinity",
                "0.0",
                "--max-swarms",
                "2",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        swarm_ids = [c["swarm_id_hex"] for c in payload["candidates"]]
        assert len(swarm_ids) == 2
        # The "medium" bucket ad should be ranked first (exact match).
        assert swarm_ids[0] == "a" * 64
        # Scores MUST come back in descending order.
        scores = [c["score"] for c in payload["candidates"]]
        assert scores == sorted(scores, reverse=True)

    def test_discover_empty_cache_returns_success_and_empty_list(
        self, tmp_path: Path, capsys
    ):
        cache_path = _dump_cache(tmp_path / "cache.json", [])
        fp_path = _write_fingerprint(tmp_path / "fp.json")

        rc = cli_main(
            [
                "--json",
                "discover",
                "--cache",
                str(cache_path),
                "--fingerprint",
                str(fp_path),
                "--min-affinity",
                "0.0",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["candidates"] == []
