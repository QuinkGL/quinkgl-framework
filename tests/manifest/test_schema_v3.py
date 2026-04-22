"""Phase 1 schema v3 extensions (spec §4.7).

Covers the new top-level fields (`name`, `description`, `created_at`,
`expires_at`, `task`, `model`, `byzantine`, `round_limit`, `bootstrap_peers`,
`tracker_urls`, `creator_pubkey`, `signature`), their validation order
(§4.7.8), and backward compatibility with `schema_version=2` payloads loaded
under ``strict=False`` (§20).
"""

from __future__ import annotations

import copy

import pytest

from quinkgl.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SwarmManifest,
)
from quinkgl.manifest import errors as E


# --- Sample helpers ---------------------------------------------------------


def _good_v3_dict() -> dict:
    """Return a minimal but fully-valid v3 manifest dict."""
    m = SwarmManifest(
        model_arch_fingerprint="abc123",
        data_schema_hash="def456",
    )
    return m.to_dict()


def _good_v2_dict() -> dict:
    """A legacy v2 manifest dict (no v3 fields, schema_version=2)."""
    return {
        "schema_version": 2,
        "model_arch_fingerprint": "abc",
        "data_schema_hash": "def",
        "aggregation": {"name": "FedAvg", "params": {}},
        "topology": {"name": "Random", "params": {}},
        "compression": {"enabled": False, "params": {}},
        "data_policy": {
            **{
                k: v for k, v in _good_v3_dict()["data_policy"].items()
                if k != "schema_version"
            },
            "schema_version": 2,
        },
    }


# --- Schema version ---------------------------------------------------------


class TestSchemaVersion:
    def test_constant_is_three(self):
        assert MANIFEST_SCHEMA_VERSION == 3

    def test_default_manifest_has_v3(self):
        assert SwarmManifest().schema_version == 3
        assert SwarmManifest().to_dict()["schema_version"] == 3

    def test_strict_rejects_v4(self):
        d = _good_v3_dict()
        d["schema_version"] = 4
        with pytest.raises(ValueError) as exc_info:
            SwarmManifest.from_dict(d, strict=True)
        assert exc_info.value.args[0] == E.ERR_MANIFEST_SCHEMA_VERSION


# --- New field roundtrip ----------------------------------------------------


class TestNewFieldRoundtrip:
    def test_all_new_fields_survive_to_dict_from_dict(self):
        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
            name="ImageSwarm",
            description="Multi-class image classifier",
            created_at="2026-04-21T15:00:00Z",
            expires_at="2099-01-01T00:00:00Z",
            round_limit=100,
        )
        # Mutate nested v3 sub-objects.
        m.task.input_shape = [3, 224, 224]
        m.task.output_shape = [1000]
        m.task.tags = ["vision", "classification"]
        m.model.framework = "pytorch"
        m.model.arch_hash = "sha256:" + "f" * 64
        m.byzantine.f = 2
        m.byzantine.enforce_n_gt_2f_plus_2 = True
        m.bootstrap_peers = [
            {"kind": "ipv8", "peer_id": "ab" * 10, "address": "198.51.100.1:8090"},
        ]
        m.tracker_urls = [["https://tracker.a/announce"], ["https://tracker.b/announce"]]

        restored = SwarmManifest.from_dict(m.to_dict(), strict=True)

        assert restored.name == "ImageSwarm"
        assert restored.description == "Multi-class image classifier"
        assert restored.created_at == "2026-04-21T15:00:00Z"
        assert restored.expires_at == "2099-01-01T00:00:00Z"
        assert restored.round_limit == 100
        assert restored.task.input_shape == [3, 224, 224]
        assert restored.task.output_shape == [1000]
        assert restored.task.tags == ["vision", "classification"]
        assert restored.model.framework == "pytorch"
        assert restored.model.arch_hash == "sha256:" + "f" * 64
        assert restored.byzantine.f == 2
        assert restored.byzantine.enforce_n_gt_2f_plus_2 is True
        assert restored.bootstrap_peers[0]["address"] == "198.51.100.1:8090"
        assert restored.tracker_urls == [
            ["https://tracker.a/announce"],
            ["https://tracker.b/announce"],
        ]


# --- Validation order (§4.7.8) ---------------------------------------------


class TestValidationOrder:
    def test_unknown_top_level_key_strict(self):
        d = _good_v3_dict()
        d["surprise"] = True
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_UNKNOWN_KEYS

    def test_missing_required_key_strict(self):
        d = _good_v3_dict()
        d.pop("name")
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_MISSING_KEYS

    @pytest.mark.parametrize(
        "field,value",
        [
            ("created_at", "not-a-date"),
            ("created_at", "2026/04/21"),
            ("expires_at", "yesterday"),
            ("name", ""),
            ("name", "x" * 200),
        ],
    )
    def test_field_regex_invalid(self, field, value):
        d = _good_v3_dict()
        d[field] = value
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_expires_before_created_invalid(self):
        d = _good_v3_dict()
        d["created_at"] = "2030-01-01T00:00:00Z"
        d["expires_at"] = "2020-01-01T00:00:00Z"
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_expired_manifest_rejected(self):
        d = _good_v3_dict()
        d["created_at"] = "2020-01-01T00:00:00Z"
        d["expires_at"] = "2020-12-31T00:00:00Z"
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_EXPIRED

    def test_arch_hash_bad_format(self):
        d = _good_v3_dict()
        d["model"]["arch_hash"] = "md5:abc"
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_task_type_invalid(self):
        d = _good_v3_dict()
        d["task"]["type"] = "telepathy"
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_task_input_shape_must_be_positive_ints(self):
        d = _good_v3_dict()
        d["task"]["input_shape"] = [0, 224]
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_task_tag_regex_enforced(self):
        d = _good_v3_dict()
        d["task"]["tags"] = ["Not-Allowed"]
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_task_tags_max_sixteen(self):
        d = _good_v3_dict()
        d["task"]["tags"] = [f"t{i}" for i in range(17)]
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_byzantine_f_must_be_non_negative(self):
        d = _good_v3_dict()
        d["byzantine"]["f"] = -1
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_bootstrap_peer_kind_invalid(self):
        d = _good_v3_dict()
        d["bootstrap_peers"] = [
            {"kind": "carrier-pigeon", "peer_id": "ab", "address": "x:1"}
        ]
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID

    def test_tracker_urls_must_be_two_level(self):
        d = _good_v3_dict()
        d["tracker_urls"] = ["https://tracker.a/announce"]  # single-level = wrong
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(d, strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_FIELD_INVALID


# --- Backward compatibility (§20) -------------------------------------------


class TestV2BackwardCompat:
    def test_v2_payload_rejected_strict(self):
        with pytest.raises(ValueError) as exc:
            SwarmManifest.from_dict(_good_v2_dict(), strict=True)
        assert exc.value.args[0] == E.ERR_MANIFEST_SCHEMA_VERSION

    def test_v2_payload_loaded_non_strict(self):
        m = SwarmManifest.from_dict(_good_v2_dict(), strict=False)
        # v3 fields materialise with defaults; core v2 data survives.
        assert m.model_arch_fingerprint == "abc"
        assert m.data_schema_hash == "def"
        assert m.aggregation_name == "FedAvg"
        # v3 scaffolding present with safe defaults.
        assert m.task is not None
        assert m.model is not None
        assert m.byzantine is not None
        assert isinstance(m.bootstrap_peers, list)
        assert isinstance(m.tracker_urls, list)

    def test_v3_validate_allows_default_constructed_instance(self):
        """Default kwargs must produce an instance that passes `validate()` on
        v3 scaffolding — legacy tests depend on this so that the existing
        empty-fingerprint check remains the FIRST observable failure."""
        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
        )
        m.validate()  # should not raise


# --- Signature field (excluded from canonical bytes) ------------------------


class TestSignatureExclusion:
    def test_signature_absence_and_presence_hash_same(self):
        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
        )
        h0 = m.manifest_hash()
        m.signature = "ed25519:" + "a" * 128
        h1 = m.manifest_hash()
        assert h0 == h1, "signature MUST NOT influence canonical bytes (§5.3)"

    def test_creator_pubkey_DOES_affect_hash(self):
        """`creator_pubkey` is signed-over data → MUST change the hash."""
        m1 = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
        )
        m2 = copy.deepcopy(m1)
        m2.creator_pubkey = "ed25519:" + "b" * 64
        assert m1.manifest_hash() != m2.manifest_hash()
