"""Canonical encoding invariants (spec §5).

`SwarmManifest.canonical_bytes()` MUST be byte-identical across repeated
calls, across semantically-equivalent constructions, and across Python
process invocations. This is the foundation of the ``swarm_id`` contract.
"""

from __future__ import annotations

import copy
import json

import pytest

from quinkgl.manifest import SwarmManifest


def _sample_manifest() -> SwarmManifest:
    m = SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="StabilitySwarm",
        description="repeat-stable",
        created_at="2026-04-21T15:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
        round_limit=50,
    )
    m.task.input_shape = [3, 224, 224]
    m.task.output_shape = [10]
    m.task.tags = ["vision", "a-b"]
    m.model.framework = "pytorch"
    m.model.arch_hash = "sha256:" + "c" * 64
    m.byzantine.f = 1
    m.byzantine.enforce_n_gt_2f_plus_2 = True
    m.bootstrap_peers = [
        {"kind": "ipv8", "peer_id": "deadbeef", "address": "10.0.0.1:8090"},
        {"kind": "tunnel", "peer_id": "cafebabe", "address": "10.0.0.2:8091"},
    ]
    m.tracker_urls = [
        ["https://tracker.one/a", "https://tracker.one/b"],
        ["https://tracker.two/a"],
    ]
    m.creator_pubkey = "ed25519:" + "0" * 64
    return m


def test_hundred_iterations_byte_stable():
    m = _sample_manifest()
    first = m.canonical_bytes()
    for _ in range(100):
        assert m.canonical_bytes() == first


def test_key_insertion_order_irrelevant():
    m = _sample_manifest()
    d1 = m.to_dict()
    d2 = dict(reversed(list(d1.items())))
    # Reverse nested objects too.
    d2["task"] = dict(reversed(list(d2["task"].items())))
    d2["model"] = dict(reversed(list(d2["model"].items())))
    restored1 = SwarmManifest.from_dict(d1, strict=True)
    restored2 = SwarmManifest.from_dict(d2, strict=True)
    assert restored1.canonical_bytes() == restored2.canonical_bytes()
    assert restored1.manifest_hash() == restored2.manifest_hash()


def test_signature_pop_from_canonical():
    m = _sample_manifest()
    without_sig = m.canonical_bytes()
    m.signature = "ed25519:" + "f" * 128
    with_sig = m.canonical_bytes()
    assert without_sig == with_sig
    # `signature` must not appear in the canonical payload at all.
    assert b"signature" not in with_sig


def test_canonical_bytes_utf8_and_sorted():
    m = _sample_manifest()
    raw = m.canonical_bytes()
    decoded = raw.decode("utf-8")
    parsed = json.loads(decoded)
    # Re-encode with sort_keys and compare: canonical_bytes MUST already be
    # sorted, so re-serialising sorted should match.
    resorted = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    assert raw == resorted


def test_nan_rejected():
    m = _sample_manifest()
    m.task.input_shape = [float("nan")]  # deliberately poisoned
    with pytest.raises(ValueError):
        m.canonical_bytes()


def test_deep_copy_produces_same_hash():
    m = _sample_manifest()
    clone = copy.deepcopy(m)
    assert m.manifest_hash() == clone.manifest_hash()
