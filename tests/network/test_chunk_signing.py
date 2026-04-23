"""
regression tests — Sign model update payloads.

Validates that:
 - _chunk_sign_data builds deterministic canonical bytes.
 - _chunk_sign and _chunk_verify round-trip with real IPv8 keys.
 - A tampered chunk is rejected.
 - An impersonation attempt (wrong key) is rejected.
 - ModelChunkPayload carries the signature field.
"""

import hashlib
import struct

import pytest

from quinkgl.network.gossip_community import (
    _chunk_sign_data,
    _chunk_sign,
    _chunk_verify,
    ModelChunkPayload,
)


# ---------------------------------------------------------------------------
# Helpers — use ipv8 crypto if available, else skip
# ---------------------------------------------------------------------------

def _generate_keypair():
    """Generate an IPv8-compatible EC key pair for testing."""
    try:
        from ipv8.keyvault.crypto import default_eccrypto
        private_key = default_eccrypto.generate_key("curve25519")
        return private_key
    except Exception:
        pytest.skip("ipv8 crypto not available")


def _metadata(**overrides):
    data = {
        "sample_count": 8,
        "loss": 0.1,
        "accuracy": 0.9,
        "timestamp": 123456,
        "total_chunks": 4,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# B14-1: Canonical sign data is deterministic
# ---------------------------------------------------------------------------

def test_sign_data_deterministic():
    a = _chunk_sign_data("alice", 5, "schema-hash", 3, b"hello", **_metadata())
    b = _chunk_sign_data("alice", 5, "schema-hash", 3, b"hello", **_metadata())
    assert a == b


def test_sign_data_changes_with_sender():
    a = _chunk_sign_data("alice", 5, "s", 0, b"data", **_metadata())
    b = _chunk_sign_data("bob", 5, "s", 0, b"data", **_metadata())
    assert a != b


def test_sign_data_changes_with_round():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata())
    b = _chunk_sign_data("alice", 2, "s", 0, b"data", **_metadata())
    assert a != b


def test_sign_data_changes_with_chunk_index():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata())
    b = _chunk_sign_data("alice", 1, "s", 1, b"data", **_metadata())
    assert a != b


def test_sign_data_changes_with_chunk_data():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data-A", **_metadata())
    b = _chunk_sign_data("alice", 1, "s", 0, b"data-B", **_metadata())
    assert a != b


def test_sign_data_changes_with_sample_count():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(sample_count=8))
    b = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(sample_count=9))
    assert a != b


def test_sign_data_changes_with_total_chunks():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(total_chunks=4))
    b = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(total_chunks=5))
    assert a != b


def test_sign_data_changes_with_timestamp():
    a = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(timestamp=123))
    b = _chunk_sign_data("alice", 1, "s", 0, b"data", **_metadata(timestamp=124))
    assert a != b


# ---------------------------------------------------------------------------
# B14-2: Sign + verify round-trip with real keys
# ---------------------------------------------------------------------------

def test_sign_verify_roundtrip():
    key = _generate_keypair()
    sig = _chunk_sign(key, "alice", 10, "schema", 0, b"chunk-bytes", **_metadata())
    assert _chunk_verify(key.pub(), sig, "alice", 10, "schema", 0, b"chunk-bytes", **_metadata())


# ---------------------------------------------------------------------------
# B14-3: Tampered chunk is rejected
# ---------------------------------------------------------------------------

def test_tampered_chunk_rejected():
    key = _generate_keypair()
    sig = _chunk_sign(key, "alice", 10, "schema", 0, b"original", **_metadata())
    assert not _chunk_verify(key.pub(), sig, "alice", 10, "schema", 0, b"tampered", **_metadata())


# ---------------------------------------------------------------------------
# B14-4: Impersonation (wrong key) is rejected
# ---------------------------------------------------------------------------

def test_impersonation_rejected():
    key_alice = _generate_keypair()
    key_bob = _generate_keypair()
    sig = _chunk_sign(key_alice, "alice", 10, "schema", 0, b"data", **_metadata())
    # Verify with bob's key should fail
    assert not _chunk_verify(key_bob.pub(), sig, "alice", 10, "schema", 0, b"data", **_metadata())


def test_sample_count_tampering_rejected():
    key = _generate_keypair()
    sig = _chunk_sign(key, "alice", 10, "schema", 0, b"data", **_metadata(sample_count=8))
    assert not _chunk_verify(
        key.pub(), sig, "alice", 10, "schema", 0, b"data", **_metadata(sample_count=9)
    )


# ---------------------------------------------------------------------------
# B14-5: Empty signature returns False
# ---------------------------------------------------------------------------

def test_empty_signature_rejected():
    key = _generate_keypair()
    assert not _chunk_verify(key.pub(), b"", "alice", 1, "s", 0, b"data", **_metadata())


# ---------------------------------------------------------------------------
# B14-6: ModelChunkPayload carries signature field
# ---------------------------------------------------------------------------

def test_payload_signature_field():
    p = ModelChunkPayload(
        transfer_id="tid",
        chunk_index=0,
        total_chunks=1,
        sender_id="alice",
        data_schema_hash="schema",
        round_number=1,
        sample_count=8,
        loss=0.1,
        accuracy=0.9,
        chunk_data=b"data",
        timestamp=123,
        signature=b"sig-bytes",
    )
    assert p.signature == b"sig-bytes"

    pack_list = p.to_pack_list()
    # Last element should be the signature
    assert pack_list[-1] == ('varlenH', b"sig-bytes")


def test_payload_default_empty_signature():
    p = ModelChunkPayload(
        transfer_id="tid",
        chunk_index=0,
        total_chunks=1,
        sender_id="alice",
        data_schema_hash="schema",
        round_number=1,
        sample_count=8,
        loss=0.1,
        accuracy=0.9,
        chunk_data=b"data",
        timestamp=123,
    )
    assert p.signature == b""
