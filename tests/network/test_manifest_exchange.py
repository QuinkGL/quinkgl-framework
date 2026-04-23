"""Manifest exchange wire protocol (spec §13).

These tests exercise the protocol machinery (chunking, reassembly,
rate-limiting, hash verification) without standing up an IPv8 stack.
A two-peer live integration is deferred to the CLI/run-command test
suite where a local swarm fixture is cheap.
"""

from __future__ import annotations

import hashlib

import pytest

from quinkgl.manifest import SwarmManifest
from quinkgl.manifest import errors as E
from quinkgl.network.manifest_exchange import (
    CHUNK_DATA_SIZE,
    REQUEST_TIMEOUT_SECONDS,
    ManifestAssembler,
    ManifestRegistry,
    ManifestRequestPayload,
    ManifestResponseChunkPayload,
    ManifestResponseNackPayload,
    NackReason,
    RateLimiter,
    chunk_manifest_bytes,
)


def _sample() -> SwarmManifest:
    return SwarmManifest(
        model_arch_fingerprint="abc",
        data_schema_hash="def",
        name="ExchangeTest",
    )


# --- Chunking --------------------------------------------------------------


class TestChunking:
    def test_tiny_payload_produces_single_chunk(self):
        chunks = chunk_manifest_bytes(b"hello")
        assert len(chunks) == 1
        assert chunks[0] == b"hello"

    def test_large_payload_respects_chunk_size(self):
        raw = b"A" * (CHUNK_DATA_SIZE * 3 + 42)
        chunks = chunk_manifest_bytes(raw)
        assert len(chunks) == 4
        assert all(len(c) <= CHUNK_DATA_SIZE for c in chunks)
        assert b"".join(chunks) == raw

    def test_empty_payload_rejected(self):
        with pytest.raises(ValueError):
            chunk_manifest_bytes(b"")


# --- Assembler happy path --------------------------------------------------


class TestAssemblerHappyPath:
    def test_assemble_in_order(self):
        canonical = _sample().canonical_bytes()
        chunks = chunk_manifest_bytes(canonical)
        swarm_id = hashlib.sha256(canonical).digest()
        asm = ManifestAssembler(swarm_id=swarm_id, nonce=42)
        for idx, c in enumerate(chunks):
            asm.add_chunk(idx, len(chunks), c)
        assert asm.is_complete()
        restored = asm.assemble_and_verify()
        assert restored == canonical

    def test_assemble_out_of_order(self):
        canonical = _sample().canonical_bytes()
        chunks = chunk_manifest_bytes(canonical)
        swarm_id = hashlib.sha256(canonical).digest()
        asm = ManifestAssembler(swarm_id=swarm_id, nonce=7)
        # Reverse order on the wire should still reassemble correctly
        # because chunks carry their own index.
        for idx in reversed(range(len(chunks))):
            asm.add_chunk(idx, len(chunks), chunks[idx])
        assert asm.assemble_and_verify() == canonical


# --- Assembler error paths -------------------------------------------------


class TestAssemblerErrors:
    def test_inconsistent_total_chunks_rejected(self):
        asm = ManifestAssembler(swarm_id=b"\x01" * 32, nonce=1)
        asm.add_chunk(0, 3, b"AAA")
        with pytest.raises(ValueError) as exc:
            asm.add_chunk(1, 4, b"BBB")  # different total
        assert exc.value.args[0] == E.ERR_WIRE_CHUNK_INCONSISTENT

    def test_tampered_chunk_fails_hash_verify(self):
        canonical = _sample().canonical_bytes()
        chunks = chunk_manifest_bytes(canonical)
        swarm_id = hashlib.sha256(canonical).digest()
        asm = ManifestAssembler(swarm_id=swarm_id, nonce=1)
        # Flip one byte in the last chunk.
        chunks[-1] = chunks[-1][:-1] + bytes([chunks[-1][-1] ^ 0xFF])
        for idx, c in enumerate(chunks):
            asm.add_chunk(idx, len(chunks), c)
        with pytest.raises(ValueError) as exc:
            asm.assemble_and_verify()
        assert exc.value.args[0] == E.ERR_MANIFEST_HASH_MISMATCH

    def test_assemble_before_complete_raises(self):
        asm = ManifestAssembler(swarm_id=b"\x01" * 32, nonce=1)
        asm.add_chunk(0, 2, b"AAA")
        with pytest.raises(ValueError):
            asm.assemble_and_verify()

    def test_duplicate_chunk_ignored(self):
        """Re-delivering a chunk index MUST not corrupt the assembler or
        trip the inconsistency guard — UDP duplication is expected."""
        asm = ManifestAssembler(swarm_id=b"\x01" * 32, nonce=1)
        asm.add_chunk(0, 2, b"AAA")
        asm.add_chunk(0, 2, b"AAA")  # exact duplicate — silently accepted
        asm.add_chunk(1, 2, b"BBB")
        assert asm.is_complete()


# --- Rate limiter ----------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        clock = [1000.0]
        rl = RateLimiter(limit=4, window_seconds=60.0, clock=lambda: clock[0])
        for _ in range(4):
            assert rl.allow("peer-a") is True
        assert rl.allow("peer-a") is False

    def test_window_rolls_over(self):
        clock = [1000.0]
        rl = RateLimiter(limit=4, window_seconds=60.0, clock=lambda: clock[0])
        for _ in range(4):
            rl.allow("peer-a")
        clock[0] += 61.0
        assert rl.allow("peer-a") is True

    def test_per_peer_isolation(self):
        clock = [1000.0]
        rl = RateLimiter(limit=2, window_seconds=60.0, clock=lambda: clock[0])
        rl.allow("a")
        rl.allow("a")
        assert rl.allow("a") is False
        assert rl.allow("b") is True  # different peer, fresh budget


# --- Registry + server behavior -------------------------------------------


class TestManifestRegistry:
    def test_lookup_hit_returns_canonical_bytes(self):
        m = _sample()
        reg = ManifestRegistry()
        reg.register(m)
        assert reg.get(m.manifest_hash()) == m.canonical_bytes()

    def test_lookup_miss_returns_none(self):
        reg = ManifestRegistry()
        assert reg.get("0" * 64) is None


# --- Payload wire ----------------------------------------------------------


class TestPayloadWire:
    def test_request_fields(self):
        p = ManifestRequestPayload(swarm_id_hex="a" * 64, request_nonce=42)
        assert p.swarm_id_hex == "a" * 64
        assert p.request_nonce == 42
        assert p.msg_id == 30

    def test_chunk_fields(self):
        p = ManifestResponseChunkPayload(
            swarm_id_hex="a" * 64,
            request_nonce=1,
            chunk_index=0,
            total_chunks=3,
            chunk_data=b"xyz",
        )
        assert p.msg_id == 31
        assert p.chunk_data == b"xyz"

    def test_nack_reason_enumeration(self):
        p = ManifestResponseNackPayload(
            swarm_id_hex="a" * 64,
            request_nonce=1,
            reason=NackReason.UNKNOWN_SWARM,
        )
        assert p.msg_id == 32
        assert p.reason == NackReason.UNKNOWN_SWARM

    def test_nack_reason_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            ManifestResponseNackPayload(
                swarm_id_hex="a" * 64,
                request_nonce=1,
                reason="PIZZA_TIME",
            )


# --- Sanity: timeout constant ---------------------------------------------


def test_timeout_matches_spec():
    assert REQUEST_TIMEOUT_SECONDS == 30.0
