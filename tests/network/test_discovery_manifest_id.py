"""`DiscoveryAnnouncePayload.manifest_id` field (spec §12.1–§12.3).

These are unit tests over payload encoding/decoding and the ``PeerInfo``
compatibility check.  They do NOT spin up IPv8 — the cross-version wire
semantics (old peer emits 5 fields; new peer emits 6) are covered by
exercising ``to_pack_list`` and ``from_unpack_list`` directly.
"""

from __future__ import annotations

from quinkgl.network.gossip_community import (
    DiscoveryAnnouncePayload,
    PeerInfo,
)
from quinkgl.network.gossip_community import _manifest_id_blocks_peer


HEX_MID_A = "a" * 64
HEX_MID_B = "b" * 64


# --- DiscoveryAnnouncePayload wire compat ----------------------------------


class TestPayloadWire:
    def test_format_list_has_six_fields(self):
        assert DiscoveryAnnouncePayload.format_list == ["varlenH"] * 6

    def test_roundtrip_new_peer_to_new_peer(self):
        p = DiscoveryAnnouncePayload(
            node_id="alice",
            domain="health",
            data_schema_hash="abc",
            model_version="1.2.3",
            fingerprint_json="{}",
            manifest_id=HEX_MID_A,
        )
        packed = [item[1] for item in p.to_pack_list()]
        assert len(packed) == 6
        assert packed[5] == HEX_MID_A.encode("utf-8")

        restored = DiscoveryAnnouncePayload.from_unpack_list(*packed)
        assert restored.node_id == "alice"
        assert restored.domain == "health"
        assert restored.data_schema_hash == "abc"
        assert restored.model_version == "1.2.3"
        assert restored.fingerprint_json == "{}"
        assert restored.manifest_id == HEX_MID_A

    def test_decode_old_five_field_payload_yields_empty_manifest_id(self):
        # Simulate a v2.0.0 peer that only emits 5 fields.
        args = [
            b"bob",
            b"health",
            b"abc",
            b"1.0.0",
            b"",
        ]
        restored = DiscoveryAnnouncePayload.from_unpack_list(*args)
        assert restored.node_id == "bob"
        assert restored.manifest_id == ""

    def test_manifest_id_defaults_to_empty(self):
        p = DiscoveryAnnouncePayload(
            node_id="alice", domain="h", data_schema_hash="abc"
        )
        assert p.manifest_id == ""
        packed = [item[1] for item in p.to_pack_list()]
        assert packed[5] == b""


# --- PeerInfo.is_compatible ------------------------------------------------


class TestPeerInfoCompatibility:
    def _peer(self, **kw):
        base = dict(
            peer=object(),
            node_id="n",
            domain="health",
            data_schema_hash="abc",
        )
        base.update(kw)
        return PeerInfo(**base)

    def test_manifest_id_match_overrides_domain_mismatch(self):
        """If both sides speak manifest_id and they agree, the pair is
        compatible regardless of the legacy (domain, data_schema_hash)."""
        p = self._peer(domain="other", data_schema_hash="zzz", manifest_id=HEX_MID_A)
        assert p.is_compatible(
            domain="ignored", data_schema_hash="ignored", manifest_id=HEX_MID_A
        )

    def test_manifest_id_mismatch_rejects_even_if_domain_matches(self):
        p = self._peer(manifest_id=HEX_MID_A)
        assert not p.is_compatible(
            domain="health", data_schema_hash="abc", manifest_id=HEX_MID_B
        )

    def test_legacy_fallback_when_no_manifest_id(self):
        p = self._peer()  # no manifest_id
        assert p.is_compatible(domain="health", data_schema_hash="abc")
        assert not p.is_compatible(domain="other", data_schema_hash="abc")

    def test_empty_manifest_id_falls_through_to_legacy(self):
        """When either side has an empty manifest_id, the check reverts to
        the legacy pair — cross-version peers keep talking."""
        p = self._peer(manifest_id="")
        assert p.is_compatible(
            domain="health", data_schema_hash="abc", manifest_id=HEX_MID_A
        )


# --- Pre-filter helper -----------------------------------------------------


class TestPreFilterHelper:
    """``_manifest_id_blocks_peer(local, remote)`` returns True only when
    *both* sides advertise a manifest_id AND the two values differ — this is
    the §12.3 "discovery manifest mismatch" gate.
    """

    def test_blocks_only_when_both_set_and_different(self):
        assert _manifest_id_blocks_peer(HEX_MID_A, HEX_MID_B) is True

    def test_does_not_block_when_equal(self):
        assert _manifest_id_blocks_peer(HEX_MID_A, HEX_MID_A) is False

    def test_does_not_block_when_local_missing(self):
        assert _manifest_id_blocks_peer("", HEX_MID_B) is False

    def test_does_not_block_when_remote_missing(self):
        assert _manifest_id_blocks_peer(HEX_MID_A, "") is False

    def test_does_not_block_when_both_missing(self):
        assert _manifest_id_blocks_peer("", "") is False
