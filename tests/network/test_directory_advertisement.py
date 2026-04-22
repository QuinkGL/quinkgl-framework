"""Phase 3 ``SwarmAdvertisement`` + ``SwarmAdvertisementPayload`` contract
(spec §17.2, §21.3).

Exercises only the data-layer pieces that can run without an IPv8 reactor:
the dataclass, canonical-bytes signing/verification, and the ten-field
wire payload.  :mod:`tests.network.test_directory_community` stands the
community up end-to-end.
"""

from __future__ import annotations

import json

import pytest

from quinkgl.manifest import keygen


# --- Dataclass + canonical bytes ------------------------------------------


def _dummy_ad_kwargs() -> dict:
    return {
        "swarm_id_hex": "a" * 64,
        "name": "phase3-test-swarm",
        "tags": ["vision", "pytorch"],
        "input_shape": [3, 224, 224],
        "output_shape": [10],
        "label_type": "integer",
        "data_schema_hash": "sha256:" + "b" * 64,
        "reference_fingerprint": {"version": 1, "hash": "deadbeef"},
    }


class TestSwarmAdvertisementDataclass:
    def test_canonical_bytes_is_deterministic(self):
        from quinkgl.network.directory import SwarmAdvertisement

        a = SwarmAdvertisement(**_dummy_ad_kwargs())
        b = SwarmAdvertisement(**_dummy_ad_kwargs())
        assert a.canonical_bytes() == b.canonical_bytes()

    def test_canonical_bytes_excludes_signature(self):
        """§17.2: signature MUST be excluded from the signed input."""
        from quinkgl.network.directory import SwarmAdvertisement

        ad = SwarmAdvertisement(**_dummy_ad_kwargs())
        pre = ad.canonical_bytes()
        ad.signature = "ed25519:" + "0" * 128
        assert ad.canonical_bytes() == pre

    def test_canonical_bytes_changes_when_any_signed_field_mutates(self):
        from quinkgl.network.directory import SwarmAdvertisement

        ad = SwarmAdvertisement(**_dummy_ad_kwargs())
        canonical = ad.canonical_bytes()
        ad.name = "hijacked"
        assert ad.canonical_bytes() != canonical


# --- sign / verify --------------------------------------------------------


class TestAdvertisementSignVerify:
    def test_sign_then_verify_roundtrip(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            sign_advertisement,
            verify_advertisement,
        )

        private_pem, _ = keygen(None)
        ad = SwarmAdvertisement(**_dummy_ad_kwargs())

        signed = sign_advertisement(ad, private_pem)

        assert signed.signature is not None
        assert signed.signature.startswith("ed25519:")
        assert signed.creator_pubkey is not None
        assert verify_advertisement(signed) is True

    def test_verify_rejects_tampered_ad(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            sign_advertisement,
            verify_advertisement,
        )

        private_pem, _ = keygen(None)
        signed = sign_advertisement(
            SwarmAdvertisement(**_dummy_ad_kwargs()), private_pem
        )
        signed.name = "hijacked"

        assert verify_advertisement(signed) is False

    def test_verify_rejects_unsigned_ad(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            verify_advertisement,
        )

        assert verify_advertisement(SwarmAdvertisement(**_dummy_ad_kwargs())) is False

    def test_verify_rejects_wrong_pubkey(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            sign_advertisement,
            verify_advertisement,
        )

        private_a, _ = keygen(None)
        _, pub_b = keygen(None)

        signed = sign_advertisement(
            SwarmAdvertisement(**_dummy_ad_kwargs()), private_a
        )
        signed.creator_pubkey = "ed25519:" + pub_b.hex()
        assert verify_advertisement(signed) is False


# --- SwarmAdvertisementPayload (msg_id=40) --------------------------------


class TestSwarmAdvertisementPayload:
    def test_msg_id_and_format_list_match_spec(self):
        from quinkgl.network.directory import SwarmAdvertisementPayload

        assert SwarmAdvertisementPayload.msg_id == 40
        assert SwarmAdvertisementPayload.format_list == ["varlenH"] * 10

    def test_roundtrip_bytes_via_pack_unpack(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            SwarmAdvertisementPayload,
            sign_advertisement,
        )

        private_pem, _ = keygen(None)
        signed = sign_advertisement(
            SwarmAdvertisement(**_dummy_ad_kwargs()), private_pem
        )
        payload = SwarmAdvertisementPayload.from_advertisement(signed)

        # Round-trip through the pack list — simulates what IPv8's wire
        # serializer does under the hood.
        pack = payload.to_pack_list()
        assert len(pack) == 10
        # Extract only the bytes values in order and feed them back in.
        raw_values = [v for (_fmt, v) in pack]
        restored_payload = SwarmAdvertisementPayload.from_unpack_list(*raw_values)
        restored_ad = restored_payload.to_advertisement()

        assert restored_ad.swarm_id_hex == signed.swarm_id_hex
        assert restored_ad.name == signed.name
        assert restored_ad.tags == signed.tags
        assert restored_ad.input_shape == signed.input_shape
        assert restored_ad.output_shape == signed.output_shape
        assert restored_ad.label_type == signed.label_type
        assert restored_ad.data_schema_hash == signed.data_schema_hash
        assert restored_ad.reference_fingerprint == signed.reference_fingerprint
        assert restored_ad.creator_pubkey == signed.creator_pubkey
        assert restored_ad.signature == signed.signature

    def test_payload_signature_survives_wire_and_still_verifies(self):
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            SwarmAdvertisementPayload,
            sign_advertisement,
            verify_advertisement,
        )

        private_pem, _ = keygen(None)
        signed = sign_advertisement(
            SwarmAdvertisement(**_dummy_ad_kwargs()), private_pem
        )
        pack = SwarmAdvertisementPayload.from_advertisement(signed).to_pack_list()
        raw_values = [v for (_fmt, v) in pack]
        restored = SwarmAdvertisementPayload.from_unpack_list(
            *raw_values
        ).to_advertisement()

        assert verify_advertisement(restored) is True

    def test_tags_csv_parses_back_to_list(self):
        """tags are comma-joined on the wire; empty list round-trips cleanly."""
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            SwarmAdvertisementPayload,
        )

        kwargs = _dummy_ad_kwargs()
        kwargs["tags"] = []
        ad = SwarmAdvertisement(**kwargs)
        payload = SwarmAdvertisementPayload.from_advertisement(ad)
        raw_values = [v for (_fmt, v) in payload.to_pack_list()]
        restored = SwarmAdvertisementPayload.from_unpack_list(
            *raw_values
        ).to_advertisement()
        assert restored.tags == []

    def test_shapes_are_json_encoded(self):
        """input_shape / output_shape travel as JSON to preserve dimensions."""
        from quinkgl.network.directory import (
            SwarmAdvertisement,
            SwarmAdvertisementPayload,
        )

        ad = SwarmAdvertisement(**_dummy_ad_kwargs())
        payload = SwarmAdvertisementPayload.from_advertisement(ad)
        pack = dict(
            zip(
                [
                    "swarm_id_hex",
                    "name",
                    "tags_csv",
                    "input_shape_json",
                    "output_shape_json",
                    "label_type",
                    "data_schema_hash",
                    "reference_fingerprint_json",
                    "creator_pubkey",
                    "signature",
                ],
                [v for (_fmt, v) in payload.to_pack_list()],
            )
        )
        assert json.loads(pack["input_shape_json"]) == [3, 224, 224]
        assert json.loads(pack["output_shape_json"]) == [10]
