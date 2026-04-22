"""Magnet URI grammar (spec §8).

`MagnetLink`, `parse_magnet`, `format_magnet` implement the canonical
``quinkgl:?…`` scheme.  Round-trip MUST be bit-identical for URIs emitted by
``format_magnet``.  Unknown parameters are ignored (forward compat).
"""

from __future__ import annotations

import pytest

from quinkgl.manifest import (
    MagnetLink,
    SwarmManifest,
    format_magnet,
    parse_magnet,
)
from quinkgl.manifest import errors as E


HEX_64 = "a" * 64
HEX_64_B = "b" * 64


# --- parse_magnet — happy path ---------------------------------------------


class TestParseHappyPath:
    def test_minimal_xt_only(self):
        uri = f"quinkgl:?xt=urn:qgl:{HEX_64}"
        link = parse_magnet(uri)
        assert isinstance(link, MagnetLink)
        assert link.swarm_id == bytes.fromhex(HEX_64)
        assert link.display_name is None
        assert link.keywords == []
        assert link.trackers == []
        assert link.bootstrap_peers == []
        assert link.protocol_version == 1

    def test_full_uri_with_all_params(self):
        uri = (
            f"quinkgl:?xt=urn:qgl:{HEX_64}"
            "&dn=My%20Swarm"
            "&kw=vision,medical"
            "&v=1"
            "&tr=https%3A%2F%2Ftracker.example%2Fannounce"
            "&tr=https%3A%2F%2Fbackup.example%2Fannounce"
            "&bs=10.0.0.1%3A8090"
            "&bs=10.0.0.2%3A8091"
        )
        link = parse_magnet(uri)
        assert link.swarm_id == bytes.fromhex(HEX_64)
        assert link.display_name == "My Swarm"
        assert link.keywords == ["vision", "medical"]
        assert link.protocol_version == 1
        assert link.trackers == [
            "https://tracker.example/announce",
            "https://backup.example/announce",
        ]
        assert link.bootstrap_peers == ["10.0.0.1:8090", "10.0.0.2:8091"]

    def test_unknown_params_ignored(self):
        uri = f"quinkgl:?xt=urn:qgl:{HEX_64}&futuristic=1&foo=bar"
        link = parse_magnet(uri)
        assert link.swarm_id == bytes.fromhex(HEX_64)


# --- parse_magnet — error paths --------------------------------------------


class TestParseErrors:
    @pytest.mark.parametrize(
        "uri",
        [
            "http://tracker.example/?xt=urn:qgl:" + HEX_64,
            "magnet:?xt=urn:qgl:" + HEX_64,
            "QUINKGL:?xt=urn:qgl:" + HEX_64,
            "",
            "quinkgl:",
        ],
    )
    def test_wrong_scheme(self, uri):
        with pytest.raises(ValueError) as exc:
            parse_magnet(uri)
        assert exc.value.args[0] == E.ERR_MAGNET_SCHEME

    def test_missing_xt(self):
        with pytest.raises(ValueError) as exc:
            parse_magnet("quinkgl:?dn=no-xt-here")
        assert exc.value.args[0] == E.ERR_MAGNET_XT

    def test_xt_wrong_urn(self):
        with pytest.raises(ValueError) as exc:
            parse_magnet("quinkgl:?xt=urn:btih:" + HEX_64)
        assert exc.value.args[0] == E.ERR_MAGNET_XT

    def test_xt_wrong_hex_length(self):
        with pytest.raises(ValueError) as exc:
            parse_magnet("quinkgl:?xt=urn:qgl:abc123")
        assert exc.value.args[0] == E.ERR_MAGNET_XT

    def test_xt_uppercase_rejected(self):
        with pytest.raises(ValueError) as exc:
            parse_magnet("quinkgl:?xt=urn:qgl:" + ("A" * 64))
        assert exc.value.args[0] == E.ERR_MAGNET_XT

    def test_xt_duplicate(self):
        uri = f"quinkgl:?xt=urn:qgl:{HEX_64}&xt=urn:qgl:{HEX_64_B}"
        with pytest.raises(ValueError) as exc:
            parse_magnet(uri)
        # Duplicate xt is an xt-specific error (exactly-1 requirement).
        assert exc.value.args[0] == E.ERR_MAGNET_XT

    def test_dn_duplicate(self):
        uri = f"quinkgl:?xt=urn:qgl:{HEX_64}&dn=A&dn=B"
        with pytest.raises(ValueError) as exc:
            parse_magnet(uri)
        assert exc.value.args[0] == E.ERR_MAGNET_DUPLICATE

    def test_v_duplicate(self):
        uri = f"quinkgl:?xt=urn:qgl:{HEX_64}&v=1&v=2"
        with pytest.raises(ValueError) as exc:
            parse_magnet(uri)
        assert exc.value.args[0] == E.ERR_MAGNET_DUPLICATE


# --- format_magnet ----------------------------------------------------------


class TestFormatMagnet:
    def test_minimal_format(self):
        link = MagnetLink(swarm_id=bytes.fromhex(HEX_64))
        assert format_magnet(link) == f"quinkgl:?xt=urn:qgl:{HEX_64}"

    def test_full_format_parameter_order(self):
        """Canonical output order: xt, dn, kw, v, tr*, bs*."""
        link = MagnetLink(
            swarm_id=bytes.fromhex(HEX_64),
            display_name="Demo Swarm",
            keywords=["a", "b"],
            trackers=["https://t1/announce", "https://t2/announce"],
            bootstrap_peers=["10.0.0.1:8090"],
            protocol_version=2,
        )
        out = format_magnet(link)
        # xt first, then dn, then kw, then v, then trs, then bss.
        assert out.startswith(f"quinkgl:?xt=urn:qgl:{HEX_64}&dn=")
        dn_idx = out.index("&dn=")
        kw_idx = out.index("&kw=")
        v_idx = out.index("&v=")
        tr_idx = out.index("&tr=")
        bs_idx = out.index("&bs=")
        assert dn_idx < kw_idx < v_idx < tr_idx < bs_idx
        assert "&dn=Demo%20Swarm&" in out
        assert "&kw=a,b&" in out
        assert "&v=2&" in out
        assert out.count("&tr=") == 2
        assert out.count("&bs=") == 1

    def test_roundtrip_canonical(self):
        original = MagnetLink(
            swarm_id=bytes.fromhex(HEX_64),
            display_name="Road & Rain",
            keywords=["vision", "medical-x"],
            trackers=["https://tracker.example/announce?q=1"],
            bootstrap_peers=["fe80::1:8090"],
            protocol_version=2,
        )
        s = format_magnet(original)
        parsed = parse_magnet(s)
        assert format_magnet(parsed) == s
        assert parsed == original


# --- SwarmManifest.to_magnet convenience ------------------------------------


class TestToMagnet:
    def test_manifest_to_magnet_uses_manifest_hash(self):
        m = SwarmManifest(model_arch_fingerprint="abc", data_schema_hash="def")
        uri = m.to_magnet()
        link = parse_magnet(uri)
        assert link.swarm_id.hex() == m.manifest_hash()

    def test_manifest_to_magnet_passes_trackers(self):
        m = SwarmManifest(
            model_arch_fingerprint="abc",
            data_schema_hash="def",
            name="ImageSwarm",
        )
        uri = m.to_magnet(
            trackers=["https://tracker.example/announce"],
            bootstrap_peers=["10.0.0.1:8090"],
        )
        link = parse_magnet(uri)
        assert link.trackers == ["https://tracker.example/announce"]
        assert link.bootstrap_peers == ["10.0.0.1:8090"]
        assert link.display_name == "ImageSwarm"
