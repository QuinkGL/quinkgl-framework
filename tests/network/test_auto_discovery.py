"""Phase 3 auto-discovery orchestrator + manifest registry (spec §18).

Pure-Python tests: no IPv8 reactor, no real network.  The orchestrator
drives the spec §18.1 flow against an in-process
:class:`SwarmDirectoryCommunity` and an injected async manifest loader,
so we can cover the affinity ranking, trust-policy gate, and top-K
truncation deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import pytest

from quinkgl.fingerprint import DataFingerprint
from quinkgl.manifest import keygen
from quinkgl.manifest.errors import ERR_WIRE_UNKNOWN_SWARM
from quinkgl.network.auto_discovery import (
    ManifestRegistry,
    discover_and_join,
    rank_candidates,
)
from quinkgl.network.directory import (
    SwarmAdvertisement,
    SwarmDirectoryCommunity,
    sign_advertisement,
)


# --- Helpers ---------------------------------------------------------------


def _fp(label_bucket: str = "medium", *, num_classes: int = 3) -> DataFingerprint:
    return DataFingerprint(
        label_buckets={"cls": label_bucket},
        noised_moments={"f": (0.1, 0.2)},
        sample_bucket="medium",
        num_classes=num_classes,
    )


def _ad_for(fp: DataFingerprint, swarm_id: str, priv: bytes, *, tags=None):
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
    return sign_advertisement(ad, priv)


def _seed(community: SwarmDirectoryCommunity, priv: bytes) -> dict:
    """Seed directory with three ads of varying affinity to the caller."""
    ads = {
        "exact": _ad_for(_fp("medium"), "a" * 64, priv, tags=["vision", "pytorch"]),
        "near": _ad_for(_fp("high"), "b" * 64, priv, tags=["vision"]),
        "far": _ad_for(
            DataFingerprint(
                label_buckets={"cls": "low"},
                noised_moments={"g": (5.0, 5.0)},
                sample_bucket="small",
                num_classes=2,
            ),
            "c" * 64,
            priv,
            tags=["audio"],
        ),
    }
    for ad in ads.values():
        community.ingest(ad)
    return ads


# --- rank_candidates -------------------------------------------------------


class TestRankCandidates:
    def test_returns_sorted_desc_by_score(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        results = rank_candidates(
            directory=community,
            fingerprint=_fp("medium"),
            tags=["vision"],
            min_affinity=0.0,
        )

        assert len(results) == 2  # "audio" ad excluded by tag filter
        scores = [score for score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_min_affinity_filter_drops_below_threshold(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        results = rank_candidates(
            directory=community,
            fingerprint=_fp("medium"),
            tags=None,
            min_affinity=0.99,
        )

        # Only an exact self-match stays above 0.99.
        assert len(results) == 1
        assert results[0][1].swarm_id_hex == "a" * 64

    def test_max_swarms_truncates(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        results = rank_candidates(
            directory=community,
            fingerprint=_fp("medium"),
            tags=None,
            min_affinity=0.0,
            max_swarms=1,
        )
        assert len(results) == 1

    def test_pinned_trust_policy_requires_signature_and_trusted_creator(self):
        priv_a, _ = keygen(None)
        priv_b, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        community.ingest(_ad_for(_fp("medium"), "a" * 64, priv_a))
        community.ingest(_ad_for(_fp("medium"), "b" * 64, priv_b))

        # Extract A's raw pubkey bytes for the trust set.
        pub_a_hex = community.query()[0].creator_pubkey.split(":", 1)[1]
        if community.query()[0].swarm_id_hex != "a" * 64:
            pub_a_hex = community.query()[1].creator_pubkey.split(":", 1)[1]
        trusted = {bytes.fromhex(pub_a_hex)}

        results = rank_candidates(
            directory=community,
            fingerprint=_fp("medium"),
            tags=None,
            min_affinity=0.0,
            trust_policy="pinned",
            trusted_creator_pubkeys=trusted,
        )
        assert [ad.swarm_id_hex for _, ad in results] == ["a" * 64]

    def test_open_trust_policy_accepts_any_signed_ad(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)
        results = rank_candidates(
            directory=community,
            fingerprint=_fp("medium"),
            tags=None,
            min_affinity=0.0,
            trust_policy="open",
        )
        assert len(results) == 3


# --- discover_and_join -----------------------------------------------------


class TestDiscoverAndJoin:
    @pytest.mark.asyncio
    async def test_loads_manifests_for_top_candidates(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        loaded: list = []

        async def loader(ad: SwarmAdvertisement):
            loaded.append(ad.swarm_id_hex)
            # Return a stand-in object — the orchestrator only passes
            # it through, so no real SwarmManifest is required here.
            return {"swarm_id": ad.swarm_id_hex}

        joined = await discover_and_join(
            directory=community,
            fingerprint=_fp("medium"),
            manifest_loader=loader,
            tags=["vision"],
            min_affinity=0.0,
            max_swarms=2,
        )

        assert [m["swarm_id"] for m in joined] == loaded
        assert len(joined) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_candidates_pass_filter(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        async def loader(ad):  # pragma: no cover - never called
            raise AssertionError("loader should not be called")

        joined = await discover_and_join(
            directory=community,
            fingerprint=_fp("medium"),
            manifest_loader=loader,
            tags=["nonexistent-tag"],
            min_affinity=0.0,
        )
        assert joined == []

    @pytest.mark.asyncio
    async def test_loader_exception_surfaces_but_later_candidates_still_run(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        _seed(community, priv)

        attempted: list = []

        async def loader(ad):
            attempted.append(ad.swarm_id_hex)
            if len(attempted) == 1:
                raise RuntimeError("transient fetch failure")
            return {"swarm_id": ad.swarm_id_hex}

        joined = await discover_and_join(
            directory=community,
            fingerprint=_fp("medium"),
            manifest_loader=loader,
            tags=None,
            min_affinity=0.0,
            max_swarms=3,
        )

        # Three ads were attempted (one per candidate), first failed, the
        # remaining two succeeded — the orchestrator MUST NOT stop at the
        # first transient error when max_swarms > 1.
        assert len(attempted) == 3
        assert len(joined) == 2


# --- ManifestRegistry (§18.3) ---------------------------------------------


class TestManifestRegistry:
    def test_register_and_lookup_by_bytes(self):
        registry = ManifestRegistry()
        community = object()
        swarm_id = b"\x00" * 32
        registry.register(swarm_id, community)
        assert registry.get(swarm_id) is community
        assert swarm_id in registry

    def test_register_rejects_non_32_byte_swarm_id(self):
        registry = ManifestRegistry()
        with pytest.raises(ValueError):
            registry.register(b"short", object())

    def test_unregister_removes_entry(self):
        registry = ManifestRegistry()
        swarm_id = b"\x01" * 32
        registry.register(swarm_id, "community")
        registry.unregister(swarm_id)
        assert registry.get(swarm_id) is None

    def test_route_dispatches_to_handler(self):
        registry = ManifestRegistry()
        received = {}

        class FakeCommunity:
            def on_packet(self, packet):
                received["seen"] = packet

        swarm_id = b"\x02" * 32
        registry.register(swarm_id, FakeCommunity())
        registry.route(swarm_id, packet=b"hello", handler="on_packet")
        assert received == {"seen": b"hello"}

    def test_route_unknown_swarm_raises_err_wire_unknown_swarm(self):
        registry = ManifestRegistry()
        with pytest.raises(ValueError) as exc:
            registry.route(b"\x03" * 32, packet=b"x", handler="on_packet")
        assert exc.value.args[0] == ERR_WIRE_UNKNOWN_SWARM
