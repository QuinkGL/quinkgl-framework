"""Phase 3 ``SwarmDirectoryCommunity`` contract (spec §17.1, §17.3, §17.4).

These tests exercise the transport-free parts of the directory — the
fixed community ID, local LRU cache with TTL eviction, the per-creator
and per-session rate limits, and the synchronous :meth:`query`
filtering surface.  The IPv8 reactor itself is deliberately out of
scope here.
"""

from __future__ import annotations

import hashlib

import pytest

from quinkgl.manifest import keygen
from quinkgl.manifest.errors import ERR_SIGNATURE_INVALID, ERR_WIRE_RATE_LIMITED
from quinkgl.network.directory import (
    DEFAULT_ADVERTISEMENT_TTL_SECONDS,
    DIRECTORY_COMMUNITY_ID,
    MAX_ADS_PER_CREATOR_PER_DAY,
    MAX_ADS_PER_SESSION,
    MAX_CACHE_ENTRIES,
    SwarmAdvertisement,
    SwarmDirectoryCommunity,
    sign_advertisement,
)


# --- Fixed identity --------------------------------------------------------


def test_directory_community_id_is_sha256_of_fixed_tag():
    """§17.1: first 20 bytes of SHA-256 of ``b"QuinkGL-SwarmDirectory-v1"``."""
    expected = hashlib.sha256(b"QuinkGL-SwarmDirectory-v1").digest()[:20]
    assert DIRECTORY_COMMUNITY_ID == expected
    assert len(DIRECTORY_COMMUNITY_ID) == 20


# --- Helpers ---------------------------------------------------------------


def _make_ad(swarm_id: str, *, tags=None, shape=None, label="integer"):
    return SwarmAdvertisement(
        swarm_id_hex=swarm_id,
        name=f"swarm-{swarm_id[:6]}",
        tags=list(tags or []),
        input_shape=list(shape or [3, 32, 32]),
        output_shape=[10],
        label_type=label,
        data_schema_hash="sha256:" + "0" * 64,
        reference_fingerprint={"version": 1, "hash": swarm_id[:16]},
    )


def _signed(swarm_id: str, private_pem: bytes, **kwargs) -> SwarmAdvertisement:
    return sign_advertisement(_make_ad(swarm_id, **kwargs), private_pem)


# --- Ingest / cache --------------------------------------------------------


class TestDirectoryIngest:
    def test_ingest_valid_ad_populates_cache(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        ad = _signed("a" * 64, priv)
        assert community.ingest(ad) is True
        assert len(community.all_advertisements()) == 1

    def test_ingest_rejects_unsigned_ad(self):
        community = SwarmDirectoryCommunity()
        ad = _make_ad("a" * 64)
        assert community.ingest(ad) is False
        assert community.all_advertisements() == []

    def test_ingest_stamps_received_at_and_expires_at(self):
        priv, _ = keygen(None)
        clock = [1_700_000_000.0]
        community = SwarmDirectoryCommunity(clock=lambda: clock[0])
        community.ingest(_signed("a" * 64, priv))

        [stored] = community.all_advertisements()
        assert stored.received_at == 1_700_000_000.0
        assert stored.expires_at == 1_700_000_000.0 + DEFAULT_ADVERTISEMENT_TTL_SECONDS

    def test_duplicate_swarm_id_highest_received_at_wins(self):
        """§17.3: duplicates dedupe by highest ``received_at``."""
        priv, _ = keygen(None)
        clock = [1_000.0]
        community = SwarmDirectoryCommunity(clock=lambda: clock[0])

        first = _signed("a" * 64, priv)
        first.name = "first"
        first = sign_advertisement(first, priv)
        community.ingest(first)

        # Advance the clock and ingest a newer ad for the same swarm.
        clock[0] = 2_000.0
        second = _signed("a" * 64, priv)
        second.name = "second"
        second = sign_advertisement(second, priv)
        assert community.ingest(second) is True

        [stored] = community.all_advertisements()
        assert stored.name == "second"

        # Replaying an older ad after the newer one MUST NOT win.
        # The community uses its own clock (which is still 2000.0), but
        # the already-stored entry was received at 2000.0 too, so a
        # replay now would tie.  Rewind the clock just to exercise the
        # "older wins → drop" branch deterministically.
        clock[0] = 1_500.0
        assert community.ingest(first) is False
        [stored] = community.all_advertisements()
        assert stored.name == "second"

    def test_expired_ads_are_evicted(self):
        priv, _ = keygen(None)
        clock = [1_000.0]
        community = SwarmDirectoryCommunity(
            clock=lambda: clock[0], default_ttl_seconds=10
        )
        community.ingest(_signed("a" * 64, priv))
        assert len(community.all_advertisements()) == 1
        clock[0] = 1_100.0
        assert community.all_advertisements() == []

    def test_cache_cap_evicts_oldest(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity(max_cache_entries=3)
        for swarm_id in ("a" * 64, "b" * 64, "c" * 64, "d" * 64):
            community.ingest(_signed(swarm_id, priv))
        ids = [a.swarm_id_hex for a in community.all_advertisements()]
        assert ids == ["b" * 64, "c" * 64, "d" * 64]


# --- Query filters (§17.4) -------------------------------------------------


class TestDirectoryQuery:
    def _seed(self, community):
        priv, _ = keygen(None)
        community.ingest(
            _signed("a" * 64, priv, tags=["vision", "pytorch"], shape=[3, 224, 224])
        )
        community.ingest(
            _signed("b" * 64, priv, tags=["audio"], shape=[1, 16000], label="integer")
        )
        community.ingest(
            _signed("c" * 64, priv, tags=["vision"], shape=[3, 32, 32], label="float")
        )
        return priv

    def test_query_no_filters_returns_all(self):
        community = SwarmDirectoryCommunity()
        self._seed(community)
        assert len(community.query()) == 3

    def test_query_tag_filter_is_subset_and(self):
        community = SwarmDirectoryCommunity()
        self._seed(community)
        vision_pytorch = community.query(tags=["vision", "pytorch"])
        assert {a.swarm_id_hex for a in vision_pytorch} == {"a" * 64}

    def test_query_filter_by_input_shape_and_label_type(self):
        community = SwarmDirectoryCommunity()
        self._seed(community)
        res = community.query(input_shape=[3, 224, 224], label_type="integer")
        assert {a.swarm_id_hex for a in res} == {"a" * 64}

    def test_query_trusted_creators_filters_by_pubkey(self):
        community = SwarmDirectoryCommunity()
        priv = self._seed(community)

        # Extract the signing pubkey that the seeded ads all share.
        pub_hex = community.all_advertisements()[0].creator_pubkey.split(":", 1)[1]
        good = {bytes.fromhex(pub_hex)}

        assert len(community.query(trusted_creators=good)) == 3
        # A fresh keypair that never signed anything filters everything out.
        _, fresh_pub = keygen(None)
        assert community.query(trusted_creators={fresh_pub}) == []


# --- Publish + rate limits -------------------------------------------------


class TestDirectoryPublish:
    def test_publish_ingests_locally(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity()
        community.publish(_signed("a" * 64, priv))
        assert len(community.all_advertisements()) == 1

    def test_publish_rejects_unsigned(self):
        community = SwarmDirectoryCommunity()
        with pytest.raises(ValueError) as exc:
            community.publish(_make_ad("a" * 64))
        assert exc.value.args[0] == ERR_SIGNATURE_INVALID

    def test_session_rate_limit_trips_after_threshold(self):
        priv, _ = keygen(None)
        community = SwarmDirectoryCommunity(max_ads_per_session=3)
        for i in range(3):
            community.publish(_signed(f"{i:064x}", priv))
        with pytest.raises(ValueError) as exc:
            community.publish(_signed("f" * 64, priv))
        assert exc.value.args[0] == ERR_WIRE_RATE_LIMITED
        assert "session" in exc.value.args[1]["detail"]

    def test_per_creator_daily_rate_limit_trips(self):
        priv_a, _ = keygen(None)
        priv_b, _ = keygen(None)
        community = SwarmDirectoryCommunity(
            max_ads_per_creator_per_day=2, max_ads_per_session=100
        )
        community.publish(_signed("a" * 64, priv_a))
        community.publish(_signed("b" * 64, priv_a))
        with pytest.raises(ValueError) as exc:
            community.publish(_signed("c" * 64, priv_a))
        assert exc.value.args[0] == ERR_WIRE_RATE_LIMITED
        assert "creator" in exc.value.args[1]["detail"]

        # A different creator is unaffected by A's quota.
        community.publish(_signed("d" * 64, priv_b))

    def test_creator_window_slides_after_one_day(self):
        priv, _ = keygen(None)
        clock = [1_000_000.0]
        community = SwarmDirectoryCommunity(
            clock=lambda: clock[0],
            max_ads_per_creator_per_day=1,
            max_ads_per_session=100,
        )
        community.publish(_signed("a" * 64, priv))
        with pytest.raises(ValueError):
            community.publish(_signed("b" * 64, priv))
        # 24h + 1s later, the window has slid forward.
        clock[0] += 24 * 60 * 60 + 1
        community.publish(_signed("c" * 64, priv))


# --- Constant sanity -------------------------------------------------------


def test_spec_constants_match_spec_values():
    assert MAX_CACHE_ENTRIES == 10_000
    assert MAX_ADS_PER_CREATOR_PER_DAY == 100
    assert MAX_ADS_PER_SESSION == 10
    assert DEFAULT_ADVERTISEMENT_TTL_SECONDS == 30 * 24 * 60 * 60
