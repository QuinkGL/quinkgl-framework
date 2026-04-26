# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Tests for quinkgl.testing public fixtures (B-10 acceptance)."""

from __future__ import annotations

import asyncio

import pytest

from quinkgl.testing import DummyDataLoader, local_swarm_fixture, make_dummy_manifest


def _pyipv8_default_curve_supported() -> bool:
    """pyipv8 ``medium`` keys may require OpenSSL curves dropped in newer cryptography builds."""
    try:
        from ipv8.keyvault.crypto import default_eccrypto

        default_eccrypto.generate_key("medium")
        return True
    except Exception:
        return False


_IPV8_CURVE_OK = _pyipv8_default_curve_supported()

_LOCAL_SWARM_SKIP = pytest.mark.skipif(
    not _IPV8_CURVE_OK,
    reason=(
        "IPv8/pyipv8 default EC curve unsupported by this OpenSSL/cryptography "
        "(e.g. sect409k1 on CI) — local_swarm_fixture needs real IPv8"
    ),
)


class TestMakeDummyManifest:
    def test_defaults_validate(self) -> None:
        m = make_dummy_manifest()
        assert m.name == "dummy-swarm"
        assert m.schema_version == 3

    def test_override_name(self) -> None:
        m = make_dummy_manifest(name="overridden")
        assert m.name == "overridden"


class TestDummyDataLoader:
    def test_iterates_correct_length(self) -> None:
        dl = DummyDataLoader([4, 3, 224, 224], num_batches=5)
        assert len(dl) == 5
        count = sum(1 for _ in dl)
        assert count == 5


@_LOCAL_SWARM_SKIP
class TestLocalSwarmFixture:
    @pytest.mark.asyncio
    async def test_creates_three_peers(self) -> None:
        async with local_swarm_fixture(size=3) as nodes:
            assert len(nodes) == 3
            for n in nodes:
                assert n.node_id.startswith("peer-")

    @pytest.mark.asyncio
    async def test_nodes_are_cleaned_up_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            async with local_swarm_fixture(size=2) as nodes:
                assert len(nodes) == 2
                raise RuntimeError("boom")
