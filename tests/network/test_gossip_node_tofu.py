"""GossipNode ↔ TOFU cache wiring (spec §15)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quinkgl.gossip import TrustPolicy
from quinkgl.manifest.errors import ERR_TRUST_TOFU_CONFLICT
from quinkgl.network.tofu import TofuCache


@pytest.fixture
def tofu_work_dir(tmp_path: Path, monkeypatch):
    """Redirect the TOFU cache file into a per-test work dir."""
    work = tmp_path / "ipv8"
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IPV8_WORK_DIR", str(work))
    return work


def _make_signed_manifest(pubkey_hex: str = "aa" * 32):
    """Build a minimal v3 manifest carrying a creator_pubkey + dummy signature.

    The signature bytes do not need to verify — the TOFU path only reads
    ``manifest.creator_pubkey`` and ``manifest.manifest_hash()``.  We
    still populate ``signature`` because ``trust_policy='pinned'``
    rejects unsigned manifests up-front.
    """
    from quinkgl.testing import make_dummy_manifest

    m = make_dummy_manifest()
    m = replace(
        m,
        creator_pubkey=f"ed25519:{pubkey_hex}",
        signature=f"ed25519:{'bb' * 64}",
    )
    return m


def _build_model():
    import torch.nn as nn

    from quinkgl.models import PyTorchModel

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    return PyTorchModel(_Tiny())


class TestEnumAcceptance:
    def test_gossip_node_accepts_trust_policy_enum(self, tofu_work_dir: Path):
        from quinkgl import GossipNode

        manifest = _make_signed_manifest()
        node = GossipNode(
            node_id="peer-1",
            manifest=manifest,
            model=_build_model(),
            trust_policy=TrustPolicy.OPEN,
            quiet=True,
        )
        assert node.trust_policy == "open"

    def test_gossip_node_accepts_bare_string(self, tofu_work_dir: Path):
        from quinkgl import GossipNode

        node = GossipNode(
            node_id="peer-2",
            manifest=_make_signed_manifest(),
            model=_build_model(),
            trust_policy="open",
            quiet=True,
        )
        assert node.trust_policy == "open"


class TestTofuCacheIntegration:
    def test_first_encounter_writes_tofu_cache(self, tofu_work_dir: Path):
        from quinkgl import GossipNode

        manifest = _make_signed_manifest()
        node = GossipNode(
            node_id="peer-tofu-1",
            manifest=manifest,
            model=_build_model(),
            trust_policy=TrustPolicy.TOFU,
            quiet=True,
        )

        cache_path = tofu_work_dir / "tofu_creators.json"
        assert cache_path.exists()
        snap = TofuCache(cache_path).as_dict()
        assert manifest.manifest_hash() in snap
        assert snap[manifest.manifest_hash()]["creator_pubkey"] == manifest.creator_pubkey
        # Keep a reference to prevent the fixture's teardown racing with
        # IPv8 manager __del__.
        del node

    def test_conflicting_creator_key_raises_tofu_conflict(self, tofu_work_dir: Path):
        from quinkgl import GossipNode

        first = _make_signed_manifest(pubkey_hex="aa" * 32)
        GossipNode(
            node_id="peer-first",
            manifest=first,
            model=_build_model(),
            trust_policy=TrustPolicy.TOFU,
            quiet=True,
        )

        # Same swarm_id (manifest_hash) but different creator_pubkey ——
        # to force swarm_id reuse we copy the manifest and flip just
        # the creator_pubkey.  TOFU compares against the swarm_id
        # derived from canonical bytes, which *does* include the
        # pubkey; so instead we construct a second manifest whose
        # canonical bytes match the first by preserving every field
        # except creator_pubkey and signature, then manually cache the
        # first under the new manifest's swarm_id to simulate a
        # republish-with-different-key attack.
        attacker_manifest = replace(
            first,
            creator_pubkey=f"ed25519:{'cc' * 32}",
        )
        cache = TofuCache(tofu_work_dir / "tofu_creators.json")
        # Manually seed the attacker-swarm id with the original key to
        # set up the conflict scenario deterministically.
        cache.record_or_validate(
            attacker_manifest.manifest_hash(), first.creator_pubkey
        )

        with pytest.raises(ValueError) as excinfo:
            GossipNode(
                node_id="peer-attacker",
                manifest=attacker_manifest,
                model=_build_model(),
                trust_policy=TrustPolicy.TOFU,
                quiet=True,
            )

        assert excinfo.value.args[0] == ERR_TRUST_TOFU_CONFLICT
        payload = excinfo.value.args[1]
        assert payload["expected"] == first.creator_pubkey
        assert payload["actual"] == attacker_manifest.creator_pubkey

    def test_tofu_skipped_when_manifest_unsigned(self, tofu_work_dir: Path):
        """Unsigned manifests cannot bind a creator_pubkey → TOFU is a no-op.

        This matches the spec: TOFU protects the ``swarm_id →
        creator_pubkey`` binding, and an unsigned manifest simply has
        no pubkey to bind.  The node MUST still be constructible (so
        unsigned-manifest + ``trust_policy=tofu`` degrades to
        effectively ``open``).
        """
        from quinkgl import GossipNode
        from quinkgl.testing import make_dummy_manifest

        manifest = make_dummy_manifest()  # no creator_pubkey/signature
        GossipNode(
            node_id="peer-unsigned",
            manifest=manifest,
            model=_build_model(),
            trust_policy=TrustPolicy.TOFU,
            quiet=True,
        )
        assert not (tofu_work_dir / "tofu_creators.json").exists()
