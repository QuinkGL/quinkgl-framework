"""Phase-2 residual exports on ``quinkgl.gossip`` (spec §10.6.1, §15).

Covers:
* :class:`quinkgl.gossip.TrustPolicy` enum — values, ``str`` coercion
  (so legacy ``trust_policy="open"`` callers keep working), JSON
  serialisation.
* :class:`quinkgl.gossip.TrainingMetrics` dataclass — required +
  optional fields, ``to_dict()`` omits ``None`` values, ``extra``
  flattens without overriding canonical keys.
"""

from __future__ import annotations

import json

import pytest

from quinkgl.gossip import TrainingMetrics, TrustPolicy


class TestTrustPolicyEnum:
    def test_values_match_spec(self):
        assert TrustPolicy.OPEN.value == "open"
        assert TrustPolicy.TOFU.value == "tofu"
        assert TrustPolicy.PINNED.value == "pinned"

    def test_str_mixin_equality_with_legacy_strings(self):
        # The enum is a str-mixin so existing ``trust_policy="open"``
        # call sites keep working and comparisons in GossipNode's
        # validation branch stay identical.
        assert TrustPolicy.OPEN == "open"
        assert TrustPolicy.PINNED == "pinned"

    def test_json_serialises_as_bare_string(self):
        dumped = json.dumps({"policy": TrustPolicy.TOFU})
        assert json.loads(dumped) == {"policy": "tofu"}

    def test_coerce_accepts_enum_and_string(self):
        assert TrustPolicy.coerce("open") is TrustPolicy.OPEN
        assert TrustPolicy.coerce("TOFU") is TrustPolicy.TOFU
        assert TrustPolicy.coerce(TrustPolicy.PINNED) is TrustPolicy.PINNED

    def test_coerce_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="invalid trust_policy"):
            TrustPolicy.coerce("untrusted")
        with pytest.raises(ValueError, match="invalid trust_policy"):
            TrustPolicy.coerce(42)


class TestTrainingMetrics:
    def test_only_required_field(self):
        m = TrainingMetrics(round_number=5)
        assert m.round_number == 5
        assert m.loss is None
        assert m.to_dict() == {"round_number": 5}

    def test_full_snapshot_round_trip(self):
        m = TrainingMetrics(
            round_number=3,
            loss=0.42,
            accuracy=0.81,
            samples_trained=1024,
            duration_s=1.5,
            peer_count=6,
        )
        d = m.to_dict()
        assert d == {
            "round_number": 3,
            "loss": 0.42,
            "accuracy": 0.81,
            "samples_trained": 1024,
            "duration_s": 1.5,
            "peer_count": 6,
        }

    def test_extra_is_flattened_and_does_not_override_canonical(self):
        m = TrainingMetrics(
            round_number=1,
            loss=0.1,
            extra={"gradient_norm": 2.5, "loss": 999.0},
        )
        d = m.to_dict()
        assert d["loss"] == 0.1  # canonical wins
        assert d["gradient_norm"] == 2.5

    def test_to_dict_is_json_ready(self):
        m = TrainingMetrics(round_number=1, loss=0.5, extra={"custom": [1, 2, 3]})
        # json.dumps MUST not raise for the emitted dict.
        payload = json.loads(json.dumps(m.to_dict()))
        assert payload["round_number"] == 1
        assert payload["custom"] == [1, 2, 3]
