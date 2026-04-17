"""Tests for manifest data_policy (Phase 6f)."""

import pytest

from quinkgl.manifest.schema import (
    CollaborationPolicy,
    DataPolicy,
    PersonalizationPolicy,
    PrototypePolicy,
)


# ── CollaborationPolicy ─────────────────────────────────────────────


class TestCollaborationPolicy:
    def test_defaults(self):
        cp = CollaborationPolicy()
        assert cp.mode == "personalized"
        assert cp.exploration_initial == 0.8
        assert cp.exploration_decay == 0.95
        assert cp.exploration_min == 0.1
        assert cp.ema_alpha == 0.2
        assert cp.edge_decay_factor == 0.95
        assert cp.eviction_min_weight == 0.05
        assert cp.cold_start_rounds == 3

    def test_custom(self):
        cp = CollaborationPolicy(mode="standard", cold_start_rounds=5, ema_alpha=0.3)
        assert cp.mode == "standard"
        assert cp.cold_start_rounds == 5
        assert cp.ema_alpha == 0.3

    def test_roundtrip(self):
        cp = CollaborationPolicy(mode="agnostic", exploration_initial=0.6, cold_start_rounds=7)
        d = cp.to_dict()
        cp2 = CollaborationPolicy.from_dict(d)
        assert cp2.mode == "agnostic"
        assert cp2.exploration_initial == 0.6
        assert cp2.cold_start_rounds == 7

    def test_from_dict_partial(self):
        d = {"mode": "standard"}
        cp = CollaborationPolicy.from_dict(d)
        assert cp.mode == "standard"
        assert cp.cold_start_rounds == 3

    def test_validate_valid(self):
        cp = CollaborationPolicy()
        cp.validate()

    def test_validate_bad_mode(self):
        cp = CollaborationPolicy(mode="invalid")
        with pytest.raises(ValueError, match="mode"):
            cp.validate()

    def test_validate_exploration_initial_out_of_range(self):
        cp = CollaborationPolicy(exploration_initial=1.5)
        with pytest.raises(ValueError, match="exploration_initial"):
            cp.validate()

    def test_validate_min_greater_than_initial(self):
        cp = CollaborationPolicy(exploration_min=0.9, exploration_initial=0.8)
        with pytest.raises(ValueError, match="exploration_min"):
            cp.validate()

    def test_validate_cold_start_zero(self):
        cp = CollaborationPolicy(cold_start_rounds=0)
        with pytest.raises(ValueError, match="cold_start_rounds"):
            cp.validate()

    def test_validate_ema_alpha_negative(self):
        cp = CollaborationPolicy(ema_alpha=-0.1)
        with pytest.raises(ValueError, match="ema_alpha"):
            cp.validate()


# ── PersonalizationPolicy ───────────────────────────────────────────


class TestPersonalizationPolicy:
    def test_defaults(self):
        pp = PersonalizationPolicy()
        assert pp.model_split == "auto"
        assert pp.apfl_enabled is True
        assert pp.apfl_initial_alpha == 0.5
        assert pp.fedbn_enabled is True

    def test_roundtrip(self):
        pp = PersonalizationPolicy(model_split="manual", apfl_enabled=False, apfl_initial_alpha=0.3)
        d = pp.to_dict()
        pp2 = PersonalizationPolicy.from_dict(d)
        assert pp2.model_split == "manual"
        assert pp2.apfl_enabled is False
        assert pp2.apfl_initial_alpha == 0.3

    def test_validate_valid(self):
        PersonalizationPolicy().validate()

    def test_validate_bad_model_split(self):
        pp = PersonalizationPolicy(model_split="invalid")
        with pytest.raises(ValueError, match="model_split"):
            pp.validate()

    def test_validate_alpha_out_of_range(self):
        pp = PersonalizationPolicy(apfl_initial_alpha=1.5)
        with pytest.raises(ValueError, match="apfl_initial_alpha"):
            pp.validate()


# ── PrototypePolicy ─────────────────────────────────────────────────


class TestPrototypePolicy:
    def test_defaults(self):
        pp = PrototypePolicy()
        assert pp.enabled is False
        assert pp.alignment_weight == 0.1
        assert pp.fedpac_enabled is False

    def test_roundtrip(self):
        pp = PrototypePolicy(enabled=True, alignment_weight=0.2, fedpac_enabled=True)
        d = pp.to_dict()
        pp2 = PrototypePolicy.from_dict(d)
        assert pp2.enabled is True
        assert pp2.alignment_weight == 0.2
        assert pp2.fedpac_enabled is True

    def test_validate_valid(self):
        PrototypePolicy().validate()
        PrototypePolicy(enabled=True, fedpac_enabled=True).validate()

    def test_validate_negative_alignment_weight(self):
        pp = PrototypePolicy(alignment_weight=-0.1)
        with pytest.raises(ValueError, match="alignment_weight"):
            pp.validate()

    def test_validate_fedpac_without_enabled(self):
        pp = PrototypePolicy(enabled=False, fedpac_enabled=True)
        with pytest.raises(ValueError, match="fedpac_enabled"):
            pp.validate()


# ── DataPolicy ──────────────────────────────────────────────────────


class TestDataPolicy:
    def test_defaults(self):
        dp = DataPolicy()
        assert dp.fingerprint_enabled is True
        assert dp.min_affinity == 0.3
        assert dp.privacy_level == "standard"
        assert dp.label_granularity == "bucket"
        assert dp.feature_noise_sigma == 0.1
        assert dp.gradient_fingerprint is False
        assert isinstance(dp.collaboration, CollaborationPolicy)
        assert isinstance(dp.personalization, PersonalizationPolicy)
        assert isinstance(dp.prototypes, PrototypePolicy)

    def test_roundtrip(self):
        dp = DataPolicy(
            fingerprint_enabled=False,
            min_affinity=0.5,
            privacy_level="strict",
            label_granularity="coarse",
            feature_noise_sigma=0.2,
            gradient_fingerprint=True,
            collaboration=CollaborationPolicy(cold_start_rounds=5),
            personalization=PersonalizationPolicy(apfl_enabled=False),
            prototypes=PrototypePolicy(enabled=True, fedpac_enabled=True),
        )
        d = dp.to_dict()
        dp2 = DataPolicy.from_dict(d)
        assert dp2.fingerprint_enabled is False
        assert dp2.min_affinity == 0.5
        assert dp2.privacy_level == "strict"
        assert dp2.collaboration.cold_start_rounds == 5
        assert dp2.personalization.apfl_enabled is False
        assert dp2.prototypes.enabled is True
        assert dp2.prototypes.fedpac_enabled is True

    def test_roundtrip_defaults(self):
        dp = DataPolicy()
        d = dp.to_dict()
        dp2 = DataPolicy.from_dict(d)
        assert dp2.fingerprint_enabled == dp.fingerprint_enabled
        assert dp2.collaboration.mode == dp.collaboration.mode
        assert dp2.prototypes.enabled == dp.prototypes.enabled

    def test_from_dict_partial(self):
        d = {"fingerprint_enabled": False}
        dp = DataPolicy.from_dict(d)
        assert dp.fingerprint_enabled is False
        assert dp.min_affinity == 0.3
        assert isinstance(dp.collaboration, CollaborationPolicy)

    def test_from_dict_empty(self):
        dp = DataPolicy.from_dict({})
        assert dp.fingerprint_enabled is True
        assert isinstance(dp.collaboration, CollaborationPolicy)

    def test_validate_valid(self):
        DataPolicy().validate()

    def test_validate_min_affinity_out_of_range(self):
        dp = DataPolicy(min_affinity=1.5)
        with pytest.raises(ValueError, match="min_affinity"):
            dp.validate()

    def test_validate_bad_privacy_level(self):
        dp = DataPolicy(privacy_level="invalid")
        with pytest.raises(ValueError, match="privacy_level"):
            dp.validate()

    def test_validate_bad_label_granularity(self):
        dp = DataPolicy(label_granularity="invalid")
        with pytest.raises(ValueError, match="label_granularity"):
            dp.validate()

    def test_validate_negative_sigma(self):
        dp = DataPolicy(feature_noise_sigma=-0.1)
        with pytest.raises(ValueError, match="feature_noise_sigma"):
            dp.validate()

    def test_validate_cascades_to_collaboration(self):
        dp = DataPolicy(collaboration=CollaborationPolicy(mode="invalid"))
        with pytest.raises(ValueError, match="mode"):
            dp.validate()

    def test_validate_cascades_to_personalization(self):
        dp = DataPolicy(personalization=PersonalizationPolicy(model_split="invalid"))
        with pytest.raises(ValueError, match="model_split"):
            dp.validate()

    def test_validate_cascades_to_prototypes(self):
        dp = DataPolicy(prototypes=PrototypePolicy(fedpac_enabled=True, enabled=False))
        with pytest.raises(ValueError, match="fedpac_enabled"):
            dp.validate()


class TestApplyJoinPolicy:
    def test_valid_join(self):
        dp = DataPolicy()
        dp.apply_join_policy()

    def test_invalid_join_raises(self):
        dp = DataPolicy(privacy_level="bogus")
        with pytest.raises(ValueError):
            dp.apply_join_policy()

    def test_join_with_custom_cold_start(self):
        dp = DataPolicy(collaboration=CollaborationPolicy(cold_start_rounds=10))
        dp.apply_join_policy()
        assert dp.collaboration.cold_start_rounds == 10

    def test_join_with_prototypes_enabled(self):
        dp = DataPolicy(prototypes=PrototypePolicy(enabled=True, alignment_weight=0.2))
        dp.apply_join_policy()
        assert dp.prototypes.enabled is True
