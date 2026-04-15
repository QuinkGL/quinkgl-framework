"""Tests for APFL (Adaptive Personalized Federated Learning) mixin."""

import numpy as np
import pytest

from quinkgl.models.base import APFLConfig, APFLMixin


class TestAPFLConfig:
    def test_default_config(self):
        cfg = APFLConfig()
        assert cfg.initial_alpha == 0.5
        assert cfg.alpha_lr == 0.01
        assert cfg.min_alpha == 0.1
        assert cfg.max_alpha == 0.9
        assert cfg.update_frequency == 1

    def test_custom_config(self):
        cfg = APFLConfig(initial_alpha=0.8, alpha_lr=0.05, min_alpha=0.2, max_alpha=0.95, update_frequency=3)
        assert cfg.initial_alpha == 0.8
        assert cfg.alpha_lr == 0.05
        assert cfg.update_frequency == 3


class TestAPFLMixinComputePersonalized:
    def test_balanced_mix(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5))
        local = {"w": np.array([2.0, 4.0])}
        global_ = {"w": np.array([0.0, 0.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["w"], np.array([1.0, 2.0]))

    def test_alpha_1_pure_local(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=1.0, min_alpha=0.0, max_alpha=1.0))
        local = {"w": np.array([10.0])}
        global_ = {"w": np.array([0.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["w"], np.array([10.0]))

    def test_alpha_0_pure_global(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.0, min_alpha=0.0, max_alpha=1.0))
        local = {"w": np.array([10.0])}
        global_ = {"w": np.array([5.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["w"], np.array([5.0]))

    def test_multi_key(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5))
        local = {"a": np.array([2.0]), "b": np.array([4.0])}
        global_ = {"a": np.array([0.0]), "b": np.array([0.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["a"], np.array([1.0]))
        np.testing.assert_array_almost_equal(result["b"], np.array([2.0]))

    def test_local_only_key_preserved(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5))
        local = {"a": np.array([2.0]), "local_only": np.array([99.0])}
        global_ = {"a": np.array([0.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["local_only"], np.array([99.0]))

    def test_global_only_key_preserved(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5))
        local = {"a": np.array([2.0])}
        global_ = {"a": np.array([0.0]), "global_only": np.array([77.0])}
        result = mixin.compute_personalized_weights(local, global_)
        np.testing.assert_array_almost_equal(result["global_only"], np.array([77.0]))


class TestAPFLMixinAlphaAdaptation:
    def test_alpha_increases_when_local_better(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.05))
        mixin.update_alpha(val_loss_local=0.3, val_loss_global=0.5)
        assert mixin.alpha == pytest.approx(0.55)

    def test_alpha_decreases_when_global_better(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.05))
        mixin.update_alpha(val_loss_local=0.7, val_loss_global=0.3)
        assert mixin.alpha == pytest.approx(0.45)

    def test_alpha_respects_max(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.88, alpha_lr=0.05, max_alpha=0.9))
        mixin.update_alpha(val_loss_local=0.1, val_loss_global=0.5)
        assert mixin.alpha == pytest.approx(0.9)

    def test_alpha_respects_min(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.12, alpha_lr=0.05, min_alpha=0.1))
        mixin.update_alpha(val_loss_local=0.7, val_loss_global=0.3)
        assert mixin.alpha == pytest.approx(0.1)

    def test_update_frequency(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.05, update_frequency=3))
        mixin.update_alpha(0.3, 0.5)  # round 1: skip
        assert mixin.alpha == pytest.approx(0.5)
        mixin.update_alpha(0.3, 0.5)  # round 2: skip
        assert mixin.alpha == pytest.approx(0.5)
        mixin.update_alpha(0.3, 0.5)  # round 3: update
        assert mixin.alpha == pytest.approx(0.55)

    def test_equal_loss_no_change(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.05))
        mixin.update_alpha(val_loss_local=0.5, val_loss_global=0.5)
        assert mixin.alpha == pytest.approx(0.45)  # global not better → decrease

    def test_converges_to_preferred(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.02, min_alpha=0.1, max_alpha=0.9))
        for _ in range(50):
            mixin.update_alpha(val_loss_local=0.2, val_loss_global=0.8)
        assert mixin.alpha > 0.8

    def test_converges_to_global(self):
        mixin = APFLMixin(APFLConfig(initial_alpha=0.5, alpha_lr=0.02, min_alpha=0.1, max_alpha=0.9))
        for _ in range(50):
            mixin.update_alpha(val_loss_local=0.8, val_loss_global=0.2)
        assert mixin.alpha < 0.2
