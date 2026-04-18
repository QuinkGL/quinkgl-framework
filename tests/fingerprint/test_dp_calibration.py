"""Tests for DP noise calibration (audit Task-2 finding F3).

Covers:
  - ``calibrated_noise_scale`` formula correctness (Gaussian + Laplace)
  - Parameter validation (epsilon, delta, sensitivity, mechanism)
  - ``FingerprintPrivacyConfig.effective_*_noise_scale`` dispatch
  - Fresh noise sampling per query (no cached noise)
  - Backwards compatibility when ``*_dp_epsilon`` is unset
"""

import math

import numpy as np
import pytest

from quinkgl.fingerprint import (
    FingerprintComputer,
    FingerprintPrivacyConfig,
    NOISE_MECHANISM_GAUSSIAN,
    NOISE_MECHANISM_LAPLACE,
    NOISE_MECHANISM_NONE,
    calibrated_noise_scale,
)


# ── calibrated_noise_scale ─────────────────────────────────────────


class TestCalibratedNoiseScale:
    def test_laplace_formula(self):
        assert calibrated_noise_scale(NOISE_MECHANISM_LAPLACE, 2.0, 0.5) == pytest.approx(4.0)

    def test_gaussian_formula(self):
        sens, eps, delta = 1.0, 1.0, 1e-5
        expected = sens * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        got = calibrated_noise_scale(NOISE_MECHANISM_GAUSSIAN, sens, eps, delta)
        assert got == pytest.approx(expected)

    def test_gaussian_scale_increases_as_epsilon_decreases(self):
        s_high_eps = calibrated_noise_scale(NOISE_MECHANISM_GAUSSIAN, 1.0, 1.0, 1e-5)
        s_low_eps = calibrated_noise_scale(NOISE_MECHANISM_GAUSSIAN, 1.0, 0.1, 1e-5)
        assert s_low_eps > s_high_eps

    def test_scale_proportional_to_sensitivity(self):
        s1 = calibrated_noise_scale(NOISE_MECHANISM_LAPLACE, 1.0, 1.0)
        s2 = calibrated_noise_scale(NOISE_MECHANISM_LAPLACE, 5.0, 1.0)
        assert s2 == pytest.approx(5.0 * s1)

    def test_none_mechanism_returns_zero(self):
        assert calibrated_noise_scale(NOISE_MECHANISM_NONE, 1.0, 1.0) == 0.0

    @pytest.mark.parametrize("sensitivity", [0.0, -1.0])
    def test_invalid_sensitivity(self, sensitivity):
        with pytest.raises(ValueError):
            calibrated_noise_scale(NOISE_MECHANISM_GAUSSIAN, sensitivity, 1.0)

    @pytest.mark.parametrize("epsilon", [0.0, -0.1])
    def test_invalid_epsilon(self, epsilon):
        with pytest.raises(ValueError):
            calibrated_noise_scale(NOISE_MECHANISM_LAPLACE, 1.0, epsilon)

    @pytest.mark.parametrize("delta", [0.0, 1.0, 1.5, -1e-5])
    def test_invalid_delta_for_gaussian(self, delta):
        with pytest.raises(ValueError):
            calibrated_noise_scale(NOISE_MECHANISM_GAUSSIAN, 1.0, 1.0, delta)

    def test_invalid_mechanism(self):
        with pytest.raises(ValueError):
            calibrated_noise_scale("exponential", 1.0, 1.0)


# ── FingerprintPrivacyConfig.effective_*_noise_scale ───────────────


class TestEffectiveNoiseScale:
    def test_feature_legacy_fallback(self):
        cfg = FingerprintPrivacyConfig(feature_noise_sigma=0.3)
        assert cfg.feature_dp_epsilon is None
        assert cfg.effective_feature_noise_scale() == pytest.approx(0.3)

    def test_feature_calibrated_from_epsilon(self):
        cfg = FingerprintPrivacyConfig(
            feature_noise_sigma=0.1,  # should be ignored
            feature_clip_norm=2.0,
            feature_dp_epsilon=1.0,
            feature_dp_delta=1e-5,
            feature_noise_mechanism=NOISE_MECHANISM_GAUSSIAN,
        )
        expected = 2.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert cfg.effective_feature_noise_scale() == pytest.approx(expected)
        # Legacy sigma must NOT affect the output.
        assert cfg.effective_feature_noise_scale() != pytest.approx(0.1)

    def test_feature_explicit_sensitivity_overrides_clip(self):
        cfg = FingerprintPrivacyConfig(
            feature_clip_norm=5.0,
            feature_sensitivity=1.0,
            feature_dp_epsilon=1.0,
            feature_noise_mechanism=NOISE_MECHANISM_LAPLACE,
        )
        assert cfg.effective_feature_noise_scale() == pytest.approx(1.0)

    def test_gradient_requires_sensitivity(self):
        cfg = FingerprintPrivacyConfig(gradient_dp_epsilon=1.0)
        with pytest.raises(ValueError):
            cfg.effective_gradient_noise_scale()

    def test_invalid_mechanism_rejected(self):
        with pytest.raises(ValueError):
            FingerprintPrivacyConfig(feature_noise_mechanism="laplacian")


# ── FingerprintComputer end-to-end noise behaviour ─────────────────


class TestComputerNoise:
    def _moments(self):
        return {f"k{i}": (0.1 * i, 0.01 * i) for i in range(8)}

    def test_fresh_noise_per_call(self):
        """Two successive computes on identical inputs must produce
        different noised moments — otherwise averaging attacks trivially
        recover the raw moments."""
        cfg = FingerprintPrivacyConfig(
            feature_noise_sigma=0.5,  # legacy path, still non-zero
        )
        comp = FingerprintComputer(cfg)
        fp1 = comp.compute_from_label_counts({"a": 10}, feature_moments=self._moments())
        fp2 = comp.compute_from_label_counts({"a": 10}, feature_moments=self._moments())
        assert fp1.noised_moments != fp2.noised_moments

    def test_calibrated_path_applies_calibrated_scale(self):
        """Empirical std of added noise roughly matches the calibrated σ."""
        np.random.seed(0)
        sensitivity = 1.0
        epsilon = 1.0
        cfg = FingerprintPrivacyConfig(
            feature_clip_norm=sensitivity,
            feature_sensitivity=sensitivity,
            feature_dp_epsilon=epsilon,
            feature_noise_mechanism=NOISE_MECHANISM_GAUSSIAN,
        )
        expected_sigma = cfg.effective_feature_noise_scale()

        comp = FingerprintComputer(cfg)
        # Constant moment at 0 → noised_mean IS the noise sample.
        samples = []
        for _ in range(4000):
            fp = comp.compute_from_label_counts(
                {"a": 10}, feature_moments={"x": (0.0, 0.0)}
            )
            samples.append(fp.noised_moments["x"][0])
        empirical = float(np.std(samples))
        # 10% tolerance for 4000 samples.
        assert empirical == pytest.approx(expected_sigma, rel=0.1)

    def test_none_mechanism_produces_no_noise(self):
        cfg = FingerprintPrivacyConfig(
            feature_noise_mechanism=NOISE_MECHANISM_NONE,
            feature_dp_epsilon=1.0,
            feature_sensitivity=1.0,
        )
        comp = FingerprintComputer(cfg)
        fp = comp.compute_from_label_counts(
            {"a": 10}, feature_moments={"x": (0.3, 0.2)}
        )
        # With no noise, values must equal clipped inputs exactly.
        assert fp.noised_moments["x"] == (0.3, 0.2)
