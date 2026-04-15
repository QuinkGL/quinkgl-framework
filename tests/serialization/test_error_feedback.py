"""Tests for the Error Feedback (EF) mechanism.

Verifies that the residual buffer correctly accumulates compression
error and re-injects it, turning biased compressors into effectively
unbiased ones (Alistarh et al. 2018, Richtárik et al. 2021 — EF21).
"""

import numpy as np
import pytest

from quinkgl.serialization.error_feedback import (
    ErrorFeedbackConfig,
    ErrorFeedbackState,
)


# ------------------------------------------------------------------ #
# Basic residual accumulation
# ------------------------------------------------------------------ #

class TestErrorFeedbackBasics:
    def test_first_round_passthrough(self):
        """First call to apply() should return delta unchanged (no residual yet)."""
        ef = ErrorFeedbackState()
        delta = np.array([1.0, 2.0, 3.0])
        result = ef.apply(delta)
        np.testing.assert_array_equal(result, delta)

    def test_residual_accumulates(self):
        """After update(), the residual should be non-zero."""
        ef = ErrorFeedbackState()
        corrected = np.array([1.0, 2.0, 3.0])
        # Simulate sparsification that zeroes out small values
        compressed = np.array([0.0, 0.0, 3.0])

        ef.update(corrected, compressed)
        assert ef.total_residual_norm > 0

    def test_residual_injected_in_next_round(self):
        """apply() should add the residual to the next delta."""
        ef = ErrorFeedbackState()
        delta1 = np.array([1.0, 2.0, 3.0])
        compressed1 = np.array([0.0, 0.0, 3.0])

        ef.update(delta1, compressed1)
        # residual = [1.0, 2.0, 0.0]

        delta2 = np.array([0.5, 0.5, 0.5])
        corrected2 = ef.apply(delta2)
        # expected = [0.5+1.0, 0.5+2.0, 0.5+0.0] = [1.5, 2.5, 0.5]
        np.testing.assert_allclose(corrected2, [1.5, 2.5, 0.5])

    def test_dict_weights(self):
        """EF should work with dict-of-arrays (PyTorch state_dict style)."""
        ef = ErrorFeedbackState()

        delta = {
            "layer1.weight": np.array([1.0, 2.0, 3.0]),
            "layer1.bias": np.array([0.1]),
        }
        compressed = {
            "layer1.weight": np.array([0.0, 0.0, 3.0]),
            "layer1.bias": np.array([0.0]),
        }

        ef.update(delta, compressed)
        assert ef.total_residual_norm > 0

        delta2 = {
            "layer1.weight": np.array([0.5, 0.5, 0.5]),
            "layer1.bias": np.array([0.2]),
        }
        corrected = ef.apply(delta2)
        np.testing.assert_allclose(corrected["layer1.weight"], [1.5, 2.5, 0.5])
        np.testing.assert_allclose(corrected["layer1.bias"], [0.3])

    def test_reset_clears_residuals(self):
        ef = ErrorFeedbackState()
        ef.update(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert ef.total_residual_norm > 0

        ef.reset()
        assert ef.total_residual_norm == 0.0
        assert ef.round_number == 0

    def test_disabled_is_passthrough(self):
        """When disabled, apply() returns input unchanged and update() is no-op."""
        config = ErrorFeedbackConfig(enabled=False)
        ef = ErrorFeedbackState(config=config)

        delta = np.array([1.0, 2.0])
        result = ef.apply(delta)
        np.testing.assert_array_equal(result, delta)

        ef.update(delta, np.zeros(2))
        assert ef.total_residual_norm == 0.0


# ------------------------------------------------------------------ #
# EF21-style momentum
# ------------------------------------------------------------------ #

class TestEF21Momentum:
    def test_momentum_blends_residuals(self):
        """With momentum > 0, old residual should be blended with new."""
        config = ErrorFeedbackConfig(momentum=0.5)
        ef = ErrorFeedbackState(config=config)

        # Round 1
        ef.update(np.array([10.0]), np.array([0.0]))
        r1 = ef.total_residual_norm  # residual = 10.0

        # Round 2: new residual = 5.0, blended = 0.5*10 + 0.5*5 = 7.5
        ef.update(np.array([5.0]), np.array([0.0]))
        r2 = ef.total_residual_norm

        assert r2 == pytest.approx(7.5, abs=1e-6)


# ------------------------------------------------------------------ #
# Norm capping
# ------------------------------------------------------------------ #

class TestResidualNormCap:
    def test_residual_capped(self):
        """Residual norm should not exceed max_residual_norm."""
        config = ErrorFeedbackConfig(max_residual_norm=1.0)
        ef = ErrorFeedbackState(config=config)

        ef.update(np.array([100.0, 100.0]), np.array([0.0, 0.0]))
        assert ef.total_residual_norm <= 1.0 + 1e-6

    def test_no_cap_by_default(self):
        config = ErrorFeedbackConfig()
        ef = ErrorFeedbackState(config=config)

        ef.update(np.array([100.0, 100.0]), np.array([0.0, 0.0]))
        expected = float(np.linalg.norm([100.0, 100.0]))
        assert ef.total_residual_norm == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------ #
# Conservation property: sum of all compressed + final residual = sum of all deltas
# ------------------------------------------------------------------ #

class TestConservation:
    def test_total_information_conserved_over_rounds(self):
        """
        Over K rounds, the sum of compressed outputs + final residual
        should equal the sum of raw deltas (information conservation).
        """
        rng = np.random.RandomState(42)
        ef = ErrorFeedbackState()

        total_delta = np.zeros(50)
        total_compressed = np.zeros(50)

        for _ in range(20):
            delta = rng.randn(50).astype(np.float64)
            total_delta += delta

            corrected = ef.apply(delta)

            # Simulate top-10% sparsification
            k = 5
            abs_vals = np.abs(corrected)
            topk_idx = np.argpartition(abs_vals, -k)[-k:]
            compressed = np.zeros_like(corrected)
            compressed[topk_idx] = corrected[topk_idx]

            ef.update(corrected, compressed)
            total_compressed += compressed

        # Final residual
        residual = np.zeros(50)
        for key, val in ef._residuals.items():
            residual += val

        np.testing.assert_allclose(
            total_compressed + residual,
            total_delta,
            atol=1e-10,
            err_msg="EF must conserve information: Σcompressed + residual = Σdelta",
        )


# ------------------------------------------------------------------ #
# Integration with CompressionConfig
# ------------------------------------------------------------------ #

class TestCompressionConfigIntegration:
    def test_config_has_error_feedback_field(self):
        from quinkgl.serialization.compression import CompressionConfig
        config = CompressionConfig(error_feedback=True)
        assert config.error_feedback is True

    def test_config_default_is_false(self):
        from quinkgl.serialization.compression import CompressionConfig
        config = CompressionConfig()
        assert config.error_feedback is False
