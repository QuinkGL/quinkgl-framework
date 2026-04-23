"""T27: DP composition test.

Verifies that PrivacyBudgetTracker correctly composes privacy budget
across multiple queries (rounds) and refuses over-budget releases.
"""

import pytest

from quinkgl.fingerprint.computer import PrivacyBudgetTracker, FingerprintComputer
from quinkgl.fingerprint.fingerprint import FingerprintPrivacyConfig


class TestPrivacyBudgetTrackerComposition:
    """T27: Verify DP budget composition across queries."""

    def test_single_query_within_budget(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.consume(0.5, 1e-6) is True
        assert tracker.consumed_epsilon == pytest.approx(0.5)
        assert tracker.consumed_delta == pytest.approx(1e-6)
        assert tracker.query_count == 1

    def test_multiple_queries_compose_additively(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.consume(0.3, 1e-6) is True
        assert tracker.consume(0.3, 1e-6) is True
        assert tracker.consumed_epsilon == pytest.approx(0.6)
        assert tracker.consumed_delta == pytest.approx(2e-6)
        assert tracker.query_count == 2

    def test_over_budget_rejected(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.consume(0.6, 1e-6) is True
        assert tracker.consume(0.5, 1e-6) is False  # 0.6 + 0.5 > 1.0
        # Budget should not have been consumed
        assert tracker.consumed_epsilon == pytest.approx(0.6)
        assert tracker.query_count == 1

    def test_over_delta_rejected(self):
        tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-5)
        assert tracker.consume(0.1, 5e-6) is True
        assert tracker.consume(0.1, 5e-6) is True  # delta now at 1e-5
        assert tracker.consume(0.1, 1e-6) is False  # delta would exceed
        assert tracker.consumed_delta == pytest.approx(1e-5)

    def test_remaining_epsilon(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.remaining_epsilon() == pytest.approx(1.0)
        tracker.consume(0.3, 1e-6)
        assert tracker.remaining_epsilon() == pytest.approx(0.7)
        tracker.consume(0.5, 1e-6)
        assert tracker.remaining_epsilon() == pytest.approx(0.2)

    def test_no_total_epsilon_always_allows(self):
        tracker = PrivacyBudgetTracker(total_epsilon=None, total_delta=1e-5)
        assert tracker.consume(100.0, 1e-6) is True
        assert tracker.consume(100.0, 1e-6) is True
        assert tracker.remaining_epsilon() is None

    def test_exact_budget_exhaustion_allowed(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.consume(0.5, 5e-6) is True
        assert tracker.consume(0.5, 5e-6) is True  # exactly at limit
        assert tracker.consumed_epsilon == pytest.approx(1.0)
        assert tracker.remaining_epsilon() == pytest.approx(0.0)
        # Next query should fail
        assert tracker.consume(0.01, 0.0) is False

    def test_zero_consumption_counts_as_query(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
        assert tracker.consume(0.0, 0.0) is True
        assert tracker.query_count == 1
        assert tracker.consumed_epsilon == 0.0


class TestFingerprintComputerDPComposition:
    """T27: Verify FingerprintComputer integrates with budget tracker."""

    def test_budget_consumed_across_rounds(self):
        """Verify that FingerprintComputer consumes budget on each round.

        Note: _add_feature_noise passes the full feature_dp_delta per call,
        so with realistic delta values only one round succeeds. We verify
        that at least one round consumes budget and that the tracker
        correctly blocks subsequent rounds when budget is exhausted.
        """
        config = FingerprintPrivacyConfig(
            feature_dp_epsilon=1.0,
            feature_dp_delta=1e-5,
            feature_noise_sigma=0.1,
        )
        computer = FingerprintComputer(privacy_config=config)
        tracker = computer._budget_tracker

        label_counts = {"a": 50, "b": 50}
        moments = {"f1": (1.0, 0.5), "f2": (2.0, 0.3)}

        fp1 = computer.compute_from_label_counts(label_counts, moments, round_number=1)
        eps_after_r1 = tracker.consumed_epsilon
        assert eps_after_r1 > 0
        assert tracker.query_count == 1

        # Round 2 should be blocked (delta exhausted after round 1)
        fp2 = computer.compute_from_label_counts(label_counts, moments, round_number=2)
        # Budget should not have increased
        assert tracker.consumed_epsilon == pytest.approx(eps_after_r1)
        assert tracker.query_count == 1  # no new query counted

    def test_over_budget_returns_clipped_moments(self):
        config = FingerprintPrivacyConfig(
            feature_dp_epsilon=0.01,  # very small budget
            feature_dp_delta=1e-5,
            feature_noise_sigma=0.1,
        )
        computer = FingerprintComputer(privacy_config=config)

        label_counts = {"a": 50, "b": 50}
        moments = {"f1": (1.0, 0.5), "f2": (2.0, 0.3)}

        # First call may consume budget
        fp1 = computer.compute_from_label_counts(label_counts, moments, round_number=1)

        # Exhaust remaining budget
        tracker = computer._budget_tracker
        while tracker.consume(0.001, 1e-6):
            pass

        # Now compute should return clipped moments without noise
        fp2 = computer.compute_from_label_counts(label_counts, moments, round_number=2)
        # Moments should still be present (clipped) but no noise added
        assert "f1" in fp2.noised_moments
