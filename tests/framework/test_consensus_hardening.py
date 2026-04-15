"""Regression tests: ConsensusTracker hardening.

Covers:
- min_peers_for_consensus enforcement.
- max_round_ahead clamping in record_checkpoint.
- Freeze-first policy (ballot-stuffing prevention).
- Near-zero loss edge case (absolute tolerance).
- Absolute tolerance replaces relative tolerance.
"""

import pytest

from quinkgl.gossip.consensus import ConsensusTracker, PeerCheckpoint


# ── Tests ────────────────────────────────────────────────────────────

class TestMinPeersForConsensus:
    def test_two_agreeing_peers_not_enough_with_min_3(self):
        """Consensus must not be reached with 2 peers when min=3."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.05,
            min_peers_for_consensus=3,
        )
        for pid in ["a", "b"]:
            ct.record_checkpoint(PeerCheckpoint(pid, 10, 0.1, 0.9))

        result = ct.check_consensus(10)
        assert result is not None
        assert result.agreement_ratio == 1.0
        assert result.reached is False

    def test_three_agreeing_peers_reaches_consensus(self):
        """Consensus must be reached with 3 peers when min=3."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.05,
            min_peers_for_consensus=3,
        )
        for pid in ["a", "b", "c"]:
            ct.record_checkpoint(PeerCheckpoint(pid, 10, 0.1, 0.9))

        result = ct.check_consensus(10)
        assert result.reached is True

    def test_min_1_allows_single_peer_consensus(self):
        """With min=1, even a single peer can reach consensus."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.05,
            min_peers_for_consensus=1,
        )
        ct.record_checkpoint(PeerCheckpoint("solo", 10, 0.1, 0.9))

        result = ct.check_consensus(10)
        assert result.reached is True


class TestMaxRoundAhead:
    def test_implausibly_high_round_is_clamped(self):
        """Round 99999 must be clamped to last_checkpoint_round + max_round_ahead."""
        ct = ConsensusTracker(max_round_ahead=10)
        # last_checkpoint_round starts at 0, so max_allowed = 0 + 10 = 10
        ct.record_checkpoint(PeerCheckpoint("attacker", 99999, 0.1, 0.9))

        # Should be stored at round 10 (clamped), not 99999
        assert 99999 not in ct._checkpoints
        assert 10 in ct._checkpoints
        assert "attacker" in ct._checkpoints[10]

    def test_normal_round_is_not_clamped(self):
        """A round within bounds must not be clamped."""
        ct = ConsensusTracker(max_round_ahead=50)
        ct.record_checkpoint(PeerCheckpoint("p1", 5, 0.1, 0.9))
        ct.record_checkpoint(PeerCheckpoint("p2", 30, 0.2, 0.8))

        assert 5 in ct._checkpoints
        assert 30 in ct._checkpoints

    def test_round_exactly_at_max_is_accepted(self):
        """A round exactly at max_round_ahead must be accepted without clamping."""
        ct = ConsensusTracker(max_round_ahead=20)
        # last_checkpoint_round = 0, max_allowed = 20
        ct.record_checkpoint(PeerCheckpoint("p1", 20, 0.1, 0.9))
        assert 20 in ct._checkpoints


class TestFreezeFirstPolicy:
    def test_duplicate_peer_round_is_ignored(self):
        """Second submission from same (peer, round) must be ignored."""
        ct = ConsensusTracker(min_peers_for_consensus=1)
        ct.record_checkpoint(PeerCheckpoint("p1", 10, 0.1, 0.9))
        ct.record_checkpoint(PeerCheckpoint("p1", 10, 0.5, 0.5))  # attempt overwrite

        stored = ct._checkpoints[10]["p1"]
        assert stored.loss == 0.1  # first value retained
        assert stored.accuracy == 0.9

    def test_ballot_stuffing_does_not_inflate_peer_count(self):
        """100 submissions from same peer must count as 1."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.05,
            min_peers_for_consensus=1,
        )
        for i in range(100):
            ct.record_checkpoint(PeerCheckpoint("stuffer", 10, 0.1 + i * 0.001, 0.9))

        result = ct.check_consensus(10)
        assert result.total_peers == 1

    def test_different_peers_same_round_all_recorded(self):
        """Different peers submitting to same round must all be recorded."""
        ct = ConsensusTracker(min_peers_for_consensus=1)
        for i in range(5):
            ct.record_checkpoint(PeerCheckpoint(f"p{i}", 10, 0.1, 0.9))

        assert len(ct._checkpoints[10]) == 5


class TestAbsoluteTolerance:
    def test_near_zero_loss_uses_absolute_comparison(self):
        """Near-zero losses must be compared with absolute tolerance, not relative."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.01,
            min_peers_for_consensus=1,
        )
        ct.record_checkpoint(PeerCheckpoint("a", 10, 0.0, 0.99))
        ct.record_checkpoint(PeerCheckpoint("b", 10, 0.005, 0.99))
        ct.record_checkpoint(PeerCheckpoint("c", 10, 0.003, 0.99))

        result = ct.check_consensus(10)
        # mean_loss ≈ 0.00267; all within 0.01 absolute tolerance
        assert result.agreeing_peers == 3
        assert result.reached is True

    def test_zero_loss_exact_agreement(self):
        """All peers at loss=0 must agree."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=1.0,
            loss_tolerance=0.01,
            min_peers_for_consensus=3,
        )
        for pid in ["a", "b", "c"]:
            ct.record_checkpoint(PeerCheckpoint(pid, 10, 0.0, 1.0))

        result = ct.check_consensus(10)
        assert result.agreeing_peers == 3
        assert result.reached is True

    def test_peers_outside_tolerance_not_counted(self):
        """A peer whose loss deviates > tolerance must not be counted as agreeing."""
        ct = ConsensusTracker(
            checkpoint_interval=1,
            consensus_threshold=0.5,
            loss_tolerance=0.01,
            min_peers_for_consensus=1,
        )
        ct.record_checkpoint(PeerCheckpoint("close", 10, 0.10, 0.9))
        ct.record_checkpoint(PeerCheckpoint("far", 10, 0.20, 0.9))

        result = ct.check_consensus(10)
        # mean = 0.15, |0.10 - 0.15| = 0.05 > 0.01, |0.20 - 0.15| = 0.05 > 0.01
        assert result.agreeing_peers == 0

    def test_loss_tolerance_floor_at_1e6(self):
        """loss_tolerance must be floored at 1e-6."""
        ct = ConsensusTracker(loss_tolerance=0.0)
        assert ct.loss_tolerance == 1e-6
