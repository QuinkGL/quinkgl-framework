"""
ConsensusTracker

Tracks checkpoint announcements from peers and detects
when the network has reached convergence consensus.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PeerCheckpoint:
    peer_id: str
    round_number: int
    loss: float
    accuracy: float
    model_version: str = "1.0.0"


@dataclass
class ConsensusResult:
    reached: bool
    total_peers: int
    agreeing_peers: int
    agreement_ratio: float
    mean_loss: float
    mean_accuracy: float
    checkpoint_round: int


class ConsensusTracker:
    """
    Detects convergence consensus among gossip peers.

    When a sufficient fraction of peers report similar loss values
    within a checkpoint window, consensus is declared.
    """

    def __init__(
        self,
        checkpoint_interval: int = 10,
        consensus_threshold: float = 0.5,
        loss_tolerance: float = 0.05,
    ):
        """
        Args:
            checkpoint_interval: Rounds between checkpoint collections.
            consensus_threshold: Fraction of peers required for consensus (0-1).
            loss_tolerance: Maximum relative loss difference to count as agreement.
        """
        self.checkpoint_interval = checkpoint_interval
        self.consensus_threshold = consensus_threshold
        self.loss_tolerance = loss_tolerance
        self._checkpoints: Dict[int, Dict[str, PeerCheckpoint]] = {}
        self._last_checkpoint_round: int = 0

    def should_checkpoint(self, current_round: int) -> bool:
        if current_round - self._last_checkpoint_round >= self.checkpoint_interval:
            return True
        return False

    def record_checkpoint(self, checkpoint: PeerCheckpoint) -> None:
        rnd = checkpoint.round_number
        if rnd not in self._checkpoints:
            self._checkpoints[rnd] = {}
        self._checkpoints[rnd][checkpoint.peer_id] = checkpoint
        self._last_checkpoint_round = max(self._last_checkpoint_round, rnd)

    def check_consensus(self, round_number: Optional[int] = None) -> Optional[ConsensusResult]:
        """
        Check whether consensus has been reached.

        Looks at the most recent checkpoint round (or a specific round)
        and determines if enough peers agree on loss.

        Returns:
            ConsensusResult if checkpoints exist, None otherwise.
        """
        if not self._checkpoints:
            return None

        if round_number is None:
            round_number = max(self._checkpoints.keys())

        checkpoints = self._checkpoints.get(round_number, {})
        if not checkpoints:
            return None

        peer_list = list(checkpoints.values())
        total_peers = len(peer_list)

        if total_peers == 0:
            return None

        losses = [c.loss for c in peer_list]
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(c.accuracy for c in peer_list) / len(peer_list)

        if mean_loss == 0:
            agreeing = [c for c in peer_list if c.loss == 0]
        else:
            agreeing = [
                c for c in peer_list
                if abs(c.loss - mean_loss) / max(abs(mean_loss), 1e-12) <= self.loss_tolerance
            ]

        agreement_ratio = len(agreeing) / total_peers
        reached = agreement_ratio >= self.consensus_threshold

        return ConsensusResult(
            reached=reached,
            total_peers=total_peers,
            agreeing_peers=len(agreeing),
            agreement_ratio=agreement_ratio,
            mean_loss=mean_loss,
            mean_accuracy=mean_accuracy,
            checkpoint_round=round_number,
        )

    def prune_old_checkpoints(self, keep_rounds: int = 5) -> None:
        if not self._checkpoints:
            return
        sorted_rounds = sorted(self._checkpoints.keys(), reverse=True)
        for rnd in sorted_rounds[keep_rounds:]:
            del self._checkpoints[rnd]

    @property
    def last_checkpoint_round(self) -> int:
        return self._last_checkpoint_round
