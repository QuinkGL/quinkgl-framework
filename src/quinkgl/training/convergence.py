"""
Convergence Monitoring and Early Stopping.

Detects training convergence and provides early stopping
mechanisms for the gossip learning loop.

References:
    Lian et al. 2017 — Decentralized SGD convergence bounds
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConvergenceStatus(Enum):
    IMPROVING = "improving"
    PLATEAU = "plateau"
    CONVERGED = "converged"
    DIVERGING = "diverging"


@dataclass
class ConvergenceConfig:
    window_size: int = 10
    patience: int = 20
    min_delta: float = 1e-4
    target_accuracy: Optional[float] = None
    target_loss: Optional[float] = None
    diverging_threshold: float = 0.5


@dataclass
class ConvergenceReport:
    status: ConvergenceStatus
    current_loss: float
    current_accuracy: float
    best_loss: float
    best_accuracy: float
    rounds_without_improvement: int
    loss_trend: str
    accuracy_trend: str
    reason_code: str = ""


class ConvergenceMonitor:
    """
    Monitors training metrics and detects convergence.

    Tracks loss and accuracy over a sliding window and
    determines whether training should continue or stop.

    Supports three stopping criteria:
    1. Loss plateau: Loss hasn't improved by min_delta for patience rounds
    2. Target accuracy: Accuracy has reached a specified threshold
    3. Target loss: Loss has dropped below a specified threshold
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()
        self._best_loss: float = float('inf')
        self._best_accuracy: float = 0.0
        self._rounds_without_improvement: int = 0
        self._loss_history: List[float] = []
        self._accuracy_history: List[float] = []

    def update(self, loss: float, accuracy: float, round_number: int = 0) -> ConvergenceReport:
        """
        Update monitor with latest metrics and check convergence.

        Args:
            loss: Current EMA-smoothed loss.
            accuracy: Current EMA-smoothed accuracy.
            round_number: Current training round.

        Returns:
            ConvergenceReport with current status.
        """
        self._loss_history.append(loss)
        self._accuracy_history.append(accuracy)

        if len(self._loss_history) > self.config.window_size * 2:
            self._loss_history = self._loss_history[-self.config.window_size * 2:]
            self._accuracy_history = self._accuracy_history[-self.config.window_size * 2:]

        improved = False
        if loss < self._best_loss - self.config.min_delta:
            self._best_loss = loss
            improved = True
        if accuracy > self._best_accuracy + self.config.min_delta:
            self._best_accuracy = accuracy
            improved = True

        if improved:
            self._rounds_without_improvement = 0
        else:
            self._rounds_without_improvement += 1

        status = self._determine_status(loss, accuracy)
        reason_code = self._get_reason_code(status, loss, accuracy)

        return ConvergenceReport(
            status=status,
            current_loss=loss,
            current_accuracy=accuracy,
            best_loss=self._best_loss,
            best_accuracy=self._best_accuracy,
            rounds_without_improvement=self._rounds_without_improvement,
            loss_trend=self._compute_trend(self._loss_history),
            accuracy_trend=self._compute_trend(self._accuracy_history),
            reason_code=reason_code,
        )

    def should_stop(self, report: ConvergenceReport) -> bool:
        """
        Determine whether training should stop early.

        Args:
            report: Latest convergence report.

        Returns:
            True if training should stop.
        """
        if report.status == ConvergenceStatus.CONVERGED:
            return True

        if report.rounds_without_improvement >= self.config.patience:
            return True

        return False

    def _determine_status(self, loss: float, accuracy: float) -> ConvergenceStatus:
        """Determine convergence status from current metrics.

        TASK-015: Uses a noise-robust Mann-Kendall trend test on the
        recent loss window instead of a simple first-last difference,
        making plateau detection resilient to stochastic training noise.
        """
        if self.config.target_accuracy is not None and accuracy >= self.config.target_accuracy:
            return ConvergenceStatus.CONVERGED

        if self.config.target_loss is not None and loss <= self.config.target_loss:
            return ConvergenceStatus.CONVERGED

        if len(self._loss_history) < self.config.window_size:
            return ConvergenceStatus.IMPROVING

        recent = self._loss_history[-self.config.window_size:]
        if len(recent) < 2:
            return ConvergenceStatus.IMPROVING

        # TASK-015: Noise-robust plateau detection via Mann-Kendall S statistic.
        # Counts concordant/discordant pairs; S ≈ 0 means no trend (plateau).
        is_plateau = self._mann_kendall_plateau(recent, self.config.min_delta)

        if is_plateau:
            if self._rounds_without_improvement >= self.config.patience:
                return ConvergenceStatus.CONVERGED
            return ConvergenceStatus.PLATEAU

        if loss > self._best_loss * (1 + self.config.diverging_threshold):
            return ConvergenceStatus.DIVERGING

        return ConvergenceStatus.IMPROVING

    @staticmethod
    def _mann_kendall_plateau(values: List[float], min_delta: float) -> bool:
        """TASK-015: Mann-Kendall trend test for noise-robust plateau detection.

        Computes the S statistic: for each pair (i, j) with i < j,
        sgn(x_j − x_i) is summed.  A small |S| relative to the
        maximum possible indicates no monotonic trend — i.e., a
        plateau — even when individual values are noisy.

        Returns True if the series shows no statistically significant
        monotonic trend (plateau), False otherwise.
        """
        n = len(values)
        if n < 2:
            return True

        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = values[j] - values[i]
                if diff > min_delta:
                    s += 1
                elif diff < -min_delta:
                    s -= 1
                # else: diff within noise floor, treat as tie (s += 0)

        # Normalise S to [-1, 1] range.  Maximum |S| = n*(n-1)/2.
        max_s = n * (n - 1) / 2
        if max_s == 0:
            return True

        # If |S| / max_s < 0.3, there's no strong monotonic trend → plateau.
        # The 0.3 threshold corresponds roughly to a p-value > 0.05 for n ≥ 10.
        normalized = abs(s) / max_s
        return normalized < 0.3

    def _get_reason_code(self, status: ConvergenceStatus, loss: float, accuracy: float) -> str:
        """TASK-025: Generate reason code for stopping decision."""
        if status == ConvergenceStatus.CONVERGED:
            if self.config.target_accuracy is not None and accuracy >= self.config.target_accuracy:
                return "target_accuracy_reached"
            if self.config.target_loss is not None and loss <= self.config.target_loss:
                return "target_loss_reached"
            if self._rounds_without_improvement >= self.config.patience:
                return "plateau_patience_exceeded"
        elif status == ConvergenceStatus.DIVERGING:
            return "loss_diverging"
        return "continue"

    @staticmethod
    def _compute_trend(history: List[float]) -> str:
        """Compute a simple trend description from history."""
        if len(history) < 2:
            return "stable"

        recent = history[-5:] if len(history) >= 5 else history
        if len(recent) < 2:
            return "stable"

        diff = recent[-1] - recent[0]
        if abs(diff) < 1e-6:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def best_accuracy(self) -> float:
        return self._best_accuracy

    @property
    def rounds_without_improvement(self) -> int:
        return self._rounds_without_improvement

    def state_dict(self) -> Dict:
        """Serialize state for persistence (TASK-015)."""
        return {
            "best_loss": self._best_loss,
            "best_accuracy": self._best_accuracy,
            "rounds_without_improvement": self._rounds_without_improvement,
            "loss_history": self._loss_history,
            "accuracy_history": self._accuracy_history,
            "config": {
                "window_size": self.config.window_size,
                "patience": self.config.patience,
                "min_delta": self.config.min_delta,
                "target_accuracy": self.config.target_accuracy,
                "target_loss": self.config.target_loss,
                "diverging_threshold": self.config.diverging_threshold,
            }
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from dict for persistence (TASK-015)."""
        self._best_loss = state_dict.get("best_loss", float('inf'))
        self._best_accuracy = state_dict.get("best_accuracy", 0.0)
        self._rounds_without_improvement = state_dict.get("rounds_without_improvement", 0)
        self._loss_history = state_dict.get("loss_history", [])
        self._accuracy_history = state_dict.get("accuracy_history", [])
        
        config_data = state_dict.get("config", {})
        self.config.window_size = config_data.get("window_size", 10)
        self.config.patience = config_data.get("patience", 20)
        self.config.min_delta = config_data.get("min_delta", 1e-4)
        self.config.target_accuracy = config_data.get("target_accuracy")
        self.config.target_loss = config_data.get("target_loss")
        self.config.diverging_threshold = config_data.get("diverging_threshold", 0.5)
