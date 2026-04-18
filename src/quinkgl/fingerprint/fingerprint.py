"""
Privacy-Preserving Data Fingerprint.

Lightweight data distribution summaries exchanged between peers
to compute affinity scores for domain-aware collaboration.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


_BUCKET_ORDER = {"low": 0, "medium": 1, "high": 2}

# Supported differential-privacy noise mechanisms for feature/gradient moments.
NOISE_MECHANISM_GAUSSIAN = "gaussian"
NOISE_MECHANISM_LAPLACE = "laplace"
NOISE_MECHANISM_NONE = "none"
_VALID_NOISE_MECHANISMS = {
    NOISE_MECHANISM_GAUSSIAN,
    NOISE_MECHANISM_LAPLACE,
    NOISE_MECHANISM_NONE,
}


def _adjacent_bucket(a: str, b: str) -> bool:
    return abs(_BUCKET_ORDER.get(a, -1) - _BUCKET_ORDER.get(b, -1)) == 1


def calibrated_noise_scale(
    mechanism: str,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
) -> float:
    """Compute the calibrated noise scale for a DP mechanism.

    Returns the ``σ`` (Gaussian std-dev) or ``b`` (Laplace scale) required to
    satisfy ``(ε, δ)``-DP for a query with the given ``sensitivity``.

    Formulas:
      - Gaussian (analytic, simple bound): σ = Δ · √(2 · ln(1.25 / δ)) / ε
      - Laplace:                          b = Δ / ε

    Args:
        mechanism: one of ``gaussian`` | ``laplace`` | ``none``.
        sensitivity: the per-query sensitivity Δ (L2 for Gaussian,
            L1 for Laplace).  Must be > 0.
        epsilon: privacy budget ε.  Must be > 0.
        delta: Gaussian-only failure probability δ ∈ (0, 1).

    Raises:
        ValueError: for invalid parameters.
    """
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if mechanism == NOISE_MECHANISM_NONE:
        return 0.0
    if mechanism == NOISE_MECHANISM_LAPLACE:
        return sensitivity / epsilon
    if mechanism == NOISE_MECHANISM_GAUSSIAN:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    raise ValueError(
        f"mechanism must be one of {_VALID_NOISE_MECHANISMS}, got '{mechanism}'"
    )


@dataclass
class FingerprintPrivacyConfig:
    """Controls privacy level of shared fingerprints.

    Noise calibration (feature & gradient moments):
      - When the corresponding ``*_dp_epsilon`` is set, the noise scale is
        derived from ``(sensitivity, epsilon, delta)`` via
        ``calibrated_noise_scale``.  The hardcoded ``*_noise_sigma`` is then
        ignored for privacy purposes and kept only as a backwards-compatible
        fallback when ``*_dp_epsilon`` is ``None``.
      - Sensitivity MUST reflect the maximum change a single data record can
        cause in a moment.  For clipped moments this is bounded by the
        clip norm (see ``feature_clip_norm``).  Callers are responsible for
        choosing a sensitivity consistent with their clipping contract.
    """

    label_granularity: str = "bucket"
    label_buckets: List[Tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("low", 0.0, 0.2),
            ("medium", 0.2, 0.5),
            ("high", 0.5, 1.0),
        ]
    )
    feature_noise_sigma: float = 0.1
    feature_clip_norm: float = 5.0
    feature_dp_epsilon: Optional[float] = None
    feature_dp_delta: float = 1e-5
    feature_sensitivity: Optional[float] = None  # defaults to feature_clip_norm
    feature_noise_mechanism: str = NOISE_MECHANISM_GAUSSIAN
    sample_count_buckets: List[Tuple[str, int, int]] = field(
        default_factory=lambda: [
            ("0-100", 0, 100),
            ("100-1k", 100, 1000),
            ("1k-10k", 1000, 10000),
            ("10k-100k", 10000, 100000),
            ("100k+", 100000, 10**9),
        ]
    )
    gradient_enabled: bool = False
    gradient_noise_sigma: float = 0.05
    gradient_dp_epsilon: Optional[float] = None
    gradient_dp_delta: float = 1e-5
    gradient_sensitivity: Optional[float] = None
    gradient_noise_mechanism: str = NOISE_MECHANISM_GAUSSIAN

    def __post_init__(self) -> None:
        if self.feature_noise_mechanism not in _VALID_NOISE_MECHANISMS:
            raise ValueError(
                f"feature_noise_mechanism must be one of {_VALID_NOISE_MECHANISMS}, "
                f"got '{self.feature_noise_mechanism}'"
            )
        if self.gradient_noise_mechanism not in _VALID_NOISE_MECHANISMS:
            raise ValueError(
                f"gradient_noise_mechanism must be one of {_VALID_NOISE_MECHANISMS}, "
                f"got '{self.gradient_noise_mechanism}'"
            )

    def effective_feature_noise_scale(self) -> float:
        """Return the noise scale actually applied to feature moments.

        If ``feature_dp_epsilon`` is set, the scale is calibrated from
        ``(sensitivity, epsilon, delta)``; otherwise the legacy
        ``feature_noise_sigma`` is returned unchanged.
        """
        if self.feature_dp_epsilon is None:
            return float(self.feature_noise_sigma)
        sensitivity = (
            self.feature_sensitivity
            if self.feature_sensitivity is not None
            else self.feature_clip_norm
        )
        return calibrated_noise_scale(
            self.feature_noise_mechanism,
            sensitivity,
            self.feature_dp_epsilon,
            self.feature_dp_delta,
        )

    def effective_gradient_noise_scale(self) -> float:
        """Return the noise scale actually applied to gradient moments."""
        if self.gradient_dp_epsilon is None:
            return float(self.gradient_noise_sigma)
        if self.gradient_sensitivity is None:
            raise ValueError(
                "gradient_sensitivity must be set when gradient_dp_epsilon is used"
            )
        return calibrated_noise_scale(
            self.gradient_noise_mechanism,
            self.gradient_sensitivity,
            self.gradient_dp_epsilon,
            self.gradient_dp_delta,
        )


@dataclass
class AffinityWeights:
    """Weights for multi-signal affinity computation."""
    label: float = 0.4
    feature: float = 0.3
    gradient: float = 0.15
    history: float = 0.15
    external_history_score: float = 0.0


@dataclass
class DataFingerprint:
    """Privacy-preserving data distribution summary.

    All fields are pre-processed to reduce raw data leakage risk.
    Affinity is computed from these transformed values.
    """

    label_buckets: Dict[str, str]
    noised_moments: Dict[str, Tuple[float, float]]
    sample_bucket: str
    num_classes: int
    gradient_moments: Optional[Dict[str, Tuple[float, float]]] = None

    def affinity_score(
        self,
        other: "DataFingerprint",
        weights: Optional[AffinityWeights] = None,
    ) -> float:
        if weights is None:
            weights = AffinityWeights()

        label_sim = self._label_similarity(other)
        feature_sim = self._feature_similarity(other)
        gradient_sim = (
            self._gradient_similarity(other)
            if self.gradient_moments and other.gradient_moments
            else 0.0
        )

        total_w = weights.label + weights.feature + weights.gradient + weights.history
        if total_w == 0:
            return 0.5

        active_w = weights.label + weights.feature
        if gradient_sim > 0:
            active_w += weights.gradient
        if weights.external_history_score > 0:
            active_w += weights.history

        if active_w == 0:
            active_w = total_w

        raw = (
            weights.label * label_sim
            + weights.feature * feature_sim
            + weights.gradient * gradient_sim
            + weights.history * weights.external_history_score
        )

        normalized = raw / active_w * total_w if active_w > 0 else 0.0
        return max(0.0, min(1.0, normalized / total_w))

    def _label_similarity(self, other: "DataFingerprint") -> float:
        all_labels = set(self.label_buckets.keys()) | set(other.label_buckets.keys())
        if not all_labels:
            return 1.0
        score = 0.0
        for label in all_labels:
            my_b = self.label_buckets.get(label)
            other_b = other.label_buckets.get(label)
            if my_b is None or other_b is None:
                continue
            if my_b == other_b:
                score += 1.0
            elif _adjacent_bucket(my_b, other_b):
                score += 0.5
        return score / len(all_labels)

    def _feature_similarity(self, other: "DataFingerprint") -> float:
        my_vec = self._flatten_moments()
        other_vec = other._flatten_moments()
        if my_vec is None or other_vec is None or len(my_vec) == 0:
            return 0.5
        norm_a = float(np.linalg.norm(my_vec))
        norm_b = float(np.linalg.norm(other_vec))
        if norm_a == 0 or norm_b == 0:
            return 0.5
        return float(np.dot(my_vec, other_vec) / (norm_a * norm_b))

    def _gradient_similarity(self, other: "DataFingerprint") -> float:
        my_vec = self._flatten_gradient_moments()
        other_vec = other._flatten_gradient_moments()
        if my_vec is None or other_vec is None or len(my_vec) == 0:
            return 0.0
        norm_a = float(np.linalg.norm(my_vec))
        norm_b = float(np.linalg.norm(other_vec))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(my_vec, other_vec) / (norm_a * norm_b))

    def _flatten_moments(self) -> Optional[np.ndarray]:
        parts: List[float] = []
        for key in sorted(self.noised_moments.keys()):
            mean, var = self.noised_moments[key]
            parts.extend([mean, var])
        return np.array(parts) if parts else None

    def _flatten_gradient_moments(self) -> Optional[np.ndarray]:
        if not self.gradient_moments:
            return None
        parts: List[float] = []
        for key in sorted(self.gradient_moments.keys()):
            mean, var = self.gradient_moments[key]
            parts.extend([mean, var])
        return np.array(parts) if parts else None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label_buckets": dict(self.label_buckets),
            "noised_moments": {k: [v[0], v[1]] for k, v in self.noised_moments.items()},
            "sample_bucket": self.sample_bucket,
            "num_classes": self.num_classes,
        }
        if self.gradient_moments:
            d["gradient_moments"] = {
                k: [v[0], v[1]] for k, v in self.gradient_moments.items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFingerprint":
        moments = {k: (v[0], v[1]) for k, v in data.get("noised_moments", {}).items()}
        grad_moments = None
        if "gradient_moments" in data:
            grad_moments = {
                k: (v[0], v[1]) for k, v in data["gradient_moments"].items()
            }
        return cls(
            label_buckets=data.get("label_buckets", {}),
            noised_moments=moments,
            sample_bucket=data.get("sample_bucket", "unknown"),
            num_classes=data.get("num_classes", 0),
            gradient_moments=grad_moments,
        )
