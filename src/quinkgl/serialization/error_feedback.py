"""
Error Feedback for Biased Compressors.

Implements the error feedback (EF) mechanism described by:
  - Alistarh et al. 2018 — "The Convergence of Sparsified Gradient Methods"
  - Richtárik et al. 2021 — "EF21: A New, Simpler, Theoretically Better"

When using biased compressors such as Top-k sparsification or scalar
quantization, the discarded information (the *residual*) is accumulated
and injected into the next compression round.  This turns a biased
compressor into an *effectively unbiased* one, restoring convergence
guarantees.

Usage:
    from quinkgl.serialization.error_feedback import ErrorFeedbackState

    ef = ErrorFeedbackState()
    compressed, meta = ef.compress_with_feedback(delta, sparsify_fn, config)
    # ... send `compressed` over the network ...
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ErrorFeedbackConfig:
    """Configuration for the error feedback mechanism.

    Parameters
    ----------
    enabled : bool
        Master switch.  When *False* the module is a transparent pass-through.
    momentum : float
        Momentum coefficient for the residual buffer (EF21-style).
        0.0 = classic EF (full residual), 1.0 = ignore residual (no feedback).
        Typical value: 0.0 for standard EF, 0.9 for EF21 with momentum.
    max_residual_norm : float or None
        Optional hard cap on the ℓ₂-norm of the residual buffer to prevent
        unbounded growth in adversarial settings.  ``None`` = no cap.
    """

    enabled: bool = True
    momentum: float = 0.0
    max_residual_norm: Optional[float] = None


class ErrorFeedbackState:
    """Per-node residual buffer that accumulates compression error.

    Thread-safety: this class is **not** thread-safe.  In QuinkGL each
    ``GossipNode`` runs on a single asyncio event loop, so this is fine.

    The canonical call sequence per gossip round is::

        corrected = ef.apply(raw_delta)    # inject residual
        compressed = compress(corrected)   # biased compressor
        ef.update(corrected, compressed)   # store new residual
    """

    def __init__(self, config: Optional[ErrorFeedbackConfig] = None) -> None:
        self.config = config or ErrorFeedbackConfig()
        self._residuals: Dict[str, np.ndarray] = {}
        self._round: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, delta: Any) -> Any:
        """Add the accumulated residual to the raw delta **before** compression.

        Parameters
        ----------
        delta : numpy array or dict of numpy arrays
            The raw model-weight delta (``w_current − w_previous``).

        Returns
        -------
        corrected : same type as *delta*
            ``delta + residual`` (element-wise).  If no residual exists yet
            (first round) or if EF is disabled, returns *delta* unchanged.
        """
        if not self.config.enabled:
            return delta

        if isinstance(delta, np.ndarray):
            return self._apply_array("__single__", delta)
        elif isinstance(delta, dict):
            result = {}
            for key, value in delta.items():
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                    result[key] = self._apply_array(key, value)
                else:
                    result[key] = value
            return result
        return delta

    def update(self, corrected: Any, compressed: Any) -> None:
        """Compute and store the new residual: ``residual = corrected − compressed``.

        Parameters
        ----------
        corrected : numpy array or dict
            The corrected delta that was fed into the compressor (output of
            :meth:`apply`).
        compressed : numpy array or dict
            The output of the compressor (what was actually sent over the wire).
        """
        if not self.config.enabled:
            return

        self._round += 1

        if isinstance(corrected, np.ndarray):
            self._update_array("__single__", corrected, compressed)
        elif isinstance(corrected, dict):
            for key in corrected:
                c_val = corrected[key]
                s_val = compressed.get(key, np.zeros_like(c_val)) if isinstance(compressed, dict) else compressed
                if isinstance(c_val, np.ndarray) and np.issubdtype(c_val.dtype, np.floating):
                    self._update_array(key, c_val, s_val)

    def reset(self) -> None:
        """Clear the residual buffer (e.g. after a topology change)."""
        self._residuals.clear()
        self._round = 0

    @property
    def total_residual_norm(self) -> float:
        """ℓ₂-norm of the concatenated residual buffer (global, cross-tensor)."""
        if not self._residuals:
            return 0.0
        norms_sq = [np.sum(r ** 2) for r in self._residuals.values()]
        return float(np.sqrt(sum(norms_sq)))

    @property
    def per_tensor_residual_norms(self) -> dict:
        """Per-tensor ℓ₂-norms of the residual buffer.

        S12: The global ``total_residual_norm`` and the per-tensor cap in
        ``_update_array`` operate at different granularities.  Use this
        property when you need per-tensor monitoring consistent with the cap.
        """
        return {k: float(np.linalg.norm(r)) for k, r in self._residuals.items()}

    @property
    def round_number(self) -> int:
        return self._round

    @property
    def buffer_keys(self) -> list:
        """Names of weight tensors that have a non-zero residual."""
        return list(self._residuals.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_array(self, key: str, delta: np.ndarray) -> np.ndarray:
        residual = self._residuals.get(key)
        if residual is None:
            return delta
        # S13: Keep result in float64 through sparsification; cast only at final serialization.
        return delta.astype(np.float64) + residual.astype(np.float64)

    def _update_array(
        self, key: str, corrected: np.ndarray, compressed: np.ndarray
    ) -> None:
        new_residual = corrected.astype(np.float64) - compressed.astype(np.float64)

        # EF21-style momentum blending
        beta = self.config.momentum
        if beta > 0 and key in self._residuals:
            old = self._residuals[key].astype(np.float64)
            new_residual = beta * old + (1 - beta) * new_residual

        # Optional norm cap
        if self.config.max_residual_norm is not None:
            norm = float(np.linalg.norm(new_residual))
            if norm > self.config.max_residual_norm:
                new_residual = new_residual * (self.config.max_residual_norm / norm)

        self._residuals[key] = new_residual
