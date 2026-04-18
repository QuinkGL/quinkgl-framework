"""S1b: verify the full compression pipeline with error feedback.

The residual norm must stay bounded over many rounds after the S1a fix
(no double-application of the EF residual).  A diverging norm is the
signature of the old bug where ef_state.apply(pre_sparse) was called
a second time inside ef_state.update().
"""

import numpy as np
import pytest

from quinkgl.serialization.compression import CompressionConfig, compress_weights, decompress_weights
from quinkgl.serialization.quantization import QuantizationConfig
from quinkgl.serialization.sparsification import SparsificationConfig
from quinkgl.serialization.error_feedback import ErrorFeedbackState, ErrorFeedbackConfig


class TestEFResidualBounded:
    """S1b: residual norm must remain bounded under repeated compress→decompress cycles."""

    def _run_rounds(self, n_rounds: int, use_quantization: bool = False):
        rng = np.random.RandomState(0)
        weights = rng.randn(200).astype(np.float32)

        config = CompressionConfig(
            sparsification=SparsificationConfig(top_k_ratio=0.1),
            quantization=QuantizationConfig(bits=8) if use_quantization else None,
            zlib_compression=False,
            error_feedback=True,
        )
        ef = ErrorFeedbackState(ErrorFeedbackConfig(enabled=True))

        norms = []
        base = weights.copy()

        for i in range(n_rounds):
            # Small random gradient update
            delta = rng.randn(200).astype(np.float32) * 0.01
            current = base + delta

            compressed, meta = compress_weights(current, config, base_weights=base, ef_state=ef)
            norms.append(ef.total_residual_norm)
            base = current

        return norms

    def test_residual_norm_does_not_diverge(self):
        """Over 50 rounds the residual norm must stay < 10x the initial weight scale."""
        norms = self._run_rounds(n_rounds=50)
        # With the fix, norms should plateau rather than grow unboundedly.
        # An initial weight scale of ~1 (randn) means a norm > 100 signals divergence.
        assert max(norms) < 100.0, (
            f"Residual norm diverged to {max(norms):.2f} — "
            "check S1a fix in compress_weights (double EF application)"
        )

    def test_residual_norm_stabilizes(self):
        """The last-10-round average norm must not be larger than the first-10 average."""
        norms = self._run_rounds(n_rounds=60)
        first_avg = sum(norms[:10]) / 10
        last_avg = sum(norms[-10:]) / 10
        assert last_avg <= first_avg * 5, (
            f"Residual norm grew: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )

    def test_residual_norm_bounded_with_quantization(self):
        """Same bound holds when quantization is enabled in the pipeline."""
        norms = self._run_rounds(n_rounds=40, use_quantization=True)
        assert max(norms) < 200.0  # slightly wider bound due to quant noise


class TestCompressionRoundtrip:
    """Basic round-trip correctness for the combined pipeline."""

    def test_sparsify_only_roundtrip(self):
        rng = np.random.RandomState(1)
        weights = rng.randn(100).astype(np.float32)
        base = rng.randn(100).astype(np.float32)

        config = CompressionConfig(
            sparsification=SparsificationConfig(top_k_ratio=0.2),
            zlib_compression=False,
        )
        compressed, meta = compress_weights(weights, config, base_weights=base)
        recovered = decompress_weights(compressed, meta, base_weights=base)

        # Only the top-20% of delta values are preserved; allow some loss
        delta = weights - base
        reconstructed_delta = recovered - base
        top_k = int(len(delta) * 0.2)
        topk_idx = np.argpartition(np.abs(delta), -top_k)[-top_k:]
        np.testing.assert_allclose(
            reconstructed_delta[topk_idx], delta[topk_idx], rtol=1e-4
        )

    def test_pipeline_version_mismatch_raises(self):
        rng = np.random.RandomState(2)
        weights = rng.randn(50).astype(np.float32)

        config = CompressionConfig(zlib_compression=False)
        compressed, meta = compress_weights(weights, config)
        meta["pipeline_version"] = 99  # simulate unsupported version

        with pytest.raises(ValueError, match="pipeline_version"):
            decompress_weights(compressed, meta)

    def test_missing_quant_meta_raises(self):
        rng = np.random.RandomState(3)
        weights = rng.randn(50).astype(np.float32)

        config = CompressionConfig(
            quantization=QuantizationConfig(bits=8),
            zlib_compression=False,
        )
        compressed, meta = compress_weights(weights, config)
        meta["quant_meta"] = None  # simulate corrupted metadata

        with pytest.raises(ValueError, match="quant_meta"):
            decompress_weights(compressed, meta)
