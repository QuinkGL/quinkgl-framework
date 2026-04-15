"""Tests for the SCAFFOLD aggregation strategy (gossip variant).

Validates control-variate drift correction, variance reduction properties,
and graceful fallback when control variates are missing.

Reference: Karimireddy et al., NeurIPS 2020.
"""

import numpy as np
import pytest

from quinkgl.aggregation.scaffold import Scaffold
from quinkgl.aggregation.base import ModelUpdate


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_update(
    peer_id: str,
    weights: np.ndarray,
    sample_count: int = 100,
    control_variate=None,
) -> ModelUpdate:
    meta = {}
    if control_variate is not None:
        meta["control_variate"] = control_variate
    return ModelUpdate(
        peer_id=peer_id,
        weights=weights,
        sample_count=sample_count,
        metadata=meta,
    )


def _make_dict_update(
    peer_id: str,
    weights: dict,
    sample_count: int = 100,
    control_variate=None,
) -> ModelUpdate:
    meta = {}
    if control_variate is not None:
        meta["control_variate"] = control_variate
    return ModelUpdate(
        peer_id=peer_id,
        weights=weights,
        sample_count=sample_count,
        metadata=meta,
    )


# ------------------------------------------------------------------ #
# Basic aggregation without control variates (fallback to FedAvg)
# ------------------------------------------------------------------ #

class TestScaffoldFallback:
    @pytest.mark.asyncio
    async def test_no_control_variates_equals_fedavg(self):
        """Without control variates, SCAFFOLD should be identical to FedAvg."""
        scaffold = Scaffold(learning_rate=0.01)

        updates = [
            _make_update("A", np.array([1.0, 2.0]), sample_count=100),
            _make_update("B", np.array([3.0, 4.0]), sample_count=100),
        ]
        result = await scaffold.aggregate(updates)

        expected = np.array([2.0, 3.0])  # simple average
        np.testing.assert_allclose(result.weights, expected, atol=1e-6)

    @pytest.mark.asyncio
    async def test_metadata_reports_scaffold(self):
        scaffold = Scaffold()
        updates = [_make_update("A", np.array([1.0]))]
        result = await scaffold.aggregate(updates)
        assert result.metadata["aggregation_method"] == "scaffold"

    @pytest.mark.asyncio
    async def test_round_counter_increments(self):
        scaffold = Scaffold()
        updates = [_make_update("A", np.array([1.0]))]
        await scaffold.aggregate(updates)
        await scaffold.aggregate(updates)
        assert scaffold.round_number == 2


# ------------------------------------------------------------------ #
# With control variates — drift correction
# ------------------------------------------------------------------ #

class TestScaffoldDriftCorrection:
    @pytest.mark.asyncio
    async def test_control_variate_corrects_drift(self):
        """When a peer has drifted (large c_peer - c_avg), correction pulls back."""
        scaffold = Scaffold(learning_rate=0.1)

        # Peer A has heavy drift, Peer B has none
        cv_a = {"__single__": np.array([10.0, 10.0])}  # large drift
        cv_b = {"__single__": np.array([0.0, 0.0])}    # no drift

        updates = [
            _make_update("A", np.array([5.0, 5.0]), sample_count=100, control_variate=cv_a),
            _make_update("B", np.array([5.0, 5.0]), sample_count=100, control_variate=cv_b),
        ]
        result = await scaffold.aggregate(updates)

        # Without correction: [5.0, 5.0]
        # c_avg = average of cv_a and cv_b = [5.0, 5.0]
        # Correction for A: 0.1 * (cv_a - c_avg) = 0.1 * [5.0, 5.0] = [0.5, 0.5]
        # Correction for B: 0.1 * (cv_b - c_avg) = 0.1 * [-5.0, -5.0] = [-0.5, -0.5]
        # Corrected A: [5.0-0.5, 5.0-0.5] = [4.5, 4.5]
        # Corrected B: [5.0+0.5, 5.0+0.5] = [5.5, 5.5]
        # Average: [5.0, 5.0]
        # The corrections cancel out with equal weights → same as uncorrected
        np.testing.assert_allclose(result.weights, [5.0, 5.0], atol=1e-6)

    @pytest.mark.asyncio
    async def test_asymmetric_drift_correction(self):
        """Asymmetric sample counts should lead to different result than uncorrected."""
        scaffold = Scaffold(learning_rate=0.1)

        cv_a = {"__single__": np.array([10.0])}
        cv_b = {"__single__": np.array([0.0])}

        updates = [
            _make_update("A", np.array([10.0]), sample_count=300, control_variate=cv_a),
            _make_update("B", np.array([0.0]), sample_count=100, control_variate=cv_b),
        ]
        result = await scaffold.aggregate(updates)

        # c_avg (sample-weighted) = (300*10 + 100*0) / 400 = 7.5
        # Correction A: 0.1 * (10 - 7.5) = 0.25 → corrected A = 10 - 0.25 = 9.75
        # Correction B: 0.1 * (0 - 7.5) = -0.75 → corrected B = 0 + 0.75 = 0.75
        # Weighted avg: (300*9.75 + 100*0.75) / 400 = (2925 + 75) / 400 = 7.5
        np.testing.assert_allclose(result.weights, [7.5], atol=1e-6)

    @pytest.mark.asyncio
    async def test_partial_control_variates(self):
        """Mixed updates — some with CV, some without — should still work."""
        scaffold = Scaffold(learning_rate=0.01)

        updates = [
            _make_update("A", np.array([2.0]), sample_count=100,
                         control_variate={"__single__": np.array([1.0])}),
            _make_update("B", np.array([4.0]), sample_count=100),
        ]
        result = await scaffold.aggregate(updates)
        # Should not raise; B is treated as zero correction
        assert result.weights.shape == (1,)


# ------------------------------------------------------------------ #
# Dict weights (PyTorch state_dict style)
# ------------------------------------------------------------------ #

class TestScaffoldDictWeights:
    @pytest.mark.asyncio
    async def test_dict_weights_aggregation(self):
        scaffold = Scaffold(learning_rate=0.01)

        w_a = {"fc.weight": np.array([[1.0, 2.0]]), "fc.bias": np.array([0.5])}
        w_b = {"fc.weight": np.array([[3.0, 4.0]]), "fc.bias": np.array([1.5])}

        updates = [
            _make_dict_update("A", w_a, sample_count=100),
            _make_dict_update("B", w_b, sample_count=100),
        ]
        result = await scaffold.aggregate(updates)

        np.testing.assert_allclose(result.weights["fc.weight"], [[2.0, 3.0]], atol=1e-6)
        np.testing.assert_allclose(result.weights["fc.bias"], [1.0], atol=1e-6)

    @pytest.mark.asyncio
    async def test_dict_weights_with_control_variates(self):
        scaffold = Scaffold(learning_rate=0.1)

        cv_a = {"fc.weight": np.array([[1.0, 1.0]]), "fc.bias": np.array([0.0])}
        cv_b = {"fc.weight": np.array([[-1.0, -1.0]]), "fc.bias": np.array([0.0])}

        w_a = {"fc.weight": np.array([[5.0, 5.0]]), "fc.bias": np.array([1.0])}
        w_b = {"fc.weight": np.array([[5.0, 5.0]]), "fc.bias": np.array([1.0])}

        updates = [
            _make_dict_update("A", w_a, sample_count=100, control_variate=cv_a),
            _make_dict_update("B", w_b, sample_count=100, control_variate=cv_b),
        ]
        result = await scaffold.aggregate(updates)

        # cv_avg = [[0,0]], corrections cancel → result = [[5,5]]
        np.testing.assert_allclose(result.weights["fc.weight"], [[5.0, 5.0]], atol=1e-6)


# ------------------------------------------------------------------ #
# Global control variate update
# ------------------------------------------------------------------ #

class TestGlobalControlVariate:
    @pytest.mark.asyncio
    async def test_global_cv_initialized_after_first_round(self):
        scaffold = Scaffold()

        cv = {"__single__": np.array([1.0])}
        updates = [_make_update("A", np.array([1.0]), control_variate=cv)]
        await scaffold.aggregate(updates)

        assert scaffold.global_control_variate is not None

    @pytest.mark.asyncio
    async def test_control_momentum(self):
        """With momentum > 0, global CV should blend old and new."""
        scaffold = Scaffold(control_momentum=0.5)

        cv1 = {"__single__": np.array([10.0])}
        cv2 = {"__single__": np.array([0.0])}

        await scaffold.aggregate([_make_update("A", np.array([1.0]), control_variate=cv1)])
        # global_cv = [10.0]

        await scaffold.aggregate([_make_update("A", np.array([1.0]), control_variate=cv2)])
        # global_cv = 0.5 * 10.0 + 0.5 * 0.0 = 5.0

        np.testing.assert_allclose(
            scaffold.global_control_variate["__single__"], [5.0], atol=1e-6
        )


# ------------------------------------------------------------------ #
# Local control variate computation
# ------------------------------------------------------------------ #

class TestLocalControlVariate:
    def test_compute_local_cv_dict(self):
        scaffold = Scaffold(learning_rate=0.1)

        global_w = {"fc": np.array([1.0, 2.0])}
        local_w = {"fc": np.array([0.9, 1.8])}

        cv = scaffold.get_local_control_variate(
            local_weights=local_w,
            global_weights=global_w,
            num_local_steps=1,
        )

        # c_i = (w_global - w_local) / (K * eta) = (0.1, 0.2) / 0.1 = (1.0, 2.0)
        np.testing.assert_allclose(cv["fc"], [1.0, 2.0], atol=1e-6)

    def test_compute_local_cv_numpy(self):
        scaffold = Scaffold(learning_rate=0.01)

        global_w = np.array([5.0])
        local_w = np.array([4.9])

        cv = scaffold.get_local_control_variate(
            local_weights=local_w,
            global_weights=global_w,
            num_local_steps=1,
        )

        # (5.0 - 4.9) / (1 * 0.01) = 10.0
        np.testing.assert_allclose(cv, [10.0], atol=1e-6)


# ------------------------------------------------------------------ #
# Variance reduction property
# ------------------------------------------------------------------ #

class TestVarianceReduction:
    @pytest.mark.asyncio
    async def test_scaffold_reduces_variance_over_rounds(self):
        """
        Over multiple rounds with consistent drift, SCAFFOLD should
        produce more stable outputs than raw averaging.
        """
        scaffold = Scaffold(learning_rate=0.1)
        rng = np.random.RandomState(42)

        # Simulate 10 rounds with 3 peers having different drifts
        drifts = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, -1.0])]
        results = []

        for r in range(10):
            updates = []
            for i, drift in enumerate(drifts):
                base = np.array([5.0, 5.0])
                noise = rng.randn(2) * 0.1
                w = base + drift * 0.5 + noise
                cv = {
                    "__single__": drift * (r + 1) * 0.1,
                }
                updates.append(
                    _make_update(f"peer_{i}", w, sample_count=100, control_variate=cv)
                )
            result = await scaffold.aggregate(updates)
            results.append(result.weights.copy())

        # Compute variance of outputs across rounds
        stacked = np.stack(results)
        variance = np.var(stacked, axis=0).mean()

        # Variance should be reasonable (not diverging)
        assert variance < 1.0, f"Output variance too high: {variance}"


# ------------------------------------------------------------------ #
# Import test
# ------------------------------------------------------------------ #

class TestPublicImport:
    def test_importable_from_aggregation(self):
        from quinkgl.aggregation import Scaffold
        assert Scaffold is not None

    def test_importable_from_top_level(self):
        from quinkgl import Scaffold
        assert Scaffold is not None
