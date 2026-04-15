"""
FingerprintComputer — computes DataFingerprint from local data and model.

Applies privacy transforms (quantization, noise, bucketing)
before the fingerprint leaves the node.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    FingerprintPrivacyConfig,
)


class FingerprintComputer:
    """Computes DataFingerprint from local data and model."""

    def __init__(self, privacy_config: Optional[FingerprintPrivacyConfig] = None):
        self.privacy = privacy_config or FingerprintPrivacyConfig()

    def compute_from_label_counts(
        self,
        label_counts: Dict[str, int],
        feature_moments: Optional[Dict[str, Tuple[float, float]]] = None,
        gradient_moments: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> DataFingerprint:
        total = sum(label_counts.values()) or 1
        raw_proportions = {k: v / total for k, v in label_counts.items()}
        label_buckets = self._quantize_labels(raw_proportions)

        noised_moments: Dict[str, Tuple[float, float]] = {}
        if feature_moments:
            noised_moments = self._add_feature_noise(feature_moments)

        grad_moments: Optional[Dict[str, Tuple[float, float]]] = None
        if gradient_moments and self.privacy.gradient_enabled:
            grad_moments = self._add_gradient_noise(gradient_moments)

        sample_bucket = self._bucket_sample_count(total)

        return DataFingerprint(
            label_buckets=label_buckets,
            noised_moments=noised_moments,
            sample_bucket=sample_bucket,
            num_classes=len(label_counts),
            gradient_moments=grad_moments,
        )

    def _quantize_labels(self, proportions: Dict[str, float]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for label, prop in proportions.items():
            bucketed = False
            for bucket_name, low, high in self.privacy.label_buckets:
                if low <= prop < high:
                    result[label] = bucket_name
                    bucketed = True
                    break
            if not bucketed:
                result[label] = "high"
        return result

    def _add_feature_noise(
        self, moments: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        sigma = self.privacy.feature_noise_sigma
        clip = self.privacy.feature_clip_norm
        noised: Dict[str, Tuple[float, float]] = {}
        for key, (mean, var) in moments.items():
            m = float(np.clip(mean, -clip, clip)) + float(np.random.normal(0, sigma))
            v = max(0.0, float(np.clip(var, -clip, clip)) + float(np.random.normal(0, sigma)))
            noised[key] = (m, v)
        return noised

    def _add_gradient_noise(
        self, moments: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        sigma = self.privacy.gradient_noise_sigma
        noised: Dict[str, Tuple[float, float]] = {}
        for key, (mean, var) in moments.items():
            m = mean + float(np.random.normal(0, sigma))
            v = max(0.0, var + float(np.random.normal(0, sigma)))
            noised[key] = (float(m), float(v))
        return noised

    def _bucket_sample_count(self, count: int) -> str:
        for bucket_name, low, high in self.privacy.sample_count_buckets:
            if low <= count < high:
                return bucket_name
        return "100k+"

    @staticmethod
    def extract_bn_moments(weights: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        moments: Dict[str, Tuple[float, float]] = {}
        for key, val in weights.items():
            if "running_mean" in key and isinstance(val, np.ndarray):
                base_key = key.replace("running_mean", "").rstrip(".")
                mean_val = float(np.mean(val))
                var_key = key.replace("running_mean", "running_var")
                if var_key in weights and isinstance(weights[var_key], np.ndarray):
                    var_val = float(np.mean(weights[var_key]))
                    moments[base_key] = (mean_val, var_val)
        return moments
