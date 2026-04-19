"""
FingerprintComputer — computes DataFingerprint from local data and model.

Applies privacy transforms (quantization, noise, bucketing)
before the fingerprint leaves the node.
"""

import hashlib
import hmac

import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    FingerprintPrivacyConfig,
    NOISE_MECHANISM_GAUSSIAN,
    NOISE_MECHANISM_LAPLACE,
    NOISE_MECHANISM_NONE,
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
        round_number: Optional[int] = None,
    ) -> DataFingerprint:
        total_samples = sum(label_counts.values())
        total = total_samples or 1
        raw_proportions = {k: v / total for k, v in label_counts.items()}
        raw_buckets = self._quantize_labels(raw_proportions)
        round_nonce = self._derive_round_nonce(round_number)

        num_classes = len(label_counts)
        class_count_bucket = self._bucket_class_count(num_classes)

        # Audit F4: suppress label mapping when the class count is below the
        # reveal threshold so a single-class peer is indistinguishable from
        # a peer with no data.
        if num_classes < self.privacy.min_classes_to_reveal:
            label_buckets: Dict[str, str] = {}
            revealed_num_classes = 0
        else:
            label_buckets = self._maybe_hash_label_keys(raw_buckets, round_nonce)
            # When hashing is active, the raw integer class count is also
            # suppressed; downstream consumers should use class_count_bucket.
            revealed_num_classes = (
                0 if self.privacy.hash_label_keys else num_classes
            )

        noised_moments: Dict[str, Tuple[float, float]] = {}
        if feature_moments:
            noised_moments = self._add_feature_noise(feature_moments)

        grad_moments: Optional[Dict[str, Tuple[float, float]]] = None
        if gradient_moments and self.privacy.gradient_enabled:
            grad_moments = self._add_gradient_noise(gradient_moments)

        sample_bucket = self._bucket_sample_count(total_samples)

        return DataFingerprint(
            label_buckets=label_buckets,
            noised_moments=noised_moments,
            sample_bucket=sample_bucket,
            num_classes=revealed_num_classes,
            gradient_moments=grad_moments,
            class_count_bucket=class_count_bucket,
            round_nonce=round_nonce,
        )

    def _derive_round_nonce(self, round_number: Optional[int]) -> Optional[str]:
        """Derive a stable per-round nonce string.

        The nonce is not intended to be secret; it exists to bind a
        fingerprint instance to a given round so cross-round correlation is
        harder and hash-based label keys can rotate.  Returning ``None`` for
        legacy/no-round calls preserves backwards compatibility.
        """
        if round_number is None:
            return None
        if not isinstance(round_number, int) or round_number < 0:
            raise ValueError(f"round_number must be a non-negative int, got {round_number}")
        material = f"quinkgl-fingerprint-round:{round_number}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    def _bucket_class_count(self, count: int) -> str:
        for bucket_name, low, high in self.privacy.class_count_buckets:
            if low <= count < high:
                return bucket_name
        # Count exceeds every configured bucket → fall back to the last one.
        return self.privacy.class_count_buckets[-1][0]

    def _hash_label_key(self, label: str, round_nonce: Optional[str] = None) -> str:
        """Stable obfuscation of a raw label name.

        Uses HMAC-SHA256 when ``label_key_secret`` is configured, otherwise
        plain SHA-256.  The result is truncated to ``label_key_hash_length``
        hex characters.  Peers that share the same secret (or no secret)
        produce identical hashes for identical labels.  When ``round_nonce``
        is provided, it is mixed into the digest input so label keys rotate
        across rounds and become harder to link longitudinally.
        """
        nonce_prefix = f"{round_nonce}:" if round_nonce is not None else ""
        raw = f"{nonce_prefix}{label}".encode("utf-8")
        if self.privacy.label_key_secret is not None:
            digest = hmac.new(
                self.privacy.label_key_secret, raw, hashlib.sha256
            ).hexdigest()
        else:
            digest = hashlib.sha256(raw).hexdigest()
        length = max(1, int(self.privacy.label_key_hash_length))
        return digest[:length]

    def _maybe_hash_label_keys(
        self,
        buckets: Dict[str, str],
        round_nonce: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self.privacy.hash_label_keys:
            return dict(buckets)
        hashed: Dict[str, str] = {}
        for label, bucket in buckets.items():
            hashed[self._hash_label_key(label, round_nonce)] = bucket
        return hashed

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

    def _sample_noise(self, mechanism: str, scale: float) -> float:
        """Sample a fresh noise value per call.

        Privacy invariant: noise MUST be sampled per-query, never cached,
        to prevent averaging-attack de-noising across repeated fingerprints.
        """
        if scale <= 0.0 or mechanism == NOISE_MECHANISM_NONE:
            return 0.0
        if mechanism == NOISE_MECHANISM_LAPLACE:
            return float(np.random.laplace(0.0, scale))
        # Gaussian (default)
        return float(np.random.normal(0.0, scale))

    def _add_feature_noise(
        self, moments: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        scale = self.privacy.effective_feature_noise_scale()
        mech = self.privacy.feature_noise_mechanism
        clip = self.privacy.feature_clip_norm
        noised: Dict[str, Tuple[float, float]] = {}
        for key, (mean, var) in moments.items():
            m = float(np.clip(mean, -clip, clip)) + self._sample_noise(mech, scale)
            v = max(
                0.0,
                float(np.clip(var, -clip, clip)) + self._sample_noise(mech, scale),
            )
            noised[key] = (m, v)
        return noised

    def _add_gradient_noise(
        self, moments: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        scale = self.privacy.effective_gradient_noise_scale()
        mech = self.privacy.gradient_noise_mechanism
        noised: Dict[str, Tuple[float, float]] = {}
        for key, (mean, var) in moments.items():
            m = mean + self._sample_noise(mech, scale)
            v = max(0.0, var + self._sample_noise(mech, scale))
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
