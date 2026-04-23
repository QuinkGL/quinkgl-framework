"""Privacy-preserving data fingerprinting for domain-aware collaboration."""

from quinkgl.fingerprint.fingerprint import (
    DataFingerprint,
    FINGERPRINT_SCHEMA_VERSION,
    AffinityWeights,
    FingerprintPrivacyConfig,
    calibrated_noise_scale,
    NOISE_MECHANISM_GAUSSIAN,
    NOISE_MECHANISM_LAPLACE,
    NOISE_MECHANISM_NONE,
    _BUCKET_RANGES,
    _adjacent_bucket,
)
from quinkgl.fingerprint.computer import FingerprintComputer

__all__ = [
    "DataFingerprint",
    "FINGERPRINT_SCHEMA_VERSION",
    "AffinityWeights",
    "FingerprintPrivacyConfig",
    "FingerprintComputer",
    "calibrated_noise_scale",
    "NOISE_MECHANISM_GAUSSIAN",
    "NOISE_MECHANISM_LAPLACE",
    "NOISE_MECHANISM_NONE",
    "_BUCKET_RANGES",
    "_adjacent_bucket",
]
