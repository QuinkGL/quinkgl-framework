# Data Fingerprints

**Data fingerprints** are privacy-preserving summaries of a peer's local data distribution. They enable data-aware peer selection without exposing raw data.

## Why Fingerprints?

In federated learning, peers often have **non-IID** (non-independent, identically distributed) data. Training with dissimilar peers can hurt convergence.

Data fingerprints solve this by:
- Summarizing data distribution in a privacy-safe way
- Enabling affinity-based peer matching
- Preserving confidentiality (no raw data leaves the peer)

## Fingerprint Components

| Component | Description | Privacy Transform |
|-----------|-------------|-------------------|
| Label distribution | Class frequency histogram | Quantized into buckets |
| Feature moments | Mean, variance per feature | Gaussian noise added |
| Sample count | Number of training samples | Bucketed into ranges |
| Gradient moments | Training gradient statistics | Optional, disabled by default |

## Affinity Scoring

`AffinityTopology` uses fingerprints to compute **affinity scores** between peers:

```
affinity(peer_a, peer_b) = weighted_sum(
    label_similarity,
    feature_similarity,
    gradient_similarity,
    collaboration_history
)
```

Peers with high affinity are more likely to be selected for model exchange.

## Privacy Guarantees

- **No raw data** is ever transmitted
- **Differential privacy** noise can be configured
- **Per-round nonces** prevent cross-round linkability
- **Schema validation** prevents malformed fingerprint injection

## Configuration

```python
from quinkgl import FingerprintPrivacyConfig

config = FingerprintPrivacyConfig(
    feature_noise_sigma=0.1,
    feature_dp_epsilon=1.0,
    gradient_fingerprint=False,  # disabled by default (higher risk)
)
```

## Cold Start

New peers start with a "blind" phase where they explore randomly before fingerprint-based selection activates. This ensures network connectivity even without prior fingerprint data.
