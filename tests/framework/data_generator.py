"""
Synthetic Data Generator

Generate synthetic datasets for testing the gossip learning framework.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SyntheticDataset:
    """Container for synthetic dataset."""
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    num_classes: int
    feature_dim: int


def generate_binary_classification(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: float = 0.1,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None
) -> SyntheticDataset:
    """
    Generate binary classification data (two interleaving moons/circles style).

    Args:
        n_samples: Total number of samples
        n_features: Number of features (will use first 2 for pattern, rest are noise)
        noise: Amount of noise to add
        train_ratio: Ratio of training data
        random_seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with train/test splits
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate two classes with different distributions
    n_class = n_samples // 2

    # Class 0: Gaussian centered at (-1, -1)
    class0_features = np.random.randn(n_class, n_features) * noise
    class0_features[:, :2] += [-1, -1]  # Shift first two features

    # Class 1: Gaussian centered at (1, 1)
    class1_features = np.random.randn(n_class, n_features) * noise
    class1_features[:, :2] += [1, 1]  # Shift first two features

    # Combine
    features = np.vstack([class0_features, class1_features])
    labels = np.hstack([np.zeros(n_class), np.ones(n_class)]).astype(int)

    # Shuffle
    indices = np.random.permutation(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Split
    n_train = int(n_samples * train_ratio)
    train_features, test_features = features[:n_train], features[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    return SyntheticDataset(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=2,
        feature_dim=n_features
    )


def generate_multi_class_classification(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 4,
    noise: float = 0.5,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None
) -> SyntheticDataset:
    """
    Generate multi-class classification data.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Amount of noise
        train_ratio: Ratio of training data
        random_seed: Random seed

    Returns:
        SyntheticDataset with train/test splits
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    samples_per_class = n_samples // n_classes

    features_list = []
    labels_list = []

    # Generate each class as a Gaussian cluster
    for i in range(n_classes):
        # Create a center point for this class (in a circle)
        angle = 2 * np.pi * i / n_classes
        center = [
            2 * np.cos(angle),  # First feature
            2 * np.sin(angle)   # Second feature
        ]

        class_features = np.random.randn(samples_per_class, n_features) * noise
        class_features[:, :2] += center  # Shift by center

        features_list.append(class_features)
        labels_list.append(np.full(samples_per_class, i))

    features = np.vstack(features_list)
    labels = np.hstack(labels_list).astype(int)

    # Shuffle
    indices = np.random.permutation(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Split
    n_train = int(n_samples * train_ratio)
    train_features, test_features = features[:n_train], features[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    return SyntheticDataset(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=n_classes,
        feature_dim=n_features
    )


def generate_non_iid_data(
    n_nodes: int = 5,
    n_samples_per_node: int = 200,
    n_features: int = 10,
    n_classes: int = 4,
    skew: float = 0.7,
    random_seed: Optional[int] = None
) -> dict:
    """
    Generate non-IID data for each node (each node has different class distribution).

    Args:
        n_nodes: Number of nodes
        n_samples_per_node: Samples per node
        n_features: Number of features
        n_classes: Number of classes
        skew: How skewed the distribution is (0=uniform, 1=only one class per node)
        random_seed: Random seed

    Returns:
        Dict mapping node_id to (features, labels) tuple
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    node_data = {}

    for node_id in range(n_nodes):
        # Each node has a preference for certain classes
        # Create a probability distribution for this node
        base_probs = np.ones(n_classes)
        preferred_class = node_id % n_classes
        base_probs[preferred_class] += skew * n_classes

        # Normalize
        probs = base_probs / base_probs.sum()

        # Sample labels according to this distribution
        labels = np.random.choice(
            n_classes,
            size=n_samples_per_node,
            p=probs
        ).astype(int)

        # Generate features for each label
        features_list = []
        for c in range(n_classes):
            n_c = (labels == c).sum()
            if n_c == 0:
                continue

            # Class center
            angle = 2 * np.pi * c / n_classes
            center = [
                2 * np.cos(angle),
                2 * np.sin(angle)
            ]

            class_features = np.random.randn(n_c, n_features) * 0.5
            class_features[:, :2] += center

            features_list.append(class_features)

        # Shuffle features and keep labels
        features = np.vstack(features_list)

        # Shuffle together
        indices = np.random.permutation(n_samples_per_node)
        features = features[indices]
        labels = labels[indices]

        node_data[node_id] = (features, labels)

    return node_data


def generate_image_like_data(
    n_samples: int = 1000,
    img_size: int = 8,
    n_classes: int = 10,
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None
) -> SyntheticDataset:
    """
    Generate image-like data (flattened 2D images).

    Simulates small grayscale images like MNIST.

    Args:
        n_samples: Total number of samples
        img_size: Size of image (img_size x img_size)
        n_classes: Number of classes
        train_ratio: Ratio of training data
        random_seed: Random seed

    Returns:
        SyntheticDataset with image-like data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_features = img_size * img_size
    samples_per_class = n_samples // n_classes

    features_list = []
    labels_list = []

    # Generate each class with a different pattern
    for i in range(n_classes):
        # Create a base pattern (different position for each class)
        class_features = np.random.randn(samples_per_class, n_features) * 0.3

        # Add a "signal" in a specific region
        signal_row = (i // 3) % img_size
        signal_col = i % img_size
        signal_idx = signal_row * img_size + signal_col

        class_features[:, signal_idx] += np.random.randn(samples_per_class) * 0.5 + 2

        features_list.append(class_features)
        labels_list.append(np.full(samples_per_class, i))

    features = np.vstack(features_list)
    labels = np.hstack(labels_list).astype(int)

    # Shuffle
    indices = np.random.permutation(n_samples)
    features = features[indices]
    labels = labels[indices]

    # Split
    n_train = int(n_samples * train_ratio)
    train_features, test_features = features[:n_train], features[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    return SyntheticDataset(
        train_features=train_features.reshape(-1, 1, img_size, img_size),  # N, C, H, W
        train_labels=train_labels,
        test_features=test_features.reshape(-1, 1, img_size, img_size),
        test_labels=test_labels,
        num_classes=n_classes,
        feature_dim=n_features
    )


if __name__ == "__main__":
    print("Testing data generators...")

    # Test binary classification
    print("\n1. Binary Classification:")
    data = generate_binary_classification(n_samples=100, random_seed=42)
    print(f"  Train: {data.train_features.shape}, {data.train_labels.shape}")
    print(f"  Test: {data.test_features.shape}, {data.test_labels.shape}")
    print(f"  Class distribution: {np.bincount(data.train_labels)}")

    # Test multi-class
    print("\n2. Multi-Class Classification:")
    data = generate_multi_class_classification(n_samples=200, n_classes=4, random_seed=42)
    print(f"  Train: {data.train_features.shape}, {data.train_labels.shape}")
    print(f"  Class distribution: {np.bincount(data.train_labels)}")

    # Test non-IID
    print("\n3. Non-IID Data (5 nodes):")
    node_data = generate_non_iid_data(n_nodes=5, n_samples_per_node=100, random_seed=42)
    for node_id, (features, labels) in node_data.items():
        print(f"  Node {node_id}: {np.bincount(labels)}")

    # Test image-like data
    print("\n4. Image-Like Data:")
    data = generate_image_like_data(n_samples=100, img_size=8, random_seed=42)
    print(f"  Train: {data.train_features.shape}")
    print(f"  Test: {data.test_features.shape}")
    print(f"  Class distribution: {np.bincount(data.train_labels)}")
