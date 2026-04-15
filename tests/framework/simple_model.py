"""
Simple PyTorch Model for Testing

Basic MLP for binary classification on synthetic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for testing.

    Input: n_features
    Output: Classification (logits)
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)  # Multi-class classification
            # No softmax - CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.net(x)

    def get_input_size(self) -> int:
        """Return expected input size."""
        return self.net[0].in_features


class SimpleCNN(nn.Module):
    """
    Simple CNN for image-like data (e.g., MNIST-like 8x8 grayscale).
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_simple_mlp(input_size: int = 10) -> nn.Module:
    """Factory function to create a simple MLP."""
    return SimpleMLP(input_size=input_size)


def create_simple_cnn(input_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """Factory function to create a simple CNN."""
    return SimpleCNN(input_channels=input_channels, num_classes=num_classes)


if __name__ == "__main__":
    # Test the models
    print("Testing SimpleMLP...")
    mlp = SimpleMLP(input_size=10)
    x = torch.randn(4, 10)  # Batch of 4, 10 features
    y = mlp(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    print("\nTesting SimpleCNN...")
    cnn = SimpleCNN(input_channels=1, num_classes=10)
    x = torch.randn(4, 1, 8, 8)  # Batch of 4, 1 channel, 8x8 image
    y = cnn(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
