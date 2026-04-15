"""
PyTorch Model Wrapper

Wrapper for PyTorch nn.Module models.
"""

from typing import Any, Dict, Optional
import numpy as np

from quinkgl.models.base import (
    ModelWrapper,
    TrainingConfig,
    TrainingResult,
    ModelSplit,
    PersonalizedModelWrapper,
)


class PyTorchModel(ModelWrapper):
    """
    Wrapper for PyTorch models.

    Provides get_weights/set_weights interface for PyTorch nn.Module models.
    """

    def __init__(self, model: Any, device: str = "cpu", model_version: str = "1.0.0"):
        """
        Initialize PyTorch model wrapper.

        Args:
            model: PyTorch nn.Module instance
            device: Device to train on ("cpu" or "cuda")
            model_version: Semantic version string for model architecture
        """
        super().__init__(model, model_version=model_version)
        self.device = device
        if device == "cuda":
            self.model = self.model.to(device)

        # Store model to CPU for weight extraction
        self._original_device = device

    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get model weights as numpy arrays.

        Returns:
            Dict mapping parameter names to numpy arrays
        """
        self.model.eval()

        state_dict = self.model.state_dict()
        weights = {}

        for name, param in state_dict.items():
            # Convert to numpy array
            weights[name] = param.cpu().detach().numpy().copy()

        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weights from numpy arrays.

        Args:
            weights: Dict mapping parameter names to numpy arrays

        Raises:
            ValueError: If weights are invalid (NaN, Inf, wrong shapes, etc.)
        """
        import torch
        import logging

        logger = logging.getLogger(__name__)

        # Get expected shapes from current model
        expected_state_dict = self.model.state_dict()

        # Validate weights before loading
        state_dict = {}
        for name, array in weights.items():
            # Check if parameter exists in model
            if name not in expected_state_dict:
                logger.warning(f"Unknown parameter '{name}', skipping")
                continue

            # Convert to numpy if not already
            if not isinstance(array, np.ndarray):
                try:
                    array = np.array(array)
                except Exception as e:
                    raise ValueError(f"Cannot convert weights[{name}] to numpy array: {e}")

            # Check for NaN values
            if np.isnan(array).any():
                raise ValueError(f"Weights[{name}] contains NaN values")

            # Check for Inf values
            if np.isinf(array).any():
                raise ValueError(f"Weights[{name}] contains Inf values")

            # Check shape compatibility
            expected_shape = expected_state_dict[name].shape
            if array.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {expected_shape}, got {array.shape}"
                )

            # Convert numpy array to torch tensor with same dtype as original
            original_dtype = expected_state_dict[name].dtype
            tensor = torch.from_numpy(array).to(dtype=original_dtype)

            state_dict[name] = tensor

        # Ensure all model parameters are covered (use existing for missing ones)
        for name in expected_state_dict:
            if name not in state_dict:
                logger.warning(f"Missing weights for '{name}', using existing values")
                state_dict[name] = expected_state_dict[name]

        # Load validated state dict
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise ValueError(f"Failed to load state dict: {e}")

    async def train(
        self,
        data: Any,
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Train the model on local data.

        Args:
            data: Tuple of (features, labels) or DataLoader
            config: Training configuration

        Returns:
            TrainingResult with metrics
        """
        import torch
        import logging
        from torch.utils.data import DataLoader, TensorDataset

        logger = logging.getLogger(__name__)
        config = config or TrainingConfig()

        # Convert data to DataLoader if needed
        if isinstance(data, tuple) and len(data) == 2:
            features, labels = data
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        else:
            dataloader = data  # Assume it's already a DataLoader

        # Setup training
        self.model.train()

        # Use custom loss function if provided, otherwise default to CrossEntropyLoss
        if config.loss_fn is not None:
            criterion = config.loss_fn
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # FedProx: wrap criterion with proximal term if configured
        if config.proximal_coefficient is not None and config.global_weights is not None:
            mu = config.proximal_coefficient
            global_w = config.global_weights
            base_criterion = criterion

            def fedprox_criterion(output, target):
                base_loss = base_criterion(output, target)
                proximal_term = torch.tensor(0.0, device=self.device)
                for name, param in self.model.named_parameters():
                    if name in global_w:
                        g = torch.from_numpy(
                            global_w[name] if isinstance(global_w[name], np.ndarray)
                            else np.array(global_w[name])
                        ).to(dtype=param.dtype, device=self.device)
                        proximal_term += ((param - g) ** 2).sum()
                return base_loss + (mu / 2) * proximal_term

            criterion = fedprox_criterion

        # Use custom optimizer if provided
        if config.optimizer is not None:
            # If optimizer is a class, instantiate it
            if isinstance(config.optimizer, type):
                optimizer_kwargs = config.optimizer_kwargs or {}
                optimizer = config.optimizer(self.model.parameters(), lr=config.learning_rate, **optimizer_kwargs)
            else:
                # If optimizer is already an instance
                optimizer = config.optimizer
        else:
            # Default to Adam optimizer
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate
            )

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping if configured
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.grad_clip_norm
                    )

                optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += batch_labels.size(0)
                epoch_correct += predicted.eq(batch_labels).sum().item()

            avg_loss = epoch_loss / len(dataloader)
            avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

            if config.verbose:
                print(f"Epoch {epoch + 1}/{config.epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

            total_loss = avg_loss
            correct = epoch_correct
            total = epoch_total

            if config.on_epoch_end:
                config.on_epoch_end(epoch, avg_loss, avg_acc)

        self.increment_round()

        return TrainingResult(
            epochs_completed=config.epochs,
            final_loss=total_loss,
            final_accuracy=correct / total if total > 0 else None,
            samples_trained=total
        )

    def evaluate(self, data: Any, loss_fn: Any = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            data: Tuple of (features, labels) or DataLoader
            loss_fn: Optional custom loss function

        Returns:
            Dict with loss and accuracy
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.model.eval()

        # Convert data to DataLoader if needed
        if isinstance(data, tuple) and len(data) == 2:
            features, labels = data
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        else:
            dataloader = data

        # Use custom loss function if provided, otherwise default to CrossEntropyLoss
        if loss_fn is not None:
            criterion = loss_fn
        else:
            criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def get_data_schema_hash(self) -> str:
        """
        Get a hash representing the model's input schema.

        For PyTorch, we use the full model architecture by including
        all parameter shapes and names for better compatibility checking.
        """
        import hashlib
        import torch

        # Collect all parameter information for full architecture fingerprint
        param_info = []
        for name, param in sorted(self.model.named_parameters()):
            if 'weight' in name or 'bias' in name:
                param_info.append(f"{name}:{list(param.shape)}")

        if param_info:
            schema_info = f"pytorch_{'_'.join(param_info)}"
        else:
            schema_info = "pytorch_unknown"

        return hashlib.sha256(schema_info.encode()).hexdigest()[:16]


class PyTorchPersonalizedModel(PersonalizedModelWrapper, PyTorchModel):
    """PyTorch model with FedRep backbone/head split and FedBN isolation.

    Backbone weights are shared via gossip (FedRep).
    Head weights are strictly local (personalized classifier).
    BatchNorm running stats are never shared (FedBN).
    """

    def __init__(
        self,
        model: Any,
        model_split: Optional[ModelSplit] = None,
        num_head_layers: int = 2,
        device: str = "cpu",
        model_version: str = "1.0.0",
    ):
        PyTorchModel.__init__(self, model, device=device, model_version=model_version)

        if model_split is None:
            layer_names = list(self.get_weights().keys())
            model_split = ModelSplit.auto_detect(layer_names, num_head_layers=num_head_layers)

        PersonalizedModelWrapper.__init__(self, model, model_split, model_version=model_version)
