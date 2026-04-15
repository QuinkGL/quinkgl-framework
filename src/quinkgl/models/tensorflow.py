"""
TensorFlow/Keras Model Wrapper

Wrapper for TensorFlow/Keras models.
"""

from typing import Any, Dict, Optional
import numpy as np

from quinkgl.models.base import (
    ModelWrapper,
    TrainingConfig,
    TrainingResult
)


class TensorFlowModel(ModelWrapper):
    """
    Wrapper for TensorFlow/Keras models.

    Provides get_weights/set_weights interface for Keras models.
    """

    def __init__(self, model: Any, model_version: str = "1.0.0"):
        """
        Initialize TensorFlow model wrapper.

        Args:
            model: Keras Model instance
            model_version: Semantic version string for model architecture
        """
        super().__init__(model, model_version=model_version)

    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get model weights as numpy arrays.

        Returns:
            Dict mapping layer names to numpy arrays of weights
        """
        weights = {}
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                for i, w in enumerate(layer_weights):
                    key = f"{layer.name}_{'weight' if i == 0 else 'bias'}"
                    weights[key] = w
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weights from numpy arrays.

        Args:
            weights: Dict mapping parameter names to numpy arrays

        Raises:
            ValueError: If weights are invalid (NaN, Inf, wrong types, etc.)
        """
        import logging
        logger = logging.getLogger(__name__)

        # Reconstruct layer weights from dict
        layer_weights = {}
        for key, array in weights.items():
            # Ensure array is numpy type
            if not isinstance(array, np.ndarray):
                try:
                    array = np.array(array)
                except Exception as e:
                    raise ValueError(f"Cannot convert weights[{key}] to numpy array: {e}")

            # Check for NaN values
            if np.isnan(array).any():
                raise ValueError(f"Weights[{key}] contains NaN values")

            # Check for Inf values
            if np.isinf(array).any():
                raise ValueError(f"Weights[{key}] contains Inf values")

            # Parse key: "layer_name_weight" or "layer_name_bias"
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                layer_name, weight_type = parts
                if layer_name not in layer_weights:
                    layer_weights[layer_name] = []
                # Ensure correct order: weight then bias
                if weight_type == 'weight':
                    if len(layer_weights[layer_name]) == 1:
                        layer_weights[layer_name].insert(0, array)
                    else:
                        layer_weights[layer_name].insert(0, array)
                else:  # bias
                    layer_weights[layer_name].append(array)

        # Set weights for each layer
        for layer in self.model.layers:
            if layer.name in layer_weights:
                layer.set_weights(layer_weights[layer.name])

    async def train(
        self,
        data: Any,
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Train the model on local data.

        Args:
            data: Tuple of (features, labels) or tf.data.Dataset
            config: Training configuration

        Returns:
            TrainingResult with metrics
        """
        config = config or TrainingConfig()

        # Prepare data
        if isinstance(data, tuple) and len(data) == 2:
            features, labels = data
            # Already numpy arrays or convertable
        else:
            features, labels = data  # Assume correct format

        # Compile model if needed
        if not self.model.optimizer:
            from tensorflow import keras
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        # Train
        history = self.model.fit(
            features, labels,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1 if config.verbose else 0,
            callbacks=[_EpochCallback(config.on_epoch_end)] if config.on_epoch_end else None
        )

        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history.get('accuracy', [None])[-1]

        self.increment_round()

        return TrainingResult(
            epochs_completed=config.epochs,
            final_loss=float(final_loss),
            final_accuracy=float(final_acc) if final_acc else None,
            samples_trained=len(features)
        )

    def evaluate(self, data: Any, loss_fn: Any = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            data: Tuple of (features, labels)
            loss_fn: Optional custom loss function (not used in TensorFlow)

        Returns:
            Dict with loss and accuracy
        """
        features, labels = data
        results = self.model.evaluate(features, labels, verbose=0)

        if isinstance(results, list):
            # Assuming [loss, accuracy] format
            return {"loss": float(results[0]), "accuracy": float(results[1])}
        else:
            return {"loss": float(results)}

    def get_data_schema_hash(self) -> str:
        """
        Get a hash representing the model's input schema.
        """
        import hashlib

        # Get input shape from model
        input_shape = self.model.input_shape
        schema_info = f"tensorflow_{input_shape}"

        return hashlib.sha256(schema_info.encode()).hexdigest()[:16]


class _EpochCallback:
    """Keras callback for epoch end notifications."""
    def __init__(self, callback_fn):
        # Defer keras import to avoid unnecessary dependency
        try:
            from tensorflow import keras
            self._keras = keras
        except ImportError:
            self._keras = None
        self.callback_fn = callback_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        self.callback_fn(epoch, loss, acc)
