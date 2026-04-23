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

        TASK-011: Supports multi-tensor layers (e.g. BatchNorm with
        gamma, beta, moving_mean, moving_variance).  Each tensor is
        stored with a numeric suffix:  ``{layer.name}/{i}``.

        Returns:
            Dict mapping parameter names to numpy arrays of weights
        """
        weights = {}
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                for i, w in enumerate(layer_weights):
                    # TASK-011: Use numeric index instead of guessing weight/bias
                    key = f"{layer.name}/{i}"
                    weights[key] = np.array(w, copy=True)
        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set model weights from numpy arrays.

        TASK-011: Reconstructs per-layer weight lists from the flat dict,
        supporting multi-tensor layers (e.g. BatchNorm).  Shape validation
        is performed before assignment.

        Args:
            weights: Dict mapping parameter names to numpy arrays

        Raises:
            ValueError: If weights are invalid (NaN, Inf, wrong types, etc.)
        """
        import logging
        logging.getLogger(__name__)

        # Validate all arrays first
        for key, array in weights.items():
            if not isinstance(array, np.ndarray):
                try:
                    array = np.array(array)
                    weights[key] = array
                except Exception as e:
                    raise ValueError(f"Cannot convert weights[{key}] to numpy array: {e}")
            if np.isnan(array).any():
                raise ValueError(f"Weights[{key}] contains NaN values")
            if np.isinf(array).any():
                raise ValueError(f"Weights[{key}] contains Inf values")

        # TASK-011: Group weights by layer name, preserving index order
        layer_groups: Dict[str, Dict[int, np.ndarray]] = {}
        for key, array in weights.items():
            parts = key.rsplit("/", 1)
            if len(parts) != 2:
                continue  # skip malformed keys
            layer_name, idx_str = parts
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {}
            layer_groups[layer_name][idx] = array

        # Set weights for each layer with shape validation
        for layer in self.model.layers:
            if layer.name not in layer_groups:
                continue
            group = layer_groups[layer.name]
            expected = layer.get_weights()
            if len(expected) == 0:
                continue

            # Reconstruct ordered list of weight arrays
            weight_arrays = []
            for i in range(len(expected)):
                if i not in group:
                    raise ValueError(
                        f"Missing weight index {i} for layer '{layer.name}'. "
                        f"Expected {len(expected)} tensors, got keys: {sorted(group.keys())}"
                    )
                arr = group[i]
                # TASK-011: Shape validation
                if arr.shape != expected[i].shape:
                    raise ValueError(
                        f"Shape mismatch for '{layer.name}/{i}': "
                        f"expected {expected[i].shape}, got {arr.shape}"
                    )
                # Validate dtype compatibility
                if arr.dtype != expected[i].dtype:
                    arr = arr.astype(expected[i].dtype)
                weight_arrays.append(arr)

            try:
                layer.set_weights(weight_arrays)
            except Exception as e:
                raise ValueError(
                    f"Failed to set weights for layer '{layer.name}': {e}"
                )

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
    """TASK-012: Keras callback for epoch end notifications - subclasses keras.callbacks.Callback."""
    def __init__(self, callback_fn):
        # Defer keras import to avoid unnecessary dependency
        try:
            from tensorflow import keras
            self._keras = keras
            # Subclass keras.callbacks.Callback if available
            self._Callback = keras.callbacks.Callback
        except ImportError:
            self._keras = None
            self._Callback = None
        self.callback_fn = callback_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0.0)
        acc = logs.get('accuracy', 0.0)
        self.callback_fn(epoch, loss, acc)
