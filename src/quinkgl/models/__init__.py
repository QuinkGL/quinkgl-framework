"""
Model Wrappers Module

Framework-agnostic model wrappers for different ML frameworks.
Users can wrap their PyTorch, TensorFlow, or custom models.

Usage:
    from quinkgl.models import PyTorchModel, TensorFlowModel

    # PyTorch
    model = PyTorchModel(my_pytorch_model)
    weights = model.get_weights()
    model.set_weights(new_weights)

    # TensorFlow (similar)
    model = TensorFlowModel(my_tf_model)
"""

from quinkgl.models.base import (
    ModelWrapper,
    TrainingConfig,
    TrainingResult,
    ModelSplit,
    PersonalizedModelWrapper,
    APFLConfig,
    APFLMixin,
)
from quinkgl.models.pytorch import PyTorchModel, PyTorchPersonalizedModel

# TensorFlow wrapper is optional - requires tf package
try:
    from quinkgl.models.tensorflow import TensorFlowModel
    _tf_available = True
except ImportError:
    _tf_available = False
    TensorFlowModel = None  # type: ignore

# Export main classes
__all__ = [
    "ModelWrapper",
    "TrainingConfig",
    "TrainingResult",
    "ModelSplit",
    "PersonalizedModelWrapper",
    "APFLConfig",
    "APFLMixin",
    "PyTorchModel",
    "PyTorchPersonalizedModel",
    "TensorFlowModel",
    "_tf_available",
]
