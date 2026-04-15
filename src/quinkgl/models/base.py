# Copyright 2026 Ali Seyhan, Baki Turhan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model Wrapper

Abstract base class for framework-specific model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelSplit:
    backbone_layers: List[str]
    head_layers: List[str]
    local_norm_layers: List[str]

    @classmethod
    def auto_detect(
        cls,
        layer_names: List[str],
        num_head_layers: int = 2,
    ) -> "ModelSplit":
        norm_keywords = {"bn", "batch_norm", "running_mean", "running_var", "num_batches_tracked"}
        local_norm: List[str] = []
        backbone: List[str] = []
        head: List[str] = []

        norm_set: Set[str] = set()
        trainable: List[str] = []

        for name in layer_names:
            lower = name.lower()
            is_norm = any(kw in lower for kw in norm_keywords)
            if is_norm:
                norm_set.add(name)
                local_norm.append(name)
            else:
                trainable.append(name)

        if trainable and num_head_layers > 0:
            split_idx = max(0, len(trainable) - num_head_layers)
            backbone = trainable[:split_idx]
            head = trainable[split_idx:]
        else:
            backbone = trainable

        return cls(
            backbone_layers=backbone,
            head_layers=head,
            local_norm_layers=local_norm,
        )

    def get_shared_layers(self) -> List[str]:
        return self.backbone_layers

    def get_local_layers(self) -> List[str]:
        return self.head_layers + self.local_norm_layers


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    verbose: bool = False
    on_epoch_end: Optional[Callable] = None
    loss_fn: Optional[Any] = None
    optimizer: Optional[Any] = None
    optimizer_kwargs: Optional[dict] = None
    grad_clip_norm: Optional[float] = None
    proximal_coefficient: Optional[float] = None
    global_weights: Optional[Any] = None


@dataclass
class TrainingResult:
    epochs_completed: int
    final_loss: float
    final_accuracy: Optional[float] = None
    samples_trained: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelWrapper(ABC):
    def __init__(self, model: Any, model_version: str = "1.0.0"):
        self.model = model
        self._current_round = 0
        self._model_version = model_version

    @abstractmethod
    def get_weights(self) -> Any:
        pass

    @abstractmethod
    def set_weights(self, weights: Any) -> None:
        pass

    @abstractmethod
    async def train(
        self,
        data: Any,
        config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        pass

    @abstractmethod
    def evaluate(self, data: Any, loss_fn: Any = None) -> Dict[str, float]:
        pass

    def get_data_schema_hash(self) -> str:
        import hashlib
        model_info = f"{self.__class__.__name__}_{self.model.__class__.__name__}"
        return hashlib.sha256(model_info.encode()).hexdigest()[:16]

    def get_model_version(self) -> str:
        return self._model_version

    @property
    def current_round(self) -> int:
        return self._current_round

    def increment_round(self):
        self._current_round += 1


class PersonalizedModelWrapper(ModelWrapper):
    """Model wrapper with FedRep backbone/head split and FedBN isolation."""

    def __init__(self, model: Any, model_split: ModelSplit, model_version: str = "1.0.0"):
        super().__init__(model=model, model_version=model_version)
        self.model_split = model_split
        self._backbone_set = set(model_split.backbone_layers)
        self._head_set = set(model_split.head_layers)
        self._norm_set = set(model_split.local_norm_layers)

    def get_backbone_weights(self) -> Dict[str, np.ndarray]:
        all_weights = self.get_weights()
        return {k: v for k, v in all_weights.items() if k in self._backbone_set}

    def set_backbone_weights(self, weights: Dict[str, np.ndarray]) -> None:
        current = self.get_weights()
        for k, v in weights.items():
            if k in self._backbone_set:
                current[k] = v
        self.set_weights(current)

    def get_head_weights(self) -> Dict[str, np.ndarray]:
        all_weights = self.get_weights()
        return {k: v for k, v in all_weights.items() if k in self._head_set}

    def get_local_norm_weights(self) -> Dict[str, np.ndarray]:
        all_weights = self.get_weights()
        return {k: v for k, v in all_weights.items() if k in self._norm_set}

    def get_shared_weights(self) -> Dict[str, np.ndarray]:
        return self.get_backbone_weights()

    def set_shared_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.set_backbone_weights(weights)


@dataclass
class APFLConfig:
    initial_alpha: float = 0.5
    alpha_lr: float = 0.01
    min_alpha: float = 0.1
    max_alpha: float = 0.9
    update_frequency: int = 1


class APFLMixin:
    """Adaptive Personalized Federated Learning mixin.

    Computes personalized model: v̄ = α·v_local + (1−α)·w_global
    α is adapted based on local vs. global model performance.
    """

    def __init__(self, config: APFLConfig):
        self.apfl_config = config
        self.alpha = config.initial_alpha
        self._round_count = 0

    def compute_personalized_weights(
        self,
        local_weights: Dict[str, np.ndarray],
        global_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        personalized = {}
        for key in local_weights:
            if key in global_weights:
                personalized[key] = (
                    self.alpha * local_weights[key]
                    + (1 - self.alpha) * global_weights[key]
                )
            else:
                personalized[key] = local_weights[key]
        for key in global_weights:
            if key not in personalized:
                personalized[key] = global_weights[key]
        return personalized

    def update_alpha(self, val_loss_local: float, val_loss_global: float):
        self._round_count += 1
        if self._round_count % self.apfl_config.update_frequency != 0:
            return
        if val_loss_local < val_loss_global:
            self.alpha = min(
                self.apfl_config.max_alpha,
                self.alpha + self.apfl_config.alpha_lr,
            )
        else:
            self.alpha = max(
                self.apfl_config.min_alpha,
                self.alpha - self.apfl_config.alpha_lr,
            )
