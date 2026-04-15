"""Tests for ModelSplit, PersonalizedModelWrapper, and PyTorchPersonalizedModel."""

import numpy as np
import pytest

from quinkgl.models.base import (
    ModelSplit,
    PersonalizedModelWrapper,
    TrainingResult,
)
from quinkgl.models.pytorch import PyTorchPersonalizedModel


class DictModelWrapper(PersonalizedModelWrapper):
    """Concrete PersonalizedModelWrapper backed by a dict of numpy arrays."""

    def __init__(self, weights_dict, model_split, model_version="1.0.0"):
        self._weights = dict(weights_dict)
        super().__init__(model=weights_dict, model_split=model_split, model_version=model_version)

    def get_weights(self):
        return dict(self._weights)

    def set_weights(self, weights):
        for k, v in weights.items():
            self._weights[k] = v

    async def train(self, data, config=None):
        return TrainingResult(epochs_completed=1, final_loss=0.5)

    def evaluate(self, data, loss_fn=None):
        return {"loss": 0.5, "accuracy": 0.8}


def _make_pw():
    weights = {
        "conv1.weight": np.array([1.0, 2.0]),
        "conv1.bias": np.array([0.1]),
        "bn1.running_mean": np.array([0.5]),
        "bn1.running_var": np.array([0.1]),
        "fc.weight": np.array([3.0, 4.0]),
        "fc.bias": np.array([0.2]),
    }
    split = ModelSplit(
        backbone_layers=["conv1.weight", "conv1.bias"],
        head_layers=["fc.weight", "fc.bias"],
        local_norm_layers=["bn1.running_mean", "bn1.running_var"],
    )
    return DictModelWrapper(weights, split)


# ── ModelSplit ────────────────────────────────────────────────────────

class TestModelSplit:
    def test_explicit_split(self):
        split = ModelSplit(
            backbone_layers=["conv1.weight", "conv1.bias"],
            head_layers=["fc.weight", "fc.bias"],
            local_norm_layers=["bn1.running_mean", "bn1.running_var"],
        )
        assert len(split.backbone_layers) == 2
        assert len(split.head_layers) == 2
        assert len(split.local_norm_layers) == 2

    def test_auto_detect_basic(self):
        names = [
            "conv1.weight", "conv1.bias",
            "bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
            "conv2.weight", "conv2.bias",
            "fc.weight", "fc.bias",
        ]
        split = ModelSplit.auto_detect(names, num_head_layers=2)
        assert "bn1.running_mean" in split.local_norm_layers
        assert "bn1.running_var" in split.local_norm_layers
        assert "bn1.num_batches_tracked" in split.local_norm_layers
        assert "fc.weight" in split.head_layers
        assert "fc.bias" in split.head_layers
        assert "conv1.weight" in split.backbone_layers
        assert "conv2.weight" in split.backbone_layers

    def test_auto_detect_no_norm(self):
        names = ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"]
        split = ModelSplit.auto_detect(names, num_head_layers=2)
        assert split.local_norm_layers == []
        assert "layer2.weight" in split.head_layers
        assert "layer2.bias" in split.head_layers
        assert "layer1.weight" in split.backbone_layers

    def test_auto_detect_all_norm(self):
        names = ["bn1.running_mean", "bn1.running_var"]
        split = ModelSplit.auto_detect(names, num_head_layers=2)
        assert split.backbone_layers == []
        assert split.head_layers == []
        assert len(split.local_norm_layers) == 2

    def test_auto_detect_zero_head(self):
        names = ["conv1.weight", "conv1.bias", "fc.weight", "fc.bias"]
        split = ModelSplit.auto_detect(names, num_head_layers=0)
        assert split.head_layers == []
        assert "fc.weight" in split.backbone_layers

    def test_get_shared_layers(self):
        split = ModelSplit(
            backbone_layers=["conv1.weight"],
            head_layers=["fc.weight"],
            local_norm_layers=["bn1.running_mean"],
        )
        assert split.get_shared_layers() == ["conv1.weight"]

    def test_get_local_layers(self):
        split = ModelSplit(
            backbone_layers=["conv1.weight"],
            head_layers=["fc.weight"],
            local_norm_layers=["bn1.running_mean"],
        )
        local = split.get_local_layers()
        assert "fc.weight" in local
        assert "bn1.running_mean" in local


# ── PersonalizedModelWrapper (via DictModelWrapper) ──────────────────

class TestPersonalizedModelWrapper:
    def test_backbone_isolation(self):
        pw = _make_pw()
        backbone = pw.get_backbone_weights()
        assert "conv1.weight" in backbone
        assert "conv1.bias" in backbone
        assert "fc.weight" not in backbone
        assert "bn1.running_mean" not in backbone

    def test_head_isolation(self):
        pw = _make_pw()
        head = pw.get_head_weights()
        assert "fc.weight" in head
        assert "fc.bias" in head
        assert "conv1.weight" not in head

    def test_local_norm_isolation(self):
        pw = _make_pw()
        norm = pw.get_local_norm_weights()
        assert "bn1.running_mean" in norm
        assert "bn1.running_var" in norm
        assert "conv1.weight" not in norm

    def test_set_backbone_preserves_head(self):
        pw = _make_pw()
        pw.set_backbone_weights({"conv1.weight": np.array([99.0])})
        all_w = pw.get_weights()
        assert all_w["conv1.weight"][0] == 99.0
        np.testing.assert_array_equal(all_w["fc.weight"], np.array([3.0, 4.0]))

    def test_set_backbone_ignores_head_keys(self):
        pw = _make_pw()
        pw.set_backbone_weights({
            "conv1.weight": np.array([10.0]),
            "fc.weight": np.array([999.0]),
        })
        all_w = pw.get_weights()
        assert all_w["conv1.weight"][0] == 10.0
        np.testing.assert_array_equal(all_w["fc.weight"], np.array([3.0, 4.0]))

    def test_get_shared_weights(self):
        pw = _make_pw()
        shared = pw.get_shared_weights()
        assert "conv1.weight" in shared
        assert "fc.weight" not in shared
        assert "bn1.running_mean" not in shared

    def test_partition_covers_all_keys(self):
        pw = _make_pw()
        full = set(pw.get_weights().keys())
        partition = (set(pw.get_backbone_weights().keys()) |
                     set(pw.get_head_weights().keys()) |
                     set(pw.get_local_norm_weights().keys()))
        assert full == partition


# ── PyTorchPersonalizedModel ──────────────────────────────────────────

class TestPyTorchPersonalizedModel:
    @pytest.fixture
    def simple_model(self):
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 3),
        )

    def test_auto_detect_split(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        split = pm.model_split
        assert len(split.backbone_layers) > 0
        assert len(split.head_layers) > 0
        assert any("running_mean" in n or "running_var" in n for n in split.local_norm_layers)

    def test_backbone_weights(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        backbone = pm.get_backbone_weights()
        assert isinstance(backbone, dict)
        for key in backbone:
            assert key in pm.model_split.backbone_layers

    def test_head_weights(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        head = pm.get_head_weights()
        for key in head:
            assert key in pm.model_split.head_layers

    def test_local_norm_weights(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        norm = pm.get_local_norm_weights()
        for key in norm:
            assert key in pm.model_split.local_norm_layers

    def test_set_backbone_preserves_head(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        original_head = pm.get_head_weights()
        original_norm = pm.get_local_norm_weights()

        backbone = pm.get_backbone_weights()
        modified = {k: v + 1.0 for k, v in backbone.items()}
        pm.set_backbone_weights(modified)

        new_head = pm.get_head_weights()
        new_norm = pm.get_local_norm_weights()

        for key in original_head:
            np.testing.assert_array_equal(original_head[key], new_head[key])
        for key in original_norm:
            np.testing.assert_array_equal(original_norm[key], new_norm[key])

    def test_explicit_model_split(self, simple_model):
        split = ModelSplit(
            backbone_layers=["0.weight", "0.bias"],
            head_layers=["3.weight", "3.bias"],
            local_norm_layers=["1.running_mean", "1.running_var", "1.weight", "1.bias", "1.num_batches_tracked"],
        )
        pm = PyTorchPersonalizedModel(simple_model, model_split=split)
        assert pm.model_split is split

    def test_full_model_weights_unchanged(self, simple_model):
        pm = PyTorchPersonalizedModel(simple_model, num_head_layers=2)
        full = pm.get_weights()
        backbone = pm.get_backbone_weights()
        head = pm.get_head_weights()
        norm = pm.get_local_norm_weights()

        all_partitioned = set(backbone.keys()) | set(head.keys()) | set(norm.keys())
        all_full = set(full.keys())
        assert all_partitioned == all_full
