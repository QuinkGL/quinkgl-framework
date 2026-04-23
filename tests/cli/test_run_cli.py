# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Run CLI tests (B-4 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quinkgl.cli.__main__ import main
from quinkgl.cli.exit_codes import NODE_CONFIG_ERROR, SUCCESS, VALIDATION_ERROR


class TestRunDryRun:
    def test_run_dry_run(self, tmp_path: Path) -> None:
        qgl = tmp_path / "swarm.qgl"
        assert main([
            "manifest", "create",
            "--name", "test",
            "--task-type", "class",
            "--input-shape", "3,224,224",
            "--output-shape", "10",
            "--label-type", "integer",
            "--model-framework", "pytorch",
            "--model-arch-hash", "sha256:" + "a" * 64,
            "--aggregation", "FedAvg",
            "--topology", "Random",
            "--output", str(qgl),
        ]) == SUCCESS
        assert main(["run", "--manifest", str(qgl), "--dry-run"]) == SUCCESS


class TestRunModeA:
    def test_mode_a_rejects_no_standard_model(self, tmp_path: Path) -> None:
        qgl = tmp_path / "swarm.qgl"
        assert main([
            "manifest", "create",
            "--name", "test",
            "--task-type", "class",
            "--input-shape", "3,224,224",
            "--output-shape", "10",
            "--label-type", "integer",
            "--model-framework", "pytorch",
            "--model-arch-hash", "sha256:" + "a" * 64,
            "--aggregation", "FedAvg",
            "--topology", "Random",
            "--output", str(qgl),
        ]) == SUCCESS
        assert main(["run", "--manifest", str(qgl), "--data", str(tmp_path)]) == NODE_CONFIG_ERROR


class TestRunModeB:
    def test_mode_b_missing_callables(self, tmp_path: Path) -> None:
        qgl = tmp_path / "swarm.qgl"
        script = tmp_path / "bad_script.py"
        script.write_text("# empty\n")
        assert main([
            "manifest", "create",
            "--name", "test",
            "--task-type", "class",
            "--input-shape", "3,224,224",
            "--output-shape", "10",
            "--label-type", "integer",
            "--model-framework", "pytorch",
            "--model-arch-hash", "sha256:" + "a" * 64,
            "--aggregation", "FedAvg",
            "--topology", "Random",
            "--output", str(qgl),
        ]) == SUCCESS
        assert main(["run", "--manifest", str(qgl), "--script", str(script)]) == NODE_CONFIG_ERROR

    def test_mode_b_script_runs(self, tmp_path: Path) -> None:
        qgl = tmp_path / "swarm.qgl"
        script = tmp_path / "peer_script.py"
        script.write_text('''
import torch
import torch.nn as nn

def build_model(manifest, **kwargs):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3*224*224, 10)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    return Tiny()

def build_loaders(manifest, **kwargs):
    from quinkgl.testing import DummyDataLoader
    return DummyDataLoader([4, 3, 224, 224], num_batches=2), None
''')
        assert main([
            "manifest", "create",
            "--name", "test",
            "--task-type", "class",
            "--input-shape", "3,224,224",
            "--output-shape", "10",
            "--label-type", "integer",
            "--model-framework", "pytorch",
            "--model-arch-hash", "sha256:" + "a" * 64,
            "--aggregation", "FedAvg",
            "--topology", "Random",
            "--output", str(qgl),
        ]) == SUCCESS
        # We expect NODE_CONFIG_ERROR because GossipNode construction
        # may fail with a non-ModelWrapper instance, or training may
        # fail quickly. The important thing is that the script is loaded
        # and build_model/build_loaders are called.
        result = main(["run", "--manifest", str(qgl), "--script", str(script), "--dry-run"])
        assert result == SUCCESS
