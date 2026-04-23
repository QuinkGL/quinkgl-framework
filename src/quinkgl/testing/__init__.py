# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Public testing helpers for user peer scripts (§10.7.6)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import quinkgl
from quinkgl.manifest import (
    ByzantineSpec,
    DataPolicy,
    ModelSpec,
    SwarmManifest,
    TaskSpec,
)


def make_dummy_manifest(**overrides: Any) -> SwarmManifest:
    """Produce a valid ``SwarmManifest`` for tests.

    Defaults match the simplest possible valid v3 manifest.  Override any
    field by passing it as a keyword argument.
    """
    defaults = {
        "name": "dummy-swarm",
        "model_arch_fingerprint": "sha256:" + "0" * 64,
        "data_schema_hash": "sha256:dummy-data-schema",
        "aggregation_name": "FedAvg",
        "aggregation_params": {},
        "topology_name": "Random",
        "topology_params": {},
        "compression_enabled": False,
        "compression_params": {},
        "data_policy": DataPolicy(),
        "task": TaskSpec(
            type="classification",
            input_shape=[3, 224, 224],
            output_shape=[10],
            label_type="integer",
            tags=["test"],
        ),
        "model": ModelSpec(
            framework="pytorch",
            arch_hash="sha256:" + "a" * 64,
        ),
        "byzantine": ByzantineSpec(f=0, enforce_n_gt_2f_plus_2=False),
        "round_limit": None,
        "bootstrap_peers": [],
        "tracker_urls": [],
    }
    defaults.update(overrides)
    manifest = SwarmManifest(**defaults)
    manifest.validate()
    return manifest


class DummyDataLoader:
    """Synthetic data loader yielding tensors of the requested shape.

    Works with PyTorch out of the box; returns plain tuples if torch is not
    installed (the framework should still be able to inspect shapes).
    """

    def __init__(
        self,
        shape: List[int],
        num_batches: int = 8,
        label_type: str = "integer",
    ):
        self.shape = list(shape)
        self.num_batches = num_batches
        self.label_type = label_type

    def __iter__(self):
        try:
            import torch
            for _ in range(self.num_batches):
                x = torch.randn(*self.shape)
                if self.label_type == "binary":
                    y = torch.randint(0, 2, (self.shape[0],))
                elif self.label_type == "multiclass":
                    y = torch.randint(0, self.shape[0], (self.shape[0],))
                else:
                    y = torch.randint(0, 10, (self.shape[0],))
                yield x, y
        except ImportError:
            for _ in range(self.num_batches):
                yield None, None

    def __len__(self) -> int:
        return self.num_batches


@asynccontextmanager
async def local_swarm_fixture(
    size: int = 3,
    manifest_path: Optional[str] = None,
    **manifest_overrides: Any,
) -> AsyncIterator[List[Any]]:
    """Spin up *size* in-process peers on a local loopback IPv8 network.

    Yields a list of ``GossipNode`` instances.  On exit, every node is
    stopped gracefully (including on exception).

    Parameters
    ----------
    size:
        Number of peers to create (default 3).
    manifest_path:
        Optional path to a ``.qgl`` file to load.  If omitted,
        ``make_dummy_manifest(**manifest_overrides)`` is used.
    **manifest_overrides:
        Passed to ``make_dummy_manifest`` when *manifest_path* is not given.
    """
    from quinkgl import GossipNode
    from quinkgl.models import PyTorchModel

    if manifest_path:
        manifest = SwarmManifest.from_file(manifest_path)
    else:
        manifest = make_dummy_manifest(**manifest_overrides)

    # Build a trivial PyTorch model that matches the manifest shapes
    try:
        import torch
        import torch.nn as nn

        class _TinyModel(nn.Module):
            def __init__(self, input_shape: List[int], output_dim: int):
                super().__init__()
                flat = 1
                for d in input_shape:
                    flat *= d
                self.fc = nn.Linear(flat, output_dim)

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        model = PyTorchModel(
            _TinyModel(manifest.task.input_shape, manifest.task.output_shape[0])
        )
    except Exception as exc:
        raise RuntimeError(f"Could not build dummy model for local swarm: {exc}") from exc

    nodes: List[GossipNode] = []
    try:
        for i in range(size):
            node = GossipNode(
                node_id=f"peer-{i}",
                manifest=manifest,
                model=model,
                port=0,  # random port
                quiet=True,
            )
            nodes.append(node)
            await node.start()

        # Give IPv8 a moment to discover loopback peers
        await asyncio.sleep(0.5)

        yield nodes
    finally:
        for node in nodes:
            try:
                await node.stop()
            except Exception:
                pass


__all__ = [
    "local_swarm_fixture",
    "make_dummy_manifest",
    "DummyDataLoader",
]
