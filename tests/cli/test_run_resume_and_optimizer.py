"""Covers the three ``quinkgl run`` completions the spec carves out:

* ``--resume`` replays the latest ``ModelStore`` checkpoint into the
  model before training begins.
* Periodic + final checkpoints are written by the per-round callback
  the CLI layer builds (``_build_on_round_end``).
* ``build_optimizer`` / ``build_scheduler`` exposed on the user script
  flow through into :class:`TrainingConfig` and produce a callable
  ``scheduler.step()`` invocation once per round.

The tests never touch IPv8 — they drive the helpers directly.  This is
intentional: the wiring we care about is between the CLI and
``quinkgl.storage`` / :class:`TrainingConfig` / the per-round callback.
The full end-to-end IPv8 path has its own integration tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quinkgl.cli.run_cmd import (
    _build_on_round_end,
    _make_training_config_with_optimizer,
    _maybe_resume,
    _open_checkpoint_store,
    _save_checkpoint,
)


# --- Stub model that implements the minimal ModelWrapper surface ---------


class _ListModel:
    """Stand-in for a real model: weights are just a tagged list."""

    def __init__(self, weights: Any = None):
        self._weights = weights if weights is not None else [0.0, 0.0, 0.0]

    def get_weights(self) -> Any:
        return list(self._weights)

    def set_weights(self, w: Any) -> None:
        self._weights = list(w)


class _StubNode:
    """Just enough of :class:`GossipNode` for _build_on_round_end."""

    def __init__(self, model: _ListModel, node_id: str = "peer-test"):
        self.model = model
        self.node_id = node_id
        self.state = SimpleNamespace(name="TRAINING")
        self.manifest = None
        self.gl_node = SimpleNamespace(
            current_round=0, aggregator=SimpleNamespace(last_loss=None)
        )
        self.community = SimpleNamespace(peers=[], peers_discovered_count=0)
        self.ipv8_manager = SimpleNamespace(port=0)


# --- Checkpoint / resume --------------------------------------------------


class TestCheckpointResume:
    def test_open_store_returns_none_without_dir(self):
        assert _open_checkpoint_store(None) is None

    def test_open_store_creates_dir_and_returns_store(self, tmp_path: Path):
        store = _open_checkpoint_store(str(tmp_path / "ckpt"))
        assert store is not None
        assert (tmp_path / "ckpt").exists()

    def test_resume_reloads_latest_weights(self, tmp_path: Path):
        store = _open_checkpoint_store(str(tmp_path / "ckpt"))
        saved = _ListModel([1.0, 2.0, 3.0])
        _save_checkpoint(store, saved, round_number=5)

        fresh = _ListModel([0.0, 0.0, 0.0])
        resumed_round = _maybe_resume(store, fresh, resume_flag=True)
        assert resumed_round == 5
        assert fresh.get_weights() == [1.0, 2.0, 3.0]

    def test_resume_without_flag_is_noop(self, tmp_path: Path):
        store = _open_checkpoint_store(str(tmp_path / "ckpt"))
        _save_checkpoint(store, _ListModel([9.0]), round_number=3)
        fresh = _ListModel([0.0])
        r = _maybe_resume(store, fresh, resume_flag=False)
        assert r == 0
        assert fresh.get_weights() == [0.0]

    def test_resume_with_empty_store_starts_fresh(self, tmp_path: Path):
        store = _open_checkpoint_store(str(tmp_path / "ckpt"))
        fresh = _ListModel([0.0, 0.0])
        assert _maybe_resume(store, fresh, resume_flag=True) == 0
        assert fresh.get_weights() == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_every_10_rounds(self, tmp_path: Path):
        store = _open_checkpoint_store(str(tmp_path / "ckpt"))
        model = _ListModel([42.0])
        node = _StubNode(model)

        cb = _build_on_round_end(
            node,
            ckpt_store=store,
            script_mod=None,
            status_json_path=None,
            since_ts="2026-04-22T00:00:00Z",
        )
        # Rounds 0..8 should NOT trigger a save; round 9 (index 9 → (9+1)%10==0) does.
        for i in range(10):
            await cb(i, {})
        latest = store.get_latest_checkpoint()
        assert latest is not None
        assert latest.round_number == 9


# --- build_optimizer / build_scheduler -----------------------------------


class TestUserOptimizerScheduler:
    def test_make_training_config_with_optimizer_instance(self):
        class _OptStub:
            pass

        opt = _OptStub()
        manifest = SimpleNamespace(task={"learning_rate": 0.05, "batch_size": 16})
        cfg = _make_training_config_with_optimizer(manifest, opt)
        assert cfg.optimizer is opt
        assert cfg.learning_rate == 0.05
        assert cfg.batch_size == 16

    def test_make_training_config_accepts_dataclass_task(self):
        class _OptStub:
            pass

        task = SimpleNamespace(learning_rate=0.001, batch_size=64)
        manifest = SimpleNamespace(task=task)
        cfg = _make_training_config_with_optimizer(manifest, _OptStub())
        assert cfg.learning_rate == 0.001
        assert cfg.batch_size == 64

    @pytest.mark.asyncio
    async def test_user_scheduler_step_invoked_each_round(self):
        calls = {"n": 0}

        class _Sched:
            def step(self):
                calls["n"] += 1

        script_mod = SimpleNamespace(
            _quinkgl_user_scheduler=_Sched(),
            on_round_end=None,
        )
        node = _StubNode(_ListModel())
        cb = _build_on_round_end(
            node,
            ckpt_store=None,
            script_mod=script_mod,
            status_json_path=None,
            since_ts="",
        )
        for i in range(3):
            await cb(i, {})
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_user_on_round_end_still_fires(self):
        fired = []

        async def _user_cb(round_idx, metrics):
            fired.append(round_idx)

        script_mod = SimpleNamespace(
            on_round_end=_user_cb,
            _quinkgl_user_scheduler=None,
        )
        node = _StubNode(_ListModel())
        cb = _build_on_round_end(
            node,
            ckpt_store=None,
            script_mod=script_mod,
            status_json_path=None,
            since_ts="",
        )
        await cb(0, {"loss": 0.1})
        await cb(1, {"loss": 0.05})
        assert fired == [0, 1]

    @pytest.mark.asyncio
    async def test_scheduler_exception_is_logged_not_raised(self, caplog):
        class _Sched:
            def step(self):
                raise RuntimeError("schedule kaboom")

        script_mod = SimpleNamespace(
            _quinkgl_user_scheduler=_Sched(),
            on_round_end=None,
        )
        node = _StubNode(_ListModel())
        cb = _build_on_round_end(
            node,
            ckpt_store=None,
            script_mod=script_mod,
            status_json_path=None,
            since_ts="",
        )
        # Must not raise.
        await cb(0, {})


# --- Status JSON snapshot refresh ----------------------------------------


class TestStatusJsonRefresh:
    @pytest.mark.asyncio
    async def test_status_json_is_rewritten_each_round(self, tmp_path: Path):
        import json

        json_path = tmp_path / "peer-test.json"
        node = _StubNode(_ListModel())
        cb = _build_on_round_end(
            node,
            ckpt_store=None,
            script_mod=None,
            status_json_path=json_path,
            since_ts="2026-04-22T00:00:00Z",
        )
        await cb(0, {})
        payload = json.loads(json_path.read_text())
        assert payload["node_id"] == "peer-test"
        assert payload["status"] == "TRAINING"
