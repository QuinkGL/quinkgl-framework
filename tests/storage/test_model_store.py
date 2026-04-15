"""Tests for ModelCheckpoint and ModelStore."""

import tempfile
import numpy as np
import pytest

from quinkgl.storage.model_store import ModelCheckpoint, ModelStore


class TestModelCheckpoint:
    def test_checksum_auto_generated(self):
        cp = ModelCheckpoint(round_number=1, weights={"w": np.zeros(3)})
        assert cp.checksum != ""
        assert len(cp.checksum) == 16

    def test_verify_checksum_valid(self):
        cp = ModelCheckpoint(round_number=1, weights={"w": np.zeros(3)})
        assert cp.verify_checksum() is True

    def test_verify_checksum_tampered(self):
        cp = ModelCheckpoint(round_number=1, weights={"w": np.zeros(3)})
        cp.checksum = "tampered1234567"
        assert cp.verify_checksum() is False

    def test_round_number_negative_raises(self):
        with pytest.raises(ValueError, match="round_number"):
            ModelCheckpoint(round_number=-1, weights={})

    def test_checkpoint_id_auto_generated(self):
        cp = ModelCheckpoint(round_number=1, weights={})
        assert cp.checkpoint_id != ""
        assert len(cp.checkpoint_id) == 16

    def test_metrics_default_empty(self):
        cp = ModelCheckpoint(round_number=0, weights={})
        assert cp.metrics == {}

    def test_contributing_peers_default_empty(self):
        cp = ModelCheckpoint(round_number=0, weights={})
        assert cp.contributing_peers == []


class TestModelStore:
    def test_save_and_get_by_round(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            weights = {"layer0": np.array([1.0, 2.0, 3.0])}
            cp = store.save_checkpoint(
                round_number=1,
                weights=weights,
                metrics={"loss": 0.5},
                contributing_peers=["peer_a"],
            )
            assert cp is not None

            loaded = store.get_checkpoint_by_round(1)
            assert loaded is not None
            assert loaded.round_number == 1
            assert loaded.metrics["loss"] == 0.5

    def test_get_nonexistent_round_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir)
            result = store.get_checkpoint_by_round(999)
            assert result is None

    def test_get_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})
            store.save_checkpoint(round_number=3, weights={"w": np.ones(3)})
            store.save_checkpoint(round_number=2, weights={"w": np.zeros(3)})

            latest = store.get_latest_checkpoint()
            assert latest is not None
            assert latest.round_number == 3

    def test_clear_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})
            store.save_checkpoint(round_number=2, weights={"w": np.zeros(3)})
            store.save_checkpoint(round_number=3, weights={"w": np.zeros(3)})

            store.clear_old_checkpoints(keep_last_n=1)
            latest = store.get_latest_checkpoint()
            assert latest is not None
            assert latest.round_number == 3

    def test_load_by_checkpoint_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            cp = store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})
            loaded = store.load_checkpoint(cp.checkpoint_id)
            assert loaded is not None
            assert loaded.round_number == 1

    def test_load_nonexistent_id_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir)
            result = store.load_checkpoint("nonexistent_id")
            assert result is None
