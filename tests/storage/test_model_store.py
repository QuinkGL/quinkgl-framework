"""Tests for ModelCheckpoint and ModelStore."""

import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from quinkgl.storage.model_store import ModelCheckpoint, ModelStore, ModelStoreError


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

    def test_verify_checksum_detects_weight_tamper(self):
        cp = ModelCheckpoint(round_number=1, weights={"w": np.zeros(3)})
        cp.weights["w"] = np.ones(3)
        assert cp.verify_checksum() is False

    def test_round_number_negative_raises(self):
        with pytest.raises(ValueError, match="round_number"):
            ModelCheckpoint(round_number=-1, weights={})

    def test_checkpoint_id_invalid_raises(self):
        with pytest.raises(ValueError, match="checkpoint_id"):
            ModelCheckpoint(round_number=1, weights={}, checkpoint_id="../evil")

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

    def test_load_invalid_checkpoint_id_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir)
            with pytest.raises(ValueError, match="checkpoint_id"):
                store.load_checkpoint("../evil")

    def test_delete_invalid_checkpoint_id_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir)
            with pytest.raises(ValueError, match="checkpoint_id"):
                store.delete_checkpoint("../evil")

    def test_save_checkpoint_propagates_disk_errors_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            with patch("quinkgl.storage.model_store.os.replace", side_effect=OSError("disk full")):
                with pytest.raises(ModelStoreError, match="Failed to save checkpoint to disk"):
                    store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})

            assert store.list_checkpoints() == []
            assert list(store.storage_dir.iterdir()) == []

    def test_list_checkpoint_metadata_returns_lightweight_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=False)
            store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)}, metrics={"loss": 0.5})
            store.save_checkpoint(round_number=2, weights={"w": np.ones(3)}, metrics={"loss": 0.25})

            with patch.object(store, "_load_from_disk", side_effect=AssertionError("should not full-load checkpoints")):
                metadata = store.list_checkpoint_metadata()

            assert [entry.round_number for entry in metadata] == [1, 2]
            assert metadata[0].metrics["loss"] == 0.5
            assert metadata[0].size_bytes > 0

    def test_get_storage_size_uses_cached_metadata_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=False)
            store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})
            first_size = store.get_storage_size()
            assert first_size > 0

            with patch.object(store, "list_checkpoint_metadata", side_effect=AssertionError("cache should be reused")):
                second_size = store.get_storage_size()

            assert second_size == first_size

    def test_delete_checkpoint_removes_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=False)
            checkpoint = store.save_checkpoint(round_number=1, weights={"w": np.zeros(3)})
            metadata_path = store._get_metadata_path(checkpoint.checkpoint_id)
            assert metadata_path.exists()

            deleted = store.delete_checkpoint(checkpoint.checkpoint_id)

            assert deleted is True
            assert metadata_path.exists() is False

    @pytest.mark.asyncio
    async def test_save_checkpoint_async_persists_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(storage_dir=tmpdir, keep_in_memory=True)
            checkpoint = await store.save_checkpoint_async(
                round_number=7,
                weights={"w": np.array([1.0, 2.0])},
                metrics={"loss": 0.25},
            )

            loaded = await store.load_checkpoint_async(checkpoint.checkpoint_id)
            assert loaded is not None
            assert loaded.round_number == 7
            assert loaded.metrics["loss"] == 0.25


@pytest.mark.asyncio
async def test_learning_node_save_checkpoint_uses_async_model_store():
    from quinkgl.core.learning_node import LearningNode

    node = object.__new__(LearningNode)
    async def _get_model_weights_snapshot():
        return {"w": np.array([3.0, 4.0])}

    node.aggregator = SimpleNamespace(
        current_round=11,
        get_model_weights_snapshot=_get_model_weights_snapshot,
    )
    node.model = SimpleNamespace(get_weights=lambda: {"w": np.array([3.0, 4.0])})
    node.model_store = SimpleNamespace(save_checkpoint_async=patch)

    called = {}

    async def _save_checkpoint_async(**kwargs):
        called.update(kwargs)

    node.model_store = SimpleNamespace(save_checkpoint_async=_save_checkpoint_async)

    await LearningNode.save_checkpoint(node, metrics={"accuracy": 0.9})

    assert called["round_number"] == 11
    assert called["metrics"] == {"accuracy": 0.9}
    assert np.array_equal(called["weights"]["w"], np.array([3.0, 4.0]))
