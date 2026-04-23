"""
T26: Concurrent access tests for ModelStore.

Tests verify thread-safety and correct behavior under concurrent access
using both the sync API (threading) and the async API (asyncio).
"""

import asyncio
import threading
import pytest
import numpy as np
from quinkgl.storage.model_store import ModelStore


class TestConcurrentSyncAPI:
    """Concurrent tests using the synchronous (threading) API."""

    def test_concurrent_saves_no_corruption(self):
        """Multiple threads saving checkpoints simultaneously must not corrupt data."""
        store = ModelStore()
        errors = []

        def save_checkpoint(round_num):
            try:
                weights = {"layer1": np.array([float(round_num), round_num * 2.0])}
                cp = store.save_checkpoint(
                    round_number=round_num,
                    weights=weights,
                    metrics={"loss": float(round_num)},
                )
                assert cp.round_number == round_num
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_checkpoint, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent saves: {errors}"

        for i in range(20):
            cp = store.get_checkpoint_by_round(i)
            assert cp is not None, f"Checkpoint for round {i} not found"
            np.testing.assert_array_equal(cp.weights["layer1"], np.array([float(i), i * 2.0]))

    def test_concurrent_loads_consistent(self):
        """Multiple threads loading the same checkpoint must get consistent data."""
        store = ModelStore()
        weights = {"w": np.array([42.0, 7.0])}
        cp = store.save_checkpoint(round_number=1, weights=weights)
        checkpoint_id = cp.checkpoint_id

        results = {}
        errors = []

        def load_checkpoint(thread_id):
            try:
                loaded = store.load_checkpoint(checkpoint_id)
                results[thread_id] = loaded
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_checkpoint, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for loaded in results.values():
            assert loaded is not None
            np.testing.assert_array_equal(loaded.weights["w"], np.array([42.0, 7.0]))

    def test_concurrent_save_and_delete(self):
        """Concurrent save + delete must not leave the store in an inconsistent state."""
        store = ModelStore()
        errors = []
        checkpoint_ids = []

        def save_then_delete(round_num):
            try:
                weights = {"w": np.array([float(round_num)])}
                cp = store.save_checkpoint(round_number=round_num, weights=weights)
                checkpoint_ids.append(cp.checkpoint_id)
                store.delete_checkpoint(cp.checkpoint_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_then_delete, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store._checkpoints) == 0

    def test_concurrent_list_during_saves(self):
        """Listing checkpoints while saves are in progress must not crash or corrupt."""
        store = ModelStore()
        errors = []
        list_results = []

        def save_checkpoints(start):
            try:
                for i in range(start, start + 5):
                    weights = {"w": np.array([float(i)])}
                    store.save_checkpoint(round_number=i, weights=weights)
            except Exception as e:
                errors.append(e)

        def list_checkpoints():
            try:
                result = store.list_checkpoints()
                list_results.append(len(result))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=save_checkpoints, args=(0,)),
            threading.Thread(target=save_checkpoints, args=(10,)),
            threading.Thread(target=list_checkpoints),
            threading.Thread(target=list_checkpoints),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for count in list_results:
            assert count >= 0


class TestConcurrentAsyncAPI:
    """Concurrent tests using the async API."""

    @pytest.mark.asyncio
    async def test_concurrent_async_saves(self):
        """Multiple concurrent async saves must not corrupt data."""
        store = ModelStore()

        async def save_checkpoint(round_num):
            weights = {"w": np.array([float(round_num)])}
            cp = await store.save_checkpoint_async(round_number=round_num, weights=weights)
            assert cp.round_number == round_num

        await asyncio.gather(*(save_checkpoint(i) for i in range(20)))

        for i in range(20):
            cp = store.get_checkpoint_by_round(i)
            assert cp is not None

    @pytest.mark.asyncio
    async def test_concurrent_async_loads(self):
        """Multiple concurrent async loads must return consistent data."""
        store = ModelStore()
        weights = {"w": np.array([99.0])}
        cp = await store.save_checkpoint_async(round_number=1, weights=weights)
        cid = cp.checkpoint_id

        results = await asyncio.gather(
            *(store.load_checkpoint_async(cid) for _ in range(10))
        )

        for loaded in results:
            assert loaded is not None
            np.testing.assert_array_equal(loaded.weights["w"], np.array([99.0]))

    @pytest.mark.asyncio
    async def test_concurrent_async_save_and_delete(self):
        """Concurrent async save + delete must not corrupt store state."""
        store = ModelStore()

        async def save_then_delete(round_num):
            weights = {"w": np.array([float(round_num)])}
            cp = await store.save_checkpoint_async(round_number=round_num, weights=weights)
            await asyncio.sleep(0.001)
            await store.delete_checkpoint_async(cp.checkpoint_id)

        await asyncio.gather(*(save_then_delete(i) for i in range(10)))

        assert len(store._checkpoints) == 0

    @pytest.mark.asyncio
    async def test_concurrent_async_list_during_saves(self):
        """Listing checkpoints while async saves are in progress must not crash."""
        store = ModelStore()

        async def save_checkpoints(start):
            for i in range(start, start + 5):
                weights = {"w": np.array([float(i)])}
                await store.save_checkpoint_async(round_number=i, weights=weights)

        results = await asyncio.gather(
            save_checkpoints(0),
            save_checkpoints(10),
            store.list_checkpoints_async(),
            store.list_checkpoints_async(),
        )
        for r in results:
            if isinstance(r, list):
                assert len(r) >= 0
