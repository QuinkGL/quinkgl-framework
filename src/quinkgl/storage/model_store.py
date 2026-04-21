"""
Model Store

Storage and versioning for model checkpoints.

SECURITY: Uses msgpack serialization instead of pickle to prevent
arbitrary code execution vulnerabilities when loading checkpoints.

- Thread-safe operations with threading.Lock
- Optional compression for reduced storage size
- Enhanced caching for better performance
- Helper methods to reduce code duplication
- Input validation for robustness
"""
import os
import asyncio
import hashlib
import io
import logging
import re
import threading
import tempfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
import msgpack

logger = logging.getLogger(__name__)

CHECKPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Custom Exceptions (optional, for better error handling)
class ModelStoreError(Exception):
    """Base exception for ModelStore errors."""
    pass

class CheckpointNotFoundError(ModelStoreError):
    """Raised when a checkpoint is not found."""
    pass


def _validate_checkpoint_id(checkpoint_id: str) -> str:
    if not isinstance(checkpoint_id, str) or not CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
        raise ValueError("checkpoint_id must match ^[A-Za-z0-9_-]{1,64}$")
    return checkpoint_id


def _checkpoint_payload_dict(checkpoint: "ModelCheckpoint") -> Dict[str, Any]:
    return {
        "round_number": checkpoint.round_number,
        "timestamp": _serialize_value(checkpoint.timestamp),
        "metrics": _serialize_value(checkpoint.metrics),
        "contributing_peers": _serialize_value(checkpoint.contributing_peers),
        "checkpoint_id": checkpoint.checkpoint_id,
        "weights": _serialize_value(checkpoint.weights),
    }

@dataclass
class ModelCheckpoint:
    """
    Represents a saved model checkpoint.
    """
    round_number: int
    weights: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    contributing_peers: List[str] = field(default_factory=list)
    checkpoint_id: str = ""
    checksum: str = ""  # SHA256 checksum for data integrity

    def __post_init__(self):
        # Validate round_number
        if self.round_number < 0:
            raise ValueError(f"round_number must be >= 0, got {self.round_number}")

        # Generate checkpoint_id if not provided
        if not self.checkpoint_id:
            content = f"{self.round_number}_{self.timestamp}"
            self.checkpoint_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        self.checkpoint_id = _validate_checkpoint_id(self.checkpoint_id)

        # Generate checksum if not provided (backward compatible - optional field)
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of the checkpoint metadata."""
        payload = _checkpoint_payload_dict(self)
        return hashlib.sha256(msgpack.packb(payload, use_bin_type=True)).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        """Verify that the stored checksum matches the computed one."""
        return self.checksum == self._compute_checksum()


@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    round_number: int
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    contributing_peers: List[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0

# Serialization Utilities
def _serialize_numpy_array(arr: np.ndarray) -> bytes:
    """Safely serialize a numpy array to bytes."""
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()

def _deserialize_numpy_array(data: bytes) -> np.ndarray:
    """Safely deserialize bytes to a numpy array."""
    buffer = io.BytesIO(data)
    return np.load(buffer, allow_pickle=False)

def _serialize_value(value: Any) -> Any:
    """
    Convert Python/numpy types to msgpack-serializable format.
    """
    if isinstance(value, np.ndarray):
        return {
            "__type__": "numpy.ndarray",
            "__data__": _serialize_numpy_array(value).hex(),
            "dtype": str(value.dtype),
            "shape": value.shape
        }
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, datetime):
        return {
            "__type__": "datetime",
            "value": value.isoformat()
        }
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value

def _deserialize_value(value: Any) -> Any:
    """
    Convert msgpack-deserialized data back to Python/numpy types.
    """
    if isinstance(value, dict):
        # Check for special types
        if value.get("__type__") == "numpy.ndarray":
            array_bytes = bytes.fromhex(value["__data__"])
            return _deserialize_numpy_array(array_bytes)
        elif value.get("__type__") == "datetime":
            return datetime.fromisoformat(value["value"])
        # Regular dict
        return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    return value

def _serialize_checkpoint(
    checkpoint: ModelCheckpoint,
    compress: bool = False
) -> bytes:
    """Serialize a checkpoint to bytes using msgpack."""
    data = _checkpoint_payload_dict(checkpoint)
    data["checksum"] = checkpoint.checksum
    serialized = msgpack.packb(data, use_bin_type=True)

    if compress:
        serialized = zlib.compress(serialized, level=3)

    return serialized


def _checkpoint_metadata_dict(metadata: CheckpointMetadata) -> Dict[str, Any]:
    return {
        "round_number": metadata.round_number,
        "timestamp": _serialize_value(metadata.timestamp),
        "metrics": _serialize_value(metadata.metrics),
        "contributing_peers": _serialize_value(metadata.contributing_peers),
        "checkpoint_id": metadata.checkpoint_id,
        "checksum": metadata.checksum,
        "size_bytes": metadata.size_bytes,
    }


def _serialize_checkpoint_metadata(metadata: CheckpointMetadata) -> bytes:
    return msgpack.packb(_checkpoint_metadata_dict(metadata), use_bin_type=True)


def _deserialize_checkpoint_metadata(data: bytes) -> CheckpointMetadata:
    unpacked = msgpack.unpackb(data, raw=False)
    return CheckpointMetadata(
        round_number=unpacked["round_number"],
        timestamp=_deserialize_value(unpacked["timestamp"]),
        metrics=_deserialize_value(unpacked.get("metrics", {})),
        contributing_peers=_deserialize_value(unpacked.get("contributing_peers", [])),
        checkpoint_id=_validate_checkpoint_id(unpacked["checkpoint_id"]),
        checksum=unpacked.get("checksum", ""),
        size_bytes=unpacked.get("size_bytes", 0),
    )


def _checkpoint_to_metadata(checkpoint: ModelCheckpoint, size_bytes: int = 0) -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_id=checkpoint.checkpoint_id,
        round_number=checkpoint.round_number,
        timestamp=checkpoint.timestamp,
        metrics=checkpoint.metrics,
        contributing_peers=checkpoint.contributing_peers,
        checksum=checkpoint.checksum,
        size_bytes=size_bytes,
    )

def _deserialize_checkpoint(data: bytes, compressed: bool = False) -> ModelCheckpoint:
    """Deserialize bytes to a checkpoint using msgpack."""
    try:
        if compressed:
            data = zlib.decompress(data)

        unpacked = msgpack.unpackb(data, raw=False)

        # Checksum is optional for backward compatibility
        checksum = unpacked.get("checksum", "")

        return ModelCheckpoint(
            round_number=unpacked["round_number"],
            timestamp=_deserialize_value(unpacked["timestamp"]),
            metrics=unpacked["metrics"],
            contributing_peers=unpacked["contributing_peers"],
            checkpoint_id=unpacked["checkpoint_id"],
            checksum=checksum,
            weights=_deserialize_value(unpacked["weights"])
        )
    except Exception as e:
        raise ModelStoreError(f"Failed to deserialize checkpoint: {e}")

# Main ModelStore Class
class ModelStore:
    """
    Handles model storage and versioning.

    Supports in-memory and disk-based storage with thread-safe operations.

    Args:
        storage_dir: Directory for disk-based storage (None = memory only)
        keep_in_memory: Whether to keep all checkpoints in memory
        compression: Enable zlib compression for disk storage (default: False)
        max_memory_checkpoints: Max checkpoints to keep in memory (0 = unlimited)
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        keep_in_memory: bool = True,
        compression: bool = False,
        max_memory_checkpoints: int = 0
    ):
        """
        Initialize the model store.

        Args:
            storage_dir: Directory for disk-based storage (None = memory only)
            keep_in_memory: Whether to keep all checkpoints in memory
            compression: Enable zlib compression (default: False for backward compatibility)
            max_memory_checkpoints: Max checkpoints in memory (0 = unlimited)
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.keep_in_memory = keep_in_memory
        self.compression = compression
        self.max_memory_checkpoints = max_memory_checkpoints

        # In-memory storage
        self._checkpoints: Dict[str, ModelCheckpoint] = {}
        self._round_index: Dict[int, str] = {}  # round_number -> checkpoint_id

        # Thread safety
        self._lock = threading.RLock()

        # Cache for list_checkpoints to avoid repeated disk scans
        self._list_cache: Optional[List[ModelCheckpoint]] = None
        self._cache_dirty: bool = True
        self._metadata_cache: Optional[List[CheckpointMetadata]] = None
        self._storage_size_cache: Optional[int] = None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Model store initialized with storage dir: {self.storage_dir}")

    # Helper Methods (reduce code duplication)
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint (handles compression)."""
        checkpoint_id = _validate_checkpoint_id(checkpoint_id)
        ext = ".msgpack.gz" if self.compression else ".msgpack"
        return self.storage_dir / f"checkpoint_{checkpoint_id}{ext}"

    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        checkpoint_id = _validate_checkpoint_id(checkpoint_id)
        return self.storage_dir / f"checkpoint_{checkpoint_id}.meta.msgpack"

    def _checkpoint_id_from_path(self, filepath: Path) -> str:
        name = filepath.name
        if name.endswith(".msgpack.gz"):
            checkpoint_id = name[len("checkpoint_"):-len(".msgpack.gz")]
        elif name.endswith(".msgpack"):
            checkpoint_id = name[len("checkpoint_"):-len(".msgpack")]
        elif name.endswith(".pkl"):
            checkpoint_id = name[len("checkpoint_"):-len(".pkl")]
        else:
            raise ValueError(f"Unsupported checkpoint filename: {name}")
        return _validate_checkpoint_id(checkpoint_id)

    def _find_checkpoint_files(self) -> List[Path]:
        """Find all checkpoint files on disk (handles multiple formats)."""
        if not self.storage_dir:
            return []

        files: List[Path] = []
        # New formats with compression
        if self.compression:
            files.extend(self.storage_dir.glob("checkpoint_*.msgpack.gz"))
        files.extend(
            filepath
            for filepath in self.storage_dir.glob("checkpoint_*.msgpack")
            if not filepath.name.endswith(".meta.msgpack")
        )
        # Old format (for migration warning only)
        files.extend(self.storage_dir.glob("checkpoint_*.pkl"))
        return files

    def _invalidate_cache(self) -> None:
        """Invalidate the list cache."""
        self._cache_dirty = True
        self._metadata_cache = None
        self._storage_size_cache = None

    def _write_bytes_atomically(self, filepath: Path, data: bytes) -> None:
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=str(filepath.parent),
                prefix=f".{filepath.name}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, filepath)
        except Exception:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
            raise

    def _save_metadata_to_disk(self, checkpoint: ModelCheckpoint, size_bytes: int) -> None:
        metadata = _checkpoint_to_metadata(checkpoint, size_bytes=size_bytes)
        metadata_path = self._get_metadata_path(checkpoint.checkpoint_id)
        self._write_bytes_atomically(metadata_path, _serialize_checkpoint_metadata(metadata))

    def _load_metadata_from_disk(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        metadata_path = self._get_metadata_path(checkpoint_id)
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    return _deserialize_checkpoint_metadata(f.read())
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata for {checkpoint_id}: {e}")

        checkpoint = self._load_from_disk(checkpoint_id)
        if checkpoint is None:
            return None

        size_bytes = 0
        filepath = self._get_checkpoint_path(checkpoint_id)
        if filepath.exists():
            size_bytes = filepath.stat().st_size

        metadata = _checkpoint_to_metadata(checkpoint, size_bytes=size_bytes)
        try:
            self._save_metadata_to_disk(checkpoint, size_bytes)
        except Exception as e:
            logger.warning(f"Failed to rebuild checkpoint metadata for {checkpoint_id}: {e}")
        return metadata

    def _enforce_memory_limit(self) -> None:
        """Enforce max memory checkpoints by removing oldest."""
        if self.max_memory_checkpoints <= 0:
            return

        while len(self._checkpoints) > self.max_memory_checkpoints:
            oldest = min(self._checkpoints.values(), key=lambda c: c.round_number)
            del self._checkpoints[oldest.checkpoint_id]
            self._round_index.pop(oldest.round_number, None)
            logger.debug(f"Evicted checkpoint from memory: {oldest.checkpoint_id}")

    # Public API (backward compatible - same method signatures)
    def save_checkpoint(
        self,
        round_number: int,
        weights: Any,
        metrics: Optional[Dict[str, float]] = None,
        contributing_peers: Optional[List[str]] = None
    ) -> ModelCheckpoint:
        """
        Save a model checkpoint (thread-safe).

        Args:
            round_number: Training round number
            weights: Model weights
            metrics: Optional metrics (loss, accuracy, etc.)
            contributing_peers: List of peer IDs that contributed

        Returns:
            ModelCheckpoint instance
        """
        with self._lock:
            checkpoint = ModelCheckpoint(
                round_number=round_number,
                weights=weights,
                metrics=metrics or {},
                contributing_peers=contributing_peers or []
            )

            if self.storage_dir:
                size_bytes = self._save_to_disk(checkpoint)
                try:
                    self._save_metadata_to_disk(checkpoint, size_bytes)
                except Exception as e:
                    logger.warning(f"Failed to persist checkpoint metadata for {checkpoint.checkpoint_id}: {e}")

            self._invalidate_cache()
            self._round_index[round_number] = checkpoint.checkpoint_id

            if self.keep_in_memory:
                self._checkpoints[checkpoint.checkpoint_id] = checkpoint
                self._enforce_memory_limit()

            logger.debug(f"Saved checkpoint for round {round_number}: {checkpoint.checkpoint_id}")
            return checkpoint

    async def save_checkpoint_async(
        self,
        round_number: int,
        weights: Any,
        metrics: Optional[Dict[str, float]] = None,
        contributing_peers: Optional[List[str]] = None,
    ) -> ModelCheckpoint:
        return await asyncio.to_thread(
            self.save_checkpoint,
            round_number,
            weights,
            metrics,
            contributing_peers,
        )

    def _save_to_disk(self, checkpoint: ModelCheckpoint) -> int:
        """Save checkpoint to disk using safe msgpack serialization."""
        filepath = self._get_checkpoint_path(checkpoint.checkpoint_id)

        try:
            serialized = _serialize_checkpoint(checkpoint, compress=self.compression)
            self._write_bytes_atomically(filepath, serialized)
            return len(serialized)
        except Exception as e:
            raise ModelStoreError(f"Failed to save checkpoint to disk: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        """
        Load a checkpoint by ID (thread-safe).

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            ModelCheckpoint or None if not found
        """
        with self._lock:
            checkpoint_id = _validate_checkpoint_id(checkpoint_id)

            # Check memory first
            if checkpoint_id in self._checkpoints:
                return self._checkpoints[checkpoint_id]

            # Check disk
            if self.storage_dir:
                return self._load_from_disk(checkpoint_id)

            return None

    async def load_checkpoint_async(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        return await asyncio.to_thread(self.load_checkpoint, checkpoint_id)

    def _load_from_disk(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        """Load checkpoint from disk using safe msgpack deserialization."""
        checkpoint_id = _validate_checkpoint_id(checkpoint_id)

        # Try new compressed format first
        if self.compression:
            filepath = self.storage_dir / f"checkpoint_{checkpoint_id}.msgpack.gz"
            if filepath.exists():
                return self._load_from_file(filepath, compressed=True)

        # Try standard msgpack format
        filepath = self.storage_dir / f"checkpoint_{checkpoint_id}.msgpack"
        if filepath.exists():
            return self._load_from_file(filepath, compressed=False)

        # Fall back to old pickle format (warning only, don't load for security)
        old_filepath = self.storage_dir / f"checkpoint_{checkpoint_id}.pkl"
        if old_filepath.exists():
            logger.warning(
                f"Old pickle format found for {checkpoint_id}. "
                f"Please migrate checkpoints. Not loading for security reasons."
            )

        return None

    def _load_from_file(self, filepath: Path, compressed: bool = False) -> Optional[ModelCheckpoint]:
        """Load checkpoint from a specific file."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()

            checkpoint = _deserialize_checkpoint(data, compressed=compressed)

            # Verify checksum if present
            if checkpoint.checksum and not checkpoint.verify_checksum():
                logger.warning(f"Checksum mismatch for checkpoint {checkpoint.checkpoint_id}")

            self._round_index[checkpoint.round_number] = checkpoint.checkpoint_id

            if self.keep_in_memory:
                self._checkpoints[checkpoint.checkpoint_id] = checkpoint

            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint from disk: {e}")
            return None

    def get_checkpoint_by_round(self, round_number: int) -> Optional[ModelCheckpoint]:
        """
        Get the checkpoint for a specific round (thread-safe).

        Args:
            round_number: Round number

        Returns:
            ModelCheckpoint or None if not found
        """
        with self._lock:
            checkpoint_id = self._round_index.get(round_number)
            if checkpoint_id:
                return self.load_checkpoint(checkpoint_id)

            # Search disk if not in index
            if self.storage_dir:
                for filepath in self._find_checkpoint_files():
                    try:
                        checkpoint_id = self._checkpoint_id_from_path(filepath)
                    except ValueError:
                        continue
                    checkpoint = self._load_from_disk(checkpoint_id)
                    if checkpoint and checkpoint.round_number == round_number:
                        self._round_index[round_number] = checkpoint.checkpoint_id
                        return checkpoint

            return None

    async def get_checkpoint_by_round_async(self, round_number: int) -> Optional[ModelCheckpoint]:
        return await asyncio.to_thread(self.get_checkpoint_by_round, round_number)

    def get_latest_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Get the most recent checkpoint (thread-safe)."""
        with self._lock:
            if not self._round_index:
                return None

            latest_round = max(self._round_index.keys())
            checkpoint_id = self._round_index.get(latest_round)
            if checkpoint_id:
                return self._checkpoints.get(checkpoint_id) or self._load_from_disk(checkpoint_id)

            return None

    async def get_latest_checkpoint_async(self) -> Optional[ModelCheckpoint]:
        return await asyncio.to_thread(self.get_latest_checkpoint)

    def list_checkpoints(self) -> List[ModelCheckpoint]:
        """
        List all stored checkpoints with caching (thread-safe).

        Returns:
            List of ModelCheckpoint objects sorted by round number
        """
        with self._lock:
            # Use cache if available and not dirty
            if not self._cache_dirty and self._list_cache is not None:
                return self._list_cache.copy()

            checkpoints = []

            # Add memory checkpoints
            if self.keep_in_memory:
                checkpoints.extend(self._checkpoints.values())

            # Add disk checkpoints if not in memory
            if self.storage_dir:
                for filepath in self._find_checkpoint_files():
                    try:
                        checkpoint_id = self._checkpoint_id_from_path(filepath)
                    except ValueError:
                        continue
                    # Skip .pkl files (not loaded for security)
                    if filepath.suffix == ".pkl":
                        continue
                    if checkpoint_id not in self._checkpoints:
                        checkpoint = self._load_from_disk(checkpoint_id)
                        if checkpoint:
                            checkpoints.append(checkpoint)

            # Sort by round number
            checkpoints.sort(key=lambda c: c.round_number)

            # Update cache
            self._list_cache = checkpoints
            self._cache_dirty = False

            return checkpoints.copy()

    def list_checkpoint_metadata(self) -> List[CheckpointMetadata]:
        with self._lock:
            if not self._cache_dirty and self._metadata_cache is not None:
                return self._metadata_cache.copy()

            metadata_entries: List[CheckpointMetadata] = []
            seen_checkpoint_ids = set()

            if self.keep_in_memory:
                for checkpoint in self._checkpoints.values():
                    size_bytes = 0
                    if self.storage_dir:
                        filepath = self._get_checkpoint_path(checkpoint.checkpoint_id)
                        if filepath.exists():
                            size_bytes = filepath.stat().st_size
                    metadata_entries.append(_checkpoint_to_metadata(checkpoint, size_bytes=size_bytes))
                    seen_checkpoint_ids.add(checkpoint.checkpoint_id)

            if self.storage_dir:
                for filepath in self._find_checkpoint_files():
                    try:
                        checkpoint_id = self._checkpoint_id_from_path(filepath)
                    except ValueError:
                        continue
                    if filepath.suffix == ".pkl" or checkpoint_id in seen_checkpoint_ids:
                        continue
                    metadata = self._load_metadata_from_disk(checkpoint_id)
                    if metadata is not None:
                        if metadata.size_bytes == 0 and filepath.exists():
                            metadata.size_bytes = filepath.stat().st_size
                        metadata_entries.append(metadata)

            metadata_entries.sort(key=lambda c: c.round_number)
            self._metadata_cache = metadata_entries
            self._cache_dirty = False
            return metadata_entries.copy()

    async def list_checkpoint_metadata_async(self) -> List[CheckpointMetadata]:
        return await asyncio.to_thread(self.list_checkpoint_metadata)

    async def list_checkpoints_async(self) -> List[ModelCheckpoint]:
        return await asyncio.to_thread(self.list_checkpoints)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint (thread-safe).

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            checkpoint_id = _validate_checkpoint_id(checkpoint_id)
            found = False

            # Remove from memory
            if checkpoint_id in self._checkpoints:
                checkpoint = self._checkpoints.pop(checkpoint_id)
                self._round_index.pop(checkpoint.round_number, None)
                found = True

            # Invalidate cache
            self._invalidate_cache()

            # Remove from disk (try all formats)
            if self.storage_dir:
                for ext in [".msgpack.gz", ".msgpack", ".pkl"]:
                    filepath = self.storage_dir / f"checkpoint_{checkpoint_id}{ext}"
                    if filepath.exists():
                        filepath.unlink()
                        found = True
                metadata_path = self._get_metadata_path(checkpoint_id)
                if metadata_path.exists():
                    metadata_path.unlink()
                    found = True

            return found

    async def delete_checkpoint_async(self, checkpoint_id: str) -> bool:
        return await asyncio.to_thread(self.delete_checkpoint, checkpoint_id)

    def clear_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Remove old checkpoints, keeping only the most recent N (thread-safe).

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        with self._lock:
            checkpoints = self.list_checkpoint_metadata()

            if len(checkpoints) <= keep_last_n:
                return

            # Remove oldest checkpoints
            to_delete = checkpoints[:-keep_last_n]
            for checkpoint in to_delete:
                self.delete_checkpoint(checkpoint.checkpoint_id)

            logger.info(f"Cleared {len(to_delete)} old checkpoints, kept {keep_last_n}")

    async def clear_old_checkpoints_async(self, keep_last_n: int = 5) -> None:
        await asyncio.to_thread(self.clear_old_checkpoints, keep_last_n)

    def get_storage_size(self) -> int:
        """
        Get the total size of stored checkpoints in bytes (thread-safe).

        Returns:
            Total size in bytes
        """
        with self._lock:
            if not self.storage_dir:
                # Estimate from in-memory checkpoints
                return sum(
                    len(_serialize_checkpoint(c, compress=self.compression))
                    for c in self._checkpoints.values()
                )

            if self._storage_size_cache is not None:
                return self._storage_size_cache

            total = 0
            for metadata in self.list_checkpoint_metadata():
                total += metadata.size_bytes
            self._storage_size_cache = total
            return total

    async def get_storage_size_async(self) -> int:
        return await asyncio.to_thread(self.get_storage_size)
