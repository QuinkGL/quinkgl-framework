"""Architecture-only model fingerprinting (spec §10.5.7).

:func:`compute_arch_hash` reduces a PyTorch ``nn.Module`` or TensorFlow
``tf.keras.Model`` to a device-independent ``sha256:<64-hex>`` string that
encodes only the *architecture* (parameter names, shapes, dtypes) — never the
weights, never the device, never gradients.

Rationale: two peers that hold the same model definition but were trained on
different devices or hardware MUST produce byte-identical architecture
hashes so that ``manifest.model.arch_hash`` can be used as an enrolment
gate (``ERR_NODE_ARCH_MISMATCH``).  Weight divergence is expected; it must
not poison the fingerprint.

The framework is detected by duck-typing the input; callers MAY pass any
object that quacks like either framework's model base class.  Unsupported
inputs raise :class:`TypeError` so that validation bubbles up cleanly instead
of silently producing a degenerate hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Sequence, Tuple

__all__ = ["compute_arch_hash"]


_HASH_PREFIX = "sha256:"


def compute_arch_hash(model: Any) -> str:
    """Return ``sha256:<64-hex>`` fingerprint of ``model``'s architecture.

    Supported inputs:

    * ``torch.nn.Module`` subclasses — fingerprint built from
      ``named_parameters()`` and ``named_buffers()``.
    * ``tf.keras.Model`` subclasses — fingerprint built from the ordered
      ``weights`` iterable (trainable + non-trainable).

    Anything else raises :class:`TypeError`.

    The fingerprint comprises three fields per tensor: name, dtype string,
    and integer shape tuple.  Device placement is ignored.  Weight values
    are ignored.
    """

    entries = _fingerprint_entries(model)
    canonical = json.dumps(
        {"framework": entries[0], "tensors": entries[1]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Framework dispatch
# ---------------------------------------------------------------------------


def _fingerprint_entries(model: Any) -> Tuple[str, List[Tuple[str, str, List[int]]]]:
    """Return ``(framework_tag, [(name, dtype_str, shape), ...])``.

    The list is sorted by ``name`` so that iteration order inside the
    framework cannot influence the hash.
    """
    if model is None:
        raise TypeError(
            "compute_arch_hash: model is None; pass an nn.Module or tf.keras.Model."
        )

    if _looks_like_torch_module(model):
        return "pytorch", _pytorch_entries(model)

    if _looks_like_keras_model(model):
        return "tensorflow", _keras_entries(model)

    raise TypeError(
        "compute_arch_hash: unsupported model type "
        f"{type(model).__module__}.{type(model).__name__}; expected a "
        "torch.nn.Module or tf.keras.Model."
    )


def _looks_like_torch_module(obj: Any) -> bool:
    try:
        import torch.nn as _nn
    except Exception:
        return False
    return isinstance(obj, _nn.Module)


def _looks_like_keras_model(obj: Any) -> bool:
    try:
        import tensorflow as _tf
    except Exception:
        return False
    try:
        return isinstance(obj, _tf.keras.Model)
    except Exception:  # pragma: no cover — TF present but keras import failed
        return False


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------


def _pytorch_entries(model: Any) -> List[Tuple[str, str, List[int]]]:
    """Sorted ``(name, dtype, shape)`` triples for every param + buffer.

    Buffers are included because architectural choices like BatchNorm
    running statistics change the declared parameter set (§10.5.7 expects
    a complete structural fingerprint).  Device is explicitly ignored.
    """
    entries: List[Tuple[str, str, List[int]]] = []
    for name, tensor in model.named_parameters(recurse=True):
        entries.append((name, str(tensor.dtype), list(tensor.shape)))
    for name, tensor in model.named_buffers(recurse=True):
        # Prefix buffer names to avoid collision with parameter names that
        # happen to share a path component.
        entries.append((f"__buffer__.{name}", str(tensor.dtype), list(tensor.shape)))
    entries.sort(key=lambda e: e[0])
    return entries


# ---------------------------------------------------------------------------
# TensorFlow / Keras
# ---------------------------------------------------------------------------


def _keras_entries(model: Any) -> List[Tuple[str, str, List[int]]]:
    """Sorted ``(name, dtype, shape)`` triples for every Keras weight."""
    weights: Sequence[Any] = list(model.weights)
    entries: List[Tuple[str, str, List[int]]] = []
    for w in weights:
        name = getattr(w, "name", "")
        dtype = str(getattr(w, "dtype", "unknown"))
        shape = getattr(w, "shape", None)
        # Keras shapes are ``TensorShape``; convert to a plain list with
        # ``None`` entries collapsed to 0 so that the canonical JSON is
        # JSON-safe.  A ``None`` dim on a keras weight would be unusual
        # (weights are usually fully-defined once the model is built).
        shape_list = (
            [int(d) if d is not None else 0 for d in shape]  # type: ignore[union-attr]
            if shape is not None
            else []
        )
        entries.append((name, dtype, shape_list))
    entries.sort(key=lambda e: e[0])
    return entries
