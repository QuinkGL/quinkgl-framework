"""Architecture-hash computation (spec §10.5.7).

`compute_arch_hash(model)` must produce a deterministic, device-independent
fingerprint of a model's *architecture* (parameter names, shapes, dtypes) —
not its weights.  Matching `manifest.model.arch_hash` lets nodes fail fast at
`node.start()` via `ERR_NODE_ARCH_MISMATCH`.
"""

from __future__ import annotations

import re

import pytest

from quinkgl.manifest import compute_arch_hash

torch = pytest.importorskip("torch")
nn = torch.nn


ARCH_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


# --- Simple fixtures --------------------------------------------------------


def _small_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )


def _small_mlp_extra_layer() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )


def _small_mlp_wider() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
    )


# --- Format -----------------------------------------------------------------


class TestFormat:
    def test_hash_matches_required_regex(self):
        h = compute_arch_hash(_small_mlp())
        assert ARCH_HASH_RE.match(h), h

    def test_hash_is_str(self):
        assert isinstance(compute_arch_hash(_small_mlp()), str)


# --- Determinism ------------------------------------------------------------


class TestDeterminism:
    def test_same_architecture_same_hash(self):
        torch.manual_seed(0)
        a = _small_mlp()
        torch.manual_seed(1234)  # different weight init
        b = _small_mlp()
        # Different weights, identical architecture → identical hash.
        assert compute_arch_hash(a) == compute_arch_hash(b)

    def test_hash_stable_across_calls(self):
        m = _small_mlp()
        h0 = compute_arch_hash(m)
        for _ in range(25):
            assert compute_arch_hash(m) == h0


# --- Sensitivity to architecture ------------------------------------------


class TestSensitivity:
    def test_extra_linear_layer_changes_hash(self):
        assert compute_arch_hash(_small_mlp()) != compute_arch_hash(_small_mlp_extra_layer())

    def test_wider_hidden_changes_hash(self):
        assert compute_arch_hash(_small_mlp()) != compute_arch_hash(_small_mlp_wider())

    def test_dtype_change_changes_hash(self):
        m1 = _small_mlp()
        m2 = _small_mlp()
        m2 = m2.to(torch.float64)
        assert compute_arch_hash(m1) != compute_arch_hash(m2)


# --- Device / weight independence ------------------------------------------


class TestInvariants:
    def test_weight_mutation_does_not_change_hash(self):
        m = _small_mlp()
        h0 = compute_arch_hash(m)
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(100.0).add_(1.0)
        assert compute_arch_hash(m) == h0

    def test_cpu_device_does_not_leak_into_hash(self):
        """The canonical representation must not encode device strings."""
        m = _small_mlp()
        h0 = compute_arch_hash(m)
        # Move to a "meta" device (always available, zero-memory) — shape
        # and dtype preserved, storage replaced.  Hash must not change.
        m_meta = _small_mlp().to("meta")
        assert compute_arch_hash(m_meta) == h0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_cuda_device_parity(self):
        m_cpu = _small_mlp()
        m_gpu = _small_mlp().cuda()
        assert compute_arch_hash(m_cpu) == compute_arch_hash(m_gpu)


# --- Unsupported inputs -----------------------------------------------------


class TestUnsupported:
    def test_none_raises(self):
        with pytest.raises(TypeError):
            compute_arch_hash(None)

    def test_plain_object_raises(self):
        class Bare:
            pass

        with pytest.raises(TypeError):
            compute_arch_hash(Bare())
