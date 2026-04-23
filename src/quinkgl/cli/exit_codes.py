# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""Stable CLI exit code mapping (§11.11)."""

from __future__ import annotations

# §11.11 — ordered once, never reordered after release.
SUCCESS = 0
VALIDATION_ERROR = 1
IO_ERROR = 2
CRYPTO_ERROR = 3
TRUST_ERROR = 4
HASH_MISMATCH = 5
WIRE_ERROR = 6
NODE_CONFIG_ERROR = 7
INTERRUPTED = 130
