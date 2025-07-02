#!/usr/bin/env python3
"""Guard tests for unified math backfill functions."""

from core.unified_math_system import (
    compute_unified_entropy,
    compute_unified_drift_field,
    generate_unified_hash,
)


def test_entropy():
    p = [0.5, 0.5]
    assert abs(compute_unified_entropy(p) - 1.0) < 1e-6


def test_drift():
    assert compute_unified_drift_field(2.0, 4.0, 6.0, 8.0) == 5.0


def test_hash_length():
    h = generate_unified_hash([1.2345, 2.3456], "t1")
    assert isinstance(h, str) and len(h) == 64
