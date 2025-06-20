#!/usr/bin/env python3
"""Glyph phase anchor – map glyph hash to Ferris wheel tick phase.

Given the 42-tick Ferris wheel used in Schwabot, we often quantise down to
smaller sub-rings (default 16) for glyph anchoring.  The mapping is deterministic:

    phase_idx = int(sha256(hash_hex)[:8], 16) % wheel_size

Only the first 32 bits of the glyph hash are used so the operation is cheap.
"""

from __future__ import annotations

import hashlib
from typing import Final

__all__: list[str] = ["phase_anchor_index", "glyph_active_for_tick"]

_DEFAULT_WHEEL: Final = 16


def phase_anchor_index(glyph_hash: str, *, wheel_size: int = _DEFAULT_WHEEL) -> int:
    """Return deterministic phase index in ``[0, wheel_size)`` for *glyph_hash*."""
    if len(glyph_hash) != 64:
        raise ValueError("glyph_hash must be 64-char SHA-256 hex")
    first32 = glyph_hash[:8]
    idx = int(first32, 16) % wheel_size
    return idx


def glyph_active_for_tick(
    glyph_hash: str,
    tick: int,
    base_cycle: int = 42,
    wheel_size: int = _DEFAULT_WHEEL,
) -> bool:
    """Return True if *glyph* is active at *tick* according to phase anchor.

    Active when ``tick % base_cycle`` equals the glyph's phase index modulo the
    wheel size (sub-ring).
    """
    if tick < 0:
        raise ValueError("tick must be non-negative")
    phase_idx = phase_anchor_index(glyph_hash, wheel_size=wheel_size)
    return (tick % base_cycle) % wheel_size == phase_idx 