#!/usr/bin/env python3
"""Phase-hash switch helper.

This stub offers a deterministic *hash-phase gate* so legacy modules can
replace the placeholder previously located in the C++ back-end.  The idea is
simple: hash an integer tick (plus optional salt) and reduce it modulo a base
cycle.  When the phase is **0** the gate opens.

Current implementation
----------------------
• One public function :func:`phase_hash_gate`.
• SHA-256 hashing – cryptographically strong yet std-lib only.
• Fully typed, ≤ 79-char lines, Flake8-clean.
"""

from __future__ import annotations

import hashlib
from typing import Final

__all__: list[str] = ["phase_hash_gate"]

_BASE_CYCLE_FALLBACK: Final = 42  # Schwabot universal harmonic constant


def _hash_int(value: int, salt: str = "") -> int:
    """Return 256-bit hash of *value*||*salt* as an integer."""
    data = f"{value}{salt}".encode()
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def phase_hash_gate(
    tick: int,
    *,
    base_cycle: int = _BASE_CYCLE_FALLBACK,
    salt: str = "",
) -> bool:
    """Return ``True`` if *tick* hashes into phase **0** of *base_cycle*.

    Parameters
    ----------
    tick
        Monotonic tick counter (non-negative).  Converted to bytes before
        hashing.
    base_cycle
        Cycle length that defines the number of hash-phases.  Defaults to
        **42** in line with Schwabot's harmonic conventions.
    salt
        Optional extra entropy to decorrelate multiple parallel hash-gates.

    Notes
    -----
    • Uses SHA-256; swapping to Blake2 or SHA-3 later will not change the API.
    • Gate condition is simply ``hash(tick) mod base_cycle == 0``.
    """
    if tick < 0:
        raise ValueError("tick must be non-negative")
    if base_cycle <= 0:
        raise ValueError("base_cycle must be positive")

    hashed = _hash_int(tick, salt=salt)
    phase = hashed % base_cycle
    return phase == 0 