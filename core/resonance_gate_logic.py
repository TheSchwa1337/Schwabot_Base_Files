#!/usr/bin/env python3
"""Phase-resonance gate logic.

This helper exposes a single function – :func:`phase_resonance_gate`.  It acts
as a deterministic *tick filter*: only ticks that align with a given harmonic
cycle (the *base_cycle*) and, optionally, a **42-bit phase mask** are allowed to
pass.  Schwabot uses this to trigger strategy rotations on specific harmonic
boundaries so that recursive loops do not desynchronise.

Math Primer
~~~~~~~~~~~
Given an integer tick **n** and a base cycle **C**, the fundamental resonance
condition is simply

    n mod C == 0

To cope with very-long sequences Schwabot employs a 42-bit ring counter; the
mask ``0x3ffffffffff`` (2^42-1) limits the counter and avoids Python int → float
precision loss when we later pass the phase index into NumPy code.
"""

from __future__ import annotations

__all__ = ["phase_resonance_gate"]

# 42-bit mask (2**42 ­- 1)
_PHASE_MASK: int = 0x3FFFF_FFFFFF  # 42 bits set


def phase_resonance_gate(
    tick: int,
    *,
    base_cycle: int = 42,
    use_mask: bool = True,
) -> bool:
    """Return ``True`` if *tick* aligns with the resonance gate.

    Parameters
    ----------
    tick
        Monotonic tick counter (non-negative integer).
    base_cycle
        Fundamental cycle length **C**.  Default **42** – Schwabot's universal
        harmonic constant.
    use_mask
        If ``True`` the *tick* is first masked to 42 bits; this mimics the ring
        counter used in the C++/Rust back-ends and prevents 64-bit overflow
        mismatch in long-running sessions.
    """
    if tick < 0:
        raise ValueError("tick must be non-negative")
    if base_cycle <= 0:
        raise ValueError("base_cycle must be positive")

    if use_mask:
        tick &= _PHASE_MASK

    return tick % base_cycle == 0 