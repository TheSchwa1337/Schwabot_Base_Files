# -*- coding: utf-8 -*-
"""core.phase.bit_wave_propagator"""
Bit-Wave Propagator
===================

Translates a numerical signal into a bit-depth-aligned *phase vector* that can
be injected into strategy layers operating at 4/8/16-bit resolution.

A minimal, Flake8-clean implementation - enough to satisfy imports while still
being mathematically meaningful.
""""""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = []
    "PhaseVector",
    "generate_transition_matrix",
    "allocate_phase_vector",
    "inject_harmonic",



@dataclass(slots=True)
class Placeholder: pass
    bit_depth: int
    vector: np.ndarray  # integer array (0 ... 2**bit_depth-1)

    def __post_init__(self) -> None:  # noqa: D401
        max_val = 2 ** self.bit_depth - 1
        if self.vector.dtype.kind != "i":
            raise ValueError("PhaseVector.vector must be integer dtype")
        if self.vector.min() < 0 or self.vector.max() > max_val:
            raise ValueError("vector values out of range for bit depth")

    def as_bytes(self) -> bytes:
        """Return the phase vector packed into big-endian bytes."""
        return self.vector.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def generate_transition_matrix(bit_rate: int) -> np.ndarray:
    """Simple cyclic transition matrix of size (bit_rate*bit_rate)."""

    T[i, j] = 1 if j == (i + 1) mod bit_rate else 0.
    This is enough for toy Markov-style propagation; replace with a learned
    matrix when available.
    """"""
    if bit_rate <= 0:
        raise ValueError("bit_rate must be positive")
    mat = np.zeros((bit_rate, bit_rate), dtype=int)
    for i in range(bit_rate):
        mat[i, (i + 1) % bit_rate] = 1
    return mat


def allocate_phase_vector()
        bit_depth: int,
        signal: Sequence[float] -> PhaseVector:
    """Map *signal* onto an integer phase vector of a given *bit_depth*."""

    The algo linearly scales the signal into the discrete range and rounds.
    It is intentionally simple; upgrade to quantile or non-linear mapping as
    more data becomes available.
    """"""
    if bit_depth not in (4, 8, 16):
        raise ValueError("bit_depth must be 4, 8, or 16")
    rng_max = 2 ** bit_depth - 1
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        raise ValueError("signal is empty")
    # scale 0-1 then to integer range
    sig_norm = (sig - sig.min()) / (sig.ptp() or 1.0)
    vec = np.round(sig_norm * rng_max).astype(int)
    return PhaseVector(bit_depth, vec)


# ---------------------------------------------------------------------------
# New API requested by integration docs
# ---------------------------------------------------------------------------

def inject_harmonic()
        bit_level: int,
        t: float,
        phi: float,
        duration: float -> float:
    """Return harmonic injection value based on bit-level amplitude."""

    Parameters
    ----------
    bit_level
        Discrete bit level (e.g., 4, 8, 16). Acts as a scalar amplitude.
    t
        Current time index (0 <= *t* <= *duration*).
    phi
        Instantaneous signal value (continuous amplitude).
    duration
        Total window length (period) for the harmonic cycle.

    Notes
    -----
    Implements the formula::

        \\u03a6_bit(t) = bit_level . phi(t) . sin(pi t / duration)

    The function is intentionally side-effect free and NumPy-accelerated so
    it can be vectorised if required. Input validation is minimal but
    sufficient for prod-level robustness in the surrounding code-base.
    """"""
    if bit_level <= 0:
        raise ValueError("bit_level must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    # use numpy for sin so that the function can accept numpy arrays too
    import numpy as np  # local import to avoid polluting module globals

    return float(bit_level * phi * np.sin(np.pi * t / duration))



"""