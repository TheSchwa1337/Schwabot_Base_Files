#!/usr/bin/env python3
"""Spectral lattice generator for quantised wave analysis.

The purpose of :func:`wave_lattice_generator` is to project an input signal onto
an FFT basis, quantise the spectrum into a *lattice grid* and return both the
quantised magnitude matrix and the frequency bins.  Down-stream GAN or pattern
matching modules can then perform discrete neighbour look-ups instead of costly
continuous interpolation.

Current implementation
----------------------
1. Performs an FFT (or rFFT for real input).
2. Splits the spectrum's absolute magnitude into *levels* logarithmic bins.
3. Quantises each bin into an integer cell on a 2-D lattice *(level, index)*.

This is intentionally lightweight.  Later extensions can add wavelet tiling or
non-uniform lattices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["wave_lattice_generator"]


def _logspace_levels(magnitudes: np.ndarray, levels: int) -> np.ndarray:
    """Compute logarithmic thresholds for *levels* bins."""
    mag_nonzero = magnitudes[magnitudes > 0]
    if mag_nonzero.size == 0:
        return np.zeros(levels + 1)
    mag_min = float(np.min(mag_nonzero))
    mag_max = float(np.max(magnitudes))
    return np.logspace(np.log10(mag_min), np.log10(mag_max), num=levels + 1)


def wave_lattice_generator(
    signal: np.ndarray,
    *,
    levels: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project *signal* onto a logarithmic spectral lattice.

    Parameters
    ----------
    signal
        1-D NumPy array containing the raw waveform.
    levels
        Number of logarithmic magnitude bins.  Defaults to **3**.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(lattice, freqs)`` where *lattice* is an integer matrix of shape
        ``(levels, n_bins)`` holding quantised magnitude tier indices (0 / 1),
        and *freqs* is the 1-D array of FFT frequency bins (Hz units with
        normalised sampling of 1.0).
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if levels < 1:
        raise ValueError("levels must be ≥ 1")

    # Compute FFT spectrum (real input ⇒ rfft).
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0)  # assume unit sampling interval
    magnitudes = np.abs(spectrum)

    # Build lattice grid: levels × n_bins boolean/int matrix.
    thresholds = _logspace_levels(magnitudes, levels)
    lattice = np.zeros((levels, magnitudes.size), dtype=int)

    for lvl in range(levels):
        lo = thresholds[lvl]
        hi = thresholds[lvl + 1]
        mask = (magnitudes >= lo) & (magnitudes < hi)
        lattice[lvl, mask] = 1

    return lattice, freqs 