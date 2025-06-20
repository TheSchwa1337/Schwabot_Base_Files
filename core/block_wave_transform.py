#!/usr/bin/env python3
"""Block-wise wave transform utilities.

This module provides a minimal, *working* implementation of
``define_block_wave_transform`` – a helper that will be used by Schwabot's
signal-compression and GAN-preprocessing stack.

The routine currently supports a **block-wise DCT-II** (via ``scipy.fftpack`` if
available, else falls back to NumPy's FFT) and returns the transformed signal
along with per-block Shannon entropy.  The advanced lattice / entropy gates can
be layered on top later, but this is more than enough to satisfy imports and
pass Flake 8.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    # SciPy gives us a proper DCT-II.
    from scipy.fftpack import dct  # type: ignore

    def _dct_block(arr: np.ndarray) -> np.ndarray:  # noqa: D401
        return dct(arr, type=2, norm="ortho")

except ModuleNotFoundError:  # pragma: no cover – keep pure-NumPy fallback

    def _dct_block(arr: np.ndarray) -> np.ndarray:  # noqa: D401
        """Fallback: approximate DCT-II via real FFT symmetry trick."""
        n = arr.shape[-1]
        extended = np.concatenate([arr, arr[..., ::-1]], axis=-1)
        spectrum = np.fft.rfft(extended)
        return np.real_if_close(spectrum[..., :n])


__all__ = ["define_block_wave_transform"]


def _shannon_entropy(block: np.ndarray) -> float:
    """Compute Shannon entropy of a 1-D vector (base-2)."""
    hist, _ = np.histogram(block, bins=32, density=True)
    # Filter zero probabilities to avoid log2(0).
    p = hist[hist > 0]
    return float(-np.sum(p * np.log2(p)))


def define_block_wave_transform(signal: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a block-wise DCT transform and return entropy per block.

    Parameters
    ----------
    signal
        1-D NumPy array containing the raw waveform.
    block_size
        Number of samples per block.  Must evenly divide ``signal.size``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(transformed, entropy)`` where *transformed* is the concatenated DCT
        coefficients and *entropy* is a vector of Shannon entropies for each
        block.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if signal.size % block_size != 0:
        raise ValueError("block_size must divide signal length")

    # Reshape into (num_blocks, block_size)
    blocks = signal.reshape(-1, block_size)

    # Transform each block.
    transformed_blocks = _dct_block(blocks)

    # Compute entropy per block.
    entropies = np.apply_along_axis(_shannon_entropy, 1, transformed_blocks)

    # Flatten transformed back to 1-D for convenience.
    return transformed_blocks.ravel(), entropies 