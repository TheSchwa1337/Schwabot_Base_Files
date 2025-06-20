#!/usr/bin/env python3
"""Entropy-based signal validator.

This module exposes :func:`validate_entropy_envelope` – a helper that checks
whether a signal (vector) lies inside an acceptable Shannon- or spectral-entropy
band.  GAN filters and anomaly detectors will use this as a quick-reject gate
before engaging heavier models.

Features implemented now
------------------------
1. **Spectral-entropy** via Welch PSD (SciPy optional, NumPy fallback).
2. User-supplied *min_entropy* / *max_entropy* band with sane defaults.
3. Returns a boolean *is_valid* **and** the computed entropy so callers can
   adaptively tune thresholds.

The implementation is intentionally lightweight to pass Flake8; deeper
statistical tests (Jensen-Shannon divergence, permutation entropy) can be added
later under the marked TODO sections.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy.signal import welch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – pure-NumPy fallback

    def welch(x: np.ndarray, *, fs: float = 1.0, nperseg: int | None = None):  # type: ignore  # noqa: D401
        """Rudimentary Welch PSD replacement (Hann + overlap=0)."""
        if nperseg is None:
            nperseg = min(256, x.size)
        window = np.hanning(nperseg)
        num_segments = x.size // nperseg
        if num_segments == 0:
            raise ValueError("nperseg larger than signal length")
        psd_acc = np.zeros(nperseg // 2 + 1)
        for i in range(num_segments):
            seg = x[i * nperseg : (i + 1) * nperseg]
            seg = seg * window
            spec = np.fft.rfft(seg)
            psd_acc += (np.abs(spec) ** 2) / (np.sum(window**2) * fs)
        psd = psd_acc / num_segments
        freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
        return freqs, psd

__all__ = ["validate_entropy_envelope"]


def _spectral_entropy(signal: np.ndarray, *, fs: float = 1.0) -> float:
    """Compute spectral entropy (base-2) of a 1-D real signal."""
    freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def validate_entropy_envelope(
    signal: np.ndarray,
    *,
    fs: float = 1.0,
    min_entropy: float = 2.0,
    max_entropy: float = 8.0,
) -> Tuple[bool, float]:
    """Validate a waveform's entropy against an allowed envelope.

    Parameters
    ----------
    signal
        1-D NumPy array of the raw or transformed signal.
    fs
        Sampling frequency (Hz).  Only used if *scipy* is present for PSD
        estimation; default **1.0** suffices for unit-less data.
    min_entropy / max_entropy
        Inclusive bounds for acceptable entropy.  Defaults are chosen to be
        lax and should be tuned by the caller.

    Returns
    -------
    Tuple[bool, float]
        ``(is_valid, entropy)`` where *is_valid* is ``True`` if the spectral
        entropy lies inside the given envelope.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    entropy = _spectral_entropy(signal, fs=fs)
    return (min_entropy <= entropy <= max_entropy), entropy

# -----------------------------------------------------------------------------
# TODO: future improvements
# -----------------------------------------------------------------------------
# • Add permutation entropy
# • Add Jensen–Shannon divergence to a reference distribution
# • Dynamic threshold adaptation based on rolling statistics 