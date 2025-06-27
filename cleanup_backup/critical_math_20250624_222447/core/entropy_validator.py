# -*- coding: utf - 8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any

# -*- coding: utf - 8 -*-
from dual_unicore_handler import DualUnicoreHandler

from core.unified_math_system import unified_math


# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""Entropy - based signal validator."

This module exposes :func:`validate_entropy_envelope` \\u2013 a helper that checks
whether a signal (vector) lies inside an acceptable Shannon- or spectral - entropy
band.  GAN filters and anomaly detectors will use this as a quick - reject gate
before engaging heavier models.

Features implemented now
------------------------
1. **Spectral - entropy** via Welch PSD (SciPy optional, NumPy fallback).
2. User - supplied *min_entropy* / *max_entropy* band with sane defaults.
3. Returns a boolean *is_valid* **and** the computed entropy so callers can
adaptively tune thresholds.

The implementation is intentionally lightweight to pass Flake8; deeper
statistical tests (Jensen - Shannon divergence, permutation entropy) can be added
later under the marked TODO sections."""
""""""
""""""
"""


try:
    from scipy.signal import welch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover \\u2013 pure - NumPy fallback
pass  # TODO: Implement except block

def welch(x: np.ndarray, *, fs: float = 1.0, nperseg: int | None = None):  # type: ignore  # noqa: D401"""
        """Rudimentary Welch PSD replacement (Hann + overlap = 0).""""""
""""""
"""
if nperseg is None:
            nperseg = unified_math.min(256, x.size)
        window = np.hanning(nperseg)
        num_segments = x.size // nperseg
        if num_segments == 0:"""
            raise ValueError("nperseg larger than signal length")
        psd_acc = np.zeros(nperseg // 2 + 1)
        for i in range(num_segments):
            seg = x[i * nperseg: (i + 1) * nperseg]
            seg = seg * window
            spec = np.fft.rfft(seg)
            psd_acc += (unified_math.unified_math.abs(spec) ** 2) / (np.sum(window**2) * fs)
        psd = psd_acc / num_segments
        freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
        return freqs, psd


__all__ = ["validate_entropy_envelope"]


def _spectral_entropy(signal: np.ndarray, *, fs: float = 1.0) -> float:
    """Compute spectral entropy (base - 2) of a 1 - D real signal."""

"""
""""""
"""
freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # avoid unified_math.log(0)
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def validate_entropy_envelope()

signal: np.ndarray,
    *,
    fs: float = 1.0,
    min_entropy: float = 2.0,
    max_entropy: float = 8.0,
) -> Tuple[bool, float]:"""
    """Validate a waveform's entropy against an allowed envelope.'"

Parameters
----------
signal
1 - D NumPy array of the raw or transformed signal.
fs
Sampling frequency (Hz).  Only used if *scipy* is present for PSD
        estimation; default **1.0** suffices for unit - less data.
min_entropy / max_entropy
Inclusive bounds for acceptable entropy.  Defaults are chosen to be
lax and should be tuned by the caller.

Returns
-------
Tuple[bool, float]
        ``(is_valid, entropy)`` where *is_valid* is ``True`` if the spectral
        entropy lies inside the given envelope."""
"""

"""
""""""
"""
if signal.ndim != 1:"""
        raise ValueError("signal must be 1 - D")
    entropy = _spectral_entropy(signal, fs=fs)
    return (min_entropy <= entropy <= max_entropy), entropy


# -----------------------------------------------------------------------------
# DONE: future improvements implemented
# -----------------------------------------------------------------------------
# \\u2022 \\u2705 Add permutation entropy
# \\u2022 \\u2705 Add Jensen\\u2013Shannon divergence to a reference distribution
# \\u2022 \\u2705 Dynamic threshold adaptation based on rolling statistics

def _permutation_entropy(signal: np.ndarray, order: int = 3) -> float:
    """Compute permutation entropy of a signal."

Parameters
----------
signal : np.ndarray
1 - D signal array
order : int
Order of permutation (default: 3)

Returns
-------
float
Permutation entropy value"""
"""

"""
""""""
"""
if len(signal) < order + 1:
        return 0.0

# Generate all possible permutations
from itertools import permutations
all_permutations = list(permutations(range(order)))
    permutation_counts = {perm: 0 for perm in all_permutations}

# Count occurrences of each permutation
for i in range(len(signal) - order):
# Get the order of the current window
window = signal[i:i + order]
        sorted_indices = np.argsort(window)
        permutation = tuple(sorted_indices)
        if permutation in permutation_counts:
            permutation_counts[permutation] += 1

# Calculate entropy
total_windows = len(signal) - order
    if total_windows == 0:
        return 0.0

entropy = 0.0
    for count in permutation_counts.values():
        if count > 0:
            p = count / total_windows
            entropy -= p * np.log2(p)

return float(entropy)


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:"""
    """Compute Jensen - Shannon divergence between two distributions."

Parameters
----------
p, q : np.ndarray
        Probability distributions (must sum to 1)

Returns
-------
float
Jensen - Shannon divergence value"""
"""

"""
""""""
"""
# Ensure they are probability distributions
p = p / np.sum(p)
    q = q / np.sum(q)

# Compute midpoint
m = 0.5 * (p + q)

# Compute KL divergences
kl_pm = np.sum(p * np.log2(p / m + 1e - 10))
    kl_qm = np.sum(q * np.log2(q / m + 1e - 10))

# Jensen - Shannon divergence
return 0.5 * (kl_pm + kl_qm)


class AdaptiveEntropyValidator:
"""
"""Entropy validator with dynamic threshold adaptation."""

"""
""""""
"""

def __init__(self, window_size: int = 100):"""
        """Initialize adaptive validator."

Parameters
----------
window_size : int
Size of rolling window for statistics"""
""""""
""""""
"""
self.window_size = window_size
        self.entropy_history = []
        self.permutation_entropy_history = []
        self.reference_distribution = None

def update_reference_distribution(self, signal: np.ndarray) -> None:"""
    """Function implementation pending."""
pass
"""
"""Update reference distribution for Jensen - Shannon divergence."

Parameters
----------
signal : np.ndarray
Signal to use as reference"""
""""""
""""""
"""
freqs, psd = welch(signal, fs = 1.0)
        self.reference_distribution = psd / np.sum(psd)

def validate_adaptive()

self,
        signal: np.ndarray,
        *,
        fs: float = 1.0,
        min_entropy: float = 2.0,
        max_entropy: float = 8.0,
        use_permutation: bool = True,
        use_js_divergence: bool = True,
        js_threshold: float = 0.5,
    ) -> Dict[str, Any]:"""
        """Validate signal with multiple entropy measures and adaptive thresholds."

Parameters
----------
signal : np.ndarray
1 - D signal array
fs : float
Sampling frequency
min_entropy, max_entropy : float
            Spectral entropy bounds
use_permutation : bool
Whether to use permutation entropy
use_js_divergence : bool
Whether to use Jensen - Shannon divergence
js_threshold : float
Threshold for JS divergence

Returns
-------
Dict[str, Any]
            Validation results with multiple measures"""
""""""
""""""
"""
if signal.ndim != 1:"""
            raise ValueError("signal must be 1 - D")

# Calculate spectral entropy
spectral_entropy = _spectral_entropy(signal, fs = fs)
        spectral_valid = min_entropy <= spectral_entropy <= max_entropy

# Calculate permutation entropy
permutation_entropy = 0.0
        permutation_valid = True
        if use_permutation:
            permutation_entropy = _permutation_entropy(signal)
# Adaptive threshold for permutation entropy
if self.permutation_entropy_history:
                mean_perm = unified_math.unified_math.mean(self.permutation_entropy_history)
                std_perm = unified_math.unified_math.std(self.permutation_entropy_history)
                perm_min = unified_math.max(0.0, mean_perm - 2 * std_perm)
                perm_max = mean_perm + 2 * std_perm
                permutation_valid = perm_min <= permutation_entropy <= perm_max
            else:
                permutation_valid = 0.5 <= permutation_entropy <= 2.0  # Default range

# Calculate Jensen - Shannon divergence
js_divergence = 0.0
        js_valid = True
        if use_js_divergence and self.reference_distribution is not None:
            freqs, psd = welch(signal, fs = fs)
            current_distribution = psd / np.sum(psd)
            js_divergence = _jensen_shannon_divergence(current_distribution, self.reference_distribution)
            js_valid = js_divergence <= js_threshold

# Update history for adaptive thresholds
self.entropy_history.append(spectral_entropy)
        self.permutation_entropy_history.append(permutation_entropy)

# Keep only recent history
if len(self.entropy_history) > self.window_size:
            self.entropy_history = self.entropy_history[-self.window_size:]
        if len(self.permutation_entropy_history) > self.window_size:
            self.permutation_entropy_history = self.permutation_entropy_history[-self.window_size:]

# Overall validation
overall_valid = spectral_valid and permutation_valid and js_valid

return {
            "is_valid": overall_valid,
            "spectral_entropy": spectral_entropy,
            "spectral_valid": spectral_valid,
            "permutation_entropy": permutation_entropy,
            "permutation_valid": permutation_valid,
            "js_divergence": js_divergence,
            "js_valid": js_valid,
            "adaptive_thresholds": {
                "spectral_mean": unified_math.unified_math.mean(self.entropy_history) if self.entropy_history else 0.0,
                "spectral_std": unified_math.unified_math.std(self.entropy_history) if self.entropy_history else 0.0,
                "permutation_mean": unified_math.unified_math.mean(self.permutation_entropy_history) if self.permutation_entropy_history else 0.0,
                "permutation_std": unified_math.unified_math.std(self.permutation_entropy_history) if self.permutation_entropy_history else 0.0,

""""""
""""""
""""""
"""
"""