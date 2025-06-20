#!/usr/bin/env python3
"""Memory-drift corrector – detect glyph hash drift and decide re-link.

Implements ΔΞ_mem logic from the design doc.  Given the *previous* profitable
glyph hash and the *current* glyph hash (plus optional price delta), returns a
scalar **drift score** in [0, 1] where values close to 1 indicate strong drift
(i.e. hashes are dissimilar and price context changed).

A simple softmax of normalised Hamming distance and price delta is used to keep
dependency footprint minimal.
"""

from __future__ import annotations

import math
from typing import Final, Tuple

__all__: list[str] = ["drift_score", "relink_required"]

_MAX_HASH_BITS: Final = 256  # SHA-256
_HAMMING_SCALE: Final = 1.0 / _MAX_HASH_BITS
_PRICE_SCALE: Final = 0.02  # normalise 2% price move → weight 1.0
_THRESHOLD: Final = 0.5  # drift score ≥ threshold ⇒ relink


def _hamming_dist(a: str, b: str) -> int:  # noqa: D401
    if len(a) != len(b):
        raise ValueError("hash strings must share length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) * 4  # hex→bits (×4)


def _softmax2(x: float, y: float) -> float:
    ex = math.exp(x)
    ey = math.exp(y)
    return max(ex, ey) / (ex + ey)


def drift_score(
    prev_hash: str,
    curr_hash: str,
    price_delta_pct: float,
) -> float:
    """Return softmax-based drift score in [0,1]."""
    hamming = _hamming_dist(prev_hash, curr_hash)
    h_norm = hamming * _HAMMING_SCALE  # 
    p_norm = abs(price_delta_pct) / _PRICE_SCALE
    return _softmax2(h_norm, p_norm)


def relink_required(score: float, threshold: float = _THRESHOLD) -> bool:  # noqa: D401
    return score >= threshold 