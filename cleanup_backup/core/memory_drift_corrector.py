# -*- coding: utf - 8 -*-
"""Memory - drift corrector \\u2013 detect glyph hash drift and decide re - link.
"""Memory - drift corrector \\u2013 detect glyph hash drift and decide re - link.
# -*- coding: utf - 8 -*-
from __future__ import annotations

"""Memory - drift corrector \\u2013 detect glyph hash drift and decide re - link.
"""Memory - drift corrector \\u2013 detect glyph hash drift and decide re - link.
# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-

from core.unified_math_system import unified_math






Implements \\u0394\\u039e_mem logic from the design doc.  Given the *previous* profitable
glyph hash and the *current* glyph hash (plus optional price delta), returns a
scalar **drift score** in [0, 1] where values close to 1 indicate strong drift
(i.e. hashes are dissimilar and price context changed).

A simple softmax of normalised Hamming distance and price delta is used to keep
dependency footprint minimal.
"""
"""
"""


from core.unified_math_system import unified_math
from typing import Final

__all__: list[str] = ["drift_score", "relink_required"]

_MAX_HASH_BITS: Final = 256  # SHA - 256
_HAMMING_SCALE: Final = 1.0 / _MAX_HASH_BITS
_PRICE_SCALE: Final = 0.02  # normalise 2% price move \\u2192 weight 1.0
_THRESHOLD: Final = 0.5  # drift score \\u2265 threshold \\u21d2 relink


def _hamming_dist(a: str, b: str) -> int:  # noqa: D401

    """TODO: document _hamming_dist."""
"""
"""
    if len(a) != len(b):
        raise ValueError("hash strings must share length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) * 4  # hex\\u2192bits (\\u00d74)


def _softmax2(x: float, y: float) -> float:

    """TODO: document _softmax2."""
"""
"""
    ex = unified_math.unified_math.exp(x)
    ey = unified_math.unified_math.exp(y)
    return unified_math.max(ex, ey) / (ex + ey)


def drift_score(

    prev_hash: str,
    curr_hash: str,
    price_delta_pct: float,
) -> float:
    """Return softmax - based drift score in [0,1]."""
"""
"""
    hamming = _hamming_dist(prev_hash, curr_hash)
    h_norm = hamming * _HAMMING_SCALE  #
    p_norm = unified_math.abs(price_delta_pct) / _PRICE_SCALE
    return _softmax2(h_norm, p_norm)


def relink_required(

    score: float, threshold: float = _THRESHOLD
) -> bool:  # noqa: D401
    """TODO: document relink_required."""
"""
"""
    return score >= threshold
