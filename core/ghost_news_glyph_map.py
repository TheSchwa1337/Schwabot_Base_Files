#!/usr/bin/env python3
"""Map news sentiment vector to glyph weight Ξᵦ = ζ(news) · μᵍ(glyph).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["news_to_glyph_weight"]

def news_to_glyph_weight(news_vec: Sequence[float], glyph_mu: Sequence[float]) -> float:
    """Return dot-product weight between news vector and glyph mean vector."""
    if len(news_vec) != len(glyph_mu):
        raise ValueError("vector length mismatch")
    return float(np.dot(news_vec, glyph_mu)) 