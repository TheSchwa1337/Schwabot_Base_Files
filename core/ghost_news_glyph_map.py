# -*- coding: utf-8 -*-\nfrom __future__ import annotations

from core.unified_math_system import unified_math
import math
# #!/usr/bin/env python3
"""Map news sentiment vector to glyph weight.

Ξ_b = ζ(news) · μ_g(glyph)
"""

from typing import Sequence

# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = ["news_to_glyph_weight"]


def news_to_glyph_weight(


    news_vec: Sequence[float], glyph_mu: Sequence[float]
) -> float:

"""Return dot-product weight between news vector and glyph mean vector."""
   if len(news_vec) != len(glyph_mu):
        raise ValueError("vector length mismatch")
    return float(unified_math.unified_math.dot_product(news_vec, glyph_mu))
