from __future__ import annotations

from core.unified_math_system import unified_math
import numpy as np
# #!/usr/bin/env python3
"""Vectorise news with weighting matrix.

V_n = ∇·(Θ · φ(news))
"""


from typing import Sequence

# from core.unified_math_system import unified_math  # F811: duplicate import

__all__: list[str] = ["vectorize_news"]


def vectorize_news(
    theta: np.ndarray, phi_news: Sequence[float]
) -> np.ndarray:  # noqa: D401
"""Return V_n vector = theta @ phi_news (gradient-like projection)."""
    phi = np.asarray(phi_news, dtype=float)
    if theta.shape[1] != phi.size:
        raise ValueError("theta column dim mismatch with phi_news length")
    return theta @ phi
