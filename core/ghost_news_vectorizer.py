#!/usr/bin/env python3
"""Vectorise news with weighting matrix: Vₙ = ∇·(Θ · φ(news))."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["vectorize_news"]


def vectorize_news(theta: np.ndarray, phi_news: Sequence[float]) -> np.ndarray:  # noqa: D401
    """Return Vₙ vector = theta @ phi_news (gradient-like projection)."""
    phi = np.asarray(phi_news, dtype=float)
    if theta.shape[1] != phi.size:
        raise ValueError("theta column dim mismatch with phi_news length")
    return theta @ phi 