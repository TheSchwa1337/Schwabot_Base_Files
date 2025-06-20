#!/usr/bin/env python3
"""Pool-volume translator – link news sentiment to USDC pool behaviour.

Implements Ω_pool_vector from the design notes.  We derive a *scalar influence
score* that quantifies how strongly a news‐sentiment signal should steer the
pool allocation gate.  The current heuristic is:

    influence = sentiment * (σ_pool / μ_pool)  (clipped to [-1, 1])

where
• *sentiment* ∈ [−1, 1] comes from glyph sentiment analysis.
• σ_pool / μ_pool is the *relative volatility* of recent USDC volume.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__: list[str] = ["translate_news_to_pool_vector"]

_CLIP_MIN: Final = -1.0
_CLIP_MAX: Final = 1.0
_EPS: Final = 1e-9


def _relative_volatility(volumes: np.ndarray) -> float:
    if volumes.size == 0:
        return 0.0
    mu = float(np.mean(volumes))
    if mu < _EPS:
        return 0.0
    sigma = float(np.std(volumes))
    return sigma / mu


def translate_news_to_pool_vector(
    sentiment: float,
    pool_volumes: np.ndarray,
) -> float:
    """Return influence score in range [-1, 1].

    Positive score ⇒ bullish (risk-on), negative ⇒ bearish (risk-off).
    """
    sentiment_clipped = max(min(sentiment, _CLIP_MAX), _CLIP_MIN)
    rel_vol = _relative_volatility(pool_volumes)
    raw = sentiment_clipped * rel_vol
    return max(min(raw, _CLIP_MAX), _CLIP_MIN) 