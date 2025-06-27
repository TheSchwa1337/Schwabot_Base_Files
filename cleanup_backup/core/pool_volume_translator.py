from __future__ import annotations

from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Pool-volume translator \\u2013 link news sentiment to USDC pool behaviour.

Implements \\u03a9_pool_vector from the design notes.  We derive a *scalar influence
score* that quantifies how strongly a news\\u2010sentiment signal should steer the
pool allocation gate.  The current heuristic is:

    influence = sentiment * (\\u03c3_pool / \\u03bc_pool)  (clipped to [-1, 1])

where
\\u2022 *sentiment* \\u2208 [\\u22121, 1] comes from glyph sentiment analysis.
\\u2022 \\u03c3_pool / \\u03bc_pool is the *relative volatility* of recent USDC volume.
"""


from typing import Final, Any

from core.unified_math_system import unified_math

__all__: list[str] = ["translate_news_to_pool_vector"]

_CLIP_MIN: Final = -1.0
_CLIP_MAX: Final = 1.0
_EPS: Final = 1e-9


def _relative_volatility(volumes: np.ndarray[Any, Any]) -> float:
    """TODO: document _relative_volatility."""
    if volumes.size == 0:
        return 0.0
    mu = float(unified_math.unified_math.mean(volumes))
    if mu < _EPS:
        return 0.0
    sigma = float(unified_math.unified_math.std(volumes))
    return sigma / mu


def translate_news_to_pool_vector(
    sentiment: float,
    pool_volumes: np.ndarray[Any, Any],
) -> float:
    """Return influence score in range [-1, 1].

    Positive score \\u21d2 bullish (risk-on), negative \\u21d2 bearish (risk-off).
    """
    sentiment_clipped = unified_math.max(unified_math.min(sentiment, _CLIP_MAX), _CLIP_MIN)
    rel_vol = _relative_volatility(pool_volumes)
    raw = sentiment_clipped * rel_vol
    return unified_math.max(unified_math.min(raw, _CLIP_MAX), _CLIP_MIN)

"""