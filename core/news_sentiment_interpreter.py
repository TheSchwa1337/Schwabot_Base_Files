#!/usr/bin/env python3
"""News sentiment interpreter – converts news into activation signals.

Implements the formula:
    λ_news = Σ(sentiment_score · drift_bias · σ_event)

This module processes financial news streams and converts them into weighted
sentiment signals that can influence ghost router decisions and strategy
matrix updates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = ["interpret_news_sentiment", "weight_sentiment_events"]

# ---------------------------------------------------------------------------
# Core sentiment processing
# ---------------------------------------------------------------------------


def interpret_news_sentiment(
    sentiment_scores: Sequence[float],
    drift_biases: Sequence[float],
    event_sigmas: Sequence[float],
) -> float:  # noqa: D401
    """Return λ_news weighted sentiment activation signal.

    Parameters
    ----------
    sentiment_scores
        Raw sentiment values (typically in [-1, 1] range).
    drift_biases
        Drift correction factors for each news item.
    event_sigmas
        Event significance weights (volatility-like measure).

    Returns
    -------
    float
        Combined sentiment signal λ_news.
    """
    if not (len(sentiment_scores) == len(drift_biases) == len(event_sigmas)):
        raise ValueError("input sequences must share length")

    scores = np.asarray(sentiment_scores, dtype=float)
    biases = np.asarray(drift_biases, dtype=float)
    sigmas = np.asarray(event_sigmas, dtype=float)

    weighted_signals = scores * biases * sigmas
    return float(np.sum(weighted_signals))


def weight_sentiment_events(
    raw_sentiment: float,
    event_importance: float,
    *,
    decay_factor: float = 0.95,
    base_weight: float = 1.0,
) -> float:  # noqa: D401
    """Apply time-decay and importance weighting to single sentiment.

    Returns weighted sentiment suitable for inclusion in λ_news calculation.
    """
    importance_weight = base_weight * event_importance
    decayed_sentiment = raw_sentiment * (decay_factor ** abs(raw_sentiment))
    return decayed_sentiment * importance_weight 