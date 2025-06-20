#!/usr/bin/env python3
"""Vector-state mapper – align news glyph vectors with live BTC state space.

The purpose of this component is to project a *glyph vector* (semantic signal
extracted from news) onto a *market state matrix* so that downstream routers
can reason about narrative alignment using a **single scalar similarity score**
or a small set of channel weights.

Current implementation (lightweight, no ML deps):
1. The BTC state space is represented by a feature matrix ``S`` of shape
   ``(n_channels, dim)`` – e.g. 4×128 for [price-delta, volume-delta, RSI,
   on-chain sentiment].  The dimensionality *dim* must match the glyph vector
   produced by :func:`glyph_news_parser.parse_news_to_glyph`.
2. Computes cosine similarity between the glyph vector and each channel row of
   ``S`` yielding an ``n_channels`` similarity vector.
3. Returns ``(similarities, best_idx)`` where *best_idx* is the channel with
   maximal similarity.  This keeps downstream gating trivial.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__: list[str] = ["map_glyph_to_state"]


def _cosine(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Return cosine similarity of *v* against each row of *m* (1-D array)."""
    dot = m @ v  # (n,) vector
    v_norm = np.linalg.norm(v)
    m_norm = np.linalg.norm(m, axis=1)
    denom = v_norm * m_norm
    # Prevent division by zero – if a norm is zero, similarity is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom == 0, 0.0, dot / denom)
    return sim.astype(float)


def map_glyph_to_state(
    glyph_vec: np.ndarray,
    state_matrix: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Project *glyph_vec* onto *state_matrix* and return similarities.

    Parameters
    ----------
    glyph_vec
        1-D NumPy array of length *dim* – L2-normalised.
    state_matrix
        2-D NumPy array of shape ``(n_channels, dim)`` representing live BTC
        feature channels.  Must share the trailing dimension with *glyph_vec*.

    Returns
    -------
    Tuple[np.ndarray, int]
        ``(similarities, best_idx)`` where *similarities* is the 1-D array of
        cosine scores, and *best_idx* is the index of the row with the highest
        similarity.  Values lie in ``[-1, 1]``.
    """
    if glyph_vec.ndim != 1:
        raise ValueError("glyph_vec must be 1-D")
    if state_matrix.ndim != 2:
        raise ValueError("state_matrix must be 2-D")
    if state_matrix.shape[1] != glyph_vec.shape[0]:
        raise ValueError("state_matrix dim mismatch with glyph_vec length")

    sims = _cosine(glyph_vec, state_matrix)
    best_idx = int(np.argmax(sims))
    return sims, best_idx 