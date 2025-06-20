#!/usr/bin/env python3
"""Glyph vector executor – executes strategic moves from glyph instructions.

Implements the formula:
    G_out = Σ ω_i · G_i_vector[t] · ζ_weighting[t]

This module takes weighted glyph vectors and converts them into executable
trade instructions that can be consumed by the routing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__: list[str] = ["GlyphInstruction", "execute_glyph_vectors"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GlyphInstruction:
    """Executable instruction derived from glyph vector processing."""

    action: str  # "buy", "sell", "hold", "wait"
    volume: float
    confidence: float
    glyph_signature: str


# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------


def execute_glyph_vectors(
    omega_weights: Sequence[float],
    glyph_vectors: Sequence[Sequence[float]],
    zeta_weightings: Sequence[float],
    *,
    action_threshold: float = 0.5,
    volume_scale: float = 1.0,
) -> GlyphInstruction:  # noqa: D401
    """Return executable instruction from weighted glyph vectors.

    Parameters
    ----------
    omega_weights
        Weighting coefficients ω_i for each glyph vector.
    glyph_vectors
        Sequence of glyph state vectors G_i_vector[t].
    zeta_weightings
        Time-varying weights ζ_weighting[t] for each vector.
    action_threshold
        Minimum confidence required to generate non-hold action.
    volume_scale
        Scaling factor for computed volume.

    Returns
    -------
    GlyphInstruction
        Executable instruction with action, volume, confidence.
    """
    if not (len(omega_weights) == len(glyph_vectors) == len(zeta_weightings)):
        raise ValueError("input sequences must share length")

    if not glyph_vectors:
        return GlyphInstruction("hold", 0.0, 0.0, "empty")

    # Convert inputs to arrays
    omega = np.asarray(omega_weights, dtype=float)
    zeta = np.asarray(zeta_weightings, dtype=float)

    # Compute weighted sum: Σ ω_i · G_i · ζ_i
    weighted_sum = np.zeros_like(glyph_vectors[0], dtype=float)
    for i, g_vec in enumerate(glyph_vectors):
        g_array = np.asarray(g_vec, dtype=float)
        weighted_sum += omega[i] * g_array * zeta[i]

    # Extract action signals (assume first 4 components are [buy, sell, hold, wait])
    if len(weighted_sum) < 4:
        return GlyphInstruction("hold", 0.0, 0.0, "insufficient_dims")

    buy_signal = weighted_sum[0]
    sell_signal = weighted_sum[1]
    hold_signal = weighted_sum[2]
    wait_signal = weighted_sum[3]

    # Determine action
    signals = [buy_signal, sell_signal, hold_signal, wait_signal]
    actions = ["buy", "sell", "hold", "wait"]
    max_idx = int(np.argmax(np.abs(signals)))
    max_signal = signals[max_idx]
    confidence = float(np.abs(max_signal))

    if confidence < action_threshold:
        action = "hold"
        volume = 0.0
    else:
        action = actions[max_idx]
        volume = confidence * volume_scale

    # Generate signature from vector hash
    vector_hash = hash(tuple(weighted_sum.round(6)))
    signature = f"glyph_{vector_hash & 0xFFFF:04x}"

    return GlyphInstruction(action, volume, confidence, signature) 