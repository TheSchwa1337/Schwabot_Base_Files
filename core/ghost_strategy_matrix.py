#!/usr/bin/env python3
"""Ghost strategy matrix utilities.

This module now provides a complete set of helpers that implement the
mathematical specification outlined in the design note.  They are intentionally
kept *stateless* – callers supply previous-tick matrices / vectors and receive
updated ones.

Public API
----------
1. build_strategy_matrix           – basic outer-product helper.
2. strategy_match_matrix           – binary match map M[i,j].
3. reward_matrix                   – profit-weighted reinforcement scores.
4. dynamic_strategy_switch         – arg-max selection via softmax.
5. update_strategy_matrix          – echo-band & volatility adaptation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__: list[str] = [
    "build_strategy_matrix",
    "strategy_match_matrix",
    "reward_matrix",
    "dynamic_strategy_switch",
    "update_strategy_matrix",
]

# ---------------------------------------------------------------------------
# Basic outer-product helper (legacy)
# ---------------------------------------------------------------------------


def build_strategy_matrix(phi: np.ndarray, kappa: np.ndarray) -> np.ndarray:  # noqa: D401
    """Return outer product S = phi[:, None] * kappa[None, :]."""
    return np.outer(phi, kappa)


# ---------------------------------------------------------------------------
# (1) Strategy match mapping – binary matrix M
# ---------------------------------------------------------------------------


def _find_band_idx(value: float | int, edges: Sequence[float | int]) -> int:
    """Return index i such that edges[i] ≤ value < edges[i+1]."""
    if not (len(edges) >= 2):
        raise ValueError("edges must contain at least two elements")
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    # If value beyond last edge, snap to last band
    return len(edges) - 2


def strategy_match_matrix(
    H_t: int,
    zeta_t: float,
    hash_edges: Sequence[int],
    zeta_edges: Sequence[float],
) -> np.ndarray:  # noqa: D401
    """Return binary M with a single 1 where current state falls.

    The matrix shape is (len(hash_edges)-1, len(zeta_edges)-1).
    """
    i = _find_band_idx(H_t, hash_edges)
    j = _find_band_idx(zeta_t, zeta_edges)
    M = np.zeros((len(hash_edges) - 1, len(zeta_edges) - 1), dtype=int)
    M[i, j] = 1
    return M


# ---------------------------------------------------------------------------
# (2) Profit optimisation reward matrix R_ij
# ---------------------------------------------------------------------------


def reward_matrix(
    P: np.ndarray,
    delta_G: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:  # noqa: D401
    """Return element-wise product R = P * delta_G * zeta.

    Arrays must share the same shape.
    """
    if not (P.shape == delta_G.shape == zeta.shape):
        raise ValueError("input arrays must share shape")
    return P * delta_G * zeta


# ---------------------------------------------------------------------------
# (3) Dynamic strategy switching – softmax & argmax
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:  # noqa: D401
    x_shift = x - np.max(x)
    e_x = np.exp(x_shift)
    return e_x / np.sum(e_x)


def dynamic_strategy_switch(
    Q: np.ndarray,
    T: np.ndarray,
    lam: np.ndarray,
) -> int:  # noqa: D401
    """Return strategy index i that maximises softmax(Q * T * λ)."""
    if not (Q.shape == T.shape == lam.shape):
        raise ValueError("arrays Q, T, lam must share shape")
    score = _softmax(Q * T * lam)
    return int(np.argmax(score))


# ---------------------------------------------------------------------------
# (4) Echo-band reinforcement & volatility adjustment
# ---------------------------------------------------------------------------


def update_strategy_matrix(
    M_prev: np.ndarray,
    R: np.ndarray,
    E: np.ndarray,
    *,
    gamma: float = 0.1,
    beta: float = 0.05,
    sigma: np.ndarray | None = None,
    eta_noise: np.ndarray | None = None,
) -> np.ndarray:  # noqa: D401
    """Return updated matrix according to ΔM formulation.

    Parameters
    ----------
    M_prev, R, E
        Previous matrix, reward matrix and EchoBand cluster activations.
    gamma
        Damping factor (resistance to switch).
    beta
        Volatility gain coefficient.
    sigma, eta_noise
        Optional volatility σ_ij and noise η arrays. If omitted zeros are used.
    """
    if not (M_prev.shape == R.shape == E.shape):
        raise ValueError("M_prev, R, E must share shape")

    # Echo-band reinforcement
    alpha = 1.0 / (1.0 + np.exp(-E))  # logistic scaling α(E_i)
    delta_M = alpha * (R - gamma * M_prev)
    M_new = M_prev + delta_M

    # Volatility & noise adjustment
    if sigma is None:
        sigma = np.zeros_like(M_new)
    if eta_noise is None:
        eta_noise = np.zeros_like(M_new)
    M_new = M_new + beta * sigma - 0.01 * eta_noise

    return M_new 