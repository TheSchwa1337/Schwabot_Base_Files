#!/usr/bin/env python3
"""Ghost Phase Integrator – trust-weighted phase correction logic.

Implements equations (1)…(9) from the design note, returning a
:class:`GhostPhasePacket` tuple (eq. 10).

All computations are purely functional, fully typed and NumPy-backed while
remaining free of heavy external dependencies.  The public helper
:func:`compute_ghost_phase_packet` is the single entry-point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

__all__: list[str] = ["GhostPhasePacket", "compute_ghost_phase_packet"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:  # noqa: D401
    """Return Levenshtein edit distance (simple O(n²) DP).

    The strings are expected to be hex-encoded hashes of identical length
    (64 chars for a 256-bit digest).  A ValueError is raised if lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("hash strings must have equal length")

    prev_row = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr_row = [i]
        for j, ch_b in enumerate(b, start=1):
            ins = prev_row[j] + 1
            del_ = curr_row[j - 1] + 1
            sub = prev_row[j - 1] + (ch_a != ch_b)
            curr_row.append(min(ins, del_, sub))
        prev_row = curr_row
    return prev_row[-1]


def _clip(x: float, lo: float, hi: float) -> float:  # noqa: D401
    return max(lo, min(hi, x))


def _hash_to_int(hex_digest: str) -> int:  # noqa: D401
    return int(hex_digest, 16)


# ---------------------------------------------------------------------------
# Dataclass container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GhostPhasePacket:
    """Output ⟨Cₜ , ζ_final , H_echo , μ_echo , δ_corr⟩."""

    C_t: float
    zeta_final: float
    H_echo: Sequence[str]
    mu_echo: float
    delta_corr: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_ghost_phase_packet(
    *,
    H_t: str,
    H_echo: Sequence[str],
    zeta_news_t: float,
    lambda_sentiment_t: float,
    alpha_t: float,
    phi_fractal_t: float,
    nu_cycle_t: float,
    delta_alt_t: float,
    grad_phi_fractal_t: float,
    delta_nu_cycle_t: float,
    drift_t: float,
    q_exec_prev: float,
    q_exec_curr: float,
    delta_t: float,
    epsilon: float = 1e-9,
) -> GhostPhasePacket:  # noqa: D401
    """Return a :class:`GhostPhasePacket` computed from live signals.

    Parameters
    ----------
    H_t, H_echo
        Current and historical 256-bit ghost hashes (hex-encoded).
    zeta_news_t, lambda_sentiment_t
        News-driven phase and sentiment scalars.
    alpha_t, phi_fractal_t, nu_cycle_t
        Altitude factor, fractal pressure and oscillator cycle scalar.
    delta_alt_t, grad_phi_fractal_t, delta_nu_cycle_t
        Altitude error, gradient of fractal pressure and cycle drift.
    drift_t
        Instantaneous drift magnitude (0 ≤ drift_t ≤ 1).
    q_exec_prev, q_exec_curr, delta_t
        Consecutive executable sizes and their time delta (sec).
    epsilon
        Small constant to stabilise log/denominator operations.
    """

    # --------------------------------------------------
    # (1) Γ_hash – similarity between current and echoed hash
    # --------------------------------------------------
    if not H_echo:
        # No echo history – assume worst case similarity (distance 64)
        lev_dist: Final = 64
    else:
        lev_dist = _levenshtein(H_t, H_echo[-1])
    gamma_hash = math.exp(-lev_dist / 64.0)

    # --------------------------------------------------
    # (2) Γ_news – sentiment-weighted news activation
    # --------------------------------------------------
    gamma_news = math.tanh(zeta_news_t * lambda_sentiment_t)

    # --------------------------------------------------
    # (3) Γ_phase – composite phase weight
    # --------------------------------------------------
    gamma_phase = alpha_t * phi_fractal_t * (1.0 + nu_cycle_t)

    # --------------------------------------------------
    # (4) μ_echo – trust index clipped to [0, 1]
    # --------------------------------------------------
    mu_echo = _clip(gamma_hash * gamma_news * gamma_phase, 0.0, 1.0)

    # --------------------------------------------------
    # (5) δ_corr – phase correction term
    # --------------------------------------------------
    delta_corr = delta_alt_t + grad_phi_fractal_t + delta_nu_cycle_t

    # --------------------------------------------------
    # (6) C_t – trust-weighted adjustment coefficient
    # --------------------------------------------------
    C_t = (1.0 - drift_t * (1.0 - mu_echo)) * math.exp(-(delta_corr ** 2))

    # --------------------------------------------------
    # (7) Θ_drift – drift entropy modulator
    # --------------------------------------------------
    if len(H_echo) < 2:
        theta_drift = 0.0
    else:
        int_series = np.fromiter((_hash_to_int(h) for h in H_echo), dtype=float)
        diff_series = np.diff(int_series)
        spectrum = np.abs(np.fft.fft(diff_series))
        theta_drift = math.log(1.0 + float(np.linalg.norm(spectrum))) / epsilon

    # --------------------------------------------------
    # (8) m_slope – execution slope
    # --------------------------------------------------
    if delta_t <= 0:
        m_slope = 0.0
    else:
        m_slope = (q_exec_curr - q_exec_prev) / delta_t

    # --------------------------------------------------
    # (9) ζ_final – phase-aligned trust correction clipped to [-1, 1]
    # --------------------------------------------------
    zeta_final = _clip(mu_echo - theta_drift + m_slope, -1.0, 1.0)

    return GhostPhasePacket(
        C_t=C_t,
        zeta_final=zeta_final,
        H_echo=tuple(H_echo),  # immutable copy
        mu_echo=mu_echo,
        delta_corr=delta_corr,
    ) 