# -*- coding: utf-8 -*-\n"""Phase packet builder for ghost routing system."""

from __future__ import annotations
from dataclasses import dataclass
from core.unified_math_system import unified_math
import math
# from core.unified_math_system import unified_math  # F811: duplicate import


@dataclass
class PhasePacket:


    """Phase packet containing hash, echo, drift and final coefficients."""

gamma: float  # Γ_hash coefficient
mu: float  # μ_echo coefficient
zeta: float  # ζ_final coefficient
theta: float  # Θ_drift coefficient


def build_packet(


    hash_seq: list[int], echo_seq: list[float], drift: float
) -> PhasePacket:
"""Compute Γ, μ, ζ, Θ from last two ticks.

Implements equations (1)-(10) from design note §3.2:
    - Γ_hash = |h_now - h_prev| / 2^256
- μ_echo = unified_math.mean(last 8 echo values)
    - ζ_final = μ * Γ (combined coefficient)
    - Θ_drift = drift * (1 - ζ) (drift compensation)

Args:
hash_seq: Sequence of hash values (need at least 2)
        echo_seq: Sequence of echo lag values
drift: Current drift measurement

Returns:
PhasePacket with computed coefficients

Raises:
ValueError: If insufficient data points
"""
    if len(hash_seq) < 2:
        raise ValueError("Need at least 2 hash values")
    if len(echo_seq) < 1:
        raise ValueError("Need at least 1 echo value")

h_now, h_prev = hash_seq[-1], hash_seq[-2]

    # Γ_hash: normalized hash difference
gamma = unified_math.abs(h_now - h_prev) / (2**256)

    # μ_echo: mean of last 8 echo values
recent_echoes = echo_seq[-8:] if len(echo_seq) >= 8 else echo_seq
    mu = float(unified_math.unified_math.mean(recent_echoes))

    # ζ_final: combined coefficient
zeta = mu * gamma

    # Θ_drift: drift compensation
theta = drift * (1 - zeta)

    return PhasePacket(gamma, mu, zeta, theta)
