#!/usr/bin/env python3
"""ZPE core matrix – zero-point energy field calculations and wave mechanics.

Implements the formulas:
    Φ_zpe(x, t) = ∇·Ψ_zpe(x, t) + λ_zpe·(∂Ψ/∂t)
    Ψ_zpe(t) = Σ_i^n A_i·sin(ω_i·t + φ_i)
    Ξ_zpe = ∫_Ω Φ_zpe(x, t) dx
    G_zpe = e^(−β·|∇Φ_zpe|²) · tanh(Φ_zpe/Ξ_zpe)

This module provides quantum-inspired field calculations for enhanced
market state analysis and phase transition detection.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__: list[str] = ["zpe_psi", "zpe_phi", "zpe_xi", "zpe_g"]

# ---------------------------------------------------------------------------
# ZPE field calculations
# ---------------------------------------------------------------------------


def zpe_psi(
    amplitudes: Sequence[float],
    frequencies: Sequence[float],
    phases: Sequence[float],
    t: float,
) -> float:  # noqa: D401
    """Return Ψ_zpe(t) = Σ_i^n A_i·sin(ω_i·t + φ_i).

    Parameters
    ----------
    amplitudes
        Amplitude coefficients A_i for each mode.
    frequencies
        Angular frequencies ω_i for each mode.
    phases
        Phase offsets φ_i for each mode.
    t
        Time parameter.
    """
    if not (len(amplitudes) == len(frequencies) == len(phases)):
        raise ValueError("amplitudes, frequencies, and phases must have same length")
    
    a_arr = np.asarray(amplitudes, dtype=float)
    w_arr = np.asarray(frequencies, dtype=float)
    p_arr = np.asarray(phases, dtype=float)
    
    # Compute sum of sinusoidal modes
    sine_terms = a_arr * np.sin(w_arr * t + p_arr)
    
    return float(np.sum(sine_terms))


def zpe_phi(
    psi_div: float,
    psi_time_deriv: float,
    lambda_zpe: float,
) -> float:  # noqa: D401
    """Return Φ_zpe(x, t) = ∇·Ψ_zpe(x, t) + λ_zpe·(∂Ψ/∂t).

    Parameters
    ----------
    psi_div
        Divergence ∇·Ψ_zpe of the wave function.
    psi_time_deriv
        Time derivative ∂Ψ/∂t of the wave function.
    lambda_zpe
        ZPE coupling constant λ_zpe.
    """
    return psi_div + lambda_zpe * psi_time_deriv


def zpe_xi(
    phi_values: Sequence[float],
    *,
    domain_width: float = 1.0,
) -> float:  # noqa: D401
    """Return Ξ_zpe = ∫_Ω Φ_zpe(x, t) dx using trapezoidal integration.

    Parameters
    ----------
    phi_values
        Discrete values of Φ_zpe at grid points.
    domain_width
        Width of integration domain Ω.
    """
    phi_arr = np.asarray(phi_values, dtype=float)
    
    if len(phi_arr) == 0:
        return 0.0
    elif len(phi_arr) == 1:
        return float(phi_arr[0] * domain_width)
    
    # Trapezoidal integration
    integral = float(np.trapz(phi_arr, dx=domain_width / (len(phi_arr) - 1)))
    
    return integral


def zpe_g(
    phi_zpe: float,
    xi_zpe: float,
    grad_phi_magnitude: float,
    beta: float,
    *,
    epsilon: float = 1e-10,
) -> float:  # noqa: D401
    """Return G_zpe = e^(−β·|∇Φ_zpe|²) · tanh(Φ_zpe/Ξ_zpe).

    Parameters
    ----------
    phi_zpe
        Field value Φ_zpe.
    xi_zpe
        Integrated field Ξ_zpe.
    grad_phi_magnitude
        Magnitude |∇Φ_zpe| of field gradient.
    beta
        Exponential decay parameter β.
    epsilon
        Small constant to prevent division by zero.
    """
    # Exponential term: e^(−β·|∇Φ_zpe|²)
    exp_term = math.exp(-beta * (grad_phi_magnitude ** 2))
    
    # Tanh term: tanh(Φ_zpe/Ξ_zpe)
    if abs(xi_zpe) < epsilon:
        tanh_term = math.tanh(phi_zpe / epsilon)
    else:
        tanh_term = math.tanh(phi_zpe / xi_zpe)
    
    return exp_term * tanh_term 