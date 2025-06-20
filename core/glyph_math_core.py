#!/usr/bin/env python3
"""Glyph math core – determinant-based glyph processing and tensor operations.

Implements the formulas:
    G_glyph(x, y) = det|∂²F/∂x∂y|
    M_glyph = Σ_i^n G_glyph(x_i, y_i) · w_i
    ψ_glyph = σ(M_glyph) · tanh(G_glyph)
    Θ_glyph = ∇ψ_glyph ⊗ ∇ψ_glyph^T

This module provides the mathematical foundation for glyph-based signal
processing and tensor field computations.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__: list[str] = ["glyph_determinant", "glyph_matrix", "glyph_psi", "glyph_tensor"]

# ---------------------------------------------------------------------------
# Core glyph computations
# ---------------------------------------------------------------------------


def glyph_determinant(
    func: Callable[[float, float], float],
    x: float,
    y: float,
    *,
    h: float = 1e-6,
) -> float:  # noqa: D401
    """Return G_glyph(x, y) = det|∂²F/∂x∂y| using finite differences.

    Parameters
    ----------
    func
        Function F(x, y) to compute second derivatives of.
    x, y
        Point at which to evaluate the determinant.
    h
        Step size for finite difference approximation.
    """
    # Compute mixed partial derivative ∂²F/∂x∂y
    f_xy = (func(x + h, y + h) - func(x + h, y - h) - 
            func(x - h, y + h) + func(x - h, y - h)) / (4 * h * h)
    
    # Compute ∂²F/∂x²
    f_xx = (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h * h)
    
    # Compute ∂²F/∂y²
    f_yy = (func(x, y + h) - 2 * func(x, y) + func(x, y - h)) / (h * h)
    
    # Hessian determinant
    hessian_det = f_xx * f_yy - f_xy * f_xy
    
    return abs(hessian_det)


def glyph_matrix(
    glyph_values: Sequence[float],
    weights: Sequence[float],
) -> float:  # noqa: D401
    """Return M_glyph = Σ_i^n G_glyph(x_i, y_i) · w_i.

    Parameters
    ----------
    glyph_values
        Sequence of G_glyph evaluations at different points.
    weights
        Corresponding weights w_i for each glyph value.
    """
    if len(glyph_values) != len(weights):
        raise ValueError("glyph_values and weights must have same length")
    
    g_array = np.asarray(glyph_values, dtype=float)
    w_array = np.asarray(weights, dtype=float)
    
    return float(np.dot(g_array, w_array))


def glyph_psi(m_glyph: float, g_glyph: float) -> float:  # noqa: D401
    """Return ψ_glyph = σ(M_glyph) · tanh(G_glyph).

    Parameters
    ----------
    m_glyph
        Matrix value M_glyph from glyph_matrix().
    g_glyph
        Glyph determinant value G_glyph.
    """
    # Sigmoid function σ(x) = 1/(1 + e^(-x))
    sigmoid = 1.0 / (1.0 + np.exp(-m_glyph))
    
    # Hyperbolic tangent
    tanh_g = np.tanh(g_glyph)
    
    return float(sigmoid * tanh_g)


def glyph_tensor(
    psi_gradient: Sequence[float],
) -> np.ndarray:  # noqa: D401
    """Return Θ_glyph = ∇ψ_glyph ⊗ ∇ψ_glyph^T outer product tensor.

    Parameters
    ----------
    psi_gradient
        Gradient vector ∇ψ_glyph as sequence of partial derivatives.
    """
    grad = np.asarray(psi_gradient, dtype=float)
    
    # Outer product: ∇ψ ⊗ ∇ψ^T
    tensor = np.outer(grad, grad)
    
    return tensor 