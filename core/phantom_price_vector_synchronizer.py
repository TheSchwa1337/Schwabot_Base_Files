#!/usr/bin/env python3
"""phantom_price_vector_synchronizer – phantom velocity adjustment and sync.

Implements the phantom price vector synchronization logic:
    Zₚ(t) = ∫₀ᵗ [α·Vₚ(t') − β·Ξₚ(t')] dt'

This module synchronizes phantom price vectors across market data streams
for ghost protocol integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__: list[str] = [
    "PhantomPriceSynchronizer",
    "compute_phantom_velocity",
    "synchronize_price_vectors",
]


@dataclass(slots=True)
class PhantomPriceSynchronizer:
    """Phantom price vector synchronizer with velocity adjustment."""
    
    alpha: float = 1.0
    beta: float = 0.5
    dt: float = 1.0

    def compute_zp_integral(
        self,
        velocity_series: Sequence[float],
        xi_series: Sequence[float],
        t_max: float,
    ) -> float:
        """Compute Zₚ(t) = ∫₀ᵗ [α·Vₚ(t') − β·Ξₚ(t')] dt'.

        Parameters
        ----------
        velocity_series
            Phantom velocity Vₚ(t') time series.
        xi_series
            Xi phantom values Ξₚ(t') time series.
        t_max
            Upper integration limit.
        """
        if len(velocity_series) != len(xi_series):
            raise ValueError("velocity and xi series must have same length")
        
        v_array = np.asarray(velocity_series, dtype=float)
        xi_array = np.asarray(xi_series, dtype=float)
        
        # Compute integrand: α·Vₚ(t') − β·Ξₚ(t')
        integrand = self.alpha * v_array - self.beta * xi_array
        
        # Trapezoidal integration from 0 to t_max
        if len(integrand) < 2:
            return 0.0
        
        dx = t_max / (len(integrand) - 1)
        integral = float(np.trapz(integrand, dx=dx))
        
        return integral

    def synchronize_vectors(
        self,
        price_vectors: Sequence[Sequence[float]],
        timestamps: Sequence[float],
    ) -> np.ndarray:
        """Synchronize multiple phantom price vectors.

        Parameters
        ----------
        price_vectors
            Sequence of price vector time series.
        timestamps
            Corresponding timestamps for synchronization.
        """
        if not price_vectors:
            return np.array([])
        
        # Convert to numpy arrays
        vectors = [np.asarray(pv, dtype=float) for pv in price_vectors]
        
        # Compute phantom velocities (simple finite difference)
        phantom_velocities = []
        for vector in vectors:
            if len(vector) < 2:
                phantom_velocities.append(np.array([0.0]))
            else:
                velocity = np.gradient(vector, self.dt)
                phantom_velocities.append(velocity)
        
        # Synchronize using weighted average
        if not phantom_velocities:
            return np.array([])
        
        # Find minimum length for synchronization
        min_length = min(len(pv) for pv in phantom_velocities)
        
        synchronized = np.zeros(min_length, dtype=float)
        for i, pv in enumerate(phantom_velocities):
            weight = 1.0 / (1.0 + i)  # Decreasing weights
            synchronized += weight * pv[:min_length]
        
        return synchronized


# Functional helpers

def compute_phantom_velocity(
    price_series: Sequence[float],
    dt: float = 1.0,
) -> np.ndarray:  # noqa: D401
    """Compute phantom velocity from price series using gradient."""
    prices = np.asarray(price_series, dtype=float)
    if len(prices) < 2:
        return np.array([0.0])
    return np.gradient(prices, dt)


def synchronize_price_vectors(
    vectors: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    alpha: float = 1.0,
    beta: float = 0.5,
) -> np.ndarray:  # noqa: D401
    """Stateless wrapper for price vector synchronization."""
    synchronizer = PhantomPriceSynchronizer(alpha=alpha, beta=beta)
    return synchronizer.synchronize_vectors(vectors, timestamps) 