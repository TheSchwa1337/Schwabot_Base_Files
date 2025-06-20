#!/usr/bin/env python3
"""zbe_position_tracker – Zalgo-position glyph evolution tracker.

Implements the Zalgo-position glyph evolution logic:
    Ψₙ = Σ ∂Zᵢ/∂t · Gᵢ(x)

This module tracks position evolution using ZBE (Zero-Based Evolution)
calculations for ghost protocol position management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__: list[str] = [
    "ZBEPositionTracker",
    "compute_zalgo_evolution",
    "track_position_state",
]


@dataclass(slots=True)
class ZBEPositionTracker:
    """Zero-Based Evolution position tracker."""
    
    dt: float = 1.0
    evolution_rate: float = 0.1

    def compute_psi_n(
        self,
        z_series: Sequence[float],
        g_functions: Sequence[Callable[[float], float]],
        x_positions: Sequence[float],
    ) -> np.ndarray:
        """Compute Ψₙ = Σ ∂Zᵢ/∂t · Gᵢ(x).

        Parameters
        ----------
        z_series
            Zalgo values Zᵢ time series.
        g_functions
            Glyph functions Gᵢ(x).
        x_positions
            Position values x to evaluate Gᵢ at.
        """
        if len(z_series) != len(g_functions):
            raise ValueError("z_series and g_functions must have same length")
        
        z_array = np.asarray(z_series, dtype=float)
        x_array = np.asarray(x_positions, dtype=float)
        
        # Compute time derivatives ∂Zᵢ/∂t using finite differences
        if len(z_array) < 2:
            dz_dt = np.array([0.0])
        else:
            dz_dt = np.gradient(z_array, self.dt)
        
        # Initialize result array
        psi_n = np.zeros_like(x_array, dtype=float)
        
        # Sum over all i: ∂Zᵢ/∂t · Gᵢ(x)
        for i, (dz_i, g_func) in enumerate(zip(dz_dt, g_functions)):
            for j, x_pos in enumerate(x_array):
                g_i_x = g_func(x_pos)
                psi_n[j] += dz_i * g_i_x
        
        return psi_n

    def evolve_position_state(
        self,
        current_state: np.ndarray,
        zalgo_derivatives: np.ndarray,
        glyph_weights: np.ndarray,
    ) -> np.ndarray:
        """Evolve position state using Zalgo-glyph evolution.

        Parameters
        ----------
        current_state
            Current position state vector.
        zalgo_derivatives
            Time derivatives of Zalgo values.
        glyph_weights
            Glyph weighting factors.
        """
        if not (len(current_state) == len(zalgo_derivatives) == len(glyph_weights)):
            raise ValueError("all arrays must have same length")
        
        # Apply evolution: state + evolution_rate * Ψₙ
        evolution_term = zalgo_derivatives * glyph_weights
        evolved_state = current_state + self.evolution_rate * evolution_term
        
        return evolved_state

    def track_position_trajectory(
        self,
        initial_state: np.ndarray,
        zalgo_time_series: Sequence[Sequence[float]],
        glyph_functions: Sequence[Callable[[float], float]],
        time_steps: int = 10,
    ) -> np.ndarray:
        """Track position trajectory over multiple time steps.

        Parameters
        ----------
        initial_state
            Initial position state.
        zalgo_time_series
            Time series of Zalgo values for each component.
        glyph_functions
            Glyph functions for evolution calculation.
        time_steps
            Number of time steps to simulate.
        """
        trajectory = np.zeros((time_steps, len(initial_state)), dtype=float)
        trajectory[0] = initial_state
        
        current_state = initial_state.copy()
        
        for step in range(1, time_steps):
            # Extract current Zalgo values
            zalgo_current = np.array([
                series[min(step, len(series) - 1)] 
                for series in zalgo_time_series
            ])
            
            # Compute derivatives
            if step > 0:
                zalgo_prev = np.array([
                    series[min(step - 1, len(series) - 1)] 
                    for series in zalgo_time_series
                ])
                zalgo_derivatives = (zalgo_current - zalgo_prev) / self.dt
            else:
                zalgo_derivatives = np.zeros_like(zalgo_current)
            
            # Compute glyph weights at current positions
            glyph_weights = np.array([
                g_func(current_state[i % len(current_state)]) 
                for i, g_func in enumerate(glyph_functions)
            ])
            
            # Evolve state
            current_state = self.evolve_position_state(
                current_state, zalgo_derivatives, glyph_weights
            )
            trajectory[step] = current_state
        
        return trajectory


# Functional helpers

def compute_zalgo_evolution(
    z_series: Sequence[float],
    glyph_weights: Sequence[float],
    dt: float = 1.0,
) -> np.ndarray:  # noqa: D401
    """Compute basic Zalgo evolution without position dependence."""
    z_array = np.asarray(z_series, dtype=float)
    weights = np.asarray(glyph_weights, dtype=float)
    
    if len(z_array) < 2:
        return np.zeros_like(weights)
    
    # Compute time derivative
    dz_dt = np.gradient(z_array, dt)
    
    # Broadcast multiplication
    if len(dz_dt) == len(weights):
        return dz_dt * weights
    else:
        # Handle length mismatch by using mean derivative
        mean_derivative = np.mean(dz_dt)
        return mean_derivative * weights


def track_position_state(
    initial_state: Sequence[float],
    zalgo_series: Sequence[float],
    glyph_weights: Sequence[float],
    evolution_rate: float = 0.1,
) -> np.ndarray:  # noqa: D401
    """Stateless position state tracking."""
    state = np.asarray(initial_state, dtype=float)
    evolution = compute_zalgo_evolution(zalgo_series, glyph_weights)
    
    # Simple single-step evolution
    return state + evolution_rate * evolution 