#!/usr/bin/env python3
"""entry_exit_vector_analyzer – routing elasticity analysis for entry/exit.

Implements the routing elasticity logic:
    Λᴿ(t) = Rᵢ(x, y) · Σ ∂P/∂t

This module analyzes entry and exit vectors using routing elasticity
calculations for optimal timing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__: list[str] = [
    "EntryExitVectorAnalyzer",
    "compute_routing_elasticity",
    "analyze_entry_exit_vectors",
]


@dataclass(slots=True)
class EntryExitVectorAnalyzer:
    """Entry/exit vector analyzer with routing elasticity."""
    
    dt: float = 1.0
    elasticity_threshold: float = 0.3

    def compute_lambda_r(
        self,
        r_function: Callable[[float, float], float],
        x_positions: Sequence[float],
        y_positions: Sequence[float],
        price_series: Sequence[float],
        timestamps: Sequence[float],
    ) -> float:
        """Compute Λᴿ(t) = Rᵢ(x, y) · Σ ∂P/∂t.

        Parameters
        ----------
        r_function
            Routing function Rᵢ(x, y).
        x_positions, y_positions
            Position coordinates for routing function.
        price_series
            Price time series P(t).
        timestamps
            Time points t.
        """
        if len(x_positions) != len(y_positions):
            raise ValueError("x_positions and y_positions must have same length")
        
        x_array = np.asarray(x_positions, dtype=float)
        y_array = np.asarray(y_positions, dtype=float)
        prices = np.asarray(price_series, dtype=float)
        times = np.asarray(timestamps, dtype=float)
        
        # Compute price derivatives Σ ∂P/∂t
        if len(prices) < 2:
            dp_dt_sum = 0.0
        else:
            dp_dt = np.gradient(prices, self.dt if len(times) <= 1 else np.gradient(times))
            dp_dt_sum = float(np.sum(dp_dt))
        
        # Compute routing function values Rᵢ(x, y)
        r_values = np.array([r_function(x, y) for x, y in zip(x_array, y_array)])
        r_sum = float(np.sum(r_values))
        
        # Compute Λᴿ(t) = Rᵢ(x, y) · Σ ∂P/∂t
        lambda_r = r_sum * dp_dt_sum
        
        return lambda_r

    def analyze_entry_signals(
        self,
        price_gradients: Sequence[float],
        volume_gradients: Sequence[float],
        elasticity_values: Sequence[float],
    ) -> np.ndarray:
        """Analyze entry signals using routing elasticity.

        Parameters
        ----------
        price_gradients
            Price gradient signals.
        volume_gradients
            Volume gradient signals.
        elasticity_values
            Computed elasticity values.
        """
        if not (len(price_gradients) == len(volume_gradients) == len(elasticity_values)):
            raise ValueError("all input sequences must have same length")
        
        price_grads = np.asarray(price_gradients, dtype=float)
        volume_grads = np.asarray(volume_gradients, dtype=float)
        elasticity = np.asarray(elasticity_values, dtype=float)
        
        # Entry signal strength: combine gradients with elasticity
        entry_strength = (price_grads + volume_grads) * elasticity
        
        # Apply threshold filtering
        entry_signals = np.where(
            entry_strength > self.elasticity_threshold,
            entry_strength,
            0.0
        )
        
        return entry_signals

    def analyze_exit_signals(
        self,
        entry_signals: np.ndarray,
        profit_targets: Sequence[float],
        risk_factors: Sequence[float],
    ) -> np.ndarray:
        """Analyze exit signals based on entry analysis.

        Parameters
        ----------
        entry_signals
            Entry signal strengths from analyze_entry_signals.
        profit_targets
            Target profit levels.
        risk_factors
            Risk assessment factors.
        """
        if len(profit_targets) != len(risk_factors):
            raise ValueError("profit_targets and risk_factors must have same length")
        
        targets = np.asarray(profit_targets, dtype=float)
        risks = np.asarray(risk_factors, dtype=float)
        
        # Ensure entry_signals matches length
        if len(entry_signals) != len(targets):
            # Broadcast or truncate to match
            min_len = min(len(entry_signals), len(targets))
            entry_signals = entry_signals[:min_len]
            targets = targets[:min_len]
            risks = risks[:min_len]
        
        # Exit signal: inverse relationship with entry strength
        # Strong entry → delayed exit, weak entry → quick exit
        exit_urgency = risks / (entry_signals + 0.1)  # avoid division by zero
        exit_opportunity = targets * np.exp(-entry_signals)
        
        exit_signals = exit_urgency + exit_opportunity
        
        return exit_signals

    def compute_vector_flow(
        self,
        entry_vectors: Sequence[Sequence[float]],
        exit_vectors: Sequence[Sequence[float]],
        elasticity_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute combined entry/exit vector flows.

        Parameters
        ----------
        entry_vectors
            Sequence of entry vector time series.
        exit_vectors
            Sequence of exit vector time series.
        elasticity_matrix
            Routing elasticity matrix.
        """
        if len(entry_vectors) != len(exit_vectors):
            raise ValueError("entry and exit vectors must have same count")
        
        # Convert to numpy arrays
        entry_arrays = [np.asarray(ev, dtype=float) for ev in entry_vectors]
        exit_arrays = [np.asarray(ev, dtype=float) for ev in exit_vectors]
        
        # Find common length
        min_entry_len = min(len(arr) for arr in entry_arrays) if entry_arrays else 0
        min_exit_len = min(len(arr) for arr in exit_arrays) if exit_arrays else 0
        common_len = min(min_entry_len, min_exit_len)
        
        if common_len == 0:
            return np.array([]), np.array([])
        
        # Stack vectors and apply elasticity transformation
        entry_matrix = np.array([arr[:common_len] for arr in entry_arrays])
        exit_matrix = np.array([arr[:common_len] for arr in exit_arrays])
        
        # Apply elasticity matrix if dimensions match
        if elasticity_matrix.shape[0] == len(entry_vectors):
            transformed_entry = elasticity_matrix @ entry_matrix
            transformed_exit = elasticity_matrix @ exit_matrix
        else:
            # Fallback: apply mean elasticity
            mean_elasticity = np.mean(elasticity_matrix)
            transformed_entry = mean_elasticity * entry_matrix
            transformed_exit = mean_elasticity * exit_matrix
        
        return transformed_entry, transformed_exit


# Functional helpers

def compute_routing_elasticity(
    r_function: Callable[[float, float], float],
    positions: Sequence[tuple[float, float]],
    price_series: Sequence[float],
    dt: float = 1.0,
) -> float:  # noqa: D401
    """Compute routing elasticity Λᴿ(t) for given positions and prices."""
    analyzer = EntryExitVectorAnalyzer(dt=dt)
    
    x_pos = [pos[0] for pos in positions]
    y_pos = [pos[1] for pos in positions]
    timestamps = list(range(len(price_series)))
    
    return analyzer.compute_lambda_r(
        r_function, x_pos, y_pos, price_series, timestamps
    )


def analyze_entry_exit_vectors(
    entry_data: Sequence[float],
    exit_data: Sequence[float],
    elasticity_values: Sequence[float],
    threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:  # noqa: D401
    """Analyze entry/exit vectors with elasticity threshold."""
    analyzer = EntryExitVectorAnalyzer(elasticity_threshold=threshold)
    
    # Use entry data as both price and volume gradients (simplified)
    entry_signals = analyzer.analyze_entry_signals(
        entry_data, entry_data, elasticity_values
    )
    
    # Use exit data for targets and risks
    exit_signals = analyzer.analyze_exit_signals(
        entry_signals, exit_data, exit_data
    )
    
    return entry_signals, exit_signals 