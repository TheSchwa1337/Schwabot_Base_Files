#!/usr/bin/env python3
"""conditional_glyph_feedback_loop – news flow scalar feedback implementation.

Implements the news flow scalar feedback logic:
    ∇Φ(t, x) = γ·∂²φ/∂x² − λ·φ

This module handles conditional glyph feedback loops for news integration
into the ghost trading system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__: list[str] = [
    "ConditionalGlyphFeedback",
    "compute_news_flow_gradient",
    "apply_feedback_loop",
]


@dataclass(slots=True)
class ConditionalGlyphFeedback:
    """Conditional glyph feedback loop processor."""
    
    gamma: float = 1.0
    lambda_param: float = 0.5
    dx: float = 0.1

    def compute_nabla_phi(
        self,
        phi_func: Callable[[float], float],
        x: float,
    ) -> float:
        """Compute ∇Φ(t, x) = γ·∂²φ/∂x² − λ·φ.

        Parameters
        ----------
        phi_func
            Function φ(x) to compute derivatives of.
        x
            Position to evaluate at.
        """
        # Compute φ(x)
        phi_x = phi_func(x)
        
        # Compute second derivative using finite differences
        phi_plus = phi_func(x + self.dx)
        phi_minus = phi_func(x - self.dx)
        phi_center = phi_func(x)
        
        # Second derivative: ∂²φ/∂x² ≈ (φ(x+h) - 2φ(x) + φ(x-h)) / h²
        second_derivative = (phi_plus - 2 * phi_center + phi_minus) / (self.dx ** 2)
        
        # Apply formula: γ·∂²φ/∂x² − λ·φ
        nabla_phi = self.gamma * second_derivative - self.lambda_param * phi_x
        
        return nabla_phi

    def process_news_feedback(
        self,
        news_values: Sequence[float],
        x_positions: Sequence[float],
    ) -> np.ndarray:
        """Process news feedback using scalar flow gradient.

        Parameters
        ----------
        news_values
            News scalar values at different positions.
        x_positions
            Spatial positions corresponding to news values.
        """
        if len(news_values) != len(x_positions):
            raise ValueError("news_values and x_positions must have same length")
        
        news_array = np.asarray(news_values, dtype=float)
        x_array = np.asarray(x_positions, dtype=float)
        
        # Create interpolation function for news values
        def news_func(x: float) -> float:
            # Simple linear interpolation
            if len(news_array) < 2:
                return news_array[0] if len(news_array) > 0 else 0.0
            return float(np.interp(x, x_array, news_array))
        
        # Compute feedback gradient at each position
        feedback_gradients = np.zeros_like(x_array, dtype=float)
        for i, x_pos in enumerate(x_array):
            feedback_gradients[i] = self.compute_nabla_phi(news_func, x_pos)
        
        return feedback_gradients

    def apply_conditional_feedback(
        self,
        glyph_state: np.ndarray,
        feedback_gradients: np.ndarray,
        condition_threshold: float = 0.5,
    ) -> np.ndarray:
        """Apply conditional feedback based on threshold.

        Parameters
        ----------
        glyph_state
            Current glyph state vector.
        feedback_gradients
            Computed feedback gradients.
        condition_threshold
            Threshold for applying feedback.
        """
        if len(glyph_state) != len(feedback_gradients):
            raise ValueError("glyph_state and feedback_gradients length mismatch")
        
        # Apply feedback only where condition is met
        condition_mask = np.abs(feedback_gradients) > condition_threshold
        
        updated_state = glyph_state.copy()
        updated_state[condition_mask] += feedback_gradients[condition_mask]
        
        return updated_state


# Functional helpers

def compute_news_flow_gradient(
    news_values: Sequence[float],
    x_positions: Sequence[float],
    gamma: float = 1.0,
    lambda_param: float = 0.5,
) -> np.ndarray:  # noqa: D401
    """Compute news flow scalar feedback gradient."""
    feedback = ConditionalGlyphFeedback(gamma=gamma, lambda_param=lambda_param)
    return feedback.process_news_feedback(news_values, x_positions)


def apply_feedback_loop(
    glyph_state: Sequence[float],
    news_values: Sequence[float],
    x_positions: Sequence[float],
    threshold: float = 0.5,
) -> np.ndarray:  # noqa: D401
    """Apply complete conditional glyph feedback loop."""
    feedback = ConditionalGlyphFeedback()
    gradients = feedback.process_news_feedback(news_values, x_positions)
    glyph_array = np.asarray(glyph_state, dtype=float)
    return feedback.apply_conditional_feedback(glyph_array, gradients, threshold) 