#!/usr/bin/env python3
"""Tick Resonance Engine - Harmony Score Calculator.

This module computes harmony scores (𝓗) that measure how well tick timing
aligns with expected phase gates (4-bit, 8-bit, 42-bit). The harmony score
feeds into the entropy-weighted entry score calculation.

Mathematical Foundation:
𝓗 = exp(-mean(|tick_i - φ_target|)^2)

Where:
- tick_i: Time deltas between consecutive ticks
- φ_target: Target phase timing for current bit depth
- Result in [0, 1] where 1 = perfect harmony

Windows CLI compatible with ASCII fallback for special characters.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Phase target timings (in seconds)
PHASE_TARGETS = {
    4: 0.25,  # 4-bit: 250ms target
    8: 0.125,  # 8-bit: 125ms target
    42: 0.024,  # 42-bit: ~24ms target (high frequency)
}

# Harmony calculation parameters
HARMONY_WINDOW_SIZE = 20  # Number of recent ticks to analyze
MIN_TICKS_REQUIRED = 3  # Minimum ticks needed for calculation


def compute_harmony_vector(
    tick_deltas: np.ndarray,
    target_phase: float,
    window_size: int = HARMONY_WINDOW_SIZE,
) -> float:
    """Compute harmony score for tick timing alignment.

    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of time deltas between consecutive ticks (in seconds)
    target_phase : float
        Target timing for current phase (in seconds)
    window_size : int, optional
        Number of recent deltas to analyze

    Returns
    -------
    float
        Harmony score in [0, 1] where 1 = perfect alignment
    """
    try:
        if len(tick_deltas) < MIN_TICKS_REQUIRED:
            logger.debug(f"Insufficient ticks for harmony: {len(tick_deltas)}")
            return 0.0

        # Use most recent window
        recent_deltas = tick_deltas[-window_size:]

        # Calculate absolute deviations from target
        deviations = np.abs(recent_deltas - target_phase)

        # Compute mean squared deviation
        mean_sq_deviation = np.mean(deviations**2)

        # Convert to harmony score using exponential decay
        harmony = float(np.exp(-mean_sq_deviation))

        # Ensure valid range
        return max(0.0, min(1.0, harmony))

    except Exception as e:
        logger.warning(f"Error computing harmony vector: {e}")
        return 0.0


def get_phase_target(bit_depth: int) -> float:
    """Get target timing for specified bit depth.

    Parameters
    ----------
    bit_depth : int
        Phase bit depth (4, 8, or 42)

    Returns
    -------
    float
        Target timing in seconds
    """
    return PHASE_TARGETS.get(bit_depth, PHASE_TARGETS[8])  # Default to 8-bit


def analyze_tick_pattern(
    tick_deltas: np.ndarray,
    bit_depth: int = 8,
) -> Tuple[float, dict]:
    """Analyze tick pattern and return harmony with diagnostics.

    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of tick time deltas
    bit_depth : int, optional
        Phase bit depth for target timing

    Returns
    -------
    Tuple[float, dict]
        - Harmony score (0-1)
        - Diagnostic information dictionary
    """
    try:
        target = get_phase_target(bit_depth)
        harmony = compute_harmony_vector(tick_deltas, target)

        # Calculate diagnostic metrics
        if len(tick_deltas) >= MIN_TICKS_REQUIRED:
            recent_deltas = tick_deltas[-HARMONY_WINDOW_SIZE:]
            mean_delta = float(np.mean(recent_deltas))
            std_delta = float(np.std(recent_deltas))
            deviation_from_target = abs(mean_delta - target)
        else:
            mean_delta = 0.0
            std_delta = 0.0
            deviation_from_target = float("inf")

        diagnostics = {
            "harmony_score": harmony,
            "target_timing": target,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "deviation_from_target": deviation_from_target,
            "tick_count": len(tick_deltas),
            "bit_depth": bit_depth,
        }

        return harmony, diagnostics

    except Exception as e:
        logger.error(f"Error analyzing tick pattern: {e}")
        return 0.0, {"error": str(e)}


def compute_multi_phase_harmony(
    tick_deltas: np.ndarray,
    phases: Optional[List[int]] = None,
) -> dict:
    """Compute harmony scores for multiple phase depths.

    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of tick time deltas
    phases : List[int], optional
        List of bit depths to analyze (default: [4, 8, 42])

    Returns
    -------
    dict
        Dictionary mapping bit depth to harmony score
    """
    if phases is None:
        phases = [4, 8, 42]

    results = {}

    for phase in phases:
        try:
            harmony, _ = analyze_tick_pattern(tick_deltas, phase)
            results[phase] = harmony
        except Exception as e:
            logger.warning(f"Error computing harmony for phase {phase}: {e}")
            results[phase] = 0.0

    return results


def get_optimal_phase(tick_deltas: np.ndarray) -> Tuple[int, float]:
    """Determine optimal phase depth based on harmony scores.

    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of tick time deltas

    Returns
    -------
    Tuple[int, float]
        - Optimal bit depth
        - Harmony score for optimal phase
    """
    try:
        harmonies = compute_multi_phase_harmony(tick_deltas)

        if not harmonies:
            return 8, 0.0  # Default fallback

        # Find phase with highest harmony
        optimal_phase = max(harmonies.items(), key=lambda x: x[1])
        return optimal_phase[0], optimal_phase[1]

    except Exception as e:
        logger.error(f"Error determining optimal phase: {e}")
        return 8, 0.0


class TickResonanceEngine:
    """Main class for tick resonance analysis."""

    def __init__(self, default_bit_depth: int = 8):
        """Initialize tick resonance engine.

        Parameters
        ----------
        default_bit_depth : int, optional
            Default phase bit depth to use
        """
        self.default_bit_depth = default_bit_depth
        self.tick_history: List[float] = []
        self.last_harmony = 0.0
        self.last_diagnostics: dict = {}

    def update_tick(self, timestamp: float) -> None:
        """Update with new tick timestamp.

        Parameters
        ----------
        timestamp : float
            Tick timestamp in seconds
        """
        self.tick_history.append(timestamp)

        # Keep reasonable history size
        if len(self.tick_history) > 100:
            self.tick_history = self.tick_history[-50:]

    def get_current_harmony(self, bit_depth: Optional[int] = None) -> float:
        """Get current harmony score.

        Parameters
        ----------
        bit_depth : int, optional
            Bit depth to use (default: instance default)

        Returns
        -------
        float
            Current harmony score
        """
        if len(self.tick_history) < 2:
            return 0.0

        # Calculate time deltas
        deltas = np.diff(self.tick_history)

        # Use specified or default bit depth
        depth = bit_depth or self.default_bit_depth

        # Compute and cache harmony
        self.last_harmony, self.last_diagnostics = analyze_tick_pattern(deltas, depth)

        return self.last_harmony

    def get_diagnostics(self) -> dict:
        """Get latest diagnostic information."""
        return self.last_diagnostics.copy()

    def reset(self) -> None:
        """Reset tick history and cached values."""
        self.tick_history.clear()
        self.last_harmony = 0.0
        self.last_diagnostics = {}


def validate_tick_deltas(tick_deltas: np.ndarray) -> bool:
    """Validate tick delta array for harmony calculation.

    Parameters
    ----------
    tick_deltas : np.ndarray
        Array of tick time deltas

    Returns
    -------
    bool
        True if valid for harmony calculation
    """
    try:
        if not isinstance(tick_deltas, np.ndarray):
            return False

        if len(tick_deltas) < MIN_TICKS_REQUIRED:
            return False

        # Check for reasonable timing values (1μs to 10s)
        if np.any(tick_deltas <= 0) or np.any(tick_deltas > 10.0):
            return False

        # Check for NaN or infinite values
        if not np.all(np.isfinite(tick_deltas)):
            return False

        return True

    except Exception:
        return False


def main() -> None:
    """Demo function for testing tick resonance engine."""
    # Create test tick pattern
    target_delta = 0.125  # 8-bit target
    num_ticks = 30

    # Perfect pattern
    perfect_deltas = np.full(num_ticks, target_delta)
    harmony_perfect = compute_harmony_vector(perfect_deltas, target_delta)

    # Noisy pattern
    noise = np.random.normal(0, 0.01, num_ticks)  # 10ms noise
    noisy_deltas = perfect_deltas + noise
    harmony_noisy = compute_harmony_vector(noisy_deltas, target_delta)

    # Random pattern
    random_deltas = np.random.uniform(0.05, 0.3, num_ticks)
    harmony_random = compute_harmony_vector(random_deltas, target_delta)

    print("Tick Resonance Engine Demo")
    print("=" * 30)
    print(f"Perfect pattern harmony: {harmony_perfect:.3f}")
    print(f"Noisy pattern harmony:   {harmony_noisy:.3f}")
    print(f"Random pattern harmony:  {harmony_random:.3f}")
    print()

    # Test engine class
    engine = TickResonanceEngine()

    # Simulate tick stream
    base_time = 1000.0
    for i in range(20):
        tick_time = base_time + i * (target_delta + np.random.normal(0, 0.005))
        engine.update_tick(tick_time)

    current_harmony = engine.get_current_harmony()
    diagnostics = engine.get_diagnostics()

    print(f"Engine current harmony: {current_harmony:.3f}")
    print(f"Mean delta: {diagnostics.get('mean_delta', 0):.3f}s")
    print(f"Target delta: {diagnostics.get('target_timing', 0):.3f}s")


if __name__ == "__main__":
    main()
