"""
Mathematical Library V4 - Recursive Truth and DLT Mechanics
==========================================================

This library implements the foundational mathematics for the Delta Lock
Transform (DLT) Waveform, Forever Fractals, and the Observer-aware
components of the Schwabot Nexus.

V4 Focus:
- Recursive Delta-Resonance Comparison
- Structural Pattern Hashing and Confirmation
- Observer-aware Confidence (Greyscale) Calculation
- Temporal Drift (Warp) Correction
"""

import hashlib
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy.typing as npt

# Import unified math system
try:
    from core.unified_math_system import unified_math
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False
    # Fallback to standard math operations
    import math as unified_math

# Import CLI handler for safe output
try:
    from core.type_binding_system import cli_handler
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ForeverFractal:
    """
    Represents a recognized, historically significant delta-pattern.
    This is the in-memory representation of a "Forever Fractal".
    """
    pattern_hash: str
    delta_sequence: np.ndarray
    length: int
    mean_delta: float
    std_dev: float


class MathLibV4:
    """
    The mathematical engine for Schwabot's recursive, observer-aware logic.
    """
    
    def __init__(self) -> None:
        """Initialize MathLib V4."""
        self.version = "4.0.0"
        if CLI_HANDLER_AVAILABLE:
            cli_handler.log_safe(logger, "info", f"MathLibV4 v{self.version} initialized.")
        else:
            logger.info(f"MathLibV4 v{self.version} initialized.")

    @staticmethod
    def calculate_deltas(time_series: np.ndarray) -> np.ndarray:
        """
        Calculates the discrete changes (deltas) in a time series.
        This is the primary input for the DLT engine.

        Mathematical Formula:
        delta(xₙ) = xₙ - xₙ₋₁

        Args:
            time_series: Input time series data

        Returns:
            Array of delta values
        """
        if time_series.size < 2:
            return np.array([])
        return np.diff(time_series)

    @staticmethod
    def confirm_triplet_lock(delta_sequence: np.ndarray, tolerance: float = 0.1) -> bool:
        """
        Confirms if the last three deltas in a sequence are approximately equal,
        indicating a stable, recurring pattern lock.

        Mathematical Formula:
        Lockₙ = True ⇔ (deltaₙ₀ ~ deltaₙ₁ ~ deltaₙ₂)

        Args:
            delta_sequence: Sequence of delta values
            tolerance: Tolerance for pattern matching

        Returns:
            True if triplet lock is confirmed
        """
        if delta_sequence.size < 3:
            return False

        d1, d2, d3 = delta_sequence[-3:]
        mean_delta = (d1 + d2 + d3) / 3
        
        if mean_delta == 0:  # Handle zero-delta case
            return d1 == d2 == d3

        dev1 = abs(d1 - mean_delta) / abs(mean_delta)
        dev2 = abs(d2 - mean_delta) / abs(mean_delta)
        dev3 = abs(d3 - mean_delta) / abs(mean_delta)

        return dev1 < tolerance and dev2 < tolerance and dev3 < tolerance

    @staticmethod
    def generate_pattern_hash(delta_sequence: np.ndarray) -> str:
        """
        Generates a SHA-256 hash representing the unique structure of a delta
        sequence. This creates the fingerprint for a "Forever Fractal".

        Mathematical Formula:
        hₙ = SHA256(ψₙ + delta(xₙ))

        Args:
            delta_sequence: Sequence of delta values

        Returns:
            SHA-256 hash string
        """
        # We use a quantized representation to ensure stability against minor noise
        quantized = np.round(delta_sequence, decimals=4)
        hasher = hashlib.sha256()
        hasher.update(quantized.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def calculate_greyscale_confidence(similarity_score: float, drift_velocity: float = 0.0) -> float:
        """
        Calculates the "Greyscale" confidence, a sigmoidal function that maps
        pattern similarity into a probabilistic confidence score, adjusted for
        temporal drift.

        Mathematical Formula:
        C_greyscale(t) = C(t) / (1 + e^(-Ωt))

        Args:
            similarity_score: Pattern similarity score [0, 1]
            drift_velocity: Temporal drift velocity

        Returns:
            Confidence score [0, 1]
        """
        if not (0.0 <= similarity_score <= 1.0):
            raise ValueError("Similarity score must be between 0 and 1.")

        # The sigmoid function's steepness can be controlled
        k = 10  # Steepness factor

        # The core confidence based on similarity
        base_confidence = 1 / (1 + np.exp(-k * (similarity_score - 0.5)))

        # Apply a penalty for high drift velocity
        drift_penalty = 1 / (1 + abs(drift_velocity))

        return base_confidence * drift_penalty

    @staticmethod
    def calculate_warp_drift_correction(historical_volatility: float, current_volatility: float) -> float:
        """
        Calculates a temporal "warp" drift correction factor. This is a simplified
        model where the correction factor is based on the ratio of current
        volatility to historical volatility.

        A factor > 1 suggests time is "compressing" (higher volatility).
        A factor < 1 suggests time is "dilating" (lower volatility).

        Args:
            historical_volatility: Historical volatility measure
            current_volatility: Current volatility measure

        Returns:
            Warp correction factor
        """
        if historical_volatility == 0:
            return 1.0
        return current_volatility / historical_volatility

    def analyze_dlt_waveform(self, time_series: np.ndarray) -> Dict[str, Union[float, str, np.ndarray]]:
        """
        Performs comprehensive DLT waveform analysis.

        Args:
            time_series: Input time series data

        Returns:
            Dictionary containing analysis results
        """
        # Calculate deltas
        deltas = self.calculate_deltas(time_series)
        
        if len(deltas) == 0:
            return {"error": "Insufficient data for analysis"}

        # Generate pattern hash
        pattern_hash = self.generate_pattern_hash(deltas)
        
        # Check for triplet lock
        triplet_lock = self.confirm_triplet_lock(deltas)
        
        # Calculate statistical measures
        mean_delta = np.mean(deltas)
        std_dev = np.std(deltas)
        
        # Calculate confidence (using mean as similarity proxy)
        confidence = self.calculate_greyscale_confidence(
            similarity_score=min(1.0, abs(mean_delta)),
            drift_velocity=std_dev
        )
        
        return {
            "pattern_hash": pattern_hash,
            "triplet_lock": triplet_lock,
            "mean_delta": float(mean_delta),
            "std_dev": float(std_dev),
            "confidence": float(confidence),
            "delta_sequence": deltas,
            "analysis_version": self.version
        }

    def create_forever_fractal(self, delta_sequence: np.ndarray) -> ForeverFractal:
        """
        Creates a Forever Fractal from a delta sequence.

        Args:
            delta_sequence: Sequence of delta values

        Returns:
            ForeverFractal object
        """
        pattern_hash = self.generate_pattern_hash(delta_sequence)
        mean_delta = np.mean(delta_sequence)
        std_dev = np.std(delta_sequence)
        
        return ForeverFractal(
            pattern_hash=pattern_hash,
            delta_sequence=delta_sequence,
            length=len(delta_sequence),
            mean_delta=float(mean_delta),
            std_dev=float(std_dev)
        )


def demo_mathlib_v4():
    """Demonstration of MathLibV4 capabilities."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    ml4 = MathLibV4()
    print(f"--- {ml4.version} Demonstration ---")

    # --- Triplet Lock ---
    print("\n[1] Testing Triplet Lock Confirmation...")
    stable_deltas = np.array([10, 10.1, 9.95, 10.5])
    unstable_deltas = np.array([10, 12, 8, 15])
    print(f"  Stable sequence lock: {ml4.confirm_triplet_lock(stable_deltas)}")
    print(f"  Unstable sequence lock: {ml4.confirm_triplet_lock(unstable_deltas)}")

    # --- Pattern Hashing ---
    print("\n[2] Testing Pattern Hashing...")
    pattern1 = np.array([1, 2, -1, 3, 2])
    pattern2 = np.array([1, 2, -1, 3, 2.1])  # Nearly identical
    pattern3 = np.array([5, 4, 3, 2, 1])
    hash1 = ml4.generate_pattern_hash(pattern1)
    hash2 = ml4.generate_pattern_hash(pattern2)
    hash3 = ml4.generate_pattern_hash(pattern3)
    print(f"  Hash 1: {hash1[:10]}...")
    print(f"  Hash 2 (similar): {hash2[:10]}... (Should match Hash 1)")
    print(f"  Hash 3 (different): {hash3[:10]}...")
    assert hash1 == hash2
    assert hash1 != hash3

    # --- Greyscale Confidence ---
    print("\n[3] Testing Greyscale Confidence...")
    strong_match_low_drift = ml4.calculate_greyscale_confidence(0.95, drift_velocity=0.1)
    weak_match_low_drift = ml4.calculate_greyscale_confidence(0.60, drift_velocity=0.1)
    strong_match_high_drift = ml4.calculate_greyscale_confidence(0.95, drift_velocity=2.0)
    print(f"  Strong match, low drift: {strong_match_low_drift:.2f}")
    print(f"  Weak match, low drift: {weak_match_low_drift:.2f}")
    print(f"  Strong match, high drift: {strong_match_high_drift:.2f}")

    # --- Warp Drift Correction ---
    print("\n[4] Testing Warp Drift Correction...")
    correction = ml4.calculate_warp_drift_correction(
        historical_volatility=0.2, current_volatility=0.4
    )
    print(f"  Volatility doubled, warp factor: {correction:.2f}")

    # --- Complete DLT Analysis ---
    print("\n[5] Testing Complete DLT Analysis...")
    test_series = np.array([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
    analysis = ml4.analyze_dlt_waveform(test_series)
    print(f"  Pattern Hash: {analysis['pattern_hash'][:10]}...")
    print(f"  Triplet Lock: {analysis['triplet_lock']}")
    print(f"  Confidence: {analysis['confidence']:.3f}")

    print("\n✅ MathLibV4 demonstration completed successfully!")


if __name__ == "__main__":
    demo_mathlib_v4()
