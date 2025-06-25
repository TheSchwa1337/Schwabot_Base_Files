from __future__ import annotations

from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Entry Gate - Mathematical Execution Confidence Evaluator.

This module provides the core mathematical gates that determine whether
a trading opportunity should be executed, deferred, or rejected based on
unified confidence metrics and entropy-weighted entry scores.

Key Functions:
- execution_confidence(): Computes Ξ scalar from fractal state
- entry_score(): Computes 𝓔ₛ from harmony, drift, liquidity, profit
- evaluate(): Main gate logic for trade execution decisions

Mathematical Foundation:
Ξ = (T · Δθ) + (ε × σ_f) + τ_p
𝓔ₛ = 𝓗 × (1 − 𝓓ₚ) × 𝓛 × P̂

Windows CLI compatible with flake8 compliance.
"""


import logging
from typing import Any, Dict, Optional

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)

# Mathematical constants
MIN_CONFIDENCE_THRESHOLD = 1.15
MIN_ENTRY_SCORE_THRESHOLD = 0.90
DEFER_ENTRY_SCORE_THRESHOLD = 0.70


def execution_confidence(
    triplet_entropy: float,
    theta_drift: float,
    coherence: float,
    loop_volatility: float,
    profit_decay: float,
) -> float:
    """Calculate execution confidence scalar Ξ.

    Parameters
    ----------
    triplet_entropy : float
        T - Information rate from triplet patterns (0-1)
    theta_drift : float
        Δθ - Normalized braid angle drift (0-1)
    coherence : float
        ε - Fractal coherence score (0-1)
    loop_volatility : float
        σ_f - Standard deviation of loop sums (0-1)
    profit_decay : float
        τ_p - Time-weighted profit modifier (0-0.3)

    Returns
    -------
    float
        Ξ - Execution confidence scalar
        >1.15: Execute immediately
        0.85-1.15: Route to GAN filter
        <0.85: Defer/cooldown
    """
    try:
        # Ξ = (T · Δθ) + (ε × σ_f) + τ_p
        confidence = (
            (triplet_entropy * theta_drift)
            + (coherence * loop_volatility)
            + profit_decay
        )

        # Ensure reasonable bounds
        return unified_math.max(0.0, unified_math.min(3.0, confidence))

    except (ValueError, TypeError) as e:
        logger.warning(f"Error computing execution confidence: {e}")
        return 0.0


def entry_score(
    harmony: float,
    drift_penalty: float,
    liquidity_score: float,
    projected_profit: float,
) -> float:
    """Calculate entropy-weighted entry score 𝓔ₛ.

    Parameters
    ----------
    harmony : float
        𝓗 - Tick harmony alignment score (0-1)
    drift_penalty : float
        𝓓ₚ - Phase drift penalty (0-1)
    liquidity_score : float
        𝓛 - Normalized liquidity depth score (0-1)
    projected_profit : float
        P̂ - Expected profit ratio (0-1)

    Returns
    -------
    float
        𝓔ₛ - Entry score
        >0.90: Execute
        0.70-0.90: Route to GAN review
        <0.70: Suppress/cooldown
    """
    try:
        # 𝓔ₛ = 𝓗 × (1 − 𝓓ₚ) × 𝓛 × P̂
        score = harmony * (1.0 - drift_penalty) * liquidity_score * projected_profit

        # Ensure valid range
        return unified_math.max(0.0, unified_math.min(1.0, score))

    except (ValueError, TypeError) as e:
        logger.warning(f"Error computing entry score: {e}")
        return 0.0


def evaluate(
    confidence: float,
    entry_score_val: float,
    gan_filter_result: Optional[bool] = None,
) -> Dict[str, Any]:
    """Main entry gate evaluation logic.

    Parameters
    ----------
    confidence : float
        Ξ - Execution confidence scalar
    entry_score_val : float
        𝓔ₛ - Entropy-weighted entry score
    gan_filter_result : bool, optional
        Result from GAN anomaly filter (if available)

    Returns
    -------
    Dict[str, Any]
        Decision dictionary with:
        - action: "execute", "defer", "gan_review", "cooldown"
        - confidence: Confidence value
        - entry_score: Entry score value
        - reason: Human-readable explanation
    """
    try:
        # Primary gate: both confidence and entry score must pass
        if (
            confidence > MIN_CONFIDENCE_THRESHOLD
            and entry_score_val > MIN_ENTRY_SCORE_THRESHOLD
        ):

            # Check GAN filter if available
            if gan_filter_result is False:
                return {
                    "action": "defer",
                    "confidence": confidence,
                    "entry_score": entry_score_val,
                    "reason": "GAN anomaly filter rejection",
                }

            return {
                "action": "execute",
                "confidence": confidence,
                "entry_score": entry_score_val,
                "reason": "High confidence and entry score",
            }

        # Secondary gate: route to GAN review if entry score in middle band
        elif confidence > 0.85 and entry_score_val > DEFER_ENTRY_SCORE_THRESHOLD:

            return {
                "action": "gan_review",
                "confidence": confidence,
                "entry_score": entry_score_val,
                "reason": "Moderate scores - route to GAN filter",
            }

        # Tertiary: cooldown for low scores
        else:
            reason_parts = []
            if confidence <= 0.85:
                reason_parts.append(f"low confidence ({confidence:.3f})")
            if entry_score_val <= DEFER_ENTRY_SCORE_THRESHOLD:
                reason_parts.append(f"low entry score ({entry_score_val:.3f})")

            return {
                "action": "cooldown",
                "confidence": confidence,
                "entry_score": entry_score_val,
                "reason": "Cooldown: " + ", ".join(reason_parts),
            }

    except Exception as e:
        logger.error(f"Error in entry gate evaluation: {e}")
        return {
            "action": "cooldown",
            "confidence": 0.0,
            "entry_score": 0.0,
            "reason": f"Evaluation error: {e}",
        }


def get_thresholds() -> Dict[str, float]:
    """Get current threshold values for monitoring/tuning."""
    return {
        "min_confidence": MIN_CONFIDENCE_THRESHOLD,
        "min_entry_score": MIN_ENTRY_SCORE_THRESHOLD,
        "defer_entry_score": DEFER_ENTRY_SCORE_THRESHOLD,
    }


# Quick validation function for testing
def validate_inputs(
    triplet_entropy: float,
    theta_drift: float,
    coherence: float,
    loop_volatility: float,
    profit_decay: float,
    harmony: float,
    drift_penalty: float,
    liquidity_score: float,
    projected_profit: float,
) -> bool:
    """Validate that all input values are in expected ranges."""
    try:
        # Check ranges for all inputs
        checks = [
            0.0 <= triplet_entropy <= 1.0,
            0.0 <= theta_drift <= 1.0,
            0.0 <= coherence <= 1.0,
            0.0 <= loop_volatility <= 1.0,
            0.0 <= profit_decay <= 0.5,  # Slightly higher bound for profit decay
            0.0 <= harmony <= 1.0,
            0.0 <= drift_penalty <= 1.0,
            0.0 <= liquidity_score <= 1.0,
            0.0 <= projected_profit <= 1.0,
        ]

        return all(checks)

    except Exception:
        return False


def main() -> None:
    """Demo function for testing entry gate logic."""
    # Test case 1: High confidence scenario
    xi = execution_confidence(0.83, 0.12, 0.92, 0.18, 0.04)
    es = entry_score(0.88, 0.12, 0.75, 0.03)
    result = evaluate(xi, es)

    safe_print(f"Test 1 - Ξ: {xi:.3f}, 𝓔ₛ: {es:.3f}")
    safe_print(f"Decision: {result['action']} - {result['reason']}")
    print()

    # Test case 2: Moderate confidence scenario
    xi2 = execution_confidence(0.65, 0.08, 0.78, 0.15, 0.02)
    es2 = entry_score(0.82, 0.08, 0.85, 0.025)
    result2 = evaluate(xi2, es2)

    safe_print(f"Test 2 - Ξ: {xi2:.3f}, 𝓔ₛ: {es2:.3f}")
    safe_print(f"Decision: {result2['action']} - {result2['reason']}")


if __name__ == "__main__":
    main()
