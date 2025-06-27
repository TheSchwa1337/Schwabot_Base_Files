# -*- coding: utf-8 -*-
"""Ghost condition - phase-aware gate for routing decisions."""

from core.unified_math_system import unified_math

def ghost_conditional(
    psi: float, xi_sent: float, phi_drift: float, phase: str = "mid", unicode_label: str = ""
) -> dict:
    """
    Phase-aware ghost conditional logic for routing decisions.

    Args:
        psi: Path coefficient (0-1)
        xi_sent: Sentiment coefficient (0-1)
        phi_drift: Drift coefficient (0-1)
        phase: 'low', 'mid', or 'high' (affects threshold/logic)
        unicode_label: Optional Unicode/emoji label for state

    Returns:
        dict: {
            'decision': bool,
            'score': float,
            'phase': str,
            'label': str
        }
    """
    z = psi * xi_sent * phi_drift
    phase_threshold = {"low": 0.3, "mid": 0.5, "high": 0.7}[phase]
    sigma = 1 / (1 + unified_math.exp(-12 * (z - phase_threshold)))
    return {
        "decision": bool(sigma >= phase_threshold),
        "score": sigma,
        "phase": phase,
        "label": unicode_label
    } 