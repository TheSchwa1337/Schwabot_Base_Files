"""
strategy_layered_gatekeeper.py
------------------------------
Combines traditional TA indicators with profit bucket overrides to create
a layered decision system that can bypass indicators when profitable
patterns are detected.
"""

from __future__ import annotations

from typing import Tuple

from .profit_bucket_registry import ProfitBucketRegistry
from .vector_band_gatekeeper import confirm_long_trend, confirm_mid_vector, confirm_short_drift


class StrategyLayeredGatekeeper:
    """Layered gatekeeper that combines TA with profit bucket overrides."""

    def __init__(self):
        self.profit_registry = ProfitBucketRegistry()
        self.override_threshold = 0.8  # Hash confidence needed for override

    def evaluate_all_gates(self,  tick_blob: str) -> Tuple[bool, str, float]:
        """
        Evaluate all gates and return (passed, reason, confidence).

        Returns:
            - passed: True if trade should execute
            - reason: Human-readable explanation
            - confidence: 0-1 confidence score
        """
        # 1. Check traditional TA indicators
        short_ok = confirm_short_drift(tick_blob)
        mid_ok = confirm_mid_vector(tick_blob)
        long_ok = confirm_long_trend(tick_blob)

        indicator_score = sum([short_ok, mid_ok, long_ok]) / 3.0

        # 2. Check for profitable pattern override
        profit_bucket = self.profit_registry.find_matching_pattern(tick_blob)
        override_available = profit_bucket is not None
        override_confidence = profit_bucket.confidence if profit_bucket else 0.0

        # 3. Decision logic
        if indicator_score >= 0.66:
            # Traditional indicators are strong - execute
            return True, f"Strong indicators ({indicator_score:.2f})", indicator_score

        elif override_confidence >= self.override_threshold:
            # Profit bucket override - execute despite weak indicators
            return True, f"Profit override ({override_confidence:.2f})", override_confidence

        elif indicator_score >= 0.33 and override_confidence >= 0.5:
            # Mixed signal but both are reasonable
            combined_confidence = (indicator_score + override_confidence) / 2
            return True, f"Mixed signal ({combined_confidence:.2f})", combined_confidence

        else:
            # No clear signal
            return (
                False,
                f"Weak signals (TA: {indicator_score:.2f}, Override: {override_confidence:.2f})",
                0.0,
            )

    def get_exit_strategy(self,  tick_blob: str) -> Tuple[float, int] | None:
        """Get exit price and time from profit bucket registry."""
        return self.profit_registry.get_exit_strategy(tick_blob)

    def record_profitable_trade(
        self,
        tick_blob: str,
        entry_price: float,
        exit_price: float,
        time_to_exit: int,
        strategy_id: str,
    ) -> None:
        """Record a successful trade for future pattern matching."""
        self.profit_registry.add_profitable_trade(tick_blob, entry_price, exit_price, time_to_exit, strategy_id)
