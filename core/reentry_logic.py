"""
Reentry Logic Module - Dynamic re-entry based on drift patterns, swing metrics, and performance.
"""

import logging
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class ReentryLogic:
    """
    Evaluates if and when to re-enter a position after exit to maximize profit.
    """
    def __init__(self, min_confidence: float = 0.5, reentry_cooldown: float = 300):
        self.min_confidence = min_confidence
        self.reentry_cooldown = reentry_cooldown
        self.last_reentry_time = 0.0

    def evaluate_reentry(self, tick_cycle: Any, swing_metrics: Dict[str, Any], drift_vector: Dict[str, float]) -> (bool, float):
        """
        Decide whether to re-enter a trade.
        Returns (should_reenter: bool, amount: float).
        """
        current_time = time.time()
        if current_time - self.last_reentry_time < self.reentry_cooldown:
            return False, 0.0

        confidence = tick_cycle.confidence_score
        swing_strength = swing_metrics.get('swing_strength', 0.0)

        # Simple rule: re-enter if confidence and swing strength high
        if confidence > self.min_confidence and swing_strength > 0.5:
            # Determine amount as fraction of available USDC
            allocation = 0.1  # 10% reentry
            amount = tick_cycle.usdc_balance * allocation
            self.last_reentry_time = current_time
            logger.info(f"ReentryLogic: triggering reentry with amount {amount}")
            return True, amount

        return False, 0.0 