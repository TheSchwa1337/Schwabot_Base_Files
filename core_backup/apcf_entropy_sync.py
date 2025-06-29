# -*- coding: utf-8 -*-
""""""
Auto-Calibrating Entropy Delta Controller
=========================================

This module acts as a safety valve for the APCF. It monitors market
entropy over several cycles. If entropy remains persistently low (indicating)
a flat, sideways market) while the APCF is still generating execution
signals, this controller can desensitize the APCF to prevent it from
trading on noise in a non-trending market.
""""""

import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)


class APCFEntropySync:
    """"""
    Auto-calibrates APCF sensitivity based on sustained entropy levels.
    """"""

    def __init__(self, entropy_window: int = 20, low_entropy_threshold: float = 0.15):
        """"""
        Initializes the entropy sync controller.

        Args:
            entropy_window: The number of recent entropy readings to average.
            low_entropy_threshold: The entropy level below which the market
                                   is considered "flat" or "sideways."
        """"""
        self.entropy_window = entropy_window
        self.low_entropy_threshold = low_entropy_threshold
        self.recent_entropy_values = deque(maxlen=self.entropy_window)
        self.is_desensitized = False
        logger.info("APCF Entropy Sync Controller initialized.")

    def record_entropy(self, entropy: float):
        """Records a new entropy value for the current tick."""
        self.recent_entropy_values.append(entropy)

    def get_market_state(self) -> Dict[str, Any]:
        """"""
        Analyzes the recent entropy trend to determine the market state.

        Returns:
            A dictionary describing the current entropy-derived market state.
        """"""
        if len(self.recent_entropy_values) < self.entropy_window:
            return {"state": "CALIBRATING", "average_entropy": -1.0}

        average_entropy = sum(self.recent_entropy_values) / len(self.recent_entropy_values)

        if average_entropy < self.low_entropy_threshold:
            state = "PERSISTENT_LOW_ENTROPY"
            self.is_desensitized = True
        else:
            state = "NORMAL_ENTROPY"
            self.is_desensitized = False

        return {"state": state, "average_entropy": average_entropy}

    def get_sensitivity_adjustment(self) -> float:
        """"""
        Returns an adjustment factor for the APCF's Θ (theta) parameter.'
        If the market is in a low-entropy state, this will return a value
        less than 1.0 to desensitize the APCF.

        Returns:
            A multiplier (e.g., 0.7 for desensitized, 1.0 for normal).
        """"""
        market_state_info = self.get_market_state()
        market_state = market_state_info["state"]

        if market_state == "PERSISTENT_LOW_ENTROPY":
            adjustment = 0.7  # Desensitize by 30%
            logger.warning(f"Market in persistent low entropy state. Applying {adjustment} sensitivity adjustment.")
            return adjustment

        return 1.0  # Normal sensitivity


# Global instance
apcf_entropy_sync = APCFEntropySync()
