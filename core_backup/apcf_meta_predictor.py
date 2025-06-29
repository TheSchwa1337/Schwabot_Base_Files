# -*- coding: utf-8 -*-
""""""
APCF Meta-Predictor Core
========================

A recursive prediction layer that learns which APCF values and patterns
produce profitable outcomes over the long term. It uses this knowledge
to refine APCF thresholds and enable AI agents to forecast future cycles.

Mathematical Model:
M(t) = sum [APCF(i) * ROI(i)] / n for i in [t-n, t]

This meta-value M(t) is used to adjust the core APCF's behavior, creating'
a powerful feedback loop.
""""""

import logging
from typing import Any, Dict, List

import numpy as np

from .adaptive_profit_cycle_function import APCFResult

logger = logging.getLogger(__name__)


class APCFMetaPredictor:
    """"""
    Analyzes historical APCF results to predict future profitable cycles
    and refine the APCF's core parameters.'
    """"""

    def __init__(self, learning_rate=0.5, window_size=100):
        """"""
        Initializes the Meta-Predictor.

        Args:
            learning_rate: How quickly the model adapts its threshold advice.
            window_size: The number of recent APCF results to consider for analysis.
        """"""
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.apcf_history: List[APCFResult] = []
        # Stores { 'apcf_signature': str, 'roi': float }
        self.trade_outcomes: List[Dict[str, Any]] = []

        # Thresholds that the meta-predictor can advise on
        self.advised_thresholds = {"execution_threshold": 1.0, "hold_threshold": 0.8}
        logger.info("APCF Meta-Predictor initialized.")

    def record_apcf_result(self, result: APCFResult):
        """Records a new APCF result for future analysis."""
        self.apcf_history.append(result)
        if len(self.apcf_history) > self.window_size:
            self.apcf_history.pop(0)

    def record_trade_outcome(self, signature: str, roi: float):
        """"""
        Records the outcome (Return on Investment) of a trade linked to a specific
        APCF calculation via its mathematical signature.
        """"""
        self.trade_outcomes.append({"apcf_signature": signature, "roi": roi})
        if len(self.trade_outcomes) > self.window_size * 2:  # Keep a larger history of outcomes
            self.trade_outcomes.pop(0)

        # A good time to run model refinement is after a new outcome is known
        self.refine_thresholds()

    def calculate_meta_value(self) -> float:
        """"""
        Calculates the historical performance meta-value M(t).
        This represents the weighted average of profitable APCF values.
        """"""
        if not self.trade_outcomes:
            return 0.0

        # Create a mapping from signature to APCF value for quick lookup
        sig_to_apcf = {res.mathematical_signature: res.apcf_value for res in self.apcf_history}

        weighted_sum = 0
        total_weight = 0

        for outcome in self.trade_outcomes:
            sig = outcome["apcf_signature"]
            roi = outcome["roi"]
            if sig in sig_to_apcf:
                apcf_val = sig_to_apcf[sig]
                # We only learn from profitable trades
                if roi > 0:
                    # Weight successful APCF values by their profitability
                    weighted_sum += apcf_val * roi
                    total_weight += roi

        if total_weight == 0:
            return 0.0

        meta_value = weighted_sum / total_weight
        logger.debug(f"Calculated new APCF meta-value: {meta_value:.3f}")
        return meta_value

    def refine_thresholds(self):
        """"""
        Adjusts the APCF execution thresholds based on the calculated
        meta-value, creating a self-correcting feedback loop.
        """"""
        meta_value = self.calculate_meta_value()

        if meta_value == 0.0:
            # Not enough data to refine
            return

        # Get current execution threshold
        current_threshold = self.advised_thresholds["execution_threshold"]

        # Move the threshold gently towards the meta_value
        # If historically profitable trades happened at APCF=1.3, we might raise the threshold
        # If they happened at 0.9, we might lower it.
        new_threshold = current_threshold * (1 - self.learning_rate) + meta_value * self.learning_rate

        # Ensure the threshold stays within a reasonable range (e.g., 0.8 to)
        # 1.5)
        new_threshold = np.clip(new_threshold, 0.8, 1.5)

        self.advised_thresholds["execution_threshold"] = new_threshold
        # Keep hold threshold relative
        self.advised_thresholds["hold_threshold"] = new_threshold - 0.2

        logger.info()
            f"APCF thresholds refined. New execution threshold: {"}
                new_threshold:.3f}""
        )

    def get_advised_thresholds(self) -> Dict[str, float]:
        """Returns the currently advised thresholds for the core APCF."""
        return self.advised_thresholds.copy()


# Global instance
apcf_meta_predictor = APCFMetaPredictor()
