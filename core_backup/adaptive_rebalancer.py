# -*- coding: utf-8 -*-
""""""
Adaptive Strategy Rebalancer
============================

This module acts as a dynamic logic gate, rerouting or modifying APCF
triggers based on real-time market context (e.g., trend strength, volume,)
entropy shifts). It provides a crucial layer of adaptability, ensuring
that APCF-driven actions are appropriate for the immediate market climate.
""""""

import logging
from typing import Any, Dict

from .adaptive_profit_cycle_function import APCFResult, APCFState

logger = logging.getLogger(__name__)


class AdaptiveRebalancer:
    """"""
    Dynamically reroutes APCF decisions based on market context.
    """"""

    def __init__(self):
        """Initializes the adaptive rebalancer."""
        logger.info("Adaptive Strategy Rebalancer initialized.")

    def review_and_reroute(self, apcf_result: APCFResult, market_context: Dict[str, Any]) -> APCFResult:
        """"""
        Reviews an APCF result against the current market context and decides
        if the original decision should be overridden or modified.

        Args:
            apcf_result: The original result from the APCF calculation.
            market_context: A dictionary with real-time market data like
                            'is_trend_flat', 'entropy', 'volume'.

        Returns:
            The original or a modified APCFResult.
        """"""

        original_state = apcf_result.state
        new_state = original_state
        reroute_reason = "None"

        is_trend_flat = market_context.get("is_trend_flat", False)
        entropy = market_context.get("entropy", 0.5)
        volume = market_context.get("volume", 0.5)

        # Rule 1: Override EXECUTE in a flat, low-volume trend to prevent
        # whipsaws.
        if original_state == APCFState.EXECUTE and is_trend_flat and volume < 0.2:
            new_state = APCFState.HOLD
            reroute_reason = "Overrode EXECUTE to HOLD due to flat, low-volume trend."

        # Rule 2: Force DEFER or VAULT_LOCK if entropy is critically high.
        elif entropy > 0.9:
            new_state = APCFState.VAULT_LOCK
            reroute_reason = f"Forced VAULT_LOCK due to critical entropy ({")}
                entropy:.2f}).""

        # Rule 3: Downgrade a high APCF signal if it occurs in a very low
        # entropy (potentially fake) market.
        elif apcf_result.apcf_value > 1.2 and entropy < 0.1:
            new_state = APCFState.HOLD
            reroute_reason = f"Downgraded high APCF signal to HOLD due to very low entropy ({")}
                entropy:.2f}).""

        # If the state was changed, create a new result object.
        if new_state != original_state:
            logger.warning()
                f"APCF rerouting occurred. Original: {"}
                    original_state.value}, New: {
                        new_state.value}. Reason: {reroute_reason}""
            )

            # Create a new result, preserving original data but updating the
            # state
            rerouted_result = APCFResult()
                apcf_value=apcf_result.apcf_value,
                    state=new_state,
                        confidence=apcf_result.confidence * 0.8,  # Reduce confidence on override
                components=apcf_result.components,
                    timestamp=apcf_result.timestamp,
                        mathematical_signature=apcf_result.mathematical_signature,
                        metadata={}
                    **apcf_result.metadata,
                        "rerouted": True,
                            "original_state": original_state.value,
                            "reroute_reason": reroute_reason,
                            },
                            )
            return rerouted_result

        logger.debug("APCF result confirmed without rerouting.")
        return apcf_result


# Global instance
adaptive_rebalancer = AdaptiveRebalancer()
