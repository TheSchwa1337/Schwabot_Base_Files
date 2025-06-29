# core/filtered_allocation_gate.py

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class FilteredAllocationGate:
    """"""
    Manages the filtering and allocation of capital based on suitability assessment,
        risk factors, and profit certainty.
    """"""

    def __init__(self, certainty_threshold: float = 0.7, risk_tolerance: float = 0.3):
        self.certainty_threshold = certainty_threshold
        self.risk_tolerance = risk_tolerance
        logger.info("FilteredAllocationGate initialized.")

    def assess_suitability(self, profit_certainty_score: float, current_risk_factor: float) -> bool:
        """"""
        Assesses if a trade or capital allocation is suitable based on profit certainty and risk factors.

        Args:
            profit_certainty_score (float): The certainty score from ProfitCertaintyMeter (0-1).
            current_risk_factor (float): The current calculated risk factor (0-1, higher is riskier).

        Returns:
            bool: True if suitable for allocation, False otherwise.
        """"""
        is_suitable = profit_certainty_score >= self.certainty_threshold and current_risk_factor <= self.risk_tolerance
        logger.debug()
            f"Suitability assessment: Certainty={"}
                profit_certainty_score:.2f} (Req: >={)
                self.certainty_threshold}), Risk={
                    current_risk_factor:.2f} (Req: <={)
                    self.risk_tolerance}). Suitable: {is_suitable}""
        )
        return is_suitable

    def determine_allocation_amount(self, available_capital: float, suitability_metrics: Dict[str, float]) -> float:
        """"""
        Determines the appropriate capital allocation amount using a threshold-based decision process
        and a weighted sum of various suitability metrics.

        Mathematical Logic: Allocation = sum w_i * M_i (where M_i are suitability metrics)

        Args:
            available_capital (float): The total capital available for allocation.
            suitability_metrics (Dict[str, float]): A dictionary of metrics influencing allocation.
                                                  Expected keys: 'profit_certainty', 'liquidity_spread',
                                                      'risk_factor', 'market_volatility'.
                                                  Weights are internal to the method.

        Returns:
            float: The calculated allocation amount.
        """"""
        profit_certainty = suitability_metrics.get("profit_certainty", 0.0)
        liquidity_spread = suitability_metrics.get("liquidity_spread", 0.0)  # From USDCPositionManager
        risk_factor = suitability_metrics.get("risk_factor", 1.0)
        market_volatility = suitability_metrics.get("market_volatility", 0.0)

        # Define weights for each metric. These are conceptual and would be tuned.
        weights = {
            "profit_certainty": 0.4,  # Higher certainty means more allocation
            "liquidity_spread": 0.3,  # Higher liquidity means more allocation
            "risk_factor": -0.2,  # Higher risk means less allocation (negative weight)
            "market_volatility": -0.1,  # Higher volatility means less allocation (negative weight)
}
}
        # Calculate a combined suitability score based on weighted sum
        # Note: Risk and volatility are inverse contributors to suitability
        combined_suitability_score = ()
            (profit_certainty * weights["profit_certainty"])
            + (liquidity_spread * weights["liquidity_spread"])
            + ((1 - risk_factor) * abs(weights["risk_factor"]))  # Invert risk impact
            + ((1 - market_volatility) * abs(weights["market_volatility"]))
        )

        # Normalize the combined score to a 0-1 range (conceptual normalization)
        # This would depend on the expected range of combined_suitability_score
        normalized_score = max(0.0, min(1.0, combined_suitability_score))

        # Threshold-based decision for actual allocation. If the normalized score is high enough.
        if normalized_score >= 0.6:  # Example threshold for active allocation
            # Allocate a percentage of available capital based on the normalized score
            allocation_percentage = normalized_score  # Simple direct mapping for example
            allocation_amount = available_capital * allocation_percentage
            logger.info(f"Allocating {allocation_amount:.2f} (Normalized Score: {normalized_score:.2f})")
            return allocation_amount
        else:
            logger.info()
                f"Not suitable for allocation based on current metrics. Normalized Score: {"}
                    normalized_score:.2f}""
            )
            return 0.0


if __name__ == "__main__":
    gate = FilteredAllocationGate(certainty_threshold=0.75, risk_tolerance=0.25)

    # Test suitability
    print("\n--- Suitability Assessment ---")
    print("Suitable (High Certainty, Low Risk):", gate.assess_suitability(0.8, 0.2))
    print("Not Suitable (Low Certainty):", gate.assess_suitability(0.6, 0.1))
    print("Not Suitable (High Risk):", gate.assess_suitability(0.9, 0.4))

    # Test allocation amount
    print("\n--- Allocation Amount Determination ---")
    available_capital = 10000.0

    # Scenario 1: Favorable metrics
    metrics_favorable = {
        "profit_certainty": 0.85,
        "liquidity_spread": 0.7,
        "risk_factor": 0.15,
        "market_volatility": 0.5,
}
}
    allocation1 = gate.determine_allocation_amount(available_capital, metrics_favorable)
    print(f"Scenario 1 Allocation: {allocation1:.2f}")

    # Scenario 2: Less favorable metrics
    metrics_less_favorable = {
        "profit_certainty": 0.6,
        "liquidity_spread": 0.3,
        "risk_factor": 0.5,
        "market_volatility": 0.3,
}
}
    allocation2 = gate.determine_allocation_amount(available_capital, metrics_less_favorable)
    print(f"Scenario 2 Allocation: {allocation2:.2f}")

    # Scenario 3: Highly favorable metrics
    metrics_highly_favorable = {
        "profit_certainty": 0.95,
        "liquidity_spread": 0.9,
        "risk_factor": 0.5,
        "market_volatility": 0.2,
}
}
    allocation3 = gate.determine_allocation_amount(available_capital, metrics_highly_favorable)
    print(f"Scenario 3 Allocation: {allocation3:.2f}")
