import logging
import math
from typing import Dict, List

logger = logging.getLogger(__name__)


class ProfitTierClassifier:
    """
    Classifies trade returns into discrete profit tiers using a logarithmic function.
    This is a core component of SFSSS for prioritizing strategy bundles.

    Mathematical Form: Tier = ⌊ log_base(Return * 100) ⌋
    """

    def __init__(
        self, log_base: float = 1.5, return_scale_factor: float = 100.0, tier_boundaries: Dict[int, float] = None
    ):  # Optional: for custom tier names/ranges
        """
        Initializes the ProfitTierClassifier.

        Args:
            log_base (float): The base for the logarithm (e.g., 1.5).
            return_scale_factor (float): Factor to scale the return before applying logarithm (e.g., 100 for percentage).
            tier_boundaries (Dict[int, float]): A dictionary mapping tier numbers to minimum return percentages
                                                to allow for custom tier definitions beyond simple log. (Not directly
                                                used by the log formula, but for conceptual mapping or overrides).
        """
        if log_base <= 1.0:
            raise ValueError("Log base must be greater than 1.0 for meaningful tiering.")
        self.log_base = log_base
        self.return_scale_factor = return_scale_factor
        self.tier_boundaries = (
            tier_boundaries
            if tier_boundaries is not None
            else {
                0: 0.00,  # <= 0% return / Experimental/Unstable
                1: 0.005,  # 0.5% return / Low Profit
                2: 0.02,  # 2% return / Medium Profit
                3: 0.05,  # 5% return / High Profit
            }
        )  # These are illustrative and can be tuned
        logger.info(f"ProfitTierClassifier initialized with log_base={self.log_base}.")

    def classify_profit_tier(self, trade_return_percentage: float) -> int:
        """
        Classifies a given trade return percentage into a discrete profit tier.

        Args:
            trade_return_percentage (float): The actual return of a trade as a percentage
                                            (e.g., 0.01 for 1%, 0.05 for 5%).

        Returns:
            int: The calculated profit tier (e.g., 0, 1, 2, 3). Returns 0 for non-positive or very low returns.
        """
        if trade_return_percentage <= 0.0:
            return 0  # Tier 0 for no profit or loss

        scaled_return = trade_return_percentage * self.return_scale_factor

        # Handle cases where scaled_return might be extremely small but positive
        if scaled_return < 1.0:  # If scaled return is less than 1 (e.g., 0.001 * 100 = 0.1)
            # We can still map it to tier 0 or a very low tier if it's positive but below log_base^1
            # log_base(X) will be < 0 for X < 1, so math.floor will give negative. Clamp to 0.
            return 0

        try:
            tier = math.floor(math.log(scaled_return, self.log_base))
            # Ensure tier is non-negative and within a reasonable max range if needed
            # Max tier can be enforced by a configuration, for now, just non-negative
            classified_tier = max(0, tier)

            # Further refine based on conceptual tier boundaries for reporting/filtering purposes
            # This part is illustrative and might be used in a wrapper or SFSSS decision logic
            if classified_tier >= 3 and trade_return_percentage >= self.tier_boundaries.get(3, 0.05):
                final_tier = 3
            elif classified_tier >= 2 and trade_return_percentage >= self.tier_boundaries.get(2, 0.02):
                final_tier = 2
            elif classified_tier >= 1 and trade_return_percentage >= self.tier_boundaries.get(1, 0.005):
                final_tier = 1
            else:
                final_tier = 0  # Default for anything below Tier 1 threshold

            logger.debug(
                f"Return {trade_return_percentage:.4f} -> Scaled: {scaled_return:.2f} -> Log Tier: {tier} -> Final Tier: {final_tier}"
            )
            return final_tier
        except ValueError as e:
            logger.error(f"Error calculating log for return {trade_return_percentage}: {e}. Returning tier 0.")
            return 0

    def get_tier_description(self, tier: int) -> str:
        """
        Provides a human-readable description for a given profit tier.
        """
        descriptions = {
            0: "T0: Experimental/Unstable (<0.5% gain)",
            1: "T1: Low Profit (0.5%-2% gain)",
            2: "T2: Medium Profit (2%-5% gain)",
            3: "T3: High Profit (>5% gain)",
        }
        return descriptions.get(tier, "Unknown Tier")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    classifier = ProfitTierClassifier(log_base=1.5, return_scale_factor=100.0)

    print("\n--- Testing ProfitTierClassifier ---")

    test_returns = [
        -0.01,  # Loss
        0.001,  # 0.1% gain (should be Tier 0)
        0.004,  # 0.4% gain (should be Tier 0)
        0.005,  # 0.5% gain (boundary, should be Tier 1)
        0.01,  # 1% gain (Tier 1)
        0.019,  # 1.9% gain (Tier 1)
        0.02,  # 2% gain (boundary, should be Tier 2)
        0.035,  # 3.5% gain (Tier 2)
        0.049,  # 4.9% gain (Tier 2)
        0.05,  # 5% gain (boundary, should be Tier 3)
        0.075,  # 7.5% gain (Tier 3)
        0.10,  # 10% gain (Tier 3, even higher log value)
        0.00001,  # Very small positive
        0.99,  # Very high return
    ]

    for ret in test_returns:
        tier = classifier.classify_profit_tier(ret)
        description = classifier.get_tier_description(tier)
        print(f"Return: {ret*100:.2f}% -> Tier: {tier} ({description})")

    print("\n--- Testing Custom Log Base ---")
    custom_classifier = ProfitTierClassifier(log_base=2.0)
    for ret in [0.01, 0.04, 0.08, 0.16]:
        tier = custom_classifier.classify_profit_tier(ret)
        print(f"Return: {ret*100:.2f}% (Log Base 2.0) -> Tier: {tier}")
