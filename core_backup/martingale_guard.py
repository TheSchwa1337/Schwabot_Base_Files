# core/martingale_guard.py

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MartingaleGuard:
    """"""
    Implements an anti-loss recursive retry model based on Martingale logic,
        but restricted to specific sigma zones and other conditions.

    Mathematical Logic: E[X_{t+1}|F_t] = X_t (Martingale property for fair game).
    Schwabot Use: If a signal's expectation is martingale-like (fair game with no edge) -> NO entry.'
    If it breaks the martingale (e.g., via volume imbalance, specific sigma zones) -> trigger ghost trade/recursive entry.
    """"""

    def __init__()
        self,
            allowed_sigma_zones: List[Any],
                min_trade_size: float = 0.1,
                max_martingale_multiplier: float = 3.0,
                profit_tier_threshold: float = 0.7,
                ):
        """"""
        Initializes the MartingaleGuard.

        Args:
            allowed_sigma_zones (List[Any]): A list of 'sigma zone' identifiers where Martingale retry is permitted.
                                             Can be integers, strings, or custom objects representing volatility bands.
            min_trade_size (float): The minimum trade size for Martingale scaling.
            max_martingale_multiplier (float): The maximum multiplier for trade size during a Martingale sequence.
            profit_tier_threshold (float): A threshold for profit tier (e.g., from Schwafit) to allow Martingale.
        """"""
        self.allowed_sigma_zones = allowed_sigma_zones
        self.min_trade_size = min_trade_size
        self.max_martingale_multiplier = max_martingale_multiplier
        self.profit_tier_threshold = profit_tier_threshold
        self.current_martingale_multiplier = 1.0
        self.consecutive_losses = 0
        logger.info(f"MartingaleGuard initialized. Allowed Sigma Zones: {allowed_sigma_zones}")

    def should_retry()
        self, sigma_zone: Any, last_trade_loss: bool, current_profit_tier_score: float, shell_echo_valid: bool = True
    ) -> bool:
        """"""
        Determines if a recursive retry (Martingale-like) is authorized.

        Args:
            sigma_zone (Any): The current sigma zone (volatility band).
            last_trade_loss (bool): True if the immediate previous trade resulted in a loss.
            current_profit_tier_score (float): The current profit tier score (e.g., from Schwafit).
            shell_echo_valid (bool): True if the current shell echo matches memory hash conditions.

        Returns:
            bool: True if a recursive retry is authorized, False otherwise.
        """"""
        if not last_trade_loss:
            self.consecutive_losses = 0  # Reset if no loss
            self.current_martingale_multiplier = 1.0
            logger.debug("Not retrying: last trade was not a loss.")
            return False

        self.consecutive_losses += 1

        if sigma_zone not in self.allowed_sigma_zones:
            logger.debug()
                f"Martingale suppressed - sigma-zone {sigma_zone} not permitted. Allowed: {self.allowed_sigma_zones}"
            )
            return False

        if current_profit_tier_score < self.profit_tier_threshold:
            logger.debug()
                f"Martingale suppressed - Profit tier score ({")}
                    current_profit_tier_score:.2f}) below threshold ({
                        self.profit_tier_threshold:.2f}).""
            )
            return False

        if not shell_echo_valid:
            logger.debug("Martingale suppressed - Shell echo not valid.")
            return False

        # Implement max multiplier check
        if self.current_martingale_multiplier * 2 > self.max_martingale_multiplier and self.consecutive_losses > 1:
            logger.warning()
                f"Martingale suppressed - Max multiplier ({self.max_martingale_multiplier:.2f}) reached or exceeded."
            )
            return False

        logger.info()
            f"Martingale retry authorized in sigma-zone {sigma_zone}. Consecutive losses: {self.consecutive_losses}"
        )
        return True

    def get_scaled_trade_size(self, base_trade_size: float) -> float:
        """"""
        Calculates the scaled trade size for a Martingale retry.

        Args:
            base_trade_size (float): The initial trade size.

        Returns:
            float: The scaled trade size, or base_trade_size if no Martingale sequence is active.
        """"""
        if self.consecutive_losses > 0:
            # Simple doubling logic; more complex strategies can be integrated
            scaled_size = base_trade_size * (2 ** (self.consecutive_losses - 1))
            # Apply max multiplier
            scaled_size = min(scaled_size, base_trade_size * self.max_martingale_multiplier)
            logger.debug()
                f"Scaled trade size for retry: {"}
                    base_trade_size:.4f} -> {
                        scaled_size:.4f} (Multiplier: {)
                    self.current_martingale_multiplier:.2f})""
            )
            self.current_martingale_multiplier = scaled_size / base_trade_size  # Update current multiplier
            return float(scaled_size)
        return float(base_trade_size)

    def reset_martingale_sequence(self):
        """"""
        Resets the Martingale sequence after a successful trade or external reset.
        """"""
        self.consecutive_losses = 0
        self.current_martingale_multiplier = 1.0
        logger.info("Martingale sequence reset.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Example Usage
    # Assume sigma zones can be integer IDs or string labels
    guard = MartingaleGuard(allowed_sigma_zones=[1, 3, "LOW_VOL"], max_martingale_multiplier=4.0)

    print("\n--- Testing MartingaleGuard ---")

    # Scenario 1: First loss, within allowed sigma zone, sufficient profit tier, valid echo
    print("\nScenario 1: First loss, allowed zone (1)")
    retry_1 = guard.should_retry()
        sigma_zone=1, last_trade_loss=True, current_profit_tier_score=0.8, shell_echo_valid=True
    )
    print(f"Should retry: {retry_1}")
    if retry_1:
        print(f"Scaled trade size: {guard.get_scaled_trade_size(10.0):.2f}")  # Expected 10.0

    # Scenario 2: Second loss, same zone
    print("\nScenario 2: Second loss, allowed zone (1)")
    retry_2 = guard.should_retry()
        sigma_zone=1, last_trade_loss=True, current_profit_tier_score=0.85, shell_echo_valid=True
    )
    print(f"Should retry: {retry_2}")
    if retry_2:
        print(f"Scaled trade size: {guard.get_scaled_trade_size(10.0):.2f}")  # Expected 20.0

    # Scenario 3: Third loss, same zone, hits max multiplier (multiplier would)
    # be 4, which is 2**2 for 3rd loss (consecutive_losses-1))
    print("\nScenario 3: Third loss, allowed zone (1), hits max multiplier")
    retry_3 = guard.should_retry()
        sigma_zone=1, last_trade_loss=True, current_profit_tier_score=0.9, shell_echo_valid=True
    )
    print(f"Should retry: {retry_3}")
    if retry_3:
        # Expected 40.0 (clamped by max_martingale_multiplier)
        print(f"Scaled trade size: {guard.get_scaled_trade_size(10.0):.2f}")

    # Scenario 4: Not allowed sigma zone
    print("\nScenario 4: Not allowed sigma zone (2)")
    guard.reset_martingale_sequence()  # Reset for clean test
    retry_4 = guard.should_retry()
        sigma_zone=2, last_trade_loss=True, current_profit_tier_score=0.8, shell_echo_valid=True
    )
    print(f"Should retry: {retry_4}")

    # Scenario 5: Profit tier too low
    print("\nScenario 5: Profit tier too low")
    guard.reset_martingale_sequence()
    retry_5 = guard.should_retry()
        sigma_zone=3, last_trade_loss=True, current_profit_tier_score=0.6, shell_echo_valid=True
    )
    print(f"Should retry: {retry_5}")

    # Scenario 6: Shell echo not valid
    print("\nScenario 6: Shell echo not valid")
    guard.reset_martingale_sequence()
    retry_6 = guard.should_retry()
        sigma_zone="LOW_VOL", last_trade_loss=True, current_profit_tier_score=0.8, shell_echo_valid=False
    )
    print(f"Should retry: {retry_6}")

    # Scenario 7: Successful trade, resets sequence
    print("\nScenario 7: Successful trade, resets sequence")
    _ = guard.should_retry()
        sigma_zone=1, last_trade_loss=True, current_profit_tier_score=0.8, shell_echo_valid=True
    )  # First loss
    print(f"Consecutive losses before reset: {guard.consecutive_losses}")
    guard.reset_martingale_sequence()
    print(f"Consecutive losses after reset: {guard.consecutive_losses}")
