# core/site_zone_controller.py

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def is_in_site_zone(tick_drift: float, orderbook_imbalance: float, liquidity_depth: float, compression: float) -> bool:
    """
    Determines if the current market conditions fall within a defined 'site zone'
    suitable for strategic activation.

    Args:
        tick_drift (float): Rolling average of price delta across ticks (momentum window).
        orderbook_imbalance (float): Ratio of bid vs ask depth over 2-5 ticks.
        liquidity_depth (float): Concentration of liquidity under price in lower shells.
        compression (float): Normalized volatility (sigma) over short/medium periods (stability field).

    Returns:
        bool: True if conditions are within the site zone, False otherwise.
    """
    # Define the thresholds for each metric based on the provided context
    # These are conceptual and would be tuned for live trading
    drift_threshold = 0.03
    # orderbook_imbalance_shell_range = (0.8, 1.2) # Use for OBI in shell_range
    liquidity_min = 12000.0
    compression_ceiling = 0.85

    # Site Zone Logical Gate: Entry Condition
    condition_tick_drift = tick_drift > drift_threshold
    condition_orderbook_imbalance = 0.8 <= orderbook_imbalance <= 1.2
    condition_liquidity_depth = liquidity_depth > liquidity_min
    condition_compression = compression < compression_ceiling

    is_within_zone = (
        condition_tick_drift and condition_orderbook_imbalance and condition_liquidity_depth and condition_compression
    )

    if is_within_zone:
        logger.info(
            f"Site Zone Active: TDZ={tick_drift:.4f}, OBI={orderbook_imbalance:.2f}, LSD={liquidity_depth:.0f}, VCI={compression:.2f}"
        )
    else:
        logger.debug(
            f"Site Zone Inactive: TDZ={tick_drift:.4f} (Req>{drift_threshold}), OBI={orderbook_imbalance:.2f} (Req 0.8-1.2), LSD={liquidity_depth:.0f} (Req>{liquidity_min}), VCI={compression:.2f} (Req<{compression_ceiling})"
        )

    return is_within_zone


if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n--- Testing Site Zone Conditions ---")

    # Scenario 1: All conditions met (should be in zone)
    print("Scenario 1: Favorable Conditions")
    in_zone_1 = is_in_site_zone(tick_drift=0.05, orderbook_imbalance=1.0, liquidity_depth=15000.0, compression=0.7)
    print(f"Is in Site Zone: {in_zone_1}")

    # Scenario 2: Tick Drift too low (should be out of zone)
    print("\nScenario 2: Low Tick Drift")
    in_zone_2 = is_in_site_zone(tick_drift=0.01, orderbook_imbalance=1.0, liquidity_depth=15000.0, compression=0.7)
    print(f"Is in Site Zone: {in_zone_2}")

    # Scenario 3: Orderbook Imbalance out of range (should be out of zone)
    print("\nScenario 3: High Orderbook Imbalance")
    in_zone_3 = is_in_site_zone(tick_drift=0.05, orderbook_imbalance=1.5, liquidity_depth=15000.0, compression=0.7)
    print(f"Is in Site Zone: {in_zone_3}")

    # Scenario 4: Liquidity Depth too low (should be out of zone)
    print("\nScenario 4: Low Liquidity Depth")
    in_zone_4 = is_in_site_zone(tick_drift=0.05, orderbook_imbalance=1.0, liquidity_depth=10000.0, compression=0.7)
    print(f"Is in Site Zone: {in_zone_4}")

    # Scenario 5: Compression too high (should be out of zone)
    print("\nScenario 5: High Compression (Volatility)")
    in_zone_5 = is_in_site_zone(tick_drift=0.05, orderbook_imbalance=1.0, liquidity_depth=15000.0, compression=0.9)
    print(f"Is in Site Zone: {in_zone_5}")

    # Scenario 6: Edge case (just barely in zone)
    print("\nScenario 6: Edge Case (Just In)")
    in_zone_6 = is_in_site_zone(tick_drift=0.031, orderbook_imbalance=0.8, liquidity_depth=12001.0, compression=0.84)
    print(f"Is in Site Zone: {in_zone_6}")
