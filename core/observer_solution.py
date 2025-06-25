# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import numpy as np
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
# #!/usr/bin/env python3
"""
Observer Solution - DLT Pattern Recognition Engine
==================================================

This module acts as the primary market observer for Schwabot. It consumes
price data, uses MathLibV4 to analyze delta sequences, and publishes
confirmed DLT patterns ("Forever Fractals") to the Fault Bus.

Core Responsibilities:
- Subscribes to market data streams via the Fault Bus.
- Maintains a rolling window of time-series data.
- Uses MathLibV4 to detect "Triplet Locks" in delta sequences.
- Generates and publishes unique pattern hashes for confirmed locks.
"""

import asyncio
import logging
from collections import deque
from typing import Deque

# from core.unified_math_system import unified_math  # F811: duplicate import

from .fault_bus import FaultBus
from .mathlib_v4 import MathLibV4

logger = logging.getLogger(__name__)


class ObserverSolution:
    """
    Watches a data stream, identifies DLT patterns, and publishes findings.
    """

    def __init__(
        self,
        fault_bus: FaultBus,
        math_lib: MathLibV4,
        window_size: int = 100,
        triplet_lock_tolerance: float = 0.05,
    ):
        """
        Initializes the ObserverSolution.

        Args:
            fault_bus: An instance of the central FaultBus.
            math_lib: An instance of MathLibV4.
            window_size: The number of recent data points to keep.
            triplet_lock_tolerance: The tolerance for confirming a Triplet Lock.
        """
        self.bus = fault_bus
        self.math = math_lib
        self.window_size = window_size
        self.triplet_lock_tolerance = triplet_lock_tolerance

        # Use a deque for an efficient rolling window of price data
        self.price_window: Deque[float] = deque(maxlen=self.window_size)

        # Keep track of hashes we've recently published to avoid spam
        self.recent_hashes: Deque[str] = deque(maxlen=20)

        logger.info(
            f"ObserverSolution initialized with window size {self.window_size}."
        )

    def subscribe_to_data_feed(self, topic: str = "new_market_price"):
        """
        Subscribes the observer's data processing handler to the Fault Bus.
        """
        self.bus.subscribe(topic, self.handle_new_price_data)
        logger.info(f"ObserverSolution subscribed to topic '{topic}'.")

    async def handle_new_price_data(self, price: float, timestamp: float):
        """
        The core callback that processes each new data point from the bus.

        Args:
            price: The new price data point.
            timestamp: The timestamp associated with the price.
        """
        self.price_window.append(price)

        # We need enough data to form at least one delta sequence
        if len(self.price_window) < 4: # 4 prices = 3 deltas
            return

        # Convert window to numpy array for our math library
        prices_np = np.array(self.price_window)

        # 1. Calculate the deltas
        deltas = self.math.calculate_deltas(prices_np)

        # 2. Check for a Triplet Lock
        is_locked = self.math.confirm_triplet_lock(
            deltas, tolerance=self.triplet_lock_tolerance
        )

        if is_locked:
            # 3. Generate a hash for the locked pattern
            # We hash the entire delta sequence that led to the lock
            pattern_hash = self.math.generate_pattern_hash(deltas)

            # Avoid publishing the same hash repeatedly
            if pattern_hash in self.recent_hashes:
                return

            self.recent_hashes.append(pattern_hash)

            # 4. Publish the confirmed hash to the bus
            logger.info(
                f"Triplet Lock CONFIRMED. Publishing hash: {pattern_hash[:10]}..."
            )
            await self.bus.publish_hash_confirmation(
                pattern_hash=pattern_hash,
                timestamp=timestamp,
                last_price=price,
                triggering_deltas=deltas[-3:].tolist()
            )


# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the ObserverSolution."""
    logging.basicConfig(level=logging.INFO)

    # 1. Setup our core components
    bus = FaultBus()
    math_lib = MathLibV4()
    observer = ObserverSolution(bus, math_lib)
    observer.subscribe_to_data_feed()

    # 2. Create a listener to react to the observer's findings
    async def trading_logic_listener(pattern_hash: str, timestamp: float, **kwargs):
        safe_print(
            "\n[TRADING LOGIC] Reacting to confirmed hash! \n"
            f"  -> Hash: {pattern_hash[:10]}...\n"
            f"  -> Timestamp: {timestamp}\n"
            f"  -> Details: {kwargs}"
        )

    # Subscribe the trading logic to a specific hash we expect to see
    EXPECTED_HASH = "4d6d9e794383141a5435e98341648a89b657956a827643e49e25a818c64a515"
    bus.subscribe_to_hash(EXPECTED_HASH, trading_logic_listener)

    # 3. Simulate a live market data feed publishing to the bus
    safe_print("--- Simulating Market Data Feed ---")

    # This sequence is designed to create a Triplet Lock
    market_prices = [
        100, 102, 101, 105, 108, 110, # Some noise
        120, # Start of pattern
        130.0, # delta = +10
        140.1, # delta = +10.1
        150.0, # delta = +9.9
    ]

    for i, price in enumerate(market_prices):
        safe_print(f"  Publishing price: {price}")
        await bus.publish("new_market_price", price=price, timestamp=1672531200 + i)
        await asyncio.sleep(0.1) # Simulate time between ticks


if __name__ == "__main__":
    asyncio.run(main())
