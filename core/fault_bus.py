# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Fault Bus - Asynchronous Event and Hash-Based Messaging System
==============================================================

Implements the central nervous system for Schwabot, enabling decoupled,
asynchronous communication between core components.

Core Features:
- Topic-based pub/sub event system.
- Special handling for hash-based routing from the DLT engine.
- Asynchronous, non-blocking listeners.
- Graceful error handling for listener callbacks.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Type definition for an asynchronous callback
AsyncCallback = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class FaultBusEvent:
    """Represents an event in the fault bus system."""
    topic: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "unknown"
    event_id: str = ""


class FaultType(Enum):
    """Types of system faults."""
    THERMAL_HIGH = "thermal_high"
    THERMAL_CRITICAL = "thermal_critical"
    PROFIT_LOW = "profit_low"
    PROFIT_CRITICAL = "profit_critical"
    BITMAP_CORRUPT = "bitmap_corrupt"
    BITMAP_OVERFLOW = "bitmap_overflow"
    GPU_OVERLOAD = "gpu_overload"
    GPU_DRIVER_CRASH = "gpu_driver_crash"
    RECURSIVE_LOOP = "recursive_loop"
    PROFIT_ANOMALY = "profit_anomaly"
    SHA_COLLISION = "sha_collision"
    # Extend this list with new categories as needed


class FaultBus:
    """
    An asynchronous, application-wide event bus for routing messages,
    events, and DLT-based hash confirmations.
    """

    def __init__(self):
        """Initializes the FaultBus."""
        # General topic-based listeners
        self._listeners: Dict[str, Set[AsyncCallback]] = defaultdict(set)
        # Listeners for specific hash confirmations from the DLT engine
        self._hash_listeners: Dict[str, Set[AsyncCallback]] = defaultdict(set)
        logger.info("Fault Bus initialized. Ready for event routing.")

    def subscribe(self, topic: str, callback: AsyncCallback):
        """
        Subscribes a listener to a specific event topic.

        Args:
            topic: The topic to subscribe to (e.g., "portfolio_update").
            callback: An async function to be called when the event is published.
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be a coroutine function (async def).")
        
        self._listeners[topic].unified_math.add(callback)
        logger.debug(f"Listener {callback.__name__} subscribed to topic '{topic}'.")

    def subscribe_to_hash(self, pattern_hash: str, callback: AsyncCallback):
        """
        Subscribes a listener to a specific DLT pattern hash. This allows
        components to react when a "Forever Fractal" is recognized.

        Args:
            pattern_hash: The SHA-256 hash of the DLT pattern.
            callback: An async function to be called when the hash is published.
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be a coroutine function (async def).")
        
        self._hash_listeners[pattern_hash].unified_math.add(callback)
        logger.debug(f"Listener {callback.__name__} subscribed to hash '{pattern_hash[:10]}...'.")

    def unsubscribe(self, topic: str, callback: AsyncCallback):
        """Unsubscribes a listener from a topic."""
        self._listeners[topic].discard(callback)
        logger.debug(f"Listener {callback.__name__} unsubscribed from topic '{topic}'.")

    def unsubscribe_from_hash(self, pattern_hash: str, callback: AsyncCallback):
        """Unsubscribes a listener from a DLT hash."""
        self._hash_listeners[pattern_hash].discard(callback)
        logger.debug(f"Listener {callback.__name__} unsubscribed from hash '{pattern_hash[:10]}...'.")

    async def publish(self, topic: str, **kwargs: Any):
        """
        Publishes an event to all subscribed listeners for a given topic.
        This is non-blocking and gathers all listener tasks.

        Args:
            topic: The topic of the event.
            **kwargs: Arbitrary data to pass to the listeners.
        """
        if topic not in self._listeners:
            logger.debug(f"No listeners for topic '{topic}'.")
            return

        tasks = [
            self._safe_execute(listener, **kwargs)
            for listener in self._listeners[topic]
        ]
        await asyncio.gather(*tasks)
        logger.debug(f"Published event to topic '{topic}' with data: {kwargs}")

    async def publish_hash_confirmation(self, pattern_hash: str, **kwargs: Any):
        """
        Publishes a DLT pattern confirmation to all listeners subscribed to
        that specific hash.

        Args:
            pattern_hash: The confirmed DLT pattern hash.
            **kwargs: Arbitrary data to pass to the listeners (e.g., analysis results).
        """
        if pattern_hash not in self._hash_listeners:
            logger.debug(f"No listeners for hash '{pattern_hash[:10]}...'.")
            return

        tasks = [
            self._safe_execute(listener, pattern_hash=pattern_hash, **kwargs)
            for listener in self._hash_listeners[pattern_hash]
        ]
        await asyncio.gather(*tasks)
        logger.info(f"Published confirmation for hash '{pattern_hash[:10]}...'.")

    async def _safe_execute(self, callback: AsyncCallback, **kwargs: Any):
        """
        Safely executes a listener coroutine and logs any exceptions
        without crashing the bus.
        """
        try:
            await callback(**kwargs)
        except Exception:
            logger.exception(
                f"Error executing listener '{callback.__name__}'. "
                f"Arguments: {kwargs}"
            )


# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the FaultBus."""
    logging.basicConfig(level=logging.INFO)
    bus = FaultBus()

    # Define some example listeners
    async def risk_manager_listener(var_95: float, volatility: float, **_):
        safe_print(
            f"[RiskManager] Received portfolio update: "
            f"VaR={var_95:.2%}, Volatility={volatility:.2%}"
        )

    async def trading_executor_listener(pattern_hash: str, **_):
        safe_print(f"[Executor] Received confirmed profitable hash: {pattern_hash[:10]}... "
              f"Preparing to execute trade.")

    async def another_trade_listener(pattern_hash: str, **_):
        safe_print(f"[Executor2] Also saw hash {pattern_hash[:10]}... Logging for confirmation.")

    # Subscribe listeners
    PROFITABLE_PATTERN_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    bus.subscribe("portfolio_metrics_updated", risk_manager_listener)
    bus.subscribe_to_hash(PROFITABLE_PATTERN_HASH, trading_executor_listener)
    bus.subscribe_to_hash(PROFITABLE_PATTERN_HASH, another_trade_listener)

    # --- Publish events ---
    safe_print("--- Publishing events ---")
    
    # Publish a general event
    await bus.publish(
        "portfolio_metrics_updated", var_95=0.025, volatility=0.18
    )
    
    # Publish a hash confirmation that has listeners
    await bus.publish_hash_confirmation(
        PROFITABLE_PATTERN_HASH,
        confidence=0.98,
        entry_price=50000.0
    )

    # Publish a hash with no listeners
    await bus.publish_hash_confirmation("f0e9d8c7b6a5f0e9d8c7b6a5f0e9d8c7b6a5f0e9d8c7b6a5f0e9d8c7b6a5f0e9")

    safe_print("\n--- Unsubscribing and re-publishing ---")
    bus.unsubscribe_from_hash(PROFITABLE_PATTERN_HASH, another_trade_listener)
    
    await bus.publish_hash_confirmation(
        PROFITABLE_PATTERN_HASH,
        confidence=0.99,
        entry_price=50100.0
    )


if __name__ == "__main__":
    asyncio.run(main())
