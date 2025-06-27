"""
Fault Bus - Asynchronous Event and Hash-Based Messaging System
=============================================================

Implements the central nervous system for Schwabot, enabling decoupled,
asynchronous communication between core components.

Core Features:
- Topic-based pub/sub event system
- Special handling for hash-based routing from the DLT engine
- Asynchronous, non-blocking listeners
- Graceful error handling for listener callbacks
"""

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Set

# Import unified math system
try:
    from core.unified_math_system import unified_math
except ImportError:
    import math as unified_math

# Import dual unicore handler
try:
    from dual_unicore_handler import DualUnicoreHandler
    unicore = DualUnicoreHandler()
except ImportError:
    unicore = None

# Configure logging
logger = logging.getLogger(__name__)

# Type definition for an asynchronous callback
AsyncCallback = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class Event:
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
            raise TypeError(
                "Callback must be a coroutine function (async def).")

        self._listeners[topic].add(callback)
        logger.debug(
            f"Listener {
                callback.__name__} subscribed to topic '{topic}'.")

    def subscribe_to_hash(self, pattern_hash: str, callback: AsyncCallback):
        """
        Subscribes a listener to a specific DLT pattern hash. This allows
        components to react when a "Forever Fractal" is recognized.

        Args:
            pattern_hash: The SHA-256 hash of the DLT pattern.
            callback: An async function to be called when the hash is published.
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError(
                "Callback must be a coroutine function (async def).")

        self._hash_listeners[pattern_hash].add(callback)
        logger.debug(
            f"Listener {callback.__name__} subscribed to hash '{pattern_hash[:10]}...'.")

    def unsubscribe(self, topic: str, callback: AsyncCallback):
        """Unsubscribes a listener from a topic."""
        self._listeners[topic].discard(callback)
        logger.debug(
            f"Listener {
                callback.__name__} unsubscribed from topic '{topic}'.")

    def unsubscribe_from_hash(
            self,
            pattern_hash: str,
            callback: AsyncCallback):
        """Unsubscribes a listener from a DLT hash."""
        self._hash_listeners[pattern_hash].discard(callback)
        logger.debug(
            f"Listener {callback.__name__} unsubscribed from hash '{pattern_hash[:10]}...'.")

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

        tasks = []
        for listener in self._listeners[topic]:
            tasks.append(self._safe_execute(listener, **kwargs))

        await asyncio.gather(*tasks)
        logger.debug(f"Published event to topic '{topic}' with data: {kwargs}")

    async def publish_hash_confirmation(
            self, pattern_hash: str, **kwargs: Any):
        """
        Publishes a DLT hash confirmation to all subscribed listeners.
        This is used when the DLT engine recognizes a "Forever Fractal" pattern.

        Args:
            pattern_hash: The SHA-256 hash of the recognized pattern.
            **kwargs: Arbitrary data to pass to the listeners.
        """
        if pattern_hash not in self._hash_listeners:
            logger.debug(f"No listeners for hash '{pattern_hash[:10]}...'.")
            return

        tasks = []
        for listener in self._hash_listeners[pattern_hash]:
            tasks.append(self._safe_execute(listener, **kwargs))

        await asyncio.gather(*tasks)
        logger.debug(
            f"Published hash confirmation '{pattern_hash[:10]}...' with data: {kwargs}")

    async def _safe_execute(self, callback: AsyncCallback, **kwargs: Any):
        """
        Safely executes a callback function with error handling.

        Args:
            callback: The async callback function to execute.
            **kwargs: Arguments to pass to the callback.

        Returns:
            The result of the callback execution.
        """
        try:
            return await callback(**kwargs)
        except Exception as e:
            logger.error(
                f"Error in fault bus callback {
                    callback.__name__}: {e}")
            # Return None to prevent task cancellation
            return None

    def get_subscriber_count(self, topic: str) -> int:
        """
        Gets the number of subscribers for a given topic.

        Args:
            topic: The topic to check.

        Returns:
            Number of subscribers.
        """
        return len(self._listeners.get(topic, set()))

    def get_hash_subscriber_count(self, pattern_hash: str) -> int:
        """
        Gets the number of subscribers for a given hash pattern.

        Args:
            pattern_hash: The hash pattern to check.

        Returns:
            Number of subscribers.
        """
        return len(self._hash_listeners.get(pattern_hash, set()))

    def list_topics(self) -> List[str]:
        """
        Lists all active topics.

        Returns:
            List of topic names.
        """
        return list(self._listeners.keys())

    def list_hash_patterns(self) -> List[str]:
        """
        Lists all active hash patterns.

        Returns:
            List of hash pattern strings.
        """
        return list(self._hash_listeners.keys())

    async def publish_fault(self, fault_type: FaultType, **kwargs: Any):
        """
        Publishes a system fault event.

        Args:
            fault_type: The type of fault.
            **kwargs: Additional fault data.
        """
        fault_data = {
            "fault_type": fault_type.value,
            "timestamp": datetime.now().isoformat(),
            "severity": self._get_fault_severity(fault_type),
            **kwargs
        }

        await self.publish("system_fault", **fault_data)
        logger.warning(f"System fault published: {fault_type.value}")

    def _get_fault_severity(self, fault_type: FaultType) -> str:
        """
        Gets the severity level for a fault type.

        Args:
            fault_type: The fault type.

        Returns:
            Severity level string.
        """
        critical_faults = {
            FaultType.THERMAL_CRITICAL,
            FaultType.PROFIT_CRITICAL,
            FaultType.GPU_DRIVER_CRASH,
            FaultType.RECURSIVE_LOOP
        }

        high_faults = {
            FaultType.THERMAL_HIGH,
            FaultType.PROFIT_LOW,
            FaultType.BITMAP_CORRUPT,
            FaultType.GPU_OVERLOAD,
            FaultType.SHA_COLLISION
        }

        if fault_type in critical_faults:
            return "critical"
        elif fault_type in high_faults:
            return "high"
        else:
            return "medium"

    async def publish_profit_update(self, profit_data: Dict[str, Any]):
        """
        Publishes a profit update event.

        Args:
            profit_data: Profit-related data.
        """
        await self.publish("profit_update", **profit_data)
        logger.debug("Profit update published")

    async def publish_thermal_update(self, thermal_data: Dict[str, Any]):
        """
        Publishes a thermal update event.

        Args:
            thermal_data: Thermal-related data.
        """
        await self.publish("thermal_update", **thermal_data)
        logger.debug("Thermal update published")

    async def publish_bitmap_update(self, bitmap_data: Dict[str, Any]):
        """
        Publishes a bitmap update event.

        Args:
            bitmap_data: Bitmap-related data.
        """
        await self.publish("bitmap_update", **bitmap_data)
        logger.debug("Bitmap update published")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Gets the current system status of the fault bus.

        Returns:
            Status information dictionary.
        """
        return {
            "total_topics": len(self._listeners),
            "total_hash_patterns": len(self._hash_listeners),
            "total_subscribers": sum(len(subscribers) for subscribers in self._listeners.values()),
            "total_hash_subscribers": sum(len(subscribers) for subscribers in self._hash_listeners.values()),
            "active_topics": list(self._listeners.keys()),
            "active_hash_patterns": list(self._hash_listeners.keys())
        }

    def clear_all_subscriptions(self):
        """Clears all subscriptions from the fault bus."""
        self._listeners.clear()
        self._hash_listeners.clear()
        logger.info("All fault bus subscriptions cleared")

    def clear_topic_subscriptions(self, topic: str):
        """
        Clears all subscriptions for a specific topic.

        Args:
            topic: The topic to clear.
        """
        if topic in self._listeners:
            del self._listeners[topic]
            logger.info(f"Cleared all subscriptions for topic '{topic}'")

    def clear_hash_subscriptions(self, pattern_hash: str):
        """
        Clears all subscriptions for a specific hash pattern.

        Args:
            pattern_hash: The hash pattern to clear.
        """
        if pattern_hash in self._hash_listeners:
            del self._hash_listeners[pattern_hash]
            logger.info(
                f"Cleared all subscriptions for hash '{pattern_hash[:10]}...'")


# Global fault bus instance
_fault_bus_instance: FaultBus = None


def get_fault_bus() -> FaultBus:
    """
    Gets the global fault bus instance.

    Returns:
        The global FaultBus instance.
    """
    global _fault_bus_instance
    if _fault_bus_instance is None:
        _fault_bus_instance = FaultBus()
    return _fault_bus_instance


def create_fault_bus() -> FaultBus:
    """
    Creates a new fault bus instance.

    Returns:
        A new FaultBus instance.
    """
    return FaultBus()


async def main():
    """Main function for testing the fault bus."""
    try:
        # Create fault bus
        fault_bus = create_fault_bus()

        # Define test callbacks
        async def test_callback(**kwargs):
            print(f"Test callback received: {kwargs}")

        async def fault_callback(**kwargs):
            print(f"Fault callback received: {kwargs}")

        # Subscribe to topics
        fault_bus.subscribe("test_topic", test_callback)
        fault_bus.subscribe("system_fault", fault_callback)

        # Publish test events
        await fault_bus.publish("test_topic", message="Hello, Fault Bus!")
        await fault_bus.publish_fault(FaultType.THERMAL_HIGH, temperature=85.0)

        # Get status
        status = fault_bus.get_system_status()
        print(f"Fault Bus Status: {status}")

    except Exception as e:
        logger.error(f"Main function failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
