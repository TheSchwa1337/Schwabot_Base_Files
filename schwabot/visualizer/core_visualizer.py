import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class VisualEvent:
    """Represents a single visual event for logging and display"""

    timestamp: float
    event_type: str
    category: str
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SchwabotVisualizer:
    """
    Core visualizer for Schwabot operations.

    Handles real-time data aggregation, filtering, and visualization
    without overwhelming the system with excessive logging.
    """

    def __init__(
        self,
        max_events: int = 1000,
        update_interval: float = 0.1,
        enable_gpu_monitoring: bool = True,
        enable_math_visualization: bool = True,
        enable_trading_visualization: bool = True,
    ):
        """
        Initialize the Schwabot Visualizer.

        Args:
            max_events: Maximum number of events to keep in memory
            update_interval: Update interval for visualization (seconds)
            enable_gpu_monitoring: Enable GPU acceleration monitoring
            enable_math_visualization: Enable mathematical calculation visualization
            enable_trading_visualization: Enable trading activity visualization
        """
        self.logger = logging.getLogger("SchwabotVisualizer")

        # Configuration
        self.max_events = max_events
        self.update_interval = update_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_math_visualization = enable_math_visualization
        self.enable_trading_visualization = enable_trading_visualization

        # Event storage
        self.events: deque = deque(maxlen=max_events)
        self.event_counters: Dict[str, int] = {}

        # Performance tracking
        self.performance_metrics = {
            "total_events": 0,
            "events_per_second": 0,
            "last_update": time.time(),
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "active_strategies": 0,
            "pending_orders": 0,
            "completed_trades": 0,
        }

        # Callbacks for external visualization systems
        self.visualization_callbacks: List[Callable] = []

        # Async state
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None

    def add_event(self, event_type: str, category: str, data: Dict[str, Any], priority: int = 1):
        """
        Add a new visual event to the system.

        Args:
            event_type: Type of event (e.g., 'trade_executed', 'math_calculation')
            category: Category of event (e.g., 'trading', 'mathematics', 'gpu')
            data: Event data dictionary
            priority: Event priority (1=low, 2=medium, 3=high)
        """
        event = VisualEvent(
            timestamp=time.time(),
            event_type=event_type,
            category=category,
            data=data,
            priority=priority,
        )

        self.events.append(event)
        self.event_counters[event_type] = self.event_counters.get(event_type, 0) + 1
        self.performance_metrics["total_events"] += 1

        # Trigger visualization callbacks
        self._trigger_visualization_callbacks(event)

    def add_math_event(self, operation: str, result: Any, duration: float = None):
        """Add a mathematical calculation event"""
        if not self.enable_math_visualization:
            return

        data = {
            "operation": operation,
            "result": str(result)[:100],  # Truncate long results
            "duration_ms": duration * 1000 if duration else None,
        }

        self.add_event("math_calculation", "mathematics", data, priority=1)

    def add_trading_event(
        self, action: str, symbol: str, amount: float, price: float, order_id: str = None
    ):
        """Add a trading activity event"""
        if not self.enable_trading_visualization:
            return

        data = {
            "action": action,
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
        }

        priority = 2 if action in ["buy", "sell"] else 1
        self.add_event("trade_activity", "trading", data, priority=priority)

    def add_gpu_event(self, operation: str, utilization: float, memory_usage: float):
        """Add a GPU-related event"""
        if not self.enable_gpu_monitoring:
            return

        data = {
            "operation": operation,
            "gpu_utilization": utilization,
            "memory_usage": memory_usage,
            "timestamp": datetime.now().isoformat(),
        }

        self.add_event("gpu_activity", "gpu", data, priority=2)

    def add_ferris_rde_event(self, operation: str, strategy_hash: str, status: str):
        """Add a Ferris RDE event"""
        data = {
            "operation": operation,
            "strategy_hash": strategy_hash,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        self.add_event("ferris_rde", "strategy", data, priority=2)

    def get_aggregated_data(self) -> Dict[str, Any]:
        """
        Get aggregated data for visualization.

        Returns:
            Dictionary containing aggregated visualization data
        """
        current_time = time.time()

        # Calculate events per second
        time_diff = current_time - self.performance_metrics["last_update"]
        if time_diff > 0:
            self.performance_metrics["events_per_second"] = (
                self.performance_metrics["total_events"] / time_diff
            )

        # Get recent events (last 100)
        recent_events = list(self.events)[-100:]

        # Aggregate by category
        category_counts = {}
        for event in recent_events:
            category_counts[event.category] = category_counts.get(event.category, 0) + 1

        return {
            "performance_metrics": self.performance_metrics.copy(),
            "event_counters": self.event_counters.copy(),
            "category_counts": category_counts,
            "recent_events": [event.to_dict() for event in recent_events],
            "system_status": {
                "total_events": len(self.events),
                "is_running": self.is_running,
                "update_interval": self.update_interval,
            },
        }

    def register_visualization_callback(self, callback: Callable):
        """Register a callback for visualization updates"""
        self.visualization_callbacks.append(callback)

    def _trigger_visualization_callbacks(self, event: VisualEvent):
        """Trigger all registered visualization callbacks"""
        for callback in self.visualization_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Visualization callback error: {e}")

    async def start(self):
        """Start the visualizer"""
        if self.is_running:
            return

        self.is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Schwabot Visualizer started")

    async def stop(self):
        """Stop the visualizer"""
        self.is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Schwabot Visualizer stopped")

    async def _update_loop(self):
        """Main update loop for the visualizer"""
        while self.is_running:
            try:
                # Update performance metrics
                self._update_performance_metrics()

                # Trigger periodic visualization updates
                aggregated_data = self.get_aggregated_data()
                for callback in self.visualization_callbacks:
                    try:
                        callback(aggregated_data)
                    except Exception as e:
                        self.logger.error(f"Periodic callback error: {e}")

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Visualizer update loop error: {e}")
                await asyncio.sleep(1)

    def _update_performance_metrics(self):
        """Update performance metrics"""
        current_time = time.time()

        # Update last update time
        self.performance_metrics["last_update"] = current_time

        # Calculate events per second over the last second
        recent_events = [event for event in self.events if current_time - event.timestamp <= 1.0]
        self.performance_metrics["events_per_second"] = len(recent_events)

    def export_data(self, filename: str = None) -> str:
        """
        Export visualization data to JSON file.

        Args:
            filename: Output filename (optional)

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"schwabot_visualization_{timestamp}.json"

        data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_metrics": self.performance_metrics,
            "event_counters": self.event_counters,
            "events": [event.to_dict() for event in self.events],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Visualization data exported to {filename}")
        return filename
