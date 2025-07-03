import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional


class DataAggregator:
    """
    Data aggregator for Schwabot visualization system.

    Collects and processes data from various system components:
    - Mathematical calculations
    - Trading activities
    - GPU operations
    - Ferris RDE strategies
    - System performance metrics
    """

    def __init__(self, visualizer=None):
        """
        Initialize the data aggregator.

        Args:
            visualizer: Reference to the main visualizer instance
        """
        self.logger = logging.getLogger("DataAggregator")
        self.visualizer = visualizer

        # Data storage
        self.math_data = defaultdict(list)
        self.trading_data = defaultdict(list)
        self.gpu_data = defaultdict(list)
        self.strategy_data = defaultdict(list)
        self.performance_data = defaultdict(list)

        # Aggregation settings
        self.max_data_points = 1000
        self.aggregation_interval = 1.0  # seconds

        # Callbacks for data processing
        self.data_callbacks: List[Callable] = []

        # Async state
        self.is_running = False
        self._aggregation_task: Optional[asyncio.Task] = None

    def add_math_data(self, operation: str, result: Any, duration: float = None):
        """Add mathematical calculation data"""
        data_point = {
            "timestamp": time.time(),
            "operation": operation,
            "result": result,
            "duration": duration,
        }

        self.math_data[operation].append(data_point)

        # Trim old data
        if len(self.math_data[operation]) > self.max_data_points:
            self.math_data[operation] = self.math_data[operation][-self.max_data_points :]

        # Notify visualizer if available
        if self.visualizer:
            self.visualizer.add_math_event(operation, result, duration)

    def add_trading_data(
        self,
        action: str,
        symbol: str,
        amount: float,
        price: float,
        order_id: str = None,
        status: str = "pending",
    ):
        """Add trading activity data"""
        data_point = {
            "timestamp": time.time(),
            "action": action,
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "order_id": order_id,
            "status": status,
        }

        self.trading_data[symbol].append(data_point)

        # Trim old data
        if len(self.trading_data[symbol]) > self.max_data_points:
            self.trading_data[symbol] = self.trading_data[symbol][-self.max_data_points :]

        # Notify visualizer if available
        if self.visualizer:
            self.visualizer.add_trading_event(action, symbol, amount, price, order_id)

    def add_gpu_data(self, operation: str, utilization: float, memory_usage: float):
        """Add GPU operation data"""
        data_point = {
            "timestamp": time.time(),
            "operation": operation,
            "utilization": utilization,
            "memory_usage": memory_usage,
        }

        self.gpu_data[operation].append(data_point)

        # Trim old data
        if len(self.gpu_data[operation]) > self.max_data_points:
            self.gpu_data[operation] = self.gpu_data[operation][-self.max_data_points :]

        # Notify visualizer if available
        if self.visualizer:
            self.visualizer.add_gpu_event(operation, utilization, memory_usage)

    def add_strategy_data(
        self, strategy_hash: str, operation: str, status: str, data: Dict[str, Any] = None
    ):
        """Add Ferris RDE strategy data"""
        data_point = {
            "timestamp": time.time(),
            "strategy_hash": strategy_hash,
            "operation": operation,
            "status": status,
            "data": data or {},
        }

        self.strategy_data[strategy_hash].append(data_point)

        # Trim old data
        if len(self.strategy_data[strategy_hash]) > self.max_data_points:
            self.strategy_data[strategy_hash] = self.strategy_data[strategy_hash][
                -self.max_data_points :
            ]

        # Notify visualizer if available
        if self.visualizer:
            self.visualizer.add_ferris_rde_event(operation, strategy_hash, status)

    def add_performance_data(self, metric: str, value: float):
        """Add system performance data"""
        data_point = {"timestamp": time.time(), "metric": metric, "value": value}

        self.performance_data[metric].append(data_point)

        # Trim old data
        if len(self.performance_data[metric]) > self.max_data_points:
            self.performance_data[metric] = self.performance_data[metric][-self.max_data_points :]

    def get_aggregated_math_data(self) -> Dict[str, Any]:
        """Get aggregated mathematical data"""
        aggregated = {}

        for operation, data_points in self.math_data.items():
            if not data_points:
                continue

            recent_data = data_points[-100:]  # Last 100 points

            # Calculate statistics
            durations = [dp["duration"] for dp in recent_data if dp["duration"] is not None]

            aggregated[operation] = {
                "total_operations": len(data_points),
                "recent_operations": len(recent_data),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "last_operation": recent_data[-1] if recent_data else None,
            }

        return aggregated

    def get_aggregated_trading_data(self) -> Dict[str, Any]:
        """Get aggregated trading data"""
        aggregated = {}

        for symbol, data_points in self.trading_data.items():
            if not data_points:
                continue

            recent_data = data_points[-100:]  # Last 100 points

            # Calculate statistics
            buy_orders = [dp for dp in recent_data if dp["action"] == "buy"]
            sell_orders = [dp for dp in recent_data if dp["action"] == "sell"]

            aggregated[symbol] = {
                "total_trades": len(data_points),
                "recent_trades": len(recent_data),
                "buy_orders": len(buy_orders),
                "sell_orders": len(sell_orders),
                "total_volume": sum(dp["amount"] for dp in recent_data),
                "avg_price": (
                    sum(dp["price"] for dp in recent_data) / len(recent_data) if recent_data else 0
                ),
                "last_trade": recent_data[-1] if recent_data else None,
            }

        return aggregated

    def get_aggregated_gpu_data(self) -> Dict[str, Any]:
        """Get aggregated GPU data"""
        aggregated = {}

        for operation, data_points in self.gpu_data.items():
            if not data_points:
                continue

            recent_data = data_points[-100:]  # Last 100 points

            # Calculate statistics
            utilizations = [dp["utilization"] for dp in recent_data]
            memory_usages = [dp["memory_usage"] for dp in recent_data]

            aggregated[operation] = {
                "total_operations": len(data_points),
                "recent_operations": len(recent_data),
                "avg_utilization": sum(utilizations) / len(utilizations) if utilizations else 0,
                "avg_memory_usage": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                "max_utilization": max(utilizations) if utilizations else 0,
                "max_memory_usage": max(memory_usages) if memory_usages else 0,
                "last_operation": recent_data[-1] if recent_data else None,
            }

        return aggregated

    def get_aggregated_strategy_data(self) -> Dict[str, Any]:
        """Get aggregated strategy data"""
        aggregated = {}

        for strategy_hash, data_points in self.strategy_data.items():
            if not data_points:
                continue

            recent_data = data_points[-100:]  # Last 100 points

            # Calculate statistics
            status_counts = defaultdict(int)
            for dp in recent_data:
                status_counts[dp["status"]] += 1

            aggregated[strategy_hash] = {
                "total_operations": len(data_points),
                "recent_operations": len(recent_data),
                "status_distribution": dict(status_counts),
                "last_operation": recent_data[-1] if recent_data else None,
            }

        return aggregated

    def get_all_aggregated_data(self) -> Dict[str, Any]:
        """Get all aggregated data"""
        return {
            "math_data": self.get_aggregated_math_data(),
            "trading_data": self.get_aggregated_trading_data(),
            "gpu_data": self.get_aggregated_gpu_data(),
            "strategy_data": self.get_aggregated_strategy_data(),
            "performance_data": dict(self.performance_data),
            "summary": {
                "total_math_operations": sum(len(data) for data in self.math_data.values()),
                "total_trades": sum(len(data) for data in self.trading_data.values()),
                "total_gpu_operations": sum(len(data) for data in self.gpu_data.values()),
                "total_strategies": len(self.strategy_data),
                "active_symbols": len(self.trading_data),
                "active_operations": len(self.math_data),
            },
        }

    def register_data_callback(self, callback: Callable):
        """Register a callback for data processing"""
        self.data_callbacks.append(callback)

    async def start(self):
        """Start the data aggregator"""
        if self.is_running:
            return

        self.is_running = True
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.logger.info("Data Aggregator started")

    async def stop(self):
        """Stop the data aggregator"""
        self.is_running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Data Aggregator stopped")

    async def _aggregation_loop(self):
        """Main aggregation loop"""
        while self.is_running:
            try:
                # Get aggregated data
                aggregated_data = self.get_all_aggregated_data()

                # Trigger data callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(aggregated_data)
                    except Exception as e:
                        self.logger.error(f"Data callback error: {e}")

                await asyncio.sleep(self.aggregation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Data aggregation loop error: {e}")
                await asyncio.sleep(1)

    def clear_old_data(self, max_age_seconds: int = 3600):
        """Clear data older than specified age"""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        # Clear old data from all collections
        for data_dict in [
            self.math_data,
            self.trading_data,
            self.gpu_data,
            self.strategy_data,
            self.performance_data,
        ]:
            for key in list(data_dict.keys()):
                data_dict[key] = [dp for dp in data_dict[key] if dp["timestamp"] > cutoff_time]

        self.logger.info(f"Cleared data older than {max_age_seconds} seconds")
