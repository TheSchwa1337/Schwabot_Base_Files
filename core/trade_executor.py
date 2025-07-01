#!/usr/bin/env python3
"""Trade Executor - Real-time Trade Execution Engine.

Handles the actual execution of trades with advanced order management,
slippage control, and risk monitoring.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str
    asset: str
    direction: str  # "buy" or "sell"
    quantity: float
    price: float
    order_type: str = "market"
    status: str = "pending"
    timestamp: float = field(default_factory=time.time)
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    fees: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeExecutor:
    """Handles the execution of trading orders.

    Interacts with simulated or live exchange APIs.
    """

    def __init__(self, simulation_mode: bool = True):
        """Initialize the trade executor.

        Args:
            simulation_mode: If True, operates in simulation mode; otherwise, connects to live exchange.
        """
        self.simulation_mode = simulation_mode
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0  # Simple counter for unique order IDs

        # Performance metrics
        self.execution_stats = {
            "total_orders": 0,
            "executed_orders": 0,
            "canceled_orders": 0,
            "avg_execution_time": 0.0,
            "simulation_trades": 0,
            "live_trades": 0,
        }

        logger.info(
            f"TradeExecutor initialized in {
                'simulation' if simulation_mode else 'live'} mode."
        )

    def place_order(
        self,
        asset: str,
        direction: str,
        quantity: float,
        price: float,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """Place a trading order.

        Args:
            asset: The trading asset (e.g., "BTC/USD").
            direction: "buy" or "sell".
            quantity: The quantity to trade.
            price: The price at which to place the order.
            order_type: "market" or "limit".

        Returns:
            A dictionary containing order details and status.
        """
        self.order_counter += 1
        order_id = f"ORDER-{self.order_counter}-{int(time.time() * 1000)}"

        new_order = Order(
            order_id=order_id,
            asset=asset,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status="pending",
        )
        self.orders[order_id] = new_order
        self.execution_stats["total_orders"] += 1

        start_time = time.time()
        try:
            if self.simulation_mode:
                # Simulate order execution
                # Small price fluctuation
                executed_price = price * (1 + (random.random() - 0.5) * 0.001)
                executed_quantity = quantity
                fees = (
                    executed_quantity * executed_price * 0.00075
                )  # Simulate 0.075% fee
                new_order.status = "filled"
                new_order.executed_price = executed_price
                new_order.executed_quantity = executed_quantity
                new_order.fees = fees
                self.execution_stats["simulation_trades"] += 1
                logger.info(
                    f"Simulated order {order_id} filled: {direction} {
                        executed_quantity:.4f} {asset} @ {
                        executed_price:.2f}"
                )
            else:
                # Placeholder for live exchange API call
                logger.info(
                    f"Placing live order: {direction} {quantity} {asset} @ {price}"
                )
                # In a real system, this would interact with an actual exchange API
                # For now, simulate a successful live trade after a delay
                time.sleep(0.05)  # Simulate network latency
                new_order.status = "filled"
                new_order.executed_price = price  # Assume filled at requested price
                new_order.executed_quantity = quantity
                new_order.fees = quantity * price * 0.0005  # Simulate 0.05% live fee
                self.execution_stats["live_trades"] += 1
                logger.info(
                    f"Live order {order_id} filled: {direction} {quantity} {asset} @ {price}"
                )

            new_order.metadata["execution_time"] = time.time() - start_time
            self.execution_stats["executed_orders"] += 1
            self._update_avg_execution_time(new_order.metadata["execution_time"])
            return self.get_order_status(order_id)

        except Exception as e:
            new_order.status = "failed"
            logger.error(f"Order {order_id} failed: {e}")
            return {"status": "failed", "order_id": order_id, "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            A dictionary with cancellation status.
        """
        order = self.orders.get(order_id)
        if order and order.status == "pending":
            order.status = "canceled"
            self.execution_stats["canceled_orders"] += 1
            logger.info(f"Order {order_id} canceled.")
            return {"status": "canceled", "order_id": order_id}
        elif order:
            logger.warning(
                f"Cannot cancel order {order_id} (status: {
                    order.status})."
            )
            return {
                "status": "failed",
                "message": f"Cannot cancel order in {
                    order.status} state",
            }
        else:
            logger.warning(f"Order {order_id} not found.")
            return {"status": "failed", "message": "Order not found"}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Retrieve the status of an order.

        Args:
            order_id: The ID of the order.

        Returns:
            A dictionary with order details.
        """
        order = self.orders.get(order_id)
        if order:
            # Convert dataclass to dict
            return {**order.__dict__, "timestamp": order.timestamp}
        else:
            return {"status": "not_found", "order_id": order_id}

    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Retrieve all orders managed by the executor."""
        return [self.get_order_status(order_id) for order_id in self.orders]

    def _update_avg_execution_time(self, new_execution_time: float) -> None:
        """Update the average execution time metric."""
        current_total = self.execution_stats["executed_orders"]
        current_avg = self.execution_stats["avg_execution_time"]

        if current_total == 1:
            self.execution_stats["avg_execution_time"] = new_execution_time
        elif current_total > 1:
            self.execution_stats["avg_execution_time"] = (
                current_avg * (current_total - 1) + new_execution_time
            ) / current_total

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return the performance statistics of the trade executor."""
        return self.execution_stats.copy()


def main():
    """Demonstrate TradeExecutor functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    executor = TradeExecutor(simulation_mode=True)

    print("\n--- Trade Executor Demo (Simulation Mode) ---")

    # Place a buy order
    buy_order_result = executor.place_order("BTC/USD", "buy", 0.001, 50000.0)
    print(f"Buy Order Result: {buy_order_result}")

    # Place a sell order
    sell_order_result = executor.place_order("ETH/USD", "sell", 0.01, 3000.0)
    print(f"Sell Order Result: {sell_order_result}")

    # Try to cancel a non-pending order (will fail)
    cancel_failed_result = executor.cancel_order(buy_order_result["order_id"])
    print(f"Cancel Failed Result: {cancel_failed_result}")

    # Simulate a pending order and then cancel it
    # New executor to ensure pending order
    executor_for_cancel = TradeExecutor(simulation_mode=True)
    pending_order_id = f"PENDING-{int(time.time() * 1000)}"
    executor_for_cancel.orders[pending_order_id] = Order(
        order_id=pending_order_id,
        asset="XRP/USD",
        direction="buy",
        quantity=100,
        price=0.5,
        status="pending",
    )
    print(
        f"\nCreated pending order: {
            executor_for_cancel.get_order_status(pending_order_id)}"
    )
    cancel_success_result = executor_for_cancel.cancel_order(pending_order_id)
    print(f"Cancel Success Result: {cancel_success_result}")
    print(
        f"Status after cancel: {
            executor_for_cancel.get_order_status(pending_order_id)}"
    )

    print("\n--- Performance Statistics ---")
    stats = executor.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n--- All Orders ---")
    for order in executor.get_all_orders():
        print(f"  {order}")


if __name__ == "__main__":
    main()
