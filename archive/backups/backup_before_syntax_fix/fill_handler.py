#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill Handler Module

Handles order fills, partial fills, and order state management for the Schwabot trading system.
Provides comprehensive fill tracking, retry logic, and state persistence.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FillStatus(Enum):
    """Order fill status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class RetryReason(Enum):
    """Retry reason enumeration."""
    PARTIAL_FILL = "partial_fill"
    NETWORK_ERROR = "network_error"
    EXCHANGE_ERROR = "exchange_error"
    TIMEOUT = "timeout"
    INSUFFICIENT_FUNDS = "insufficient_funds"


@dataclass
class FillEvent:
    """Fill event data structure."""
    order_id: str
    trade_id: str
    symbol: str
    side: str
    amount: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderState:
    """Order state tracking."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    original_amount: Decimal
    filled_amount: Decimal = Decimal('0')
    remaining_amount: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    total_cost: Decimal = Decimal('0')
    total_fee: Decimal = Decimal('0')
    status: FillStatus = FillStatus.PENDING
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    retry_count: int = 0

    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.original_amount == 0:
            return 0.0
        return float(self.filled_amount / self.original_amount * 100)


class FillHandler:
    """
    Fill Handler for managing order fills and state.
    
    Handles:
    - Order fill processing
    - Partial fill management
    - Retry logic
    - State persistence
    - Fill statistics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FillHandler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.active_orders: Dict[str, OrderState] = {}
        self.fill_history: List[FillEvent] = []
        self.retry_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Statistics
        self.total_fills_processed = 0
        self.total_retries = 0
        self.total_fees = Decimal('0')
        self.total_slippage = Decimal('0')
        
        # Configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay_base = self.config.get("retry_delay_base", 1.0)
        self.max_fill_history = self.config.get("max_fill_history", 1000)
        
        logger.info("✅ FillHandler initialized")

    async def process_fill_event(self, fill_data: Dict[str, Any]) -> FillEvent:
        """
        Process a fill event from the exchange.
        
        Args:
            fill_data: Fill data from exchange
            
        Returns:
            Processed fill event
        """
        try:
            # Extract fill data
            order_id = fill_data.get('orderId', fill_data.get('id', ''))
            trade_id = fill_data.get('tradeId', fill_data.get('trade_id', ''))
            symbol = fill_data.get('symbol', '')
            side = fill_data.get('side', '')
            amount = Decimal(str(fill_data.get('amount', 0)))
            price = Decimal(str(fill_data.get('price', 0)))
            fee = Decimal(str(fill_data.get('fee', 0)))
            fee_currency = fill_data.get('feeCurrency', fill_data.get('fee_currency', ''))
            timestamp = fill_data.get('timestamp', int(time.time() * 1000))

            # Create fill event
            fill_event = FillEvent(
                order_id=order_id,
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                fee=fee,
                fee_currency=fee_currency,
                timestamp=timestamp,
                metadata=fill_data.get('metadata', {})
            )

            # Update order state
            await self._update_order_state(fill_event)
            
            # Add to history
            self.fill_history.append(fill_event)
            if len(self.fill_history) > self.max_fill_history:
                self.fill_history.pop(0)

            # Update statistics
            self.total_fills_processed += 1
            self.total_fees += fee

            logger.info(f"✅ Fill processed: {order_id} - {amount} @ {price}")
            return fill_event

        except Exception as e:
            logger.error(f"Error processing fill event: {e}")
            raise

    async def _update_order_state(self, fill_event: FillEvent) -> None:
        """
        Update order state based on fill event.
        
        Args:
            fill_event: Fill event to process
        """
        order_id = fill_event.order_id
        
        if order_id not in self.active_orders:
            logger.warning(f"Fill for unknown order: {order_id}")
            return

        order_state = self.active_orders[order_id]
        
        # Update filled amount
        order_state.filled_amount += fill_event.amount
        order_state.remaining_amount = order_state.original_amount - order_state.filled_amount
        
        # Update average price and total cost
        if order_state.filled_amount > 0:
            total_cost = order_state.total_cost + (fill_event.amount * fill_event.price)
            order_state.average_price = total_cost / order_state.filled_amount
            order_state.total_cost = total_cost
        
        # Update fees
        order_state.total_fee += fill_event.fee
        
        # Update status
        if order_state.remaining_amount == 0:
            order_state.status = FillStatus.COMPLETE
        else:
            order_state.status = FillStatus.PARTIAL
        
        order_state.updated_at = fill_event.timestamp

    async def handle_partial_fill(
        self,
        order_id: str,
        fill_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a partial fill scenario.
        
        Args:
            order_id: Order ID
            fill_data: Fill data
            
        Returns:
            Handling result
        """
        try:
            # Process the fill
            fill_event = await self.process_fill_event(fill_data)
            
            order_state = self.active_orders.get(order_id)
            if not order_state:
                return {"status": "error", "message": "Order not found"}

            # Check if retry is needed
            if order_state.remaining_amount > 0 and order_state.retry_count < self.max_retries:
                await self._schedule_retry(order_state, RetryReason.PARTIAL_FILL)
                return {
                    "status": "partial_fill_handled",
                    "remaining_amount": str(order_state.remaining_amount),
                    "retry_scheduled": True
                }
            else:
                return {
                    "status": "partial_fill_handled",
                    "remaining_amount": str(order_state.remaining_amount),
                    "retry_scheduled": False
                }

        except Exception as e:
            logger.error(f"Error handling partial fill: {e}")
            return {"status": "error", "message": str(e)}

    async def _schedule_retry(self, order_state: OrderState, reason: RetryReason) -> None:
        """
        Schedule a retry for an order.
        
        Args:
            order_state: Order state
            reason: Retry reason
        """
        order_state.retry_count += 1
        self.total_retries += 1

        retry_info = {
            "timestamp": int(time.time() * 1000),
            "reason": reason.value,
            "retry_count": order_state.retry_count,
            "remaining_amount": str(order_state.remaining_amount),
        }

        if order_state.order_id not in self.retry_history:
            self.retry_history[order_state.order_id] = []

        self.retry_history[order_state.order_id].append(retry_info)

        logger.info(f"🔄 Scheduled retry {order_state.retry_count} for order {order_state.order_id}: {reason.value}")

    async def handle_order_update(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle order status update from exchange.
        
        Args:
            order_data: Order update data
            
        Returns:
            Update result
        """
        try:
            order_id = order_data.get('orderId', order_data.get('id', ''))
            status = order_data.get('status', '').lower()

            if order_id not in self.active_orders:
                logger.warning(f"Order update for unknown order: {order_id}")
                return {"status": "unknown_order"}

            order_state = self.active_orders[order_id]

            # Update order status
            if status in ['filled', 'closed']:
                order_state.status = FillStatus.COMPLETE
            elif status in ['canceled', 'cancelled']:
                order_state.status = FillStatus.CANCELLED
            elif status in ['expired']:
                order_state.status = FillStatus.EXPIRED
            elif status in ['rejected', 'failed']:
                order_state.status = FillStatus.FAILED

            order_state.updated_at = int(time.time() * 1000)

            return {
                "status": "order_updated",
                "order_status": order_state.status.value,
                "fill_percentage": order_state.fill_percentage,
            }

        except Exception as e:
            logger.error(f"Error handling order update: {e}")
            return {"status": "error", "message": str(e)}

    def get_order_state(self, order_id: str) -> Optional[OrderState]:
        """Get current state of an order."""
        return self.active_orders.get(order_id)

    def get_fill_statistics(self) -> Dict[str, Any]:
        """Get fill handling statistics."""
        total_orders = len(self.active_orders)
        completed_orders = sum(1 for order in self.active_orders.values() if order.status == FillStatus.COMPLETE)
        partial_orders = sum(1 for order in self.active_orders.values() if order.status == FillStatus.PARTIAL)

        return {
            "total_fills_processed": self.total_fills_processed,
            "total_retries": self.total_retries,
            "total_fees": str(self.total_fees),
            "total_slippage": str(self.total_slippage),
            "active_orders": total_orders,
            "completed_orders": completed_orders,
            "partial_orders": partial_orders,
            "completion_rate": (completed_orders / total_orders * 100) if total_orders > 0 else 0,
        }

    def clear_completed_orders(self, max_age_hours: int = 24) -> None:
        """
        Clear completed orders older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        current_time = int(time.time() * 1000)
        max_age_ms = max_age_hours * 60 * 60 * 1000

        orders_to_remove = []
        for order_id, order_state in self.active_orders.items():
            if (
                order_state.status in [FillStatus.COMPLETE, FillStatus.CANCELLED, FillStatus.FAILED]
                and current_time - order_state.updated_at > max_age_ms
            ):
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self.active_orders[order_id]

        logger.info(f"🧹 Cleared {len(orders_to_remove)} old completed orders")

    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "active_orders": {
                order_id: {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "order_type": order.order_type,
                    "original_amount": str(order.original_amount),
                    "filled_amount": str(order.filled_amount),
                    "remaining_amount": str(order.remaining_amount),
                    "average_price": str(order.average_price),
                    "total_cost": str(order.total_cost),
                    "total_fee": str(order.total_fee),
                    "status": order.status.value,
                    "created_at": order.created_at,
                    "updated_at": order.updated_at,
                    "retry_count": order.retry_count,
                }
                for order_id, order in self.active_orders.items()
            },
            "fill_history": [
                {
                    "order_id": fill.order_id,
                    "trade_id": fill.trade_id,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "amount": str(fill.amount),
                    "price": str(fill.price),
                    "fee": str(fill.fee),
                    "fee_currency": fill.fee_currency,
                    "timestamp": fill.timestamp,
                }
                for fill in self.fill_history[-100:]  # Keep last 100 fills
            ],
            "statistics": self.get_fill_statistics(),
        }

    def import_state(self, state_data: Dict[str, Any]) -> None:
        """
        Import state from persistence.
        
        Args:
            state_data: State data to import
        """
        try:
            # Import active orders
            for order_id, order_data in state_data.get("active_orders", {}).items():
                order_state = OrderState(
                    order_id=order_data["order_id"],
                    symbol=order_data["symbol"],
                    side=order_data["side"],
                    order_type=order_data["order_type"],
                    original_amount=Decimal(order_data["original_amount"]),
                    filled_amount=Decimal(order_data["filled_amount"]),
                    remaining_amount=Decimal(order_data["remaining_amount"]),
                    average_price=Decimal(order_data["average_price"]),
                    total_cost=Decimal(order_data["total_cost"]),
                    total_fee=Decimal(order_data["total_fee"]),
                    status=FillStatus(order_data["status"]),
                    created_at=order_data["created_at"],
                    updated_at=order_data["updated_at"],
                    retry_count=order_data["retry_count"],
                )
                self.active_orders[order_id] = order_state

            logger.info(f"📥 Imported {len(self.active_orders)} order states")

        except Exception as e:
            logger.error(f"Error importing state: {e}")
            raise


# Factory function for creating fill handler instances
async def create_fill_handler(config: Optional[Dict[str, Any]] = None) -> FillHandler:
    """
    Create a new fill handler instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized FillHandler instance
    """
    return FillHandler(config)


# Convenience functions for easy integration
async def process_exchange_fill(fill_handler: FillHandler, fill_data: Dict[str, Any]) -> FillEvent:
    """
    Process a fill event from any exchange.
    
    Args:
        fill_handler: Fill handler instance
        fill_data: Fill data from exchange
        
    Returns:
        Processed fill event
    """
    return await fill_handler.process_fill_event(fill_data)


async def handle_partial_fill_scenario(
    fill_handler: FillHandler,
    order_id: str,
    fill_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle a partial fill scenario.
    
    Args:
        fill_handler: Fill handler instance
        order_id: Order ID
        fill_data: Fill data
        
    Returns:
        Handling result
    """
    return await fill_handler.handle_partial_fill(order_id, fill_data) 