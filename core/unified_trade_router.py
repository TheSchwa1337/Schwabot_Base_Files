#!/usr/bin/env python3
"""
Unified Trade Router
Handles all routing between raw market data → signal construction → execution logic.
Enhanced to work with the improved trading engine integration.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Fallback implementations for missing imports
class ErrorSeverity:
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OrderSide:
    """Order side types."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType:
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class TradeSignal:
    """Trade signal with enhanced tracking."""
    id: str
    asset: str
    price: float
    volume: float
    order_type: OrderType
    order_side: OrderSide
    signal_strength: float = 0.0
    mathematical_score: float = 0.0
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "asset": self.asset,
            "price": self.price,
            "volume": self.volume,
            "order_type": self.order_type,
            "order_side": self.order_side,
            "signal_strength": self.signal_strength,
            "mathematical_score": self.mathematical_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class TradeExecution:
    """Trade execution with performance tracking."""
    id: str
    signal_id: str
    asset: str
    execution_price: float
    volume: float
    latency: float
    order_type: OrderType
    order_side: OrderSide
    performance_score: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def calculate_performance(self, entry_price: float) -> None:
        """Calculate performance score."""
        if entry_price > 0:
            price_change = (self.execution_price - entry_price) / entry_price
            self.performance_score = max(0.0, min(1.0, 0.5 + price_change))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "asset": self.asset,
            "execution_price": self.execution_price,
            "volume": self.volume,
            "latency": self.latency,
            "order_type": self.order_type,
            "order_side": self.order_side,
            "performance_score": self.performance_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ValidationError(Exception):
    """Validation error."""
    pass


class TradingError(Exception):
    """Trading error."""
    pass


def generate_trade_signal(
    asset: str,
    price: float,
    volume: float,
    metadata: Dict[str, Any]
) -> TradeSignal:
    """Generate trade signal with enhanced validation."""
    if price <= 0:
        raise ValidationError("Price must be positive")
    if volume <= 0:
        raise ValidationError("Volume must be positive")
    if not asset:
        raise ValidationError("Asset symbol is required")

    # Simple signal generation logic
    signal_strength = min(1.0, volume / 1000.0)  # Normalize volume
    mathematical_score = 0.5 + (signal_strength * 0.5)  # Base score + volume factor

    # Determine order type and side based on metadata
    order_type = OrderType.MARKET
    order_side = OrderSide.BUY if metadata.get("trend", "up") == "up" else OrderSide.SELL

    return TradeSignal(
        id=f"signal_{int(time.time() * 1000)}",
        asset=asset,
        price=price,
        volume=volume,
        order_type=order_type,
        order_side=order_side,
        signal_strength=signal_strength,
        mathematical_score=mathematical_score,
        metadata=metadata
    )


def log_trading_error(error: Exception, severity: ErrorSeverity) -> None:
    """Log trading error with severity."""
    logger.error(f"[{severity}] Trading error: {error}")


class UnifiedTradeRouter:
    """
    Enhanced unified trade router with robust error handling and validation.
    Integrates with the mathematical trading system for signal generation.
    """

    def __init__(self):
        self.signal_history: List[TradeSignal] = []
        self.execution_log: List[TradeExecution] = []
        self.error_count = 0
        self.success_count = 0
        logger.info("UnifiedTradeRouter initialized with enhanced error handling.")

    def route_trade_signal(
        self,
        price: float,
        volume: float,
        asset: str = "BTC/USDT",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeSignal:
        """
        Routes raw market data to construct a TradeSignal with enhanced validation.

        Args:
            price (float): Current market price
            volume (float): Trading volume
            asset (str): Asset symbol (default: "BTC/USDT")
            metadata (dict, optional): Additional signal metadata

        Returns:
            TradeSignal: Generated trade signal

        Raises:
            TradingError: If signal generation fails
        """
        try:
            # Use the enhanced signal generation function
            signal = generate_trade_signal(
                asset=asset, price=price, volume=volume, metadata=metadata or {}
            )

            self.signal_history.append(signal)
            self.success_count += 1

            logger.info(f"Trade Signal routed successfully: {signal.id}")
            logger.debug(f"Signal details: {signal.to_dict()}")

            return signal

        except ValidationError as ve:
            self.error_count += 1
            log_trading_error(ve, ErrorSeverity.HIGH)
            raise TradingError(f"Signal validation failed: {ve}")

        except Exception as e:
            self.error_count += 1
            log_trading_error(e, ErrorSeverity.CRITICAL)
            raise TradingError(f"Signal generation failed: {e}")

    def route_trade_execution(
        self,
        signal: TradeSignal,
        execution_price: Optional[float] = None,
        execution_latency: Optional[float] = None,
    ) -> TradeExecution:
        """
        Routes a TradeSignal to construct a TradeExecution with performance tracking.

        Args:
            signal (TradeSignal): The trade signal to execute
            execution_price (float, optional): Actual execution price (defaults to signal price)
            execution_latency (float, optional): Execution latency in seconds

        Returns:
            TradeExecution: Generated trade execution

        Raises:
            TradingError: If execution creation fails
        """
        try:
            # Use signal price if no execution price provided
            if execution_price is None:
                execution_price = signal.price

            # Calculate latency if not provided
            if execution_latency is None:
                execution_latency = 0.05  # Default 50ms latency

            # Create execution with enhanced tracking
            execution = TradeExecution(
                id=f"exec_{int(time.time() * 1000)}",
                signal_id=signal.id,
                asset=signal.asset,
                execution_price=execution_price,
                volume=signal.volume,
                latency=execution_latency,
                order_type=signal.order_type,
                order_side=signal.order_side,
            )

            # Calculate performance if we have a reference price
            if hasattr(signal, "price") and signal.price > 0:
                execution.calculate_performance(entry_price=signal.price)

            self.execution_log.append(execution)
            self.success_count += 1

            logger.info(f"Trade Execution routed successfully: {execution.id}")
            logger.debug(f"Execution details: {execution.to_dict()}")

            return execution

        except ValidationError as ve:
            self.error_count += 1
            log_trading_error(ve, ErrorSeverity.HIGH)
            raise TradingError(f"Execution validation failed: {ve}")

        except Exception as e:
            self.error_count += 1
            log_trading_error(e, ErrorSeverity.CRITICAL)
            raise TradingError(f"Execution creation failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for the router.

        Returns:
            Dict containing performance statistics
        """
        try:
            total_operations = self.success_count + self.error_count
            success_rate = (
                (self.success_count / total_operations * 100) if total_operations > 0 else 0
            )

            # Calculate average signal strength
            avg_signal_strength = 0
            if self.signal_history:
                avg_signal_strength = sum(s.signal_strength for s in self.signal_history) / len(
                    self.signal_history
                )

            # Calculate average mathematical score
            avg_math_score = 0
            if self.signal_history:
                avg_math_score = sum(s.mathematical_score for s in self.signal_history) / len(
                    self.signal_history
                )

            # Calculate average performance score
            avg_performance = 0
            valid_executions = [e for e in self.execution_log if e.performance_score is not None]
            if valid_executions:
                avg_performance = sum(e.performance_score for e in valid_executions) / len(
                    valid_executions
                )

            return {
                "total_signals": len(self.signal_history),
                "total_executions": len(self.execution_log),
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate_percent": round(success_rate, 2),
                "average_signal_strength": round(avg_signal_strength, 4),
                "average_mathematical_score": round(avg_math_score, 4),
                "average_performance_score": round(avg_performance, 4),
                "last_signal_time": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.signal_history[-1].timestamp)) 
                    if self.signal_history else None
                ),
                "last_execution_time": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.execution_log[-1].timestamp)) 
                    if self.execution_log else None
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "error": "Failed to calculate metrics",
                "total_signals": len(self.signal_history),
                "total_executions": len(self.execution_log),
                "success_count": self.success_count,
                "error_count": self.error_count,
            }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.signal_history.clear()
        self.execution_log.clear()
        self.error_count = 0
        self.success_count = 0
        logger.info("Performance metrics reset")

    def get_signal_history(self) -> List[Dict[str, Any]]:
        """Get signal history as list of dictionaries."""
        return [signal.to_dict() for signal in self.signal_history]

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log as list of dictionaries."""
        return [execution.to_dict() for execution in self.execution_log]


# Factory function
def create_unified_trade_router() -> UnifiedTradeRouter:
    """Create a new unified trade router instance."""
    return UnifiedTradeRouter()
