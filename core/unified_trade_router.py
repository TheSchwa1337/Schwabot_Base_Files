"""
Unified Trade Router
Handles all routing between raw market data → signal construction → execution logic.
Enhanced to work with the improved trading engine integration.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from core.trading_engine_integration import (
    TradeSignal, 
    TradeExecution, 
    OrderSide, 
    OrderType,
    generate_trade_signal,
    ValidationError,
    TradingError,
    log_trading_error,
    ErrorSeverity
)
from core.clean_unified_math import clean_unified_math

logger = logging.getLogger(__name__)


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
        metadata: Optional[Dict[str, Any]] = None
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
                asset=asset,
                price=price,
                volume=volume,
                metadata=metadata or {}
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
        execution_latency: Optional[float] = None
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
                signal_id=signal.id,
                asset=signal.asset,
                execution_price=execution_price,
                volume=signal.volume,
                latency=execution_latency,
                order_type=signal.order_type,
                order_side=signal.order_side
            )
            
            # Calculate performance if we have a reference price
            if hasattr(signal, 'price') and signal.price > 0:
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
            success_rate = (self.success_count / total_operations * 100) if total_operations > 0 else 0
            
            # Calculate average signal strength
            avg_signal_strength = 0
            if self.signal_history:
                avg_signal_strength = sum(s.signal_strength for s in self.signal_history) / len(self.signal_history)
            
            # Calculate average mathematical score
            avg_math_score = 0
            if self.signal_history:
                avg_math_score = sum(s.mathematical_score for s in self.signal_history) / len(self.signal_history)
            
            # Calculate average performance score
            avg_performance = 0
            valid_executions = [e for e in self.execution_log if e.performance_score is not None]
            if valid_executions:
                avg_performance = sum(e.performance_score for e in valid_executions) / len(valid_executions)
            
            return {
                "total_signals": len(self.signal_history),
                "total_executions": len(self.execution_log),
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate_percent": round(success_rate, 2),
                "average_signal_strength": round(avg_signal_strength, 4),
                "average_mathematical_score": round(avg_math_score, 4),
                "average_performance_score": round(avg_performance, 4),
                "last_signal_time": self.signal_history[-1].timestamp.isoformat() if self.signal_history else None,
                "last_execution_time": self.execution_log[-1].timestamp.isoformat() if self.execution_log else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

    def reset_metrics(self):
        """Reset performance metrics counters."""
        self.error_count = 0
        self.success_count = 0
        logger.info("Performance metrics reset.")

    def get_signal_history(self) -> List[Dict[str, Any]]:
        """Get formatted signal history."""
        return [signal.to_dict() for signal in self.signal_history]

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get formatted execution log."""
        return [execution.to_dict() for execution in self.execution_log] 