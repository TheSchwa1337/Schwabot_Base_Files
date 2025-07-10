#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Trade Router Module
============================
Provides unified trade router functionality for the Schwabot trading system.

Main Classes:
- UnifiedTradeRouter: Core trade routing functionality with mathematical analysis
- Config: Configuration data class
- Result: Result data class

Key Functions:
- route_trade_signal: Route trade signal with mathematical analysis
- route_trade_execution: Route trade execution with performance tracking
- get_performance_metrics: Get comprehensive performance metrics
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator
    from core.clean_unified_math import CleanUnifiedMathSystem
    from core.trading_engine_integration import TradeSignal, TradeExecution, generate_trade_signal, calculate_performance
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")


@dataclass
class Config:
    """Configuration data class."""

    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False


@dataclass
class Result:
    """Result data class."""

    success: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class UnifiedTradeRouter:
    """
    UnifiedTradeRouter Implementation
    Provides core unified trade router functionality with mathematical analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UnifiedTradeRouter with configuration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False

        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            try:
                self.math_config = MathConfigManager()
                self.math_cache = MathResultCache()
                self.math_orchestrator = MathOrchestrator()
                self.math_system = CleanUnifiedMathSystem()
            except Exception as e:
                logger.warning(f"Failed to initialize math infrastructure: {e}")

        # Performance tracking
        self.signal_history: List[TradeSignal] = []
        self.execution_log: List[TradeExecution] = []
        self.performance_metrics: Dict[str, float] = {}
        self.start_time = time.time()

        self._initialize_system()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
        }

    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {self.__class__.__name__}: {e}")
            self.initialized = False

    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.active = True
            self.logger.info(f"✅ {self.__class__.__name__} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {self.__class__.__name__}: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {self.__class__.__name__} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {self.__class__.__name__}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
        }

    def route_trade_signal(
        self,
        price: float,
        volume: float,
        asset: str = "BTC/USDT",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradeSignal:
        """
        Route a trade signal with mathematical analysis.
        
        Args:
            price: Current price
            volume: Trading volume
            asset: Asset pair
            metadata: Additional metadata
            
        Returns:
            TradeSignal with mathematical scoring
        """
        try:
            # Generate trade signal
            signal = generate_trade_signal(price, volume, asset, metadata)
            
            # Add to history
            self.signal_history.append(signal)
            
            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.logger.info(f"Routed trade signal: {signal.id} with score {signal.mathematical_score:.3f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error routing trade signal: {e}")
            # Return fallback signal
            return generate_trade_signal(price, volume, asset, metadata)

    def route_trade_execution(self, signal: TradeSignal) -> TradeExecution:
        """
        Route a trade execution with performance tracking.
        
        Args:
            signal: Original trade signal
            
        Returns:
            TradeExecution with performance metrics
        """
        try:
            # Simulate execution
            execution_id = f"exec_{int(time.time() * 1000)}"
            execution_time = datetime.utcnow()
            
            # Simulate execution price with some slippage
            price_impact = 0.001  # 0.1% slippage
            execution_price = signal.price * (1 + price_impact) if signal.order_side.value == "buy" else signal.price * (1 - price_impact)
            
            # Simulate latency
            latency = np.random.uniform(10, 100)  # 10-100ms
            
            # Calculate realized profit (simplified)
            realized_profit = None
            if len(self.execution_log) > 0:
                last_execution = self.execution_log[-1]
                if last_execution.order_side.value != signal.order_side.value:
                    # Calculate profit from position reversal
                    realized_profit = (execution_price - last_execution.execution_price) * signal.volume
                    if signal.order_side.value == "sell":
                        realized_profit = -realized_profit
            
            # Create execution
            execution = TradeExecution(
                id=execution_id,
                signal_id=signal.id,
                timestamp=execution_time,
                execution_price=execution_price,
                volume=signal.volume,
                asset=signal.asset,
                order_side=signal.order_side,
                order_type=signal.order_type,
                latency=latency,
                realized_profit=realized_profit,
                metadata=signal.metadata.copy()
            )
            
            # Calculate performance score
            performance_score = calculate_performance(execution, signal)
            execution.performance_score = performance_score
            
            # Add to execution log
            self.execution_log.append(execution)
            
            # Limit log size
            if len(self.execution_log) > 1000:
                self.execution_log = self.execution_log[-1000:]
            
            # Update performance metrics
            self._update_performance_metrics()
            
            self.logger.info(f"Routed trade execution: {execution_id} with performance {performance_score:.3f}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Error routing trade execution: {e}")
            # Return fallback execution
            return TradeExecution(
                id=f"fallback_{int(time.time() * 1000)}",
                signal_id=signal.id,
                timestamp=datetime.utcnow(),
                execution_price=signal.price,
                volume=signal.volume,
                asset=signal.asset,
                order_side=signal.order_side,
                order_type=signal.order_type,
                latency=50.0,
                realized_profit=None,
                performance_score=0.5,
                metadata=signal.metadata.copy()
            )

    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            if not self.execution_log:
                return
            
            # Calculate basic metrics
            total_executions = len(self.execution_log)
            successful_executions = len([e for e in self.execution_log if e.performance_score and e.performance_score > 0.5])
            
            # Calculate average metrics
            avg_performance = np.mean([e.performance_score or 0 for e in self.execution_log])
            avg_latency = np.mean([e.latency for e in self.execution_log])
            
            # Calculate profit metrics
            profits = [e.realized_profit for e in self.execution_log if e.realized_profit is not None]
            total_profit = sum(profits) if profits else 0.0
            avg_profit = np.mean(profits) if profits else 0.0
            
            # Calculate signal accuracy
            signal_accuracies = [s.confidence * s.mathematical_score for s in self.signal_history]
            avg_signal_accuracy = np.mean(signal_accuracies) if signal_accuracies else 0.0
            
            # Update metrics
            self.performance_metrics = {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
                "avg_performance_score": avg_performance,
                "avg_latency_ms": avg_latency,
                "total_profit": total_profit,
                "avg_profit": avg_profit,
                "avg_signal_accuracy": avg_signal_accuracy,
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        self._update_performance_metrics()
        return self.performance_metrics.copy()

    def route_trade(self, trade_data: Dict[str, Any]) -> Result:
        """Route a trade through the system (legacy method)."""
        try:
            if not self.active:
                return Result(success=False, error="System not active")

            # Extract trade data
            price = trade_data.get("price", 50000.0)
            volume = trade_data.get("volume", 1.0)
            asset = trade_data.get("asset", "BTC/USDT")
            metadata = trade_data.get("metadata", {})

            # Generate signal and execution
            signal = self.route_trade_signal(price, volume, asset, metadata)
            execution = self.route_trade_execution(signal)

            result_data = {
                "signal": signal.to_dict(),
                "execution": execution.to_dict(),
                "performance_metrics": self.get_performance_metrics()
            }

            result = Result(success=True, data=result_data)
            self.logger.info(f"Trade routed successfully: {signal.id}")
            return result

        except Exception as e:
            self.logger.error(f"Error routing trade: {e}")
            return Result(success=False, error=str(e))


# Factory function
def create_unified_trade_router(config: Optional[Dict[str, Any]] = None):
    """Create a unified trade router instance."""
    return UnifiedTradeRouter(config)
