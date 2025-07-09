#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Trading Strategy Executor for Schwabot
=========================================

Real trading strategy execution engine that integrates with:
- CCXT trading executor for order placement
- Order book manager for market analysis
- 2-gram pattern detection for signal generation
- Risk management and position sizing
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from core.ccxt_trading_executor import CCXTTradingExecutor, TradeOrder, OrderSide, OrderType, TradeResult
from core.order_book_manager import OrderBookManager, OrderBookSnapshot
from core.two_gram_detector import TwoGramSignal

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Trading strategy types."""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    GRID = "grid"
    FERRIS_WHEEL = "ferris_wheel"
    VOLATILITY_BREAKOUT = "volatility_breakout"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingSignal:
    """Trading signal with strategy information."""
    symbol: str
    strategy_type: StrategyType
    signal_strength: SignalStrength
    side: OrderSide
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    volume: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signal_id: str = field(default_factory=lambda: f"signal_{int(time.time() * 1000)}")


@dataclass
class StrategyExecution:
    """Strategy execution result."""
    signal_id: str
    strategy_type: StrategyType
    symbol: str
    side: OrderSide
    executed: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_amount: Optional[float] = None
    execution_time_ms: Optional[float] = None
    slippage: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class TradingStrategyExecutor:
    """
    Trading strategy executor for real trading operations.
    
    Integrates:
    - 2-gram pattern detection signals
    - Order book analysis for optimal execution
    - Risk management and position sizing
    - Multi-strategy execution and management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading strategy executor."""
        self.config = config
        
        # Core components
        self.ccxt_executor: Optional[CCXTTradingExecutor] = None
        self.order_book_manager: Optional[OrderBookManager] = None
        self.two_gram_detector: Optional[Any] = None
        
        # Strategy state
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.signal_history: List[TradingSignal] = []
        
        # Risk management
        self.max_position_size = config.get("max_position_size", 0.1)
        self.max_daily_trades = config.get("max_daily_trades", 100)
        self.max_drawdown = config.get("max_drawdown", 0.15)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        
        # Performance tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.last_reset = time.time()
        
        # Execution settings
        self.enable_real_trading = config.get("enable_real_trading", False)
        self.slippage_tolerance = config.get("slippage_tolerance", 0.001)
        self.execution_timeout = config.get("execution_timeout", 30.0)
        
        logger.info("🎯 Trading strategy executor initialized")
    
    async def initialize(self, ccxt_executor: CCXTTradingExecutor, 
                        order_book_manager: OrderBookManager,
                        two_gram_detector: Any):
        """Initialize the strategy executor with required components."""
        self.ccxt_executor = ccxt_executor
        self.order_book_manager = order_book_manager
        self.two_gram_detector = two_gram_detector
        
        # Register order book callbacks
        if self.order_book_manager:
            for symbol in self.config.get("trading_symbols", ["BTC/USDC"]):
                self.order_book_manager.register_callback(symbol, self._on_order_book_update)
        
        logger.info("✅ Trading strategy executor initialized with components")
    
    async def process_2gram_signal(self, signal: TwoGramSignal, market_data: Dict[str, Any]) -> Optional[StrategyExecution]:
        """Process 2-gram pattern signal and execute strategy."""
        try:
            # Convert 2-gram signal to trading signal
            trading_signal = await self._convert_2gram_to_trading_signal(signal, market_data)
            if not trading_signal:
                return None
            
            # Validate signal
            if not await self._validate_signal(trading_signal):
                return None
            
            # Execute strategy
            execution = await self._execute_strategy(trading_signal)
            
            # Track performance
            if execution.executed:
                await self._track_execution(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Error processing 2-gram signal: {e}")
            return None
    
    async def _convert_2gram_to_trading_signal(self, signal: TwoGramSignal, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Convert 2-gram signal to trading signal."""
        try:
            # Determine strategy type based on pattern
            strategy_type = self._determine_strategy_type(signal.pattern)
            
            # Determine signal strength
            signal_strength = self._calculate_signal_strength(signal)
            
            # Determine trading side based on pattern and burst score
            side = self._determine_trading_side(signal)
            
            # Get current market price
            symbol = market_data.get("symbol", "BTC/USDC")
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
            
            # Calculate target and stop levels
            target_price, stop_loss = self._calculate_price_levels(signal, current_price, side)
            
            # Calculate position size
            volume = await self._calculate_position_size(symbol, signal, current_price)
            
            return TradingSignal(
                symbol=symbol,
                strategy_type=strategy_type,
                signal_strength=signal_strength,
                side=side,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=target_price,
                confidence=signal.burst_score / 10.0,  # Normalize burst score
                volume=volume,
                metadata={
                    "pattern": signal.pattern,
                    "burst_score": signal.burst_score,
                    "entropy": signal.entropy,
                    "fractal_resonance": signal.fractal_resonance
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting 2-gram signal: {e}")
            return None
    
    def _determine_strategy_type(self, pattern: str) -> StrategyType:
        """Determine strategy type based on 2-gram pattern."""
        strategy_mapping = {
            "UD": StrategyType.VOLATILITY_BREAKOUT,
            "DU": StrategyType.MEAN_REVERSION,
            "BE": StrategyType.ARBITRAGE,
            "EB": StrategyType.ARBITRAGE,
            "UU": StrategyType.MOMENTUM,
            "DD": StrategyType.MOMENTUM,
            "AA": StrategyType.SCALPING,
            "EE": StrategyType.SWING,
        }
        return strategy_mapping.get(pattern, StrategyType.SCALPING)
    
    def _calculate_signal_strength(self, signal: TwoGramSignal) -> SignalStrength:
        """Calculate signal strength based on 2-gram metrics."""
        # Combine burst score, entropy, and fractal resonance
        strength_score = (
            signal.burst_score * 0.4 +
            (1.0 - signal.entropy) * 0.3 +
            (signal.fractal_resonance or 0.0) * 0.3
        )
        
        if strength_score > 3.0:
            return SignalStrength.VERY_STRONG
        elif strength_score > 2.0:
            return SignalStrength.STRONG
        elif strength_score > 1.0:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _determine_trading_side(self, signal: TwoGramSignal) -> OrderSide:
        """Determine trading side based on 2-gram pattern."""
        # Simple logic based on pattern direction
        bullish_patterns = ["UU", "DU", "BE"]
        bearish_patterns = ["DD", "UD", "EB"]
        
        if signal.pattern in bullish_patterns:
            return OrderSide.BUY
        elif signal.pattern in bearish_patterns:
            return OrderSide.SELL
        else:
            # Default based on burst score direction
            return OrderSide.BUY if signal.burst_score > 0 else OrderSide.SELL
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        try:
            if self.order_book_manager:
                order_book = self.order_book_manager.get_order_book(symbol)
                if order_book:
                    return order_book.get_mid_price()
            
            # Fallback to CCXT executor
            if self.ccxt_executor:
                order_book = await self.ccxt_executor.fetch_order_book(symbol)
                if order_book:
                    return order_book.get_mid_price()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def _calculate_price_levels(self, signal: TwoGramSignal, current_price: float, side: OrderSide) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss levels."""
        try:
            # Calculate volatility-based levels
            volatility_factor = signal.entropy * 0.02  # 2% base volatility
            
            if side == OrderSide.BUY:
                target_price = current_price * (1 + volatility_factor)
                stop_loss = current_price * (1 - volatility_factor * 0.5)
            else:
                target_price = current_price * (1 - volatility_factor)
                stop_loss = current_price * (1 + volatility_factor * 0.5)
            
            return target_price, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating price levels: {e}")
            return None, None
    
    async def _calculate_position_size(self, symbol: str, signal: TwoGramSignal, current_price: float) -> float:
        """Calculate optimal position size."""
        try:
            if not self.ccxt_executor:
                return 0.0
            
            # Get available balance
            balances = await self.ccxt_executor.fetch_balance()
            available_balance = balances.get("USDC", 0.0)
            
            if available_balance <= 0:
                return 0.0
            
            # Calculate position size based on risk management
            risk_amount = available_balance * self.risk_per_trade
            position_size = risk_amount / current_price
            
            # Apply maximum position size limit
            max_size = available_balance * self.max_position_size / current_price
            position_size = min(position_size, max_size)
            
            # Adjust based on signal strength
            strength_multiplier = {
                SignalStrength.WEAK: 0.5,
                SignalStrength.MODERATE: 0.75,
                SignalStrength.STRONG: 1.0,
                SignalStrength.VERY_STRONG: 1.25
            }
            
            signal_strength = self._calculate_signal_strength(signal)
            position_size *= strength_multiplier.get(signal_strength, 1.0)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal before execution."""
        try:
            # Check daily limits
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("Daily trade limit exceeded")
                return False
            
            # Check drawdown
            if self.total_pnl < -(self.max_drawdown * 10000):  # Assuming $10k base
                logger.warning("Maximum drawdown exceeded")
                return False
            
            # Check signal confidence
            if signal.confidence < 0.3:
                logger.warning("Signal confidence too low")
                return False
            
            # Check position size
            if signal.volume <= 0:
                logger.warning("Invalid position size")
                return False
            
            # Check if real trading is enabled
            if not self.enable_real_trading:
                logger.info("Real trading disabled - signal would be executed in live mode")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    async def _execute_strategy(self, signal: TradingSignal) -> StrategyExecution:
        """Execute trading strategy."""
        try:
            start_time = time.time()
            
            # Create trade order
            order = TradeOrder(
                symbol=signal.symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                amount=signal.volume,
                price=signal.entry_price,
                stop_price=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata=signal.metadata
            )
            
            # Place order
            if self.ccxt_executor:
                result = await self.ccxt_executor.place_order(order)
                
                execution_time = (time.time() - start_time) * 1000
                
                if result.success:
                    # Calculate slippage
                    slippage = abs(result.price - signal.entry_price) / signal.entry_price if result.price else 0
                    
                    return StrategyExecution(
                        signal_id=signal.signal_id,
                        strategy_type=signal.strategy_type,
                        symbol=signal.symbol,
                        side=signal.side,
                        executed=True,
                        order_id=result.order_id,
                        fill_price=result.price,
                        fill_amount=result.amount,
                        execution_time_ms=execution_time,
                        slippage=slippage
                    )
                else:
                    return StrategyExecution(
                        signal_id=signal.signal_id,
                        strategy_type=signal.strategy_type,
                        symbol=signal.symbol,
                        side=signal.side,
                        executed=False,
                        error_message=result.error_message
                    )
            else:
                return StrategyExecution(
                    signal_id=signal.signal_id,
                    strategy_type=signal.strategy_type,
                    symbol=signal.symbol,
                    side=signal.side,
                    executed=False,
                    error_message="CCXT executor not available"
                )
                
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return StrategyExecution(
                signal_id=signal.signal_id,
                strategy_type=signal.strategy_type,
                symbol=signal.symbol,
                side=signal.side,
                executed=False,
                error_message=str(e)
            )
    
    async def _track_execution(self, execution: StrategyExecution):
        """Track strategy execution performance."""
        try:
            # Update daily counters
            self.daily_trades += 1
            
            # Calculate PnL (simplified)
            if execution.fill_price and execution.fill_amount:
                # This is a simplified PnL calculation
                # In a real system, you'd track actual position PnL
                pass
            
            # Store execution in history
            self.strategy_performance[execution.strategy_type.value].append({
                "timestamp": execution.timestamp,
                "symbol": execution.symbol,
                "side": execution.side.value,
                "executed": execution.executed,
                "fill_price": execution.fill_price,
                "fill_amount": execution.fill_amount,
                "execution_time_ms": execution.execution_time_ms,
                "slippage": execution.slippage
            })
            
        except Exception as e:
            logger.error(f"Error tracking execution: {e}")
    
    async def _on_order_book_update(self, order_book: OrderBookSnapshot):
        """Handle order book updates."""
        try:
            # Analyze order book for trading opportunities
            imbalance = self.order_book_manager.detect_order_book_imbalance(order_book.symbol)
            
            if imbalance and not imbalance.get("error"):
                # Check for significant imbalance
                if abs(imbalance["imbalance"]) > 0.1:  # 10% imbalance
                    logger.info(f"Significant order book imbalance detected: {imbalance}")
                    
                    # Could trigger additional trading signals here
                    
        except Exception as e:
            logger.error(f"Error handling order book update: {e}")
    
    def _reset_daily_counters(self):
        """Reset daily trading counters."""
        current_time = time.time()
        if current_time - self.last_reset > 86400:  # 24 hours
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = current_time
    
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        try:
            performance = {}
            
            for strategy_name, executions in self.strategy_performance.items():
                if executions:
                    total_executions = len(executions)
                    successful_executions = len([e for e in executions if e["executed"]])
                    win_rate = successful_executions / total_executions if total_executions > 0 else 0
                    
                    avg_execution_time = np.mean([e.get("execution_time_ms", 0) for e in executions])
                    avg_slippage = np.mean([e.get("slippage", 0) for e in executions])
                    
                    performance[strategy_name] = {
                        "total_executions": total_executions,
                        "successful_executions": successful_executions,
                        "win_rate": win_rate,
                        "avg_execution_time_ms": avg_execution_time,
                        "avg_slippage": avg_slippage
                    }
            
            return {
                "strategy_performance": performance,
                "daily_trades": self.daily_trades,
                "daily_pnl": self.daily_pnl,
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}


def create_trading_strategy_executor(config: Dict[str, Any]) -> TradingStrategyExecutor:
    """Factory function to create trading strategy executor."""
    return TradingStrategyExecutor(config) 