#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Execution Engine for Schwabot Trading System.

Continuous market monitoring, signal generation, and strategy execution
with quantum mathematical integration and advanced risk management.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .real_time_market_data import RealTimeMarketDataStream, MarketDataEvent, DataType  # noqa: F401
from .smart_order_executor import SmartOrderExecutor, OrderExecution
from .advanced_risk_manager import AdvancedRiskManager
from .order_book_analyzer import OrderBookAnalyzer
from .clean_trading_pipeline import CleanTradingPipeline
from .zpe_zbe_core import create_zpe_zbe_core
from .advanced_tensor_algebra import AdvancedTensorAlgebra

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class TradingSignal:
    """Trading signal with comprehensive analysis."""
    signal_type: SignalType
    symbol: str
    exchange: str
    strength: SignalStrength
    confidence: float
    price: float
    quantity: float
    timestamp: float
    stop_loss: float
    take_profit: float
    market_conditions: Dict[str, Any]
    quantum_signals: Dict[str, Any]
    tensor_signals: Dict[str, Any]
    zpe_zbe_signals: Dict[str, Any]
    order_book_signals: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of signal execution."""
    signal: TradingSignal
    execution: Optional[OrderExecution]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    total_signals: int
    successful_signals: int
    failed_signals: int
    total_pnl: float
    win_rate: float
    average_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    current_positions: int
    risk_metrics: Dict[str, Any]


class RealTimeExecutionEngine:
    """
    Real-time execution engine for continuous market monitoring and strategy execution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the real-time execution engine."""
        self.config = config or self._default_config()
        
        # Core components
        self.market_data_stream: Optional[RealTimeMarketDataStream] = None
        self.order_executor: Optional[SmartOrderExecutor] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.order_book_analyzer: Optional[OrderBookAnalyzer] = None
        self.trading_pipeline: Optional[CleanTradingPipeline] = None
        
        # Quantum components
        self.zpe_zbe_core = create_zpe_zbe_core()
        self.tensor_algebra = AdvancedTensorAlgebra()
        
        # Signal processing
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            total_signals=0,
            successful_signals=0,
            failed_signals=0,
            total_pnl=0.0,
            win_rate=0.0,
            average_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            current_positions=0,
            risk_metrics={},
        )
        
        # Control flags
        self.running = False
        self.monitoring_active = False
        self.tasks: Set[asyncio.Task] = set()
        
        # Callbacks
        self.signal_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
        logger.info("RealTimeExecutionEngine initialized with config: %s", self.config)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for real-time execution."""
        return {
            "exchanges": ["binance", "coinbase"],
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "monitoring_interval": 1.0,  # seconds
            "signal_threshold": 0.7,  # Minimum confidence for signal execution
            "max_concurrent_positions": 5,
            "enable_quantum_analysis": True,
            "enable_tensor_analysis": True,
            "enable_zpe_zbe_analysis": True,
            "enable_order_book_analysis": True,
            "risk_management": {
                "max_position_size": 0.1,  # 10% of capital
                "max_daily_loss": 0.05,  # 5% daily loss limit
                "max_drawdown": 0.15,  # 15% max drawdown
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_risk_reward": 2.0,
            },
            "signal_generation": {
                "min_confidence": 0.6,
                "min_strength": "moderate",
                "max_signals_per_hour": 10,
                "cooldown_period": 300,  # 5 minutes
            },
            "execution": {
                "max_slippage": 0.001,  # 0.1%
                "execution_timeout": 30.0,
                "retry_attempts": 3,
            },
        }

    async def initialize(self):
        """Initialize all components and start the execution engine."""
        try:
            logger.info("Initializing real-time execution engine...")
            
            # Initialize market data stream
            self.market_data_stream = RealTimeMarketDataStream(self.config)
            await self.market_data_stream.initialize()
            
            # Initialize order executor
            self.order_executor = SmartOrderExecutor(self.config)
            await self.order_executor.initialize()
            
            # Initialize risk manager
            self.risk_manager = AdvancedRiskManager(self.config.get("risk_management", {}))
            
            # Initialize order book analyzer
            self.order_book_analyzer = OrderBookAnalyzer()
            
            # Initialize trading pipeline
            self.trading_pipeline = CleanTradingPipeline(
                symbol=self.config["symbols"][0],
                initial_capital=10000.0
            )
            
            # Register market data callbacks
            self.market_data_stream.register_callback(
                DataType.TICKER, self._on_ticker_update
            )
            self.market_data_stream.register_callback(
                DataType.ORDER_BOOK, self._on_order_book_update
            )
            
            self.running = True
            logger.info("Real-time execution engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize real-time execution engine: %s", e)
            raise

    async def start_monitoring(self):
        """Start real-time market monitoring and strategy execution."""
        try:
            if not self.running:
                raise RuntimeError("Execution engine not initialized")
            
            logger.info("Starting real-time market monitoring...")
            self.monitoring_active = True
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._market_monitoring_task()),
                asyncio.create_task(self._signal_processing_task()),
                asyncio.create_task(self._risk_monitoring_task()),
                asyncio.create_task(self._performance_tracking_task()),
                asyncio.create_task(self._position_management_task()),
            ]
            
            for task in tasks:
                self.tasks.add(task)
            
            logger.info("Real-time monitoring started successfully")
            
        except Exception as e:
            logger.error("Failed to start monitoring: %s", e)
            raise

    async def _market_monitoring_task(self):
        """Continuous market monitoring task."""
        while self.monitoring_active:
            try:
                # Get current market states
                market_states = self.market_data_stream.get_all_market_states()
                
                for market_key, market_state in market_states.items():
                    # Generate signals for each market
                    signals = await self._generate_signals(market_state)
                    
                    # Queue signals for processing
                    for signal in signals:
                        await self.signal_queue.put(signal)
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error("Market monitoring task error: %s", e)
                await asyncio.sleep(5)

    async def _signal_processing_task(self):
        """Process signals from the queue."""
        while self.monitoring_active:
            try:
                # Get signal from queue
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                # Validate signal
                if not self._validate_signal(signal):
                    continue
                
                # Check risk limits
                if not self._check_risk_limits(signal):
                    logger.info("Signal rejected due to risk limits: %s", signal.symbol)
                    continue
                
                # Execute signal
                execution_result = await self._execute_signal(signal)
                
                # Update performance metrics
                self._update_performance_metrics(execution_result)
                
                # Trigger callbacks
                await self._trigger_execution_callbacks(execution_result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Signal processing error: %s", e)

    async def _risk_monitoring_task(self):
        """Continuous risk monitoring task."""
        while self.monitoring_active:
            try:
                # Check portfolio risk
                portfolio_risk = self.risk_manager.assess_portfolio_risk(
                    list(self.active_positions.values()),
                    {}  # Market data would be passed here
                )
                
                # Check if risk limits are exceeded
                if portfolio_risk.total_risk > self.config["risk_management"]["max_drawdown"]:
                    logger.warning(
                        "Portfolio risk limit exceeded: %.2f%%", 
                        portfolio_risk.total_risk * 100
                    )
                    await self._reduce_risk_exposure()
                
                # Update performance metrics
                self.performance_metrics.risk_metrics = {
                    "portfolio_risk": portfolio_risk.total_risk,
                    "max_drawdown": portfolio_risk.max_portfolio_drawdown,
                    "diversification_score": portfolio_risk.diversification_score,
                }
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("Risk monitoring task error: %s", e)
                await asyncio.sleep(10)

    async def _performance_tracking_task(self):
        """Track and update performance metrics."""
        while self.monitoring_active:
            try:
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                # Log performance summary
                if self.performance_metrics.total_signals % 10 == 0:  # Every 10 signals
                    logger.info("Performance Summary: %s", self._get_performance_summary())
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error("Performance tracking task error: %s", e)
                await asyncio.sleep(30)

    async def _position_management_task(self):
        """Manage active positions and stop-loss/take-profit orders."""
        while self.monitoring_active:
            try:
                # Check active positions
                for position_id, position in self.active_positions.items():
                    # Check stop-loss and take-profit
                    current_price = self._get_current_price(position["symbol"])
                    
                    if current_price <= position["stop_loss"]:
                        await self._close_position(position_id, "stop_loss")
                    elif current_price >= position["take_profit"]:
                        await self._close_position(position_id, "take_profit")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error("Position management task error: %s", e)
                await asyncio.sleep(5)

    async def _generate_signals(self, market_state) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        try:
            signals = []
            
            # Basic signal generation from trading pipeline
            if self.trading_pipeline:
                pipeline_signals = await self._generate_pipeline_signals(market_state)
                signals.extend(pipeline_signals)
            
            # Quantum-enhanced signal generation
            if self.config["enable_quantum_analysis"]:
                quantum_signals = await self._generate_quantum_signals(market_state)
                signals.extend(quantum_signals)
            
            # Tensor-based signal generation
            if self.config["enable_tensor_analysis"]:
                tensor_signals = await self._generate_tensor_signals(market_state)
                signals.extend(tensor_signals)
            
            # ZPE-ZBE signal generation
            if self.config["enable_zpe_zbe_analysis"]:
                zpe_zbe_signals = await self._generate_zpe_zbe_signals(market_state)
                signals.extend(zpe_zbe_signals)
            
            # Order book-based signals
            if self.config["enable_order_book_analysis"]:
                order_book_signals = await self._generate_order_book_signals(market_state)
                signals.extend(order_book_signals)
            
            return signals
            
        except Exception as e:
            logger.error("Signal generation failed: %s", e)
            return []

    async def _generate_pipeline_signals(self, market_state) -> List[TradingSignal]:
        """Generate signals using the trading pipeline."""
        try:
            signals = []
            
            # Create market data for pipeline
            market_data = {
                "price": market_state.current_price,
                "volume": market_state.volume_24h,
                "change": market_state.change_24h,
                "volatility": market_state.volatility,
                "bid": market_state.bid,
                "ask": market_state.ask,
                "spread": market_state.spread,
            }
            
            # Get pipeline decision
            decision = self.trading_pipeline._make_trading_decision(market_data)
            
            if decision["action"] in ["buy", "sell"]:
                # Calculate signal strength and confidence
                confidence = decision.get("confidence", 0.5)
                strength = self._calculate_signal_strength(confidence)
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    {"confidence": confidence},
                    market_data
                )
                
                # Calculate stop-loss and take-profit
                stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                    market_state.current_price,
                    market_data,
                    position_size
                )
                
                take_profit = self.risk_manager.calculate_dynamic_take_profit(
                    market_state.current_price,
                    stop_loss,
                    market_data,
                    position_size
                )
                
                signal = TradingSignal(
                    signal_type=SignalType.BUY if decision["action"] == "buy" else SignalType.SELL,
                    symbol=market_state.symbol,
                    exchange=market_state.exchange,
                    strength=strength,
                    confidence=confidence,
                    price=market_state.current_price,
                    quantity=position_size,
                    timestamp=time.time(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    market_conditions=market_data,
                    quantum_signals=market_state.quantum_signals,
                    tensor_signals=market_state.tensor_signals,
                    zpe_zbe_signals=market_state.zpe_zbe_signals,
                    order_book_signals={},
                    risk_metrics={},
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("Pipeline signal generation failed: %s", e)
            return []

    async def _generate_quantum_signals(self, market_state) -> List[TradingSignal]:
        """Generate signals using quantum analysis."""
        try:
            signals = []
            
            # Extract quantum signals
            quantum_signals = market_state.quantum_signals
            
            # Analyze quantum entanglement
            entanglement = quantum_signals.get("quantum_entanglement", 0.0)
            coherence = quantum_signals.get("quantum_coherence", 0.0)
            
            # Generate signal based on quantum metrics
            if entanglement > 0.7 and coherence > 0.6:
                confidence = min(entanglement * coherence, 0.95)
                strength = self._calculate_signal_strength(confidence)
                
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    symbol=market_state.symbol,
                    exchange=market_state.exchange,
                    strength=strength,
                    confidence=confidence,
                    price=market_state.current_price,
                    quantity=0.05,  # Small position for quantum signals
                    timestamp=time.time(),
                    stop_loss=market_state.current_price * 0.98,
                    take_profit=market_state.current_price * 1.02,
                    market_conditions={},
                    quantum_signals=quantum_signals,
                    tensor_signals={},
                    zpe_zbe_signals={},
                    order_book_signals={},
                    risk_metrics={},
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("Quantum signal generation failed: %s", e)
            return []

    async def _generate_tensor_signals(self, market_state) -> List[TradingSignal]:
        """Generate signals using tensor analysis."""
        try:
            signals = []
            
            # Extract tensor signals
            tensor_signals = market_state.tensor_signals
            
            # Analyze tensor metrics
            tensor_rank = tensor_signals.get("tensor_rank", 0)
            tensor_norm = tensor_signals.get("tensor_norm", 0.0)
            
            # Generate signal based on tensor analysis
            if tensor_rank > 2 and tensor_norm > 0.5:
                confidence = min(tensor_norm * 0.8, 0.9)
                strength = self._calculate_signal_strength(confidence)
                
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    symbol=market_state.symbol,
                    exchange=market_state.exchange,
                    strength=strength,
                    confidence=confidence,
                    price=market_state.current_price,
                    quantity=0.03,  # Small position for tensor signals
                    timestamp=time.time(),
                    stop_loss=market_state.current_price * 0.98,
                    take_profit=market_state.current_price * 1.02,
                    market_conditions={},
                    quantum_signals={},
                    tensor_signals=tensor_signals,
                    zpe_zbe_signals={},
                    order_book_signals={},
                    risk_metrics={},
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("Tensor signal generation failed: %s", e)
            return []

    async def _generate_zpe_zbe_signals(self, market_state) -> List[TradingSignal]:
        """Generate signals using ZPE-ZBE analysis."""
        try:
            signals = []
            
            # Extract ZPE-ZBE signals
            zpe_signals = market_state.zpe_zbe_signals
            
            # Analyze ZPE metrics
            zpe_energy = zpe_signals.get("zpe_energy", 0.0)
            zpe_frequency = zpe_signals.get("zpe_frequency", 0.0)
            
            # Generate signal based on ZPE analysis
            if zpe_energy > 0.6 and 7.0 <= zpe_frequency <= 8.0:  # Schumann resonance range
                confidence = min(zpe_energy * 0.7, 0.85)
                strength = self._calculate_signal_strength(confidence)
                
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    symbol=market_state.symbol,
                    exchange=market_state.exchange,
                    strength=strength,
                    confidence=confidence,
                    price=market_state.current_price,
                    quantity=0.02,  # Small position for ZPE signals
                    timestamp=time.time(),
                    stop_loss=market_state.current_price * 0.98,
                    take_profit=market_state.current_price * 1.02,
                    market_conditions={},
                    quantum_signals={},
                    tensor_signals={},
                    zpe_zbe_signals=zpe_signals,
                    order_book_signals={},
                    risk_metrics={},
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("ZPE-ZBE signal generation failed: %s", e)
            return []

    async def _generate_order_book_signals(self, market_state) -> List[TradingSignal]:
        """Generate signals using order book analysis."""
        try:
            signals = []
            
            if not market_state.order_book_snapshot:
                return signals
            
            # Analyze order book
            snapshot = market_state.order_book_snapshot
            
            # Check for strong buy/sell walls
            buy_walls = [
                w for w in snapshot.walls if w.wall_type.value == "buy_wall"
            ]
            # sell_walls = [w for w in snapshot.walls if w.wall_type.value == "sell_wall"]  # Future implementation
            
            # Generate signal based on wall strength
            if buy_walls and max(w.strength_score for w in buy_walls) > 0.8:
                confidence = 0.75
                strength = self._calculate_signal_strength(confidence)
                
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    symbol=market_state.symbol,
                    exchange=market_state.exchange,
                    strength=strength,
                    confidence=confidence,
                    price=market_state.current_price,
                    quantity=0.04,  # Small position for order book signals
                    timestamp=time.time(),
                    stop_loss=market_state.current_price * 0.98,
                    take_profit=market_state.current_price * 1.02,
                    market_conditions={},
                    quantum_signals={},
                    tensor_signals={},
                    zpe_zbe_signals={},
                    order_book_signals={"wall_analysis": "strong_buy_wall"},
                    risk_metrics={},
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error("Order book signal generation failed: %s", e)
            return []

    def _calculate_signal_strength(self, confidence: float) -> SignalStrength:
        """Calculate signal strength from confidence level."""
        if confidence >= 0.9:
            return SignalStrength.EXTREME
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal."""
        try:
            # Check minimum confidence
            if signal.confidence < self.config["signal_generation"]["min_confidence"]:
                return False
            
            # Check minimum strength
            min_strength = SignalStrength(self.config["signal_generation"]["min_strength"])
            if signal.strength.value < min_strength.value:
                return False
            
            # Check cooldown period
            if not self._check_signal_cooldown(signal):
                return False
            
            # Check maximum signals per hour
            if not self._check_signal_frequency():
                return False
            
            return True
            
        except Exception as e:
            logger.error("Signal validation failed: %s", e)
            return False

    def _check_signal_cooldown(self, signal: TradingSignal) -> bool:
        """Check if enough time has passed since last signal for this symbol."""
        try:
            cooldown_period = self.config["signal_generation"]["cooldown_period"]
            current_time = time.time()
            
            # Check recent signals for this symbol
            for recent_signal in self.signal_history[-10:]:  # Check last 10 signals
                if (recent_signal.symbol == signal.symbol and 
                    current_time - recent_signal.timestamp < cooldown_period):
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Signal cooldown check failed: %s", e)
            return True

    def _check_signal_frequency(self) -> bool:
        """Check if maximum signals per hour limit is not exceeded."""
        try:
            max_signals_per_hour = self.config["signal_generation"]["max_signals_per_hour"]
            current_time = time.time()
            one_hour_ago = current_time - 3600
            
            # Count signals in the last hour
            recent_signals = [s for s in self.signal_history if s.timestamp > one_hour_ago]
            
            return len(recent_signals) < max_signals_per_hour
            
        except Exception as e:
            logger.error("Signal frequency check failed: %s", e)
            return True

    def _check_risk_limits(self, signal: TradingSignal) -> bool:
        """Check if signal complies with risk limits."""
        try:
            # Check maximum concurrent positions
            if len(self.active_positions) >= self.config["max_concurrent_positions"]:
                return False
            
            # Check if we already have a position in this symbol
            if signal.symbol in self.active_positions:
                return False
            
            # Check daily loss limit
            if self.performance_metrics.total_pnl < -self.config["risk_management"]["max_daily_loss"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error("Risk limit check failed: %s", e)
            return False

    async def _execute_signal(self, signal: TradingSignal) -> ExecutionResult:
        """Execute trading signal."""
        try:
            start_time = time.time()
            
            # Create signal dictionary for order executor
            signal_dict = {
                "symbol": signal.symbol,
                "direction": signal.signal_type.value,
                "confidence": signal.confidence,
                "current_price": signal.price,
                "quantity": signal.quantity,
                "urgency": "normal",
                "market_conditions": signal.market_conditions,
            }
            
            # Execute signal
            execution = await self.order_executor.execute_signal(
                signal_dict, signal.quantity
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Calculate P&L (for now, assume 0 for new positions)
            pnl = 0.0
            
            # Create execution result
            result = ExecutionResult(
                signal=signal,
                execution=execution,
                success=execution.status == "closed",
                execution_time=execution_time,
                pnl=pnl,
            )
            
            # Add to active positions if successful
            if result.success:
                self.active_positions[signal.symbol] = {
                    "signal": signal,
                    "execution": execution,
                    "entry_price": execution.average_price,
                    "quantity": execution.executed_quantity,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "entry_time": time.time(),
                }
            
            # Add to history
            self.signal_history.append(signal)
            self.execution_history.append(result)
            
            logger.info("Signal executed: %s %s %s (confidence: %.2f)", 
                       signal.signal_type.value, signal.quantity, signal.symbol, 
                       signal.confidence)
            
            return result
            
        except Exception as e:
            logger.error("Signal execution failed: %s", e)
            return ExecutionResult(
                signal=signal,
                execution=None,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def _close_position(self, position_id: str, reason: str):
        """Close an active position."""
        try:
            position = self.active_positions.get(position_id)
            if not position:
                return
            
            # Create close signal
            close_signal = TradingSignal(
                signal_type=SignalType.SELL if position["signal"].signal_type == SignalType.BUY else SignalType.BUY,
                symbol=position_id,
                exchange=position["signal"].exchange,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                price=self._get_current_price(position_id),
                quantity=position["quantity"],
                timestamp=time.time(),
                stop_loss=0.0,
                take_profit=0.0,
                market_conditions={},
                quantum_signals={},
                tensor_signals={},
                zpe_zbe_signals={},
                order_book_signals={},
                risk_metrics={},
                metadata={"close_reason": reason},
            )
            
            # Execute close order
            close_result = await self._execute_signal(close_signal)
            
            # Calculate P&L
            if close_result.success and close_result.execution:
                entry_price = position["entry_price"]
                exit_price = close_result.execution.average_price
                quantity = position["quantity"]
                
                if position["signal"].signal_type == SignalType.BUY:
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity
                
                close_result.pnl = pnl
                self.performance_metrics.total_pnl += pnl
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            logger.info("Position closed: %s (reason: %s, P&L: %.2f)", 
                       position_id, reason, close_result.pnl)
            
        except Exception as e:
            logger.error("Failed to close position %s: %s", position_id, e)

    async def _reduce_risk_exposure(self):
        """Reduce risk exposure by closing some positions."""
        try:
            # Close positions with lowest confidence
            positions_by_confidence = sorted(
                self.active_positions.items(),
                key=lambda x: x[1]["signal"].confidence
            )
            
            # Close bottom 50% of positions
            positions_to_close = positions_by_confidence[:len(positions_by_confidence) // 2]
            
            for position_id, _ in positions_to_close:
                await self._close_position(position_id, "risk_reduction")
            
            logger.info("Reduced risk exposure by closing %d positions", 
                       len(positions_to_close))
            
        except Exception as e:
            logger.error("Failed to reduce risk exposure: %s", e)

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            # This would typically get from market data stream
            # For now, return a default price
            return 50000.0  # Default BTC price
        except Exception as e:
            logger.error("Failed to get current price for %s: %s", symbol, e)
            return 0.0

    def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        try:
            total_signals = len(self.execution_history)
            successful_signals = len([r for r in self.execution_history if r.success])
            
            self.performance_metrics.total_signals = total_signals
            self.performance_metrics.successful_signals = successful_signals
            self.performance_metrics.failed_signals = total_signals - successful_signals
            self.performance_metrics.win_rate = successful_signals / total_signals if total_signals > 0 else 0.0
            self.performance_metrics.average_pnl = (
                self.performance_metrics.total_pnl / total_signals 
                if total_signals > 0 else 0.0
            )
            self.performance_metrics.current_positions = len(self.active_positions)
            
            # Calculate Sharpe ratio (simplified)
            if total_signals > 0:
                returns = [r.pnl for r in self.execution_history]
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    self.performance_metrics.sharpe_ratio = (
                        avg_return / std_return if std_return > 0 else 0.0
                    )
            
        except Exception as e:
            logger.error("Performance metrics calculation failed: %s", e)

    def _update_performance_metrics(self, execution_result: ExecutionResult):
        """Update performance metrics with new execution result."""
        try:
            self.performance_metrics.total_pnl += execution_result.pnl
            
            # Update max drawdown
            if execution_result.pnl < 0:
                current_drawdown = abs(execution_result.pnl)
                if current_drawdown > self.performance_metrics.max_drawdown:
                    self.performance_metrics.max_drawdown = current_drawdown
            
        except Exception as e:
            logger.error("Performance metrics update failed: %s", e)

    async def _trigger_execution_callbacks(self, execution_result: ExecutionResult):
        """Trigger execution callbacks."""
        try:
            for callback in self.execution_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(execution_result)
                    else:
                        callback(execution_result)
                except Exception as e:
                    logger.error("Execution callback error: %s", e)
                    
        except Exception as e:
            logger.error("Failed to trigger execution callbacks: %s", e)

    def register_signal_callback(self, callback: Callable):
        """Register callback for signal generation."""
        self.signal_callbacks.append(callback)

    def register_execution_callback(self, callback: Callable):
        """Register callback for signal execution."""
        self.execution_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_signals": self.performance_metrics.total_signals,
            "successful_signals": self.performance_metrics.successful_signals,
            "failed_signals": self.performance_metrics.failed_signals,
            "win_rate": self.performance_metrics.win_rate,
            "total_pnl": self.performance_metrics.total_pnl,
            "average_pnl": self.performance_metrics.average_pnl,
            "max_drawdown": self.performance_metrics.max_drawdown,
            "sharpe_ratio": self.performance_metrics.sharpe_ratio,
            "current_positions": self.performance_metrics.current_positions,
            "risk_metrics": self.performance_metrics.risk_metrics,
        }

    def _get_performance_summary(self) -> str:
        """Get formatted performance summary string."""
        summary = self.get_performance_summary()
        return (
            f"Signals: {summary['total_signals']}, "
            f"Win Rate: {summary['win_rate']:.2%}, "
            f"P&L: ${summary['total_pnl']:.2f}, "
            f"Positions: {summary['current_positions']}"
        )

    async def stop(self):
        """Stop the real-time execution engine."""
        try:
            logger.info("Stopping real-time execution engine...")
            
            self.monitoring_active = False
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close all positions
            for position_id in list(self.active_positions.keys()):
                await self._close_position(position_id, "system_shutdown")
            
            # Stop components
            if self.market_data_stream:
                await self.market_data_stream.stop()
            
            if self.order_executor:
                await self.order_executor.stop()
            
            logger.info("Real-time execution engine stopped")
            
        except Exception as e:
            logger.error("Failed to stop real-time execution engine: %s", e)


# Convenience functions for external use
def create_real_time_execution_engine(
    config: Optional[Dict[str, Any]] = None
) -> RealTimeExecutionEngine:
    """Create a new real-time execution engine instance."""
    return RealTimeExecutionEngine(config)


async def start_real_time_execution_engine(
    config: Optional[Dict[str, Any]] = None
) -> RealTimeExecutionEngine:
    """Start a real-time execution engine."""
    engine = RealTimeExecutionEngine(config)
    await engine.initialize()
    await engine.start_monitoring()
    return engine 