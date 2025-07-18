#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Executor - Execute trading strategies with Math + Memory Fusion Core
============================================================================

This module handles the execution of trading strategies, signal generation,
and coordination between different strategies with full integration of the
Math + Memory Fusion Core for enhanced decision making.

Features:
- Unified signal generation with profit vector integration
- Entropy-corrected strategy execution
- Signal lineage tracking for recursive learning
- Bridge functions between strategy and mathematical confidence
- Real-time market data analysis with mathematical context
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Define a simple protocol for trading strategies
class TradingStrategy(Protocol):
    """Protocol for trading strategies."""
    is_initialized: bool

    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals."""
        ...

# Import Math + Memory Fusion Core
try:
    from core.clean_unified_math import CleanUnifiedMathSystem, UnifiedSignal
    from core.unified_profit_vectorization_system import UnifiedProfitVectorizationSystem, ProfitVector
    from core.profit_scaling_optimizer import create_profit_scaling_optimizer, RiskProfile
    from core.schwafit_overfitting_prevention import create_schwafit_overfitting_prevention, SanitizationLevel
    MATH_FUSION_AVAILABLE = True
except ImportError:
    MATH_FUSION_AVAILABLE = False
    logger.warning("Math + Memory Fusion Core not available - using fallback mode")

@dataclass
class EnhancedTradingSignal:
    """Enhanced trading signal with mathematical fusion context."""

    # Basic signal data
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    entry_price: float
    amount: float
    strategy_id: str

    # Mathematical fusion context
    unified_signal: Optional[Any] = None
    profit_vectors: List[Any] = None
    mathematical_confidence: float = 0.0
    entropy_correction: float = 0.0
    vector_confidence: float = 0.0

    # Market context
    volatility: float = 0.0
    volume: float = 0.0
    market_conditions: Dict[str, Any] = None

    # Signal lineage
    timestamp: float = time.time()
    signal_hash: Optional[str] = None
    parent_signals: List[str] = None
    metadata: Dict[str, Any] = None  # Added for profit scaling metadata

class StrategyExecutor:
    """
    Execute trading strategies and generate signals with Math + Memory Fusion Core.

    This class coordinates the execution of multiple trading strategies,
    combines their signals with mathematical fusion, and provides a unified
    interface for enhanced signal generation with profit vector memory.
    """

    def __init__(self) -> None:
        """Initialize the strategy executor with Math + Memory Fusion Core."""
        self.active_strategies: Dict[str, TradingStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.is_running = False
        self.is_initialized = False
        self.execution_task: Optional[asyncio.Task] = None

        # Math + Memory Fusion Core integration
        if MATH_FUSION_AVAILABLE:
            self.math_system = CleanUnifiedMathSystem()
            self.profit_system = UnifiedProfitVectorizationSystem()
            self.profit_scaling_optimizer = create_profit_scaling_optimizer()
            self.overfitting_prevention_system = create_schwafit_overfitting_prevention()
            logger.info("🧠 Math + Memory Fusion Core integrated")
        else:
            self.math_system = None
            self.profit_system = None
            self.profit_scaling_optimizer = None
            self.overfitting_prevention_system = None
            logger.warning("🧠 Math + Memory Fusion Core not available")

        # Enhanced signal tracking
        self.signal_history: List[EnhancedTradingSignal] = []
        self.profit_vector_history: List[Any] = []
        self.max_signal_history = 1000
        self.max_profit_history = 500

        # Integration parameters
        self.min_unified_confidence = 0.6
        self.entropy_correction_threshold = 0.3
        self.vector_confidence_weight = 0.4
        self.mathematical_confidence_weight = 0.6

        logger.info("Strategy Executor initialized with enhanced capabilities")

    async def initialize(self) -> bool:
        """Initialize the strategy executor with Math + Memory Fusion Core."""
        try:
            logger.info("Initializing Enhanced Strategy Executor...")

            # Set default strategy weights
            self.strategy_weights = {
                "ExampleStrategy": 1.0,
                "VolumeWeightedHashOscillator": 0.8,
                "MultiPhaseStrategyWeightTensor": 0.9,
                "ZygotZalgoEntropyDualKeyGate": 0.7
            }

            # Initialize Math + Memory Fusion Core if available
            if MATH_FUSION_AVAILABLE:
                logger.info("🧠 Initializing Math + Memory Fusion Core...")
                # Load historical profit vectors if available
                await self._load_historical_profit_vectors()

            self.is_initialized = True
            logger.info("Enhanced Strategy Executor initialized successfully")
            return True
        except Exception as e:
            logger.error(
                f"Failed to initialize Enhanced Strategy Executor: {e}")
            return False

    async def _load_historical_profit_vectors(self):
        """Load historical profit vectors for mathematical context."""
        try:
            # Generate some sample profit vectors for testing
            # In production, this would load from persistent storage
            sample_vectors = []
            for i in range(10):
                vector = self.profit_system.generate_profit_vector(
                    entry_tick=1000 + i * 100,
                    profit=0.02 + (i % 3) * 0.01,
                    strategy_hash=f"historical_{i}",
                    drawdown=0.01 + (i % 2) * 0.005,
                    entropy_delta=0.1 + (i % 3) * 0.05,
                    exit_type="stack_hold",
                    risk_profile="low"
                )
                sample_vectors.append(vector)

            self.profit_vector_history.extend(sample_vectors)
            logger.info(
                f"📊 Loaded {len(sample_vectors)} historical profit vectors")
        except Exception as e:
            logger.error(
                f"Error loading historical profit vectors: {e}")

    async def generate_unified_signals(
        self, market_data: Dict[str, Any]) -> List[EnhancedTradingSignal]:
        """
        Generate unified trading signals using Math + Memory Fusion Core.

        Args:
        market_data: Market data for analysis

        Returns:
        List of enhanced trading signals with mathematical fusion
        """
        try:
            if not MATH_FUSION_AVAILABLE:
                logger.warning(
                    "Math + Memory Fusion Core not available, using fallback")
                return await self.generate_signals(market_data)

            enhanced_signals = []

            # Generate unified signal using Math +
            # Memory Fusion Core
            unified_signal = self.math_system.generate_unified_signal(
                market_data=market_data,
                profit_vectors=self.profit_vector_history
            )

            # Generate enhanced strategy signals
            strategy_signals = await self._generate_enhanced_strategy_signals(market_data, unified_signal)

            # Combine and enhance signals
            for signal in strategy_signals:
                enhanced_signal = self._enhance_signal_with_fusion(signal, unified_signal, market_data)
                enhanced_signals.append(enhanced_signal)

            # Store signals in history
            self.signal_history.extend(enhanced_signals)
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]

            logger.info(f"Generated {len(enhanced_signals)} enhanced signals with Math + Memory Fusion Core")
            return enhanced_signals

        except Exception as e:
            logger.error(f"Error generating unified signals: {e}")
            return []

    def _enhance_signal_with_fusion(self, signal: EnhancedTradingSignal, unified_signal: Any, market_data: Dict[str, Any]) -> EnhancedTradingSignal:
        """Enhance signal with mathematical fusion context."""
        try:
            # Apply mathematical confidence
            signal.mathematical_confidence = unified_signal.confidence

            # Apply entropy correction
            signal.entropy_correction = unified_signal.entropy_correction

            # Apply vector confidence
            signal.vector_confidence = unified_signal.vector_confidence

            # Generate signal hash for lineage tracking
            signal.signal_hash = self._generate_signal_hash(signal)

            # Add market context
            signal.volatility = market_data.get('volatility', 0.0)
            signal.volume = market_data.get('volume', 0.0)
            signal.market_conditions = market_data

            return signal
        except Exception as e:
            logger.error(f"Error enhancing signal with fusion: {e}")
            return signal

    def _generate_signal_hash(self, signal_data: Any) -> str:
        """Generate a hash for signal lineage tracking."""
        try:
            import hashlib
            signal_str = f"{signal_data.symbol}{signal_data.action}{signal_data.entry_price}{signal_data.timestamp}"
            return hashlib.sha256(signal_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error generating signal hash: {e}")
            return ""

    async def start(self) -> bool:
        """Start the strategy executor."""
        try:
            if not self.is_initialized:
                logger.error("Strategy executor not initialized")
                return False

            self.is_running = True
            self.execution_task = asyncio.create_task(self._process_strategies())
            logger.info("Strategy executor started")
            return True
        except Exception as e:
            logger.error(f"Failed to start strategy executor: {e}")
            return False

    async def _process_strategies(self):
        """Main processing loop for strategies."""
        try:
            while self.is_running:
                # Generate simulated market data for testing
                market_data = self._generate_simulated_market_data()

                # Generate unified signals
                signals = await self.generate_unified_signals(market_data)

                # Process signals with profit scaling
                for signal in signals:
                    scaled_signal = self._apply_profit_scaling(signal, market_data)
                    await self._execute_scaled_trade(scaled_signal, market_data)

                # Wait before next iteration
                await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in strategy processing loop: {e}")

    def _generate_simulated_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data for testing."""
        return {
            'symbol': 'BTC/USD',
            'price': 50000.0 + np.random.normal(0, 100),
            'volume': 1000.0 + np.random.normal(0, 100),
            'volatility': 0.02 + np.random.normal(0, 0.01),
            'timestamp': time.time(),
            'market_conditions': {
                'trend': 'bullish' if np.random.random() > 0.5 else 'bearish',
                'liquidity': 'high' if np.random.random() > 0.3 else 'low'
            }
        }

    def _apply_profit_scaling(self, signal: EnhancedTradingSignal, market_data: Dict[str, Any]) -> EnhancedTradingSignal:
        """Apply profit scaling to the signal."""
        try:
            if self.profit_scaling_optimizer:
                # Get optimal scaling parameters
                scaling_params = self.profit_scaling_optimizer.optimize_scaling(
                    signal=signal,
                    market_data=market_data,
                    risk_profile=RiskProfile.MODERATE
                )

                # Apply scaling
                signal.amount *= scaling_params.get('amount_multiplier', 1.0)
                signal.metadata = scaling_params

            return signal
        except Exception as e:
            logger.error(f"Error applying profit scaling: {e}")
            return signal

    async def _execute_scaled_trade(self, signal: EnhancedTradingSignal, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scaled trade."""
        try:
            # Simulate trade execution
            execution_result = self._simulate_scaled_trade_execution(signal, market_data)

            # Generate profit vector for memory
            if self.profit_system and execution_result.get('executed', False):
                profit_vector = self.profit_system.generate_profit_vector(
                    entry_tick=int(time.time() * 1000),
                    profit=execution_result.get('profit', 0.0),
                    strategy_hash=signal.strategy_id,
                    drawdown=execution_result.get('drawdown', 0.0),
                    entropy_delta=signal.entropy_correction,
                    exit_type=execution_result.get('exit_type', 'manual'),
                    risk_profile=execution_result.get('risk_profile', 'medium')
                )

                # Store profit vector
                self.profit_vector_history.append(profit_vector)
                if len(self.profit_vector_history) > self.max_profit_history:
                    self.profit_vector_history = self.profit_vector_history[-self.max_profit_history:]

            return execution_result
        except Exception as e:
            logger.error(f"Error executing scaled trade: {e}")
            return {'executed': False, 'error': str(e)}

    def _simulate_scaled_trade_execution(self, signal: EnhancedTradingSignal, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the execution of a scaled trade."""
        try:
            # Simulate execution delay
            execution_delay = np.random.normal(0.1, 0.05)
            time.sleep(execution_delay)

            # Simulate execution success/failure
            execution_success = np.random.random() > 0.1  # 90% success rate

            if execution_success:
                # Simulate profit/loss
                base_profit = 0.02  # 2% base profit
                profit_variation = np.random.normal(0, 0.01)  # ±1% variation
                profit = base_profit + profit_variation

                # Simulate drawdown
                drawdown = abs(np.random.normal(0, 0.005))  # 0.5% average drawdown

                return {
                    'executed': True,
                    'profit': profit,
                    'drawdown': drawdown,
                    'exit_type': 'stack_hold',
                    'risk_profile': 'medium',
                    'execution_time': execution_delay,
                    'signal_hash': signal.signal_hash
                }
            else:
                return {
                    'executed': False,
                    'error': 'Simulated execution failure',
                    'execution_time': execution_delay
                }

        except Exception as e:
            logger.error(f"Error in simulated trade execution: {e}")
            return {'executed': False, 'error': str(e)}

    def add_strategy(self, strategy_name: str, strategy: TradingStrategy, weight: float = 1.0) -> bool:
        """Add a strategy to the executor."""
        try:
            self.active_strategies[strategy_name] = strategy
            self.strategy_weights[strategy_name] = weight
            logger.info(f"Added strategy: {strategy_name} with weight {weight}")
            return True
        except Exception as e:
            logger.error(f"Error adding strategy {strategy_name}: {e}")
            return False

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from the executor."""
        try:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
                del self.strategy_weights[strategy_name]
                logger.info(f"Removed strategy: {strategy_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing strategy {strategy_name}: {e}")
            return False

    def set_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """Set the weight for a strategy."""
        try:
            if strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] = weight
                logger.info(f"Set weight for {strategy_name}: {weight}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error setting weight for {strategy_name}: {e}")
            return False

    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from all active strategies."""
        try:
            all_signals = []
            for strategy_name, strategy in self.active_strategies.items():
                if strategy.is_initialized:
                    try:
                        signals = await strategy.generate_signals(analysis)
                        for signal in signals:
                            signal['strategy'] = strategy_name
                            signal['weight'] = self.strategy_weights.get(strategy_name, 1.0)
                            all_signals.extend(signals)
                    except Exception as e:
                        logger.error(f"Error generating signals from {strategy_name}: {e}")

            return all_signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get signal history."""
        history = self.signal_history.copy()
        if limit:
            history = history[-limit:]
        return [signal.__dict__ for signal in history]

    def get_active_strategies(self) -> Dict[str, TradingStrategy]:
        """Get active strategies."""
        return self.active_strategies.copy()

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights."""
        return self.strategy_weights.copy()

    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status."""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'active_strategies': len(self.active_strategies),
            'signal_history_count': len(self.signal_history),
            'profit_vector_count': len(self.profit_vector_history),
            'math_fusion_available': MATH_FUSION_AVAILABLE
        }

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.is_running = False
            if self.execution_task:
                self.execution_task.cancel()
            logger.info("Strategy executor cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _generate_enhanced_strategy_signals(self, market_data: Dict[str, Any], unified_signal: Any) -> List[EnhancedTradingSignal]:
        """Generate enhanced strategy signals."""
        try:
            # This is a placeholder for enhanced strategy signal generation
            # In a real implementation, this would integrate with actual strategies
            enhanced_signals = []
            
            # Create a sample enhanced signal
            sample_signal = EnhancedTradingSignal(
                symbol=market_data.get('symbol', 'BTC/USD'),
                action='hold',
                entry_price=market_data.get('price', 50000.0),
                amount=0.0,
                strategy_id='enhanced_strategy',
                unified_signal=unified_signal
            )
            
            enhanced_signals.append(sample_signal)
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced strategy signals: {e}")
            return [] 