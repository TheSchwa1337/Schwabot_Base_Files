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
    MATH_FUSION_AVAILABLE = True
except ImportError:
    MATH_FUSION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Math + Memory Fusion Core not available - using fallback mode")

logger = logging.getLogger(__name__)


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
    unified_signal: Optional[UnifiedSignal] = None
    profit_vectors: List[ProfitVector] = None
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


class StrategyExecutor:
    """
    Execute trading strategies and generate signals with Math + Memory Fusion Core.
    
    This class coordinates the execution of multiple trading strategies,
    combines their signals with mathematical fusion, and provides a unified
    interface for enhanced signal generation with profit vector memory.
    """
    
    def __init__(self):
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
            logger.info("🧠 Math + Memory Fusion Core integrated")
        else:
            self.math_system = None
            self.profit_system = None
            logger.warning("🧠 Math + Memory Fusion Core not available")
        
        # Enhanced signal tracking
        self.signal_history: List[EnhancedTradingSignal] = []
        self.profit_vector_history: List[ProfitVector] = []
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
            logger.error(f"Failed to initialize Enhanced Strategy Executor: {e}")
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
            logger.info(f"📊 Loaded {len(sample_vectors)} historical profit vectors")
            
        except Exception as e:
            logger.error(f"Error loading historical profit vectors: {e}")

    async def generate_unified_signals(self, market_data: Dict[str, Any]) -> List[EnhancedTradingSignal]:
        """
        Generate unified trading signals using Math + Memory Fusion Core.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of enhanced trading signals with mathematical fusion
        """
        try:
            if not MATH_FUSION_AVAILABLE:
                logger.warning("Math + Memory Fusion Core not available, using fallback")
                return await self.generate_signals(market_data)
            
            enhanced_signals = []
            
            # Generate unified signal using Math + Memory Fusion Core
            unified_signal = self.math_system.generate_unified_signal(
                market_data, 
                self.profit_vector_history
            )
            
            # Convert unified signal to enhanced trading signal
            if unified_signal.signal != "HOLD":
                enhanced_signal = EnhancedTradingSignal(
                    symbol=market_data.get("symbol", "BTC/USDC"),
                    action=unified_signal.signal.lower(),
                    entry_price=market_data.get("price", 50000.0),
                    amount=self._calculate_position_size(unified_signal),
                    strategy_id="unified_math_fusion",
                    unified_signal=unified_signal,
                    profit_vectors=self.profit_vector_history[-5:],  # Last 5 vectors
                    mathematical_confidence=unified_signal.mathematical_confidence,
                    entropy_correction=unified_signal.entropy_correction,
                    vector_confidence=unified_signal.vector_confidence,
                    volatility=market_data.get("volatility", 0.0),
                    volume=market_data.get("volume", 0.0),
                    market_conditions=market_data,
                    signal_hash=self._generate_signal_hash(unified_signal)
                )
                enhanced_signals.append(enhanced_signal)
            
            # Generate signals from individual strategies with mathematical enhancement
            strategy_signals = await self._generate_enhanced_strategy_signals(market_data, unified_signal)
            enhanced_signals.extend(strategy_signals)
            
            # Store signals in history
            self._store_enhanced_signals(enhanced_signals)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error generating unified signals: {e}")
            return []

    async def _generate_enhanced_strategy_signals(self, market_data: Dict[str, Any], 
                                                 unified_signal: UnifiedSignal) -> List[EnhancedTradingSignal]:
        """Generate enhanced signals from individual strategies with mathematical fusion."""
        enhanced_signals = []
        
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # Generate base signals from strategy
                base_signals = await strategy.generate_signals(market_data)
                
                for base_signal in base_signals:
                    # Enhance signal with mathematical fusion
                    enhanced_signal = self._enhance_strategy_signal(
                        base_signal, strategy_name, unified_signal
                    )
                    
                    if enhanced_signal:
                        enhanced_signals.append(enhanced_signal)
                        
            except Exception as e:
                logger.error(f"Error generating enhanced signals for {strategy_name}: {e}")
        
        return enhanced_signals

    def _enhance_strategy_signal(self, base_signal: Dict[str, Any], strategy_name: str, 
                                unified_signal: UnifiedSignal) -> Optional[EnhancedTradingSignal]:
        """Enhance a strategy signal with mathematical fusion context."""
        try:
            # Extract base signal data
            action = base_signal.get("action", "hold")
            if action == "hold":
                return None
            
            # Calculate mathematical enhancement
            strategy_weight = self.strategy_weights.get(strategy_name, 1.0)
            enhanced_confidence = (
                base_signal.get("confidence", 0.5) * strategy_weight * 
                self.mathematical_confidence_weight +
                unified_signal.confidence * self.vector_confidence_weight
            )
            
            # Apply entropy correction
            entropy_correction = 1 - unified_signal.entropy_correction
            final_confidence = enhanced_confidence * entropy_correction
            
            # Create enhanced signal
            enhanced_signal = EnhancedTradingSignal(
                symbol=base_signal.get("symbol", "BTC/USDC"),
                action=action,
                entry_price=base_signal.get("entry_price", 50000.0),
                amount=base_signal.get("amount", 0.0),
                strategy_id=strategy_name,
                unified_signal=unified_signal,
                profit_vectors=self.profit_vector_history[-3:],  # Last 3 vectors
                mathematical_confidence=enhanced_confidence,
                entropy_correction=entropy_correction,
                vector_confidence=unified_signal.vector_confidence,
                volatility=base_signal.get("volatility", 0.0),
                volume=base_signal.get("volume", 0.0),
                market_conditions=base_signal.get("market_conditions", {}),
                signal_hash=self._generate_signal_hash(base_signal)
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error enhancing strategy signal: {e}")
            return None

    def _calculate_position_size(self, unified_signal: UnifiedSignal) -> float:
        """Calculate position size based on unified signal confidence."""
        try:
            base_amount = 1000.0  # Base position size
            confidence_multiplier = unified_signal.confidence
            vector_multiplier = unified_signal.vector_confidence
            
            # Apply profit vector insights
            if unified_signal.profit_weight > 0.5:
                profit_multiplier = 1.2
            else:
                profit_multiplier = 0.8
            
            final_amount = base_amount * confidence_multiplier * vector_multiplier * profit_multiplier
            
            return max(100.0, min(5000.0, final_amount))  # Clamp between 100 and 5000
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0

    def _generate_signal_hash(self, signal_data: Any) -> str:
        """Generate a unique hash for signal tracking."""
        try:
            import hashlib
            signal_str = str(signal_data) + str(time.time())
            return hashlib.md5(signal_str.encode()).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Error generating signal hash: {e}")
            return "unknown"

    def _store_enhanced_signals(self, signals: List[EnhancedTradingSignal]):
        """Store enhanced signals in history."""
        try:
            self.signal_history.extend(signals)
            
            # Keep history within limits
            if len(self.signal_history) > self.max_signal_history:
                excess = len(self.signal_history) - self.max_signal_history
                self.signal_history = self.signal_history[excess:]
            
            logger.debug(f"Stored {len(signals)} enhanced signals")
            
        except Exception as e:
            logger.error(f"Error storing enhanced signals: {e}")

    async def update_profit_vectors(self, trade_result: Dict[str, Any]):
        """Update profit vectors with new trade results."""
        try:
            if not MATH_FUSION_AVAILABLE:
                return
            
            # Generate new profit vector from trade result
            new_vector = self.profit_system.generate_profit_vector(
                entry_tick=int(time.time()),
                profit=trade_result.get("profit", 0.0),
                strategy_hash=trade_result.get("strategy_id", "unknown"),
                drawdown=trade_result.get("drawdown", 0.0),
                entropy_delta=trade_result.get("volatility", 0.1),
                exit_type=trade_result.get("exit_type", "unknown"),
                risk_profile=trade_result.get("risk_profile", "medium")
            )
            
            # Add to history
            self.profit_vector_history.append(new_vector)
            
            # Keep history within limits
            if len(self.profit_vector_history) > self.max_profit_history:
                excess = len(self.profit_vector_history) - self.max_profit_history
                self.profit_vector_history = self.profit_vector_history[excess:]
            
            logger.info(f"📊 Updated profit vectors with new trade result")
            
        except Exception as e:
            logger.error(f"Error updating profit vectors: {e}")

    def get_mathematical_insights(self) -> Dict[str, Any]:
        """Get mathematical insights from the fusion core."""
        try:
            if not MATH_FUSION_AVAILABLE:
                return {"error": "Math + Memory Fusion Core not available"}
            
            # Get profit vector insights
            profit_insights = self.math_system.bridge_profit_to_math(self.profit_vector_history)
            
            # Get signal history insights
            recent_signals = self.signal_history[-10:] if self.signal_history else []
            signal_confidence_avg = np.mean([s.mathematical_confidence for s in recent_signals]) if recent_signals else 0.0
            
            return {
                "profit_insights": profit_insights,
                "signal_confidence_avg": signal_confidence_avg,
                "total_signals": len(self.signal_history),
                "total_profit_vectors": len(self.profit_vector_history),
                "fusion_core_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Error getting mathematical insights: {e}")
            return {"error": str(e)}
    
    async def start(self) -> bool:
        """Start the strategy executor."""
        if not self.is_initialized:
            logger.error("Strategy Executor not initialized")
            return False
        
        try:
            logger.info("Starting Strategy Executor...")
            
            self.is_running = True
            
            # Start execution task
            self.execution_task = asyncio.create_task(self._execution_loop())
            
            logger.info("Strategy Executor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Strategy Executor: {e}")
            return False
    
    async def stop(self):
        """Stop the strategy executor."""
        if not self.is_running:
            return
        
        logger.info("Stopping Strategy Executor...")
        
        try:
            self.is_running = False
            
            # Cancel execution task
            if self.execution_task:
                self.execution_task.cancel()
                try:
                    await self.execution_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Strategy Executor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Strategy Executor: {e}")
    
    async def _execution_loop(self):
        """Main execution loop for strategies with Math + Memory Fusion Core."""
        logger.info("Enhanced strategy execution loop started")
        
        try:
            while self.is_running:
                # Generate market data
                market_data = await self._generate_market_data()
                
                # Generate unified signals using Math + Memory Fusion Core
                unified_signals = await self.generate_unified_signals(market_data)
                
                # Process enhanced strategies
                await self._process_enhanced_strategies(market_data, unified_signals)
                
                # Sleep for next iteration
                await asyncio.sleep(1)  # 1 second interval
                
        except asyncio.CancelledError:
            logger.info("Enhanced strategy execution loop cancelled")
        except Exception as e:
            logger.error(f"Error in enhanced strategy execution loop: {e}")

    async def _generate_market_data(self) -> Dict[str, Any]:
        """Generate or fetch market data for analysis."""
        try:
            # Simulate market data for testing
            # In production, this would fetch from real market data sources
            current_time = time.time()
            base_price = 50000.0
            price_variation = np.sin(current_time * 0.001) * 1000
            current_price = base_price + price_variation
            
            return {
                "symbol": "BTC/USDC",
                "price": current_price,
                "volume": 1000.0 + np.random.normal(0, 100),
                "volatility": 0.02 + np.random.normal(0, 0.005),
                "timestamp": current_time,
                "prices": [current_price - 100, current_price - 50, current_price],
                "volumes": [950, 975, 1000]
            }
        except Exception as e:
            logger.error(f"Error generating market data: {e}")
            return {
                "symbol": "BTC/USDC",
                "price": 50000.0,
                "volume": 1000.0,
                "volatility": 0.02,
                "timestamp": time.time(),
                "prices": [50000.0],
                "volumes": [1000.0]
            }

    async def _process_enhanced_strategies(self, market_data: Dict[str, Any], 
                                         unified_signals: List[EnhancedTradingSignal]):
        """Process enhanced strategies with mathematical fusion."""
        try:
            # Process unified signals
            for signal in unified_signals:
                if signal.confidence > self.min_unified_confidence:
                    logger.info(f"🔗 Processing unified signal: {signal.action} {signal.symbol} "
                              f"(confidence: {signal.confidence:.3f})")
                    
                    # Simulate trade execution
                    trade_result = await self._simulate_trade_execution(signal)
                    
                    # Update profit vectors with trade result
                    await self.update_profit_vectors(trade_result)
            
            # Process individual strategies
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Check if strategy is still valid
                    if not hasattr(strategy, 'is_initialized') or not strategy.is_initialized:
                        logger.warning(f"Strategy {strategy_name} not initialized, skipping")
                        continue
                    
                    # Process strategy with mathematical context
                    logger.debug(f"Processing enhanced strategy: {strategy_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing enhanced strategy {strategy_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in enhanced strategy processing: {e}")

    async def _simulate_trade_execution(self, signal: EnhancedTradingSignal) -> Dict[str, Any]:
        """Simulate trade execution for testing purposes."""
        try:
            # Simulate trade result
            profit = np.random.normal(0.02, 0.01)  # 2% average profit with 1% std
            drawdown = abs(np.random.normal(0.01, 0.005))  # 1% average drawdown
            
            trade_result = {
                "symbol": signal.symbol,
                "action": signal.action,
                "entry_price": signal.entry_price,
                "amount": signal.amount,
                "profit": profit,
                "drawdown": drawdown,
                "volatility": signal.volatility,
                "strategy_id": signal.strategy_id,
                "exit_type": "simulated",
                "risk_profile": "medium",
                "timestamp": time.time()
            }
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error simulating trade execution: {e}")
            return {"error": str(e)}

    async def _process_strategies(self):
        """Process all active strategies (legacy method for compatibility)."""
        for strategy_name, strategy in self.active_strategies.items():
            try:
                # Check if strategy is still valid
                if not hasattr(strategy, 'is_initialized') or not strategy.is_initialized:
                    logger.warning(f"Strategy {strategy_name} not initialized, skipping")
                    continue
                
                # Process strategy (this would typically involve getting market data)
                # For now, we'll just log that we're processing
                logger.debug(f"Processing strategy: {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")

    def add_strategy(self, strategy_name: str, strategy: TradingStrategy, weight: float = 1.0) -> bool:
        """Add a strategy to the executor."""
        try:
            if not hasattr(strategy, 'is_initialized') or not strategy.is_initialized:
                logger.error(f"Strategy {strategy_name} is not initialized")
                return False
            
            self.active_strategies[strategy_name] = strategy
            self.strategy_weights[strategy_name] = weight
            
            logger.info(f"Added strategy: {strategy_name} with weight {weight}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add strategy {strategy_name}: {e}")
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from the executor."""
        try:
            if strategy_name in self.active_strategies:
                del self.active_strategies[strategy_name]
                if strategy_name in self.strategy_weights:
                    del self.strategy_weights[strategy_name]
                
                logger.info(f"Removed strategy: {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove strategy {strategy_name}: {e}")
            return False
    
    def set_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """Set the weight for a strategy."""
        try:
            if strategy_name in self.active_strategies:
                self.strategy_weights[strategy_name] = weight
                logger.info(f"Set weight for {strategy_name}: {weight}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set weight for strategy {strategy_name}: {e}")
            return False
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from all active strategies."""
        try:
            all_signals = []
            
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Generate signals from this strategy
                    strategy_signals = await strategy.generate_signals(analysis)
                    
                    # Apply strategy weight
                    weight = self.strategy_weights.get(strategy_name, 1.0)
                    for signal in strategy_signals:
                        signal['strategy'] = strategy_name
                        signal['weight'] = weight
                        signal['confidence'] = signal.get('confidence', 0.5) * weight
                        all_signals.append(signal)
                    
                except Exception as e:
                    logger.error(f"Error generating signals from strategy {strategy_name}: {e}")
            
            # Combine and rank signals
            combined_signals = await self._combine_signals(all_signals)
            
            # Store in history
            self._store_signals(combined_signals)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _combine_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and rank signals from multiple strategies."""
        try:
            if not signals:
                return []
            
            # Group signals by symbol and type
            signal_groups = {}
            for signal in signals:
                key = (signal.get('symbol', 'UNKNOWN'), signal.get('type', 'UNKNOWN'))
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append(signal)
            
            # Combine signals for each group
            combined_signals = []
            for (symbol, signal_type), group_signals in signal_groups.items():
                if len(group_signals) == 1:
                    # Single signal, use as is
                    combined_signals.append(group_signals[0])
                else:
                    # Multiple signals, combine them
                    combined_signal = await self._combine_signal_group(group_signals)
                    combined_signals.append(combined_signal)
            
            # Sort by confidence
            combined_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return signals
    
    async def _combine_signal_group(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine a group of signals for the same symbol and type."""
        try:
            if not signals:
                return {}
            
            # Weighted average of quantities and confidences
            total_weight = sum(signal.get('weight', 1.0) for signal in signals)
            weighted_quantity = sum(
                signal.get('quantity', 0) * signal.get('weight', 1.0) 
                for signal in signals
            ) / total_weight if total_weight > 0 else 0
            
            weighted_confidence = sum(
                signal.get('confidence', 0) * signal.get('weight', 1.0) 
                for signal in signals
            ) / total_weight if total_weight > 0 else 0
            
            # Use the first signal as base and update with combined values
            combined_signal = signals[0].copy()
            combined_signal['quantity'] = weighted_quantity
            combined_signal['confidence'] = weighted_confidence
            combined_signal['strategies'] = [s.get('strategy', 'unknown') for s in signals]
            combined_signal['combined_from'] = len(signals)
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error combining signal group: {e}")
            return signals[0] if signals else {}
    
    def _store_signals(self, signals: List[Dict[str, Any]]):
        """Store signals in history."""
        try:
            timestamp = datetime.now()
            
            for signal in signals:
                signal_record = {
                    'timestamp': timestamp,
                    'signal': signal.copy()
                }
                
                self.signal_history.append(signal_record)
            
            # Trim history if too long
            if len(self.signal_history) > self.max_signal_history:
                self.signal_history = self.signal_history[-self.max_signal_history:]
                
        except Exception as e:
            logger.error(f"Error storing signals: {e}")
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get signal history."""
        try:
            history = self.signal_history.copy()
            if limit:
                history = history[-limit:]
            return history
            
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []
    
    def get_active_strategies(self) -> Dict[str, TradingStrategy]:
        """Get all active strategies."""
        return self.active_strategies.copy()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights."""
        return self.strategy_weights.copy()
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Get executor status."""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "active_strategies": list(self.active_strategies.keys()),
            "strategy_weights": self.strategy_weights.copy(),
            "signal_history_count": len(self.signal_history),
            "execution_task_running": self.execution_task is not None and not self.execution_task.done()
        }
    
    async def test_strategy(self, strategy_name: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific strategy with test data."""
        try:
            if strategy_name not in self.active_strategies:
                return {"error": f"Strategy {strategy_name} not found"}
            
            strategy = self.active_strategies[strategy_name]
            
            # Analyze test data
            analysis = await strategy.generate_signals(test_data) # Changed from analyze to generate_signals
            
            # Generate signals
            signals = await strategy.generate_signals(analysis)
            
            return {
                "strategy_name": strategy_name,
                "analysis": analysis,
                "signals": signals,
                "signal_count": len(signals)
            }
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name}: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("Cleaning up Strategy Executor...")
            
            # Stop executor
            await self.stop()
            
            # Clean up strategies
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'cleanup'):
                    await strategy.cleanup()
            
            self.active_strategies.clear()
            self.strategy_weights.clear()
            self.signal_history.clear()
            
            logger.info("Strategy Executor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Strategy Executor cleanup: {e}") 