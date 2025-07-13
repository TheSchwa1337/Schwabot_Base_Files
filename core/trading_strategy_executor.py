#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Strategy Executor Module
=================================
Provides trading strategy executor functionality for the Schwabot trading system.

Mathematical Core:
f(s_i) = {
    Aggressive Market Buy,  if δP > θ
    Passive Maker Sell,     if δP < -θ
    Hold,                   else
}

This module receives hash or signal triggers and dynamically selects execution paths.
It integrates strategy_loader.py, strategy_logic.py, and strategy_router.py.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json
import hashlib

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
    from core.unified_mathematical_bridge import UnifiedMathematicalBridge
    from core.unified_mathematical_integration_methods import UnifiedMathematicalIntegrationMethods
    from core.unified_mathematical_performance_monitor import UnifiedMathematicalPerformanceMonitor
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Mathematical infrastructure not available - using fallback")


class ExecutionPath(Enum):
    """Execution path types."""
    AGGRESSIVE_MARKET_BUY = "aggressive_market_buy"
    PASSIVE_MAKER_SELL = "passive_maker_sell"
    HOLD = "hold"
    SCALPING = "scalping"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"


class StrategyType(Enum):
    """Strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    QUANTUM = "quantum"
    PHANTOM = "phantom"
    HYBRID = "hybrid"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class StrategySignal:
    """Strategy signal with mathematical properties."""
    signal_hash: str
    strategy_type: StrategyType
    execution_path: ExecutionPath
    symbol: str
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    price_delta: float  # δP
    threshold: float  # θ
    timestamp: float = field(default_factory=time.time)
    mathematical_signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionDecision:
    """Execution decision with mathematical analysis."""
    signal_hash: str
    selected_path: ExecutionPath
    confidence: float
    reasoning: str
    mathematical_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_type: StrategyType
    total_signals: int
    successful_executions: int
    total_pnl: float
    win_rate: float
    average_confidence: float
    mathematical_signature: str = ""


@dataclass
class TradingStrategyExecutorConfig:
    """Configuration for trading strategy executor."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False
    max_concurrent_strategies: int = 10
    execution_threshold: float = 0.7  # Minimum confidence for execution
    mathematical_analysis_enabled: bool = True
    performance_tracking_enabled: bool = True
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.3,
        'mean_reversion': 0.25,
        'scalping': 0.2,
        'arbitrage': 0.15,
        'grid': 0.1
    })


class TradingStrategyExecutor:
    """
    Trading Strategy Executor System
    
    Implements dynamic execution path selection:
    f(s_i) = {
        Aggressive Market Buy,  if δP > θ
        Passive Maker Sell,     if δP < -θ
        Hold,                   else
    }
    
    Receives hash or signal triggers and dynamically selects execution paths.
    Integrates strategy_loader.py, strategy_logic.py, and strategy_router.py.
    """
    
    def __init__(self, config: Optional[TradingStrategyExecutorConfig] = None):
        """Initialize the trading strategy executor system."""
        self.config = config or TradingStrategyExecutorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.active_strategies: Dict[str, StrategyType] = {}
        self.strategy_signals: Dict[str, StrategySignal] = {}
        self.execution_decisions: List[ExecutionDecision] = []
        self.strategy_performance: Dict[StrategyType, StrategyPerformance] = {}
        
        # Signal processing
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        self.decision_queue: asyncio.Queue = asyncio.Queue()
        
        # Mathematical infrastructure
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_bridge = UnifiedMathematicalBridge()
            self.math_integration = UnifiedMathematicalIntegrationMethods()
            self.math_monitor = UnifiedMathematicalPerformanceMonitor()
        else:
            self.math_bridge = None
            self.math_integration = None
            self.math_monitor = None
        
        # Performance tracking
        self.performance_metrics = {
            'signals_processed': 0,
            'decisions_made': 0,
            'executions_triggered': 0,
            'average_processing_time': 0.0,
            'strategy_accuracy': 0.0
        }
        
        # System state
        self.initialized = False
        self.active = False
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the trading strategy executor system."""
        try:
            self.logger.info("Initializing Trading Strategy Executor System")
            
            # Initialize strategy performance tracking
            for strategy_type in StrategyType:
                self.strategy_performance[strategy_type] = StrategyPerformance(
                    strategy_type=strategy_type,
                    total_signals=0,
                    successful_executions=0,
                    total_pnl=0.0,
                    win_rate=0.0,
                    average_confidence=0.0
                )
            
            self.initialized = True
            self.logger.info("✅ Trading Strategy Executor System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Trading Strategy Executor System: {e}")
            self.initialized = False
    
    async def start_executor(self) -> bool:
        """Start the strategy executor."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            
            # Start processing tasks
            asyncio.create_task(self._process_signal_queue())
            asyncio.create_task(self._process_decision_queue())
            
            self.logger.info("✅ Trading Strategy Executor started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting strategy executor: {e}")
            return False
    
    async def stop_executor(self) -> bool:
        """Stop the strategy executor."""
        try:
            self.active = False
            self.logger.info("✅ Trading Strategy Executor stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping strategy executor: {e}")
            return False
    
    async def submit_strategy_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Submit a strategy signal for processing."""
        if not self.active:
            self.logger.error("Strategy executor not active")
            return False
        
        try:
            # Create strategy signal
            signal = self._create_strategy_signal(signal_data)
            
            # Validate signal
            if not self._validate_signal(signal):
                self.logger.error(f"Invalid signal: {signal}")
                return False
            
            # Add mathematical analysis
            if self.config.mathematical_analysis_enabled:
                await self._analyze_signal_mathematically(signal)
            
            # Store signal
            self.strategy_signals[signal.signal_hash] = signal
            
            # Queue for processing
            await self.signal_queue.put(signal)
            
            self.logger.info(f"✅ Strategy signal submitted: {signal.strategy_type.value} for {signal.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error submitting strategy signal: {e}")
            return False
    
    def _create_strategy_signal(self, signal_data: Dict[str, Any]) -> StrategySignal:
        """Create a strategy signal from input data."""
        try:
            # Generate signal hash
            signal_content = f"{signal_data.get('strategy_type', '')}{signal_data.get('symbol', '')}{time.time()}"
            signal_hash = hashlib.sha256(signal_content.encode()).hexdigest()[:16]
            
            # Determine strategy type
            strategy_type_str = signal_data.get('strategy_type', 'momentum').lower()
            strategy_type = self._map_strategy_type(strategy_type_str)
            
            # Calculate price delta and threshold
            current_price = signal_data.get('current_price', 0.0)
            reference_price = signal_data.get('reference_price', current_price)
            price_delta = current_price - reference_price
            threshold = signal_data.get('threshold', 0.01)  # 1% default
            
            # Determine execution path based on mathematical model
            execution_path = self._determine_execution_path(price_delta, threshold)
            
            # Determine signal strength
            strength = self._determine_signal_strength(abs(price_delta), threshold)
            
            # Calculate confidence
            confidence = signal_data.get('confidence', 0.5)
            
            return StrategySignal(
                signal_hash=signal_hash,
                strategy_type=strategy_type,
                execution_path=execution_path,
                symbol=signal_data.get('symbol', ''),
                strength=strength,
                confidence=confidence,
                price_delta=price_delta,
                threshold=threshold,
                metadata=signal_data.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating strategy signal: {e}")
            # Return default signal
            return StrategySignal(
                signal_hash="default",
                strategy_type=StrategyType.MOMENTUM,
                execution_path=ExecutionPath.HOLD,
                symbol="",
                strength=SignalStrength.WEAK,
                confidence=0.0,
                price_delta=0.0,
                threshold=0.01
            )
    
    def _map_strategy_type(self, strategy_type_str: str) -> StrategyType:
        """Map string to strategy type."""
        mapping = {
            'momentum': StrategyType.MOMENTUM,
            'mean_reversion': StrategyType.MEAN_REVERSION,
            'scalping': StrategyType.SCALPING,
            'arbitrage': StrategyType.ARBITRAGE,
            'grid': StrategyType.GRID,
            'quantum': StrategyType.QUANTUM,
            'phantom': StrategyType.PHANTOM,
            'hybrid': StrategyType.HYBRID
        }
        return mapping.get(strategy_type_str, StrategyType.MOMENTUM)
    
    def _determine_execution_path(self, price_delta: float, threshold: float) -> ExecutionPath:
        """Determine execution path based on mathematical model."""
        try:
            # Mathematical model: f(s_i) = {Aggressive Market Buy if δP > θ, Passive Maker Sell if δP < -θ, Hold else}
            if price_delta > threshold:
                return ExecutionPath.AGGRESSIVE_MARKET_BUY
            elif price_delta < -threshold:
                return ExecutionPath.PASSIVE_MAKER_SELL
            else:
                return ExecutionPath.HOLD
                
        except Exception as e:
            self.logger.error(f"❌ Error determining execution path: {e}")
            return ExecutionPath.HOLD
    
    def _determine_signal_strength(self, abs_delta: float, threshold: float) -> SignalStrength:
        """Determine signal strength based on price delta."""
        try:
            ratio = abs_delta / threshold if threshold > 0 else 0
            
            if ratio >= 3.0:
                return SignalStrength.EXTREME
            elif ratio >= 2.0:
                return SignalStrength.STRONG
            elif ratio >= 1.5:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
                
        except Exception as e:
            self.logger.error(f"❌ Error determining signal strength: {e}")
            return SignalStrength.WEAK
    
    def _validate_signal(self, signal: StrategySignal) -> bool:
        """Validate strategy signal."""
        try:
            # Check basic requirements
            if not signal.symbol or signal.confidence < 0.0 or signal.confidence > 1.0:
                return False
            
            # Check strategy type
            if signal.strategy_type not in StrategyType:
                return False
            
            # Check execution path
            if signal.execution_path not in ExecutionPath:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating signal: {e}")
            return False
    
    async def _analyze_signal_mathematically(self, signal: StrategySignal) -> None:
        """Perform mathematical analysis on signal."""
        try:
            if not self.math_bridge:
                return
            
            # Prepare signal data for mathematical analysis
            signal_data = {
                'signal_hash': signal.signal_hash,
                'strategy_type': signal.strategy_type.value,
                'execution_path': signal.execution_path.value,
                'symbol': signal.symbol,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'price_delta': signal.price_delta,
                'threshold': signal.threshold,
                'timestamp': signal.timestamp,
                'metadata': signal.metadata
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                signal_data, {}
            )
            
            # Update signal with mathematical analysis
            signal.mathematical_signature = result.mathematical_signature
            signal.metadata['mathematical_analysis'] = {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing signal mathematically: {e}")
    
    async def _process_signal_queue(self) -> None:
        """Process signals from the queue."""
        try:
            while self.active:
                try:
                    # Get signal from queue
                    signal = await asyncio.wait_for(
                        self.signal_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process signal
                    await self._process_signal(signal)
                    
                    # Mark task as done
                    self.signal_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing signal: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in signal processing loop: {e}")
    
    async def _process_signal(self, signal: StrategySignal) -> None:
        """Process a strategy signal."""
        try:
            start_time = time.time()
            
            # Update performance metrics
            self.performance_metrics['signals_processed'] += 1
            
            # Make execution decision
            decision = await self._make_execution_decision(signal)
            
            # Store decision
            self.execution_decisions.append(decision)
            
            # Update strategy performance
            self._update_strategy_performance(signal, decision)
            
            # Queue decision for execution
            await self.decision_queue.put(decision)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['decisions_made'] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics['average_processing_time']
            total_signals = self.performance_metrics['signals_processed']
            self.performance_metrics['average_processing_time'] = (
                (current_avg * (total_signals - 1) + processing_time) / total_signals
            )
            
            self.logger.info(f"✅ Signal processed: {signal.signal_hash} -> {decision.selected_path.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal: {e}")
    
    async def _make_execution_decision(self, signal: StrategySignal) -> ExecutionDecision:
        """Make execution decision based on signal."""
        try:
            # Apply strategy weights
            strategy_weight = self.config.strategy_weights.get(signal.strategy_type.value, 0.1)
            
            # Calculate weighted confidence
            weighted_confidence = signal.confidence * strategy_weight
            
            # Determine if execution should be triggered
            should_execute = weighted_confidence >= self.config.execution_threshold
            
            # Select execution path
            if should_execute:
                selected_path = signal.execution_path
                reasoning = f"Signal confidence {signal.confidence:.3f} * strategy weight {strategy_weight:.3f} = {weighted_confidence:.3f} >= threshold {self.config.execution_threshold}"
            else:
                selected_path = ExecutionPath.HOLD
                reasoning = f"Signal confidence {signal.confidence:.3f} * strategy weight {strategy_weight:.3f} = {weighted_confidence:.3f} < threshold {self.config.execution_threshold}"
            
            # Generate execution parameters
            execution_parameters = self._generate_execution_parameters(signal, selected_path)
            
            # Perform mathematical analysis on decision
            mathematical_analysis = await self._analyze_decision_mathematically(signal, selected_path, weighted_confidence)
            
            return ExecutionDecision(
                signal_hash=signal.signal_hash,
                selected_path=selected_path,
                confidence=weighted_confidence,
                reasoning=reasoning,
                mathematical_analysis=mathematical_analysis,
                execution_parameters=execution_parameters
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error making execution decision: {e}")
            return ExecutionDecision(
                signal_hash=signal.signal_hash,
                selected_path=ExecutionPath.HOLD,
                confidence=0.0,
                reasoning=f"Error in decision making: {e}",
                execution_parameters={}
            )
    
    def _generate_execution_parameters(self, signal: StrategySignal, selected_path: ExecutionPath) -> Dict[str, Any]:
        """Generate execution parameters based on signal and path."""
        try:
            base_params = {
                'symbol': signal.symbol,
                'strategy_type': signal.strategy_type.value,
                'signal_strength': signal.strength.value,
                'price_delta': signal.price_delta,
                'threshold': signal.threshold
            }
            
            # Add path-specific parameters
            if selected_path == ExecutionPath.AGGRESSIVE_MARKET_BUY:
                base_params.update({
                    'order_type': 'market',
                    'side': 'buy',
                    'urgency': 'high',
                    'slippage_tolerance': 0.002  # 0.2%
                })
            elif selected_path == ExecutionPath.PASSIVE_MAKER_SELL:
                base_params.update({
                    'order_type': 'limit',
                    'side': 'sell',
                    'urgency': 'low',
                    'price_offset': -0.001  # 0.1% below market
                })
            elif selected_path == ExecutionPath.SCALPING:
                base_params.update({
                    'order_type': 'market',
                    'side': 'buy' if signal.price_delta > 0 else 'sell',
                    'urgency': 'high',
                    'position_size': 'small',
                    'timeout': 30  # seconds
                })
            elif selected_path == ExecutionPath.MEAN_REVERSION:
                base_params.update({
                    'order_type': 'limit',
                    'side': 'sell' if signal.price_delta > 0 else 'buy',
                    'urgency': 'medium',
                    'mean_reversion_strength': abs(signal.price_delta) / signal.threshold
                })
            else:  # HOLD
                base_params.update({
                    'action': 'hold',
                    'reason': 'Below execution threshold'
                })
            
            return base_params
            
        except Exception as e:
            self.logger.error(f"❌ Error generating execution parameters: {e}")
            return {'error': str(e)}
    
    async def _analyze_decision_mathematically(self, signal: StrategySignal, selected_path: ExecutionPath, confidence: float) -> Dict[str, Any]:
        """Perform mathematical analysis on decision."""
        try:
            if not self.math_bridge:
                return {}
            
            # Prepare decision data for mathematical analysis
            decision_data = {
                'signal_hash': signal.signal_hash,
                'selected_path': selected_path.value,
                'confidence': confidence,
                'strategy_type': signal.strategy_type.value,
                'price_delta': signal.price_delta,
                'threshold': signal.threshold,
                'mathematical_signature': signal.mathematical_signature
            }
            
            # Perform mathematical integration
            result = self.math_bridge.integrate_all_mathematical_systems(
                decision_data, {}
            )
            
            return {
                'confidence': result.overall_confidence,
                'connections': len(result.connections),
                'performance_metrics': result.performance_metrics,
                'mathematical_signature': result.mathematical_signature
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing decision mathematically: {e}")
            return {}
    
    def _update_strategy_performance(self, signal: StrategySignal, decision: ExecutionDecision) -> None:
        """Update strategy performance metrics."""
        try:
            if not self.config.performance_tracking_enabled:
                return
            
            performance = self.strategy_performance[signal.strategy_type]
            
            # Update metrics
            performance.total_signals += 1
            performance.average_confidence = (
                (performance.average_confidence * (performance.total_signals - 1) + decision.confidence) / 
                performance.total_signals
            )
            
            # Update mathematical signature
            performance.mathematical_signature = signal.mathematical_signature
            
            # Note: PnL and win rate would be updated after execution results are received
            
        except Exception as e:
            self.logger.error(f"❌ Error updating strategy performance: {e}")
    
    async def _process_decision_queue(self) -> None:
        """Process decisions from the queue."""
        try:
            while self.active:
                try:
                    # Get decision from queue
                    decision = await asyncio.wait_for(
                        self.decision_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process decision (send to execution engine)
                    await self._execute_decision(decision)
                    
                    # Mark task as done
                    self.decision_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ Error processing decision: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in decision processing loop: {e}")
    
    async def _execute_decision(self, decision: ExecutionDecision) -> None:
        """Execute a decision (send to execution engine)."""
        try:
            # Update performance metrics
            self.performance_metrics['executions_triggered'] += 1
            
            # Log execution
            self.logger.info(f"🚀 Executing decision: {decision.signal_hash} -> {decision.selected_path.value}")
            
            # Here you would send the decision to the execution engine
            # For now, we'll just log it
            execution_data = {
                'decision_id': decision.signal_hash,
                'execution_path': decision.selected_path.value,
                'confidence': decision.confidence,
                'parameters': decision.execution_parameters,
                'timestamp': decision.timestamp
            }
            
            self.logger.info(f"Execution data: {json.dumps(execution_data, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing decision: {e}")
    
    def get_strategy_performance(self, strategy_type: Optional[StrategyType] = None) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        try:
            if strategy_type:
                performance = self.strategy_performance[strategy_type]
                return {
                    'strategy_type': performance.strategy_type.value,
                    'total_signals': performance.total_signals,
                    'successful_executions': performance.successful_executions,
                    'total_pnl': performance.total_pnl,
                    'win_rate': performance.win_rate,
                    'average_confidence': performance.average_confidence,
                    'mathematical_signature': performance.mathematical_signature
                }
            else:
                return {
                    strategy_type.value: {
                        'total_signals': perf.total_signals,
                        'successful_executions': perf.successful_executions,
                        'total_pnl': perf.total_pnl,
                        'win_rate': perf.win_rate,
                        'average_confidence': perf.average_confidence
                    }
                    for strategy_type, perf in self.strategy_performance.items()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting strategy performance: {e}")
            return {}
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent execution decisions."""
        try:
            recent_decisions = self.execution_decisions[-limit:]
            return [
                {
                    'signal_hash': decision.signal_hash,
                    'selected_path': decision.selected_path.value,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'timestamp': decision.timestamp,
                    'execution_parameters': decision.execution_parameters
                }
                for decision in recent_decisions
            ]
        except Exception as e:
            self.logger.error(f"❌ Error getting recent decisions: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return self.performance_metrics.copy()
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info("✅ Trading Strategy Executor System activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating Trading Strategy Executor System: {e}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info("✅ Trading Strategy Executor System deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating Trading Strategy Executor System: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'signals_queued': self.signal_queue.qsize(),
            'decisions_queued': self.decision_queue.qsize(),
            'active_strategies': len(self.active_strategies),
            'total_signals': len(self.strategy_signals),
            'total_decisions': len(self.execution_decisions),
            'performance_metrics': self.performance_metrics,
            'config': {
                'enabled': self.config.enabled,
                'max_concurrent_strategies': self.config.max_concurrent_strategies,
                'execution_threshold': self.config.execution_threshold,
                'mathematical_analysis_enabled': self.config.mathematical_analysis_enabled,
                'performance_tracking_enabled': self.config.performance_tracking_enabled
            }
        }


def create_trading_strategy_executor(config: Optional[TradingStrategyExecutorConfig] = None) -> TradingStrategyExecutor:
    """Factory function to create TradingStrategyExecutor instance."""
    return TradingStrategyExecutor(config)


async def main():
    """Main function for testing."""
    # Create configuration
    config = TradingStrategyExecutorConfig(
        enabled=True,
        debug=True,
        max_concurrent_strategies=5,
        execution_threshold=0.7,
        mathematical_analysis_enabled=True,
        performance_tracking_enabled=True
    )
    
    # Create executor
    executor = create_trading_strategy_executor(config)
    
    # Activate system
    executor.activate()
    
    # Start executor
    await executor.start_executor()
    
    # Submit test signals
    test_signals = [
        {
            'strategy_type': 'momentum',
            'symbol': 'BTCUSDT',
            'current_price': 50000.0,
            'reference_price': 49500.0,
            'confidence': 0.85,
            'threshold': 0.01
        },
        {
            'strategy_type': 'mean_reversion',
            'symbol': 'ETHUSDT',
            'current_price': 3000.0,
            'reference_price': 3100.0,
            'confidence': 0.75,
            'threshold': 0.02
        },
        {
            'strategy_type': 'scalping',
            'symbol': 'BTCUSDT',
            'current_price': 50100.0,
            'reference_price': 50050.0,
            'confidence': 0.6,
            'threshold': 0.005
        }
    ]
    
    # Submit signals
    for signal_data in test_signals:
        await executor.submit_strategy_signal(signal_data)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get status
    status = executor.get_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Get strategy performance
    performance = executor.get_strategy_performance()
    print(f"Strategy Performance: {json.dumps(performance, indent=2)}")
    
    # Get recent decisions
    decisions = executor.get_recent_decisions()
    print(f"Recent Decisions: {json.dumps(decisions, indent=2)}")
    
    # Stop executor
    await executor.stop_executor()
    
    # Deactivate system
    executor.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
