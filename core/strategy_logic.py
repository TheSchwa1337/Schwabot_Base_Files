#!/usr/bin/env python3
"""
Strategy Logic - Core Trading Strategy Implementation
===================================================

Core strategy implementation logic for the Schwabot mathematical trading framework.
Provides strategy execution, signal processing, and decision-making capabilities.

Key Features:
- Strategy execution engine
- Signal processing and analysis
- Decision-making algorithms
- Risk-aware position sizing
- Performance tracking and optimization

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import time
import math
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enumeration"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    QUANTUM_ENHANCED = "quantum_enhanced"


class SignalType(Enum):
    """Signal type enumeration"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    HEDGE = "hedge"


class SignalStrength(Enum):
    """Signal strength enumeration"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """Trading signal container"""
    
    signal_type: SignalType
    strength: SignalStrength
    asset: str
    price: float
    volume: float
    confidence: float  # 0.0 to 1.0
    timestamp: float
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    
    strategy_type: StrategyType
    name: str
    enabled: bool = True
    max_position_size: float = 0.1
    risk_tolerance: float = 0.05
    lookback_period: int = 100
    min_signal_confidence: float = 0.6
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    last_updated: float = field(default_factory=time.time)


class StrategyLogic:
    """Core strategy logic implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize strategy logic"""
        self.version = "1.0.0"
        self.config = config or self._default_config()
        
        # Strategy registry
        self.strategies: Dict[str, StrategyConfig] = {}
        self.performance: Dict[str, StrategyPerformance] = {}
        
        # Signal processing
        self.signal_history: List[TradingSignal] = []
        self.max_signals_history = self.config.get('max_signals_history', 1000)
        
        # Performance tracking
        self.total_signals_generated = 0
        self.total_signals_executed = 0
        self.last_signal_time = 0.0
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info(f"StrategyLogic v{self.version} initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'max_signals_history': 1000,
            'default_risk_tolerance': 0.05,
            'default_max_position_size': 0.1,
            'min_signal_confidence': 0.6,
            'enable_performance_tracking': True,
            'enable_signal_filtering': True,
            'signal_cooldown_period': 1.0  # seconds
        }
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default trading strategies"""
        default_strategies = [
            StrategyConfig(
                strategy_type=StrategyType.MEAN_REVERSION,
                name="mean_reversion_v1",
                enabled=True,
                max_position_size=0.1,
                risk_tolerance=0.05,
                lookback_period=100,
                min_signal_confidence=0.6,
                parameters={
                    'z_score_threshold': 2.0,
                    'mean_reversion_strength': 0.8,
                    'volatility_lookback': 20
                }
            ),
            StrategyConfig(
                strategy_type=StrategyType.MOMENTUM,
                name="momentum_v1",
                enabled=True,
                max_position_size=0.15,
                risk_tolerance=0.08,
                lookback_period=50,
                min_signal_confidence=0.7,
                parameters={
                    'momentum_threshold': 0.02,
                    'trend_strength': 0.6,
                    'volume_weight': 0.3
                }
            ),
            StrategyConfig(
                strategy_type=StrategyType.STATISTICAL_ARBITRAGE,
                name="stat_arb_v1",
                enabled=True,
                max_position_size=0.2,
                risk_tolerance=0.03,
                lookback_period=200,
                min_signal_confidence=0.8,
                parameters={
                    'correlation_threshold': 0.8,
                    'cointegration_threshold': 0.05,
                    'pair_trading_enabled': True
                }
            )
        ]
        
        for strategy in default_strategies:
            self.register_strategy(strategy)
    
    def register_strategy(self, strategy_config: StrategyConfig) -> bool:
        """Register a new trading strategy"""
        try:
            self.strategies[strategy_config.name] = strategy_config
            
            # Initialize performance tracking
            self.performance[strategy_config.name] = StrategyPerformance(
                strategy_name=strategy_config.name
            )
            
            logger.info(f"Registered strategy: {strategy_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_config.name}: {e}")
            return False
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Process market data and generate trading signals"""
        try:
            signals = []
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_signal_time < self.config.get('signal_cooldown_period', 1.0):
                return signals
            
            # Process each enabled strategy
            for strategy_name, strategy_config in self.strategies.items():
                if not strategy_config.enabled:
                    continue
                
                # Generate signals based on strategy type
                strategy_signals = self._generate_strategy_signals(strategy_config, market_data)
                signals.extend(strategy_signals)
            
            # Filter and rank signals
            filtered_signals = self._filter_signals(signals)
            
            # Update signal history
            self.signal_history.extend(filtered_signals)
            self.total_signals_generated += len(filtered_signals)
            self.last_signal_time = current_time
            
            # Trim signal history if needed
            if len(self.signal_history) > self.max_signals_history:
                self.signal_history = self.signal_history[-self.max_signals_history:]
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return []
    
    def _generate_strategy_signals(self, strategy_config: StrategyConfig, 
                                 market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals for a specific strategy"""
        try:
            signals = []
            
            if strategy_config.strategy_type == StrategyType.MEAN_REVERSION:
                signals = self._mean_reversion_signals(strategy_config, market_data)
            elif strategy_config.strategy_type == StrategyType.MOMENTUM:
                signals = self._momentum_signals(strategy_config, market_data)
            elif strategy_config.strategy_type == StrategyType.STATISTICAL_ARBITRAGE:
                signals = self._statistical_arbitrage_signals(strategy_config, market_data)
            elif strategy_config.strategy_type == StrategyType.MACHINE_LEARNING:
                signals = self._ml_signals(strategy_config, market_data)
            elif strategy_config.strategy_type == StrategyType.QUANTUM_ENHANCED:
                signals = self._quantum_enhanced_signals(strategy_config, market_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {strategy_config.name}: {e}")
            return []
    
    def _mean_reversion_signals(self, strategy_config: StrategyConfig, 
                              market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate mean reversion signals"""
        try:
            signals = []
            
            # Extract price data
            prices = market_data.get('prices', [])
            if len(prices) < strategy_config.lookback_period:
                return signals
            
            prices = np.array(prices[-strategy_config.lookback_period:])
            
            # Calculate z-score
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price == 0:
                return signals
            
            current_price = prices[-1]
            z_score = (current_price - mean_price) / std_price
            
            # Get parameters
            z_threshold = strategy_config.parameters.get('z_score_threshold', 2.0)
            strength = strategy_config.parameters.get('mean_reversion_strength', 0.8)
            
            # Generate signal based on z-score
            if z_score > z_threshold:
                # Price is high, expect reversion down
                signal_type = SignalType.SELL
                confidence = min(abs(z_score) / z_threshold * strength, 1.0)
            elif z_score < -z_threshold:
                # Price is low, expect reversion up
                signal_type = SignalType.BUY
                confidence = min(abs(z_score) / z_threshold * strength, 1.0)
            else:
                return signals
            
            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                strength=SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE,
                asset=market_data.get('asset', 'UNKNOWN'),
                price=current_price,
                volume=market_data.get('volume', 0.0),
                confidence=confidence,
                timestamp=time.time(),
                strategy_name=strategy_config.name,
                metadata={
                    'z_score': z_score,
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'strategy_type': 'mean_reversion'
                }
            )
            
            signals.append(signal)
            return signals
            
        except Exception as e:
            logger.error(f"Error in mean reversion signals: {e}")
            return []
    
    def _momentum_signals(self, strategy_config: StrategyConfig, 
                         market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate momentum signals"""
        try:
            signals = []
            
            # Extract price data
            prices = market_data.get('prices', [])
            if len(prices) < strategy_config.lookback_period:
                return signals
            
            prices = np.array(prices[-strategy_config.lookback_period:])
            
            # Calculate momentum indicators
            short_period = min(20, len(prices) // 4)
            long_period = min(50, len(prices) // 2)
            
            if len(prices) < long_period:
                return signals
            
            short_ma = np.mean(prices[-short_period:])
            long_ma = np.mean(prices[-long_period:])
            
            # Calculate momentum
            momentum = (short_ma - long_ma) / long_ma
            
            # Get parameters
            threshold = strategy_config.parameters.get('momentum_threshold', 0.02)
            strength = strategy_config.parameters.get('trend_strength', 0.6)
            
            # Generate signal based on momentum
            if momentum > threshold:
                # Upward momentum
                signal_type = SignalType.BUY
                confidence = min(abs(momentum) / threshold * strength, 1.0)
            elif momentum < -threshold:
                # Downward momentum
                signal_type = SignalType.SELL
                confidence = min(abs(momentum) / threshold * strength, 1.0)
            else:
                return signals
            
            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                strength=SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE,
                asset=market_data.get('asset', 'UNKNOWN'),
                price=prices[-1],
                volume=market_data.get('volume', 0.0),
                confidence=confidence,
                timestamp=time.time(),
                strategy_name=strategy_config.name,
                metadata={
                    'momentum': momentum,
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'strategy_type': 'momentum'
                }
            )
            
            signals.append(signal)
            return signals
            
        except Exception as e:
            logger.error(f"Error in momentum signals: {e}")
            return []
    
    def _statistical_arbitrage_signals(self, strategy_config: StrategyConfig, 
                                     market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate statistical arbitrage signals"""
        try:
            signals = []
            
            # This is a simplified implementation
            # In a real system, you'd need pairs of assets and cointegration analysis
            
            # For now, return empty signals
            return signals
            
        except Exception as e:
            logger.error(f"Error in statistical arbitrage signals: {e}")
            return []
    
    def _ml_signals(self, strategy_config: StrategyConfig, 
                   market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate machine learning signals"""
        try:
            signals = []
            
            # Placeholder for ML-based signal generation
            # This would integrate with your ML models
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in ML signals: {e}")
            return []
    
    def _quantum_enhanced_signals(self, strategy_config: StrategyConfig, 
                                market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate quantum-enhanced signals"""
        try:
            signals = []
            
            # Placeholder for quantum-enhanced signal generation
            # This would integrate with quantum computing components
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in quantum-enhanced signals: {e}")
            return []
    
    def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank signals"""
        try:
            if not signals:
                return signals
            
            # Filter by confidence threshold
            min_confidence = self.config.get('min_signal_confidence', 0.6)
            filtered_signals = [s for s in signals if s.confidence >= min_confidence]
            
            # Sort by confidence (highest first)
            filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit number of signals per asset
            asset_signals: Dict[str, List[TradingSignal]] = {}
            for signal in filtered_signals:
                if signal.asset not in asset_signals:
                    asset_signals[signal.asset] = []
                asset_signals[signal.asset].append(signal)
            
            # Take only the best signal per asset
            final_signals = []
            for asset, asset_signal_list in asset_signals.items():
                if asset_signal_list:
                    final_signals.append(asset_signal_list[0])
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals
    
    def update_performance(self, strategy_name: str, trade_result: Dict[str, Any]) -> None:
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.performance:
                return
            
            performance = self.performance[strategy_name]
            
            # Update basic metrics
            performance.total_trades += 1
            performance.total_pnl += trade_result.get('pnl', 0.0)
            
            if trade_result.get('pnl', 0.0) > 0:
                performance.winning_trades += 1
            else:
                performance.losing_trades += 1
            
            # Calculate derived metrics
            if performance.total_trades > 0:
                performance.win_rate = performance.winning_trades / performance.total_trades
            
            if performance.losing_trades > 0:
                performance.profit_factor = abs(performance.total_pnl) / abs(
                    sum(t.get('pnl', 0.0) for t in self.signal_history if t.get('pnl', 0.0) < 0)
                )
            
            performance.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Error updating performance for {strategy_name}: {e}")
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy"""
        return self.performance.get(strategy_name)
    
    def get_all_performance(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies"""
        return self.performance.copy()
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name].enabled = True
                logger.info(f"Enabled strategy: {strategy_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error enabling strategy {strategy_name}: {e}")
            return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy"""
        try:
            if strategy_name in self.strategies:
                self.strategies[strategy_name].enabled = False
                logger.info(f"Disabled strategy: {strategy_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error disabling strategy {strategy_name}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'version': self.version,
            'total_strategies': len(self.strategies),
            'enabled_strategies': len([s for s in self.strategies.values() if s.enabled]),
            'total_signals_generated': self.total_signals_generated,
            'total_signals_executed': self.total_signals_executed,
            'last_signal_time': self.last_signal_time,
            'signal_history_size': len(self.signal_history)
        }


def main() -> None:
    """Main function for testing strategy logic"""
    try:
        print("🎯 Strategy Logic Test")
        print("=" * 40)
        
        # Initialize strategy logic
        strategy_logic = StrategyLogic()
        
        # Test market data
        market_data = {
            'asset': 'BTC',
            'prices': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': 1000.0
        }
        
        # Process market data
        signals = strategy_logic.process_market_data(market_data)
        print(f"✅ Generated {len(signals)} signals")
        
        # Display signals
        for i, signal in enumerate(signals):
            print(f"   Signal {i+1}: {signal.signal_type.value} {signal.asset} "
                  f"@ {signal.price:.2f} (confidence: {signal.confidence:.2f})")
        
        # Get system status
        status = strategy_logic.get_system_status()
        print(f"✅ System status: {status['enabled_strategies']} strategies enabled")
        
        print("\n🎉 Strategy logic test completed successfully!")
        
    except Exception as e:
        print(f"❌ Strategy logic test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
