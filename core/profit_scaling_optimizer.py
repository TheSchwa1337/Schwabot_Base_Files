#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profit Scaling Optimizer - Mathematical Position Sizing & Risk Management
=========================================================================

Implements advanced profit scaling optimization using mathematical frameworks:
- Kelly criterion for optimal position sizing
- Win rate optimization with historical data
- Volatility adjustment for market conditions
- Volume-based scaling for liquidity
- Mathematical confidence integration
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import mathematical infrastructure
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator

    # Import mathematical modules for optimization
    from core.math.volume_weighted_hash_oscillator import VolumeWeightedHashOscillator
    from core.math.zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate
    from core.math.qsc_quantum_signal_collapse_gate import QSCGate
    from core.math.unified_tensor_algebra import UnifiedTensorAlgebra
    from core.math.galileo_tensor_field_entropy_drift import GalileoTensorField
    from core.math.advanced_tensor_algebra import AdvancedTensorAlgebra
    from core.math.entropy_math import EntropyMath

    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")

class ScalingMode(Enum):
    """Profit scaling modes."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class RiskProfile(Enum):
    """Risk profiles for position sizing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


@dataclass
class ScalingParameters:
    """Parameters for profit scaling optimization."""

    # Base parameters
    base_position_size: float = 0.01  # 1% of capital
    max_position_size: float = 0.25   # 25% of capital
    min_position_size: float = 0.001  # 0.1% of capital

    # Kelly criterion parameters
    max_kelly_fraction: float = 0.25  # Maximum Kelly fraction
    conservative_factor: float = 0.5  # Conservative Kelly multiplier

    # Risk parameters
    risk_tolerance: float = 0.1       # 10% risk tolerance
    volatility_penalty: float = 10.0  # Volatility penalty multiplier
    volume_bonus: float = 1.5         # Volume bonus multiplier

    # Mathematical confidence thresholds
    min_confidence: float = 0.6       # Minimum confidence for scaling
    confidence_multiplier: float = 1.5 # Confidence scaling multiplier

    # Win rate parameters
    min_win_rate: float = 0.3         # Minimum win rate
    max_win_rate: float = 0.9         # Maximum win rate
    win_rate_weight: float = 0.4      # Weight for win rate in scaling

    # Market condition parameters
    spread_threshold: float = 0.001   # Spread threshold for order type selection
    volume_threshold: float = 1_000_000_000  # Volume threshold for scaling


@dataclass
class ScalingResult:
    """Result of profit scaling optimization."""

    # Original parameters
    original_amount: float
    original_confidence: float

    # Scaled parameters
    scaled_amount: float
    scaling_factor: float

    # Mathematical components
    kelly_fraction: float
    confidence_factor: float
    volatility_adjustment: float
    volume_factor: float
    win_rate_factor: float

    # Risk metrics
    risk_score: float
    expected_profit: float
    max_loss: float

    # Market context
    market_conditions: Dict[str, Any]
    scaling_mode: ScalingMode

    # Metadata
    timestamp: float = field(default_factory=time.time)
    optimization_time: float = 0.0


@dataclass
class WinRateData:
    """Win rate data for strategy optimization."""

    strategy_id: str
    total_trades: int
    winning_trades: int
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: float = field(default_factory=time.time)


class ProfitScalingOptimizer:
    """
    Profit Scaling Optimizer

    Implements mathematical profit scaling optimization using:
    - Kelly criterion for position sizing
    - Win rate optimization
    - Volatility adjustment
    - Volume-based scaling
    - Mathematical confidence integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the profit scaling optimizer."""
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config or self._default_config()
        self.scaling_params = ScalingParameters(**self.config.get('scaling_params', {}))

        # Initialize mathematical infrastructure
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()

            # Initialize mathematical modules
            self.vwho = VolumeWeightedHashOscillator()
            self.zygot_zalgo = ZygotZalgoEntropyDualKeyGate()
            self.qsc = QSCGate()
            self.tensor_algebra = UnifiedTensorAlgebra()
            self.galileo = GalileoTensorField()
            self.advanced_tensor = AdvancedTensorAlgebra()
            self.entropy_math = EntropyMath()

        # Win rate tracking
        self.win_rate_data: Dict[str, WinRateData] = {}
        self.strategy_performance: Dict[str, List[Dict[str, Any]]] = {}

        # Performance metrics
        self.optimization_count = 0
        self.total_scaling_time = 0.0
        self.average_scaling_time = 0.0

        # Mathematical constants
        self.GOLDEN_RATIO = 1.618033988749
        self.PI = 3.141592653589793
        self.EULER = 2.718281828459

        self.logger.info("✅ Profit Scaling Optimizer initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'scaling_params': {
                'base_position_size': 0.01,
                'max_position_size': 0.25,
                'min_position_size': 0.001,
                'max_kelly_fraction': 0.25,
                'conservative_factor': 0.5,
                'risk_tolerance': 0.1,
                'volatility_penalty': 10.0,
                'volume_bonus': 1.5,
                'min_confidence': 0.6,
                'confidence_multiplier': 1.5,
                'min_win_rate': 0.3,
                'max_win_rate': 0.9,
                'win_rate_weight': 0.4,
                'spread_threshold': 0.001,
                'volume_threshold': 1_000_000_000
            },
            'enable_mathematical_optimization': True,
            'enable_win_rate_tracking': True,
            'enable_risk_management': True,
            'enable_market_adaptation': True
        }

    def optimize_position_size(self, base_amount: float, confidence: float, strategy_id: str, market_data: Dict[str, Any], risk_profile: RiskProfile = RiskProfile.MEDIUM) -> ScalingResult:
        """
        Optimize position size using mathematical profit scaling.

        Args:
        base_amount: Base position amount
        confidence: Signal confidence (0-1)
        strategy_id: Strategy identifier
        market_data: Market data dictionary
        risk_profile: Risk profile for scaling

        Returns:
        ScalingResult with optimized parameters
        """
        try:
            start_time = time.time()

            # Get win rate data for strategy
            win_rate_data = self.win_rate_data.get(strategy_id, self._create_default_win_rate(strategy_id))

            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(win_rate_data)

            # Apply risk profile adjustments
            kelly_fraction = self._apply_risk_profile(kelly_fraction, risk_profile)

            # Calculate confidence factor
            confidence_factor = self._calculate_confidence_factor(confidence)

            # Calculate volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(market_data)

            # Calculate volume factor
            volume_factor = self._calculate_volume_factor(market_data)

            # Calculate win rate factor
            win_rate_factor = self._calculate_win_rate_factor(win_rate_data)

            # Combine all factors
            scaling_factor = (
                kelly_fraction *
                confidence_factor *
                volatility_adjustment *
                volume_factor *
                win_rate_factor
            )

            # Apply bounds
            scaling_factor = np.clip(scaling_factor, 0.0, self.scaling_params.max_position_size / self.scaling_params.base_position_size)

            # Calculate scaled amount
            scaled_amount = base_amount * scaling_factor

            # Ensure minimum position size
            if scaled_amount < self.scaling_params.min_position_size:
                scaled_amount = self.scaling_params.min_position_size

            # Calculate risk metrics
            risk_score = self._calculate_risk_score(scaled_amount, confidence, market_data, win_rate_data)
            expected_profit = self._calculate_expected_profit(scaled_amount, confidence, market_data, win_rate_data)
            max_loss = self._calculate_max_loss(scaled_amount, market_data)

            # Determine scaling mode
            scaling_mode = self._determine_scaling_mode(scaling_factor, confidence, risk_score)

            optimization_time = time.time() - start_time

            # Update performance metrics
            self.optimization_count += 1
            self.total_scaling_time += optimization_time
            self.average_scaling_time = self.total_scaling_time / self.optimization_count

            return ScalingResult(
                original_amount=base_amount,
                original_confidence=confidence,
                scaled_amount=scaled_amount,
                scaling_factor=scaling_factor,
                kelly_fraction=kelly_fraction,
                confidence_factor=confidence_factor,
                volatility_adjustment=volatility_adjustment,
                volume_factor=volume_factor,
                win_rate_factor=win_rate_factor,
                risk_score=risk_score,
                expected_profit=expected_profit,
                max_loss=max_loss,
                market_conditions=market_data,
                scaling_mode=scaling_mode,
                optimization_time=optimization_time
            )

        except Exception as e:
            self.logger.error(f"Error optimizing position size: {e}")
            # Return conservative scaling on error
            return ScalingResult(
                original_amount=base_amount,
                original_confidence=confidence,
                scaled_amount=self.scaling_params.min_position_size,
                scaling_factor=0.1,
                kelly_fraction=0.1,
                confidence_factor=0.5,
                volatility_adjustment=1.0,
                volume_factor=1.0,
                win_rate_factor=0.5,
                risk_score=0.5,
                expected_profit=0.0,
                max_loss=base_amount,
                market_conditions=market_data,
                scaling_mode=ScalingMode.CONSERVATIVE,
                optimization_time=0.0
            )

    def _calculate_kelly_fraction(self, win_rate_data: WinRateData) -> float:
        """Calculate Kelly criterion fraction."""
        try:
            if win_rate_data.total_trades < 10:
                return 0.1  # Conservative default

            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            win_rate = win_rate_data.win_rate
            avg_profit = win_rate_data.average_profit
            avg_loss = abs(win_rate_data.average_loss)

            if avg_loss == 0:
                return 0.1

            # Calculate odds received (b)
            b = avg_profit / avg_loss

            # Kelly fraction
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b

            # Apply conservative factor
            kelly_fraction *= self.scaling_params.conservative_factor

            # Ensure positive and within bounds
            kelly_fraction = max(0.0, min(kelly_fraction, self.scaling_params.max_kelly_fraction))

            return kelly_fraction

        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.1

    def _apply_risk_profile(self, kelly_fraction: float, risk_profile: RiskProfile) -> float:
        """Apply risk profile adjustments to Kelly fraction."""
        try:
            if risk_profile == RiskProfile.LOW:
                return kelly_fraction * 0.5
            elif risk_profile == RiskProfile.MEDIUM:
                return kelly_fraction * 1.0
            elif risk_profile == RiskProfile.HIGH:
                return kelly_fraction * 1.5
            else:
                return kelly_fraction

        except Exception as e:
            self.logger.error(f"Error applying risk profile: {e}")
            return kelly_fraction

    def _calculate_confidence_factor(self, confidence: float) -> float:
        """Calculate confidence factor for scaling."""
        try:
            if confidence < self.scaling_params.min_confidence:
                return 0.5

            # Linear scaling from min_confidence to 1.0
            confidence_factor = (confidence - self.scaling_params.min_confidence) / (1.0 - self.scaling_params.min_confidence)
            confidence_factor = max(0.5, min(confidence_factor, self.scaling_params.confidence_multiplier))

            return confidence_factor

        except Exception as e:
            self.logger.error(f"Error calculating confidence factor: {e}")
            return 0.5

    def _calculate_volatility_adjustment(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility adjustment factor."""
        try:
            volatility = market_data.get('volatility', 0.02)
            
            # Higher volatility = lower position size
            volatility_penalty = 1.0 / (1.0 + volatility * self.scaling_params.volatility_penalty)
            
            return max(0.1, min(volatility_penalty, 2.0))

        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0

    def _calculate_volume_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume-based scaling factor."""
        try:
            volume = market_data.get('volume', 0)
            
            if volume > self.scaling_params.volume_threshold:
                return self.scaling_params.volume_bonus
            else:
                # Linear scaling based on volume
                volume_ratio = volume / self.scaling_params.volume_threshold
                return 1.0 + (self.scaling_params.volume_bonus - 1.0) * volume_ratio

        except Exception as e:
            self.logger.error(f"Error calculating volume factor: {e}")
            return 1.0

    def _calculate_win_rate_factor(self, win_rate_data: WinRateData) -> float:
        """Calculate win rate factor for scaling."""
        try:
            win_rate = win_rate_data.win_rate
            
            # Normalize win rate to scaling factor
            if win_rate < self.scaling_params.min_win_rate:
                return 0.5
            elif win_rate > self.scaling_params.max_win_rate:
                return 1.5
            else:
                # Linear interpolation
                normalized_rate = (win_rate - self.scaling_params.min_win_rate) / (self.scaling_params.max_win_rate - self.scaling_params.min_win_rate)
                return 0.5 + normalized_rate

        except Exception as e:
            self.logger.error(f"Error calculating win rate factor: {e}")
            return 0.5

    def _calculate_risk_score(self, amount: float, confidence: float, market_data: Dict[str, Any], win_rate_data: WinRateData) -> float:
        """Calculate risk score for the position."""
        try:
            # Combine multiple risk factors
            volatility_risk = market_data.get('volatility', 0.02) * 10
            confidence_risk = 1.0 - confidence
            win_rate_risk = 1.0 - win_rate_data.win_rate
            
            # Weighted average
            risk_score = (volatility_risk * 0.4 + confidence_risk * 0.3 + win_rate_risk * 0.3)
            
            return min(1.0, max(0.0, risk_score))

        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def _calculate_expected_profit(self, amount: float, confidence: float, market_data: Dict[str, Any], win_rate_data: WinRateData) -> float:
        """Calculate expected profit for the position."""
        try:
            # Expected profit = amount * win_rate * avg_profit - amount * (1 - win_rate) * avg_loss
            win_rate = win_rate_data.win_rate
            avg_profit = win_rate_data.average_profit
            avg_loss = abs(win_rate_data.average_loss)
            
            expected_profit = amount * (win_rate * avg_profit - (1 - win_rate) * avg_loss)
            
            return expected_profit

        except Exception as e:
            self.logger.error(f"Error calculating expected profit: {e}")
            return 0.0

    def _calculate_max_loss(self, amount: float, market_data: Dict[str, Any]) -> float:
        """Calculate maximum potential loss."""
        try:
            # Simple calculation: max loss = position amount
            # In practice, this could include stop-loss levels, etc.
            return amount

        except Exception as e:
            self.logger.error(f"Error calculating max loss: {e}")
            return amount

    def _determine_scaling_mode(self, scaling_factor: float, confidence: float, risk_score: float) -> ScalingMode:
        """Determine the appropriate scaling mode."""
        try:
            if risk_score > 0.7 or confidence < 0.5:
                return ScalingMode.CONSERVATIVE
            elif scaling_factor > 2.0 and confidence > 0.8:
                return ScalingMode.AGGRESSIVE
            elif scaling_factor > 1.5:
                return ScalingMode.MODERATE
            else:
                return ScalingMode.ADAPTIVE

        except Exception as e:
            self.logger.error(f"Error determining scaling mode: {e}")
            return ScalingMode.CONSERVATIVE

    def _create_default_win_rate(self, strategy_id: str) -> WinRateData:
        """Create default win rate data for a strategy."""
        return WinRateData(
            strategy_id=strategy_id,
            total_trades=0,
            winning_trades=0,
            win_rate=0.5,
            average_profit=0.02,
            average_loss=0.01,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0
        )

    def update_win_rate(self, strategy_id: str, trade_result: Dict[str, Any]) -> None:
        """Update win rate data for a strategy."""
        try:
            if strategy_id not in self.win_rate_data:
                self.win_rate_data[strategy_id] = self._create_default_win_rate(strategy_id)

            win_rate_data = self.win_rate_data[strategy_id]
            
            # Update trade counts
            win_rate_data.total_trades += 1
            if trade_result.get('profit', 0) > 0:
                win_rate_data.winning_trades += 1

            # Update win rate
            win_rate_data.win_rate = win_rate_data.winning_trades / win_rate_data.total_trades

            # Update other metrics
            profit = trade_result.get('profit', 0)
            if profit > 0:
                win_rate_data.average_profit = (win_rate_data.average_profit * (win_rate_data.winning_trades - 1) + profit) / win_rate_data.winning_trades
            else:
                loss = abs(profit)
                win_rate_data.average_loss = (win_rate_data.average_loss * (win_rate_data.total_trades - win_rate_data.winning_trades - 1) + loss) / (win_rate_data.total_trades - win_rate_data.winning_trades)

            # Update profit factor
            if win_rate_data.average_loss > 0:
                win_rate_data.profit_factor = win_rate_data.average_profit / win_rate_data.average_loss

            win_rate_data.last_updated = time.time()

        except Exception as e:
            self.logger.error(f"Error updating win rate: {e}")

    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get optimizer status and performance metrics."""
        return {
            'optimization_count': self.optimization_count,
            'average_scaling_time': self.average_scaling_time,
            'total_scaling_time': self.total_scaling_time,
            'win_rate_data_count': len(self.win_rate_data),
            'math_infrastructure_available': MATH_INFRASTRUCTURE_AVAILABLE,
            'scaling_params': self.scaling_params.__dict__
        }


# Factory function
def create_profit_scaling_optimizer(config: Optional[Dict[str, Any]] = None) -> ProfitScalingOptimizer:
    """Create a Profit Scaling Optimizer instance."""
    return ProfitScalingOptimizer(config) 