# -*- coding: utf-8 -*-
"""
Clean Trading Pipeline for Schwabot System.

This module provides a clean, working implementation of the unified trading
pipeline that integrates all components while maintaining proper code structure
and error handling.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .clean_profit_vectorization import CleanProfitVectorization, ProfitVector, VectorizationMode

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyBranch(Enum):
    """Strategy branches."""

    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    GRID = "grid"


class MarketRegime(Enum):
    """Market regimes."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class MarketData:
    """Market data snapshot."""

    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volatility: float = 0.5
    trend_strength: float = 0.5
    entropy_level: float = 4.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingDecision:
    """Trading decision output."""

    timestamp: float
    symbol: str
    action: TradingAction
    quantity: float
    price: float
    confidence: float
    strategy_branch: StrategyBranch
    profit_potential: float
    risk_score: float
    thermal_state: ThermalState
    bit_phase: BitPhase
    profit_vector: ProfitVector
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Current state of the trading pipeline."""

    timestamp: float
    active_strategy: StrategyBranch
    current_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    current_risk_level: float
    market_regime: MarketRegime
    thermal_state: ThermalState
    bit_phase: BitPhase
    last_market_data: Optional[MarketData] = None


@dataclass
class RiskParameters:
    """Risk management parameters."""

    max_position_size: float = 0.1  # 10% max position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_daily_loss: float = 0.05  # 5% max daily loss
    volatility_threshold: float = 0.8  # High volatility threshold
    correlation_threshold: float = 0.9  # High correlation threshold


class CleanTradingPipeline:
    """
    Clean trading pipeline that integrates all Schwabot components.

    This pipeline provides:
    - Mathematical foundation for all calculations
    - Profit vectorization with multiple modes
    - Strategy switching based on market conditions
    - Risk management and position sizing
    - Real-time market analysis and decision making
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_params: Optional[RiskParameters] = None,
        default_strategy: StrategyBranch = StrategyBranch.MEAN_REVERSION,
        default_vectorization: VectorizationMode = VectorizationMode.STANDARD,
    ):
        """Initialize the clean trading pipeline."""
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()
        self.default_strategy = default_strategy
        self.default_vectorization = default_vectorization

        # Initialize core components
        self.math_foundation = CleanMathFoundation(precision=64)
        self.profit_vectorization = CleanProfitVectorization(
            risk_free_rate=0.02, default_mode=default_vectorization
        )

        # Pipeline state
        self.state = PipelineState(
            timestamp=time.time(),
            active_strategy=default_strategy,
            current_capital=initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0.0,
            current_risk_level=0.02,
            market_regime=MarketRegime.SIDEWAYS,
            thermal_state=ThermalState.WARM,
            bit_phase=BitPhase.THIRTY_TWO_BIT,
        )

        # Trading history
        self.trading_history: List[TradingDecision] = []
        self.market_data_history: List[MarketData] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Strategy weights for dynamic switching
        self.strategy_weights = {
            StrategyBranch.MEAN_REVERSION: 0.3,
            StrategyBranch.MOMENTUM: 0.25,
            StrategyBranch.ARBITRAGE: 0.15,
            StrategyBranch.SCALPING: 0.1,
            StrategyBranch.SWING: 0.15,
            StrategyBranch.GRID: 0.05,
        }

        logger.info(f"Clean Trading Pipeline initialized with " f"${initial_capital:,.2f} capital")

    async def process_market_data(self, market_data: MarketData) -> Optional[TradingDecision]:
        """
        Process market data through the complete pipeline.

        Args:
            market_data: Current market data snapshot

        Returns:
            Trading decision or None if no action recommended
        """
        try:
            start_time = time.time()

            # 1. Update market data history
            self.market_data_history.append(market_data)
            self.state.last_market_data = market_data

            # Keep history manageable
            if len(self.market_data_history) > 1000:
                self.market_data_history = self.market_data_history[-500:]

            # 2. Analyze market regime
            market_regime = self._analyze_market_regime(market_data)
            self.state.market_regime = market_regime

            # 3. Determine optimal strategy based on market regime
            optimal_strategy = self._determine_optimal_strategy(market_regime, market_data)
            if optimal_strategy != self.state.active_strategy:
                logger.info(
                    f"Strategy switch: {self.state.active_strategy.value} -> "
                    f"{optimal_strategy.value}"
                )
                self.state.active_strategy = optimal_strategy

            # 4. Adjust thermal state and bit phase based on market conditions
            self._adjust_mathematical_parameters(market_data)

            # 5. Calculate profit vectorization
            profit_vector = self.profit_vectorization.calculate_profit_vectorization(
                btc_price=market_data.price,
                volume=market_data.volume,
                market_data=self._market_data_to_dict(market_data),
                mode=self._select_vectorization_mode(market_regime),
            )

            # 6. Generate trading signal
            trading_signal = self._generate_trading_signal(market_data, profit_vector)

            # 7. Apply risk management
            risk_adjusted_signal = self._apply_risk_management(trading_signal, market_data)

            # 8. Create trading decision
            if risk_adjusted_signal and risk_adjusted_signal["action"] != TradingAction.HOLD:
                decision = TradingDecision(
                    timestamp=time.time(),
                    symbol=market_data.symbol,
                    action=risk_adjusted_signal["action"],
                    quantity=risk_adjusted_signal["quantity"],
                    price=market_data.price,
                    confidence=risk_adjusted_signal["confidence"],
                    strategy_branch=self.state.active_strategy,
                    profit_potential=profit_vector.profit_score,
                    risk_score=self._calculate_risk_score(market_data),
                    thermal_state=self.state.thermal_state,
                    bit_phase=self.state.bit_phase,
                    profit_vector=profit_vector,
                    metadata={
                        "processing_time": time.time() - start_time,
                        "market_regime": market_regime.value,
                        "vectorization_mode": profit_vector.mode,
                    },
                )

                # Track the decision
                self.trading_history.append(decision)
                self._update_pipeline_metrics(decision)

                return decision

            return None

        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
            return None

    def _analyze_market_regime(self, market_data: MarketData) -> MarketRegime:
        """Analyze current market regime based on historical data."""
        try:
            if len(self.market_data_history) < 10:
                return MarketRegime.SIDEWAYS

            # Get recent price history
            recent_prices = [md.price for md in self.market_data_history[-20:]]

            # Calculate trend strength
            price_changes = np.diff(recent_prices)
            trend_direction = np.mean(price_changes)
            trend_strength = (
                abs(trend_direction) / np.std(recent_prices) if np.std(recent_prices) > 0 else 0
            )

            # Calculate volatility
            volatility = (
                np.std(price_changes) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0
            )

            # Determine regime
            if volatility > 0.05:  # High volatility threshold
                return MarketRegime.VOLATILE
            elif volatility < 0.01:  # Low volatility threshold
                return MarketRegime.CALM
            elif trend_strength > 0.02 and trend_direction > 0:
                return MarketRegime.TRENDING_UP
            elif trend_strength > 0.02 and trend_direction < 0:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime.SIDEWAYS

    def _determine_optimal_strategy(
        self, regime: MarketRegime, market_data: MarketData
    ) -> StrategyBranch:
        """Determine optimal strategy based on market regime."""
        strategy_preferences = {
            MarketRegime.TRENDING_UP: StrategyBranch.MOMENTUM,
            MarketRegime.TRENDING_DOWN: StrategyBranch.MEAN_REVERSION,
            MarketRegime.SIDEWAYS: StrategyBranch.GRID,
            MarketRegime.VOLATILE: StrategyBranch.SCALPING,
            MarketRegime.CALM: StrategyBranch.SWING,
        }

        optimal = strategy_preferences.get(regime, self.default_strategy)

        # Add some randomness based on profit vector performance
        if hasattr(self.profit_vectorization, "mode_performance"):
            mode_performance = self.profit_vectorization.mode_performance
            best_performing_modes = sorted(
                mode_performance.items(), key=lambda x: x[1].get("total_profit", 0), reverse=True
            )

            # Occasionally switch to complement best performing modes
            if len(best_performing_modes) > 0 and np.random.random() < 0.1:
                # Some logic to map vectorization performance to strategy
                pass

        return optimal

    def _adjust_mathematical_parameters(self, market_data: MarketData) -> None:
        """Adjust thermal state and bit phase based on market conditions."""
        # Adjust thermal state based on volatility
        if market_data.volatility > 0.8:
            new_thermal = ThermalState.CRITICAL
        elif market_data.volatility > 0.6:
            new_thermal = ThermalState.HOT
        elif market_data.volatility > 0.3:
            new_thermal = ThermalState.WARM
        else:
            new_thermal = ThermalState.COOL

        if new_thermal != self.state.thermal_state:
            self.math_foundation.set_thermal_state(new_thermal)
            self.state.thermal_state = new_thermal

        # Adjust bit phase based on price precision requirements
        price_magnitude = np.log10(market_data.price) if market_data.price > 0 else 0
        if price_magnitude > 5:  # Very high prices
            new_bit_phase = BitPhase.FORTY_TWO_BIT
        elif price_magnitude > 4:
            new_bit_phase = BitPhase.THIRTY_TWO_BIT
        elif price_magnitude > 3:
            new_bit_phase = BitPhase.SIXTEEN_BIT
        elif price_magnitude > 2:
            new_bit_phase = BitPhase.EIGHT_BIT
        else:
            new_bit_phase = BitPhase.FOUR_BIT

        if new_bit_phase != self.state.bit_phase:
            self.math_foundation.set_bit_phase(new_bit_phase)
            self.state.bit_phase = new_bit_phase

    def _select_vectorization_mode(self, regime: MarketRegime) -> VectorizationMode:
        """Select appropriate vectorization mode based on market regime."""
        mode_preferences = {
            MarketRegime.TRENDING_UP: VectorizationMode.MOMENTUM,
            MarketRegime.TRENDING_DOWN: VectorizationMode.ENTROPY_WEIGHTED,
            MarketRegime.SIDEWAYS: VectorizationMode.CONSENSUS_VOTING,
            MarketRegime.VOLATILE: VectorizationMode.BIT_PHASE_TRIGGER,
            MarketRegime.CALM: VectorizationMode.HYBRID_BLEND,
        }

        return mode_preferences.get(regime, VectorizationMode.STANDARD)

    def _generate_trading_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on strategy and profit vector."""
        try:
            strategy = self.state.active_strategy

            if strategy == StrategyBranch.MEAN_REVERSION:
                return self._mean_reversion_signal(market_data, profit_vector)
            elif strategy == StrategyBranch.MOMENTUM:
                return self._momentum_signal(market_data, profit_vector)
            elif strategy == StrategyBranch.ARBITRAGE:
                return self._arbitrage_signal(market_data, profit_vector)
            elif strategy == StrategyBranch.SCALPING:
                return self._scalping_signal(market_data, profit_vector)
            elif strategy == StrategyBranch.SWING:
                return self._swing_signal(market_data, profit_vector)
            elif strategy == StrategyBranch.GRID:
                return self._grid_signal(market_data, profit_vector)
            else:
                return self._default_signal(market_data, profit_vector)

        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return None

    def _mean_reversion_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Dict[str, Any]:
        """Generate mean reversion trading signal."""
        if len(self.market_data_history) < 20:
            return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

        # Calculate mean and deviation
        recent_prices = [md.price for md in self.market_data_history[-20:]]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)

        current_price = market_data.price
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

        # Generate signal based on z-score
        if z_score > 2:  # Price too high, sell
            action = TradingAction.SELL
            confidence = min(abs(z_score) / 3, 1.0)
        elif z_score < -2:  # Price too low, buy
            action = TradingAction.BUY
            confidence = min(abs(z_score) / 3, 1.0)
        else:
            action = TradingAction.HOLD
            confidence = 0

        # Calculate position size based on confidence and profit vector
        base_quantity = self.state.current_capital * 0.1  # 10% of capital
        quantity = base_quantity * confidence * profit_vector.confidence_score

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {"z_score": z_score, "mean_price": mean_price},
        }

    def _momentum_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Dict[str, Any]:
        """Generate momentum trading signal."""
        if len(self.market_data_history) < 10:
            return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

        # Calculate momentum
        recent_prices = [md.price for md in self.market_data_history[-10:]]
        momentum = (
            (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        )

        # Generate signal based on momentum
        if momentum > 0.02:  # Strong upward momentum
            action = TradingAction.BUY
            confidence = min(abs(momentum) * 10, 1.0)
        elif momentum < -0.02:  # Strong downward momentum
            action = TradingAction.SELL
            confidence = min(abs(momentum) * 10, 1.0)
        else:
            action = TradingAction.HOLD
            confidence = 0

        base_quantity = self.state.current_capital * 0.15  # 15% for momentum trades
        quantity = base_quantity * confidence * profit_vector.confidence_score

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {"momentum": momentum},
        }

    def _arbitrage_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Dict[str, Any]:
        """Generate arbitrage trading signal."""
        # Simplified arbitrage logic - would need multiple exchange data in practice
        spread = (market_data.ask - market_data.bid) if market_data.ask and market_data.bid else 0

        if spread > market_data.price * 0.001:  # Spread > 0.1% of price
            action = TradingAction.BUY  # Buy at bid, sell at ask
            confidence = min(spread / (market_data.price * 0.005), 1.0)
            quantity = self.state.current_capital * 0.05 * confidence  # Small positions
        else:
            action = TradingAction.HOLD
            confidence = 0
            quantity = 0

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {"spread": spread},
        }

    def _scalping_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Dict[str, Any]:
        """Generate scalping trading signal."""
        # Quick in-and-out trades based on small price movements
        if len(self.market_data_history) < 5:
            return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

        # Look for quick price movements
        recent_prices = [md.price for md in self.market_data_history[-5:]]
        short_term_change = (
            (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
            if recent_prices[-3] > 0
            else 0
        )

        if abs(short_term_change) > 0.005:  # 0.5% quick movement
            action = TradingAction.BUY if short_term_change > 0 else TradingAction.SELL
            confidence = min(abs(short_term_change) * 50, 1.0)
            # Larger positions for quick trades
            quantity = self.state.current_capital * 0.2 * confidence
        else:
            action = TradingAction.HOLD
            confidence = 0
            quantity = 0

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {"short_term_change": short_term_change},
        }

    def _swing_signal(self, market_data: MarketData, profit_vector: ProfitVector) -> Dict[str, Any]:
        """Generate swing trading signal."""
        # Medium-term trades based on larger price swings
        if len(self.market_data_history) < 50:
            return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

        # Calculate swing indicators
        recent_prices = [md.price for md in self.market_data_history[-50:]]
        high = max(recent_prices)
        low = min(recent_prices)
        current = recent_prices[-1]

        # Position within range
        position_in_range = (current - low) / (high - low) if (high - low) > 0 else 0.5

        if position_in_range < 0.2:  # Near bottom of range
            action = TradingAction.BUY
            confidence = 1.0 - position_in_range * 2
        elif position_in_range > 0.8:  # Near top of range
            action = TradingAction.SELL
            confidence = (position_in_range - 0.8) * 5
        else:
            action = TradingAction.HOLD
            confidence = 0

        quantity = self.state.current_capital * 0.25 * confidence  # Larger swing positions

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {
                "position_in_range": position_in_range,
                "range_high": high,
                "range_low": low,
            },
        }

    def _grid_signal(self, market_data: MarketData, profit_vector: ProfitVector) -> Dict[str, Any]:
        """Generate grid trading signal."""
        # Grid trading with multiple levels
        base_price = 50000  # Example base price - would be dynamically set
        grid_spacing = 100  # $100 grid spacing

        current_price = market_data.price
        price_level = round((current_price - base_price) / grid_spacing)

        # Alternate buy/sell at different levels
        if price_level % 2 == 0:  # Even levels - buy
            action = TradingAction.BUY
            confidence = 0.5  # Moderate confidence for grid trades
        else:  # Odd levels - sell
            action = TradingAction.SELL
            confidence = 0.5

        quantity = self.state.current_capital * 0.05  # Small grid positions

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence * profit_vector.confidence_score,
            "metadata": {"price_level": price_level, "grid_spacing": grid_spacing},
        }

    def _default_signal(
        self, market_data: MarketData, profit_vector: ProfitVector
    ) -> Dict[str, Any]:
        """Default trading signal when no specific strategy applies."""
        # Conservative default approach
        if profit_vector.confidence_score > 0.7 and profit_vector.profit_score > 0:
            action = TradingAction.BUY
            confidence = profit_vector.confidence_score
            quantity = self.state.current_capital * 0.05 * confidence
        else:
            action = TradingAction.HOLD
            confidence = 0
            quantity = 0

        return {
            "action": action,
            "quantity": quantity,
            "confidence": confidence,
            "metadata": {"default_strategy": True},
        }

    def _apply_risk_management(
        self, signal: Dict[str, Any], market_data: MarketData
    ) -> Optional[Dict[str, Any]]:
        """Apply risk management rules to trading signal."""
        if not signal or signal["action"] == TradingAction.HOLD:
            return signal

        try:
            # Check position size limits
            max_position = self.state.current_capital * self.risk_params.max_position_size
            if signal["quantity"] > max_position:
                signal["quantity"] = max_position
                logger.warning(f"Position size reduced to {max_position} due to risk limits")

            # Check volatility limits
            if market_data.volatility > self.risk_params.volatility_threshold:
                signal["quantity"] *= 0.5  # Reduce position size in high volatility
                logger.info(
                    f"Position size reduced due to high volatility: " f"{market_data.volatility}"
                )

            # Check daily loss limits
            daily_pnl = self._calculate_daily_pnl()
            max_daily_loss = -self.state.current_capital * self.risk_params.max_daily_loss
            if daily_pnl < max_daily_loss:
                logger.warning("Daily loss limit reached, blocking new trades")
                return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

            # Risk score check
            risk_score = self._calculate_risk_score(market_data)
            if risk_score > 0.8:  # High risk
                signal["quantity"] *= 0.3  # Significantly reduce position
                logger.info(f"Position size reduced due to high risk score: {risk_score}")

            return signal

        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return {"action": TradingAction.HOLD, "quantity": 0, "confidence": 0}

    def _calculate_risk_score(self, market_data: MarketData) -> float:
        """Calculate overall risk score for current market conditions."""
        try:
            # Volatility component
            volatility_risk = min(market_data.volatility, 1.0)

            # Trend uncertainty component
            trend_risk = 1.0 - abs(market_data.trend_strength - 0.5) * 2

            # Portfolio concentration risk
            concentration_risk = 0.3  # Simplified - would calculate actual concentration

            # Recent performance risk
            performance_risk = 0.0
            if len(self.trading_history) > 5:
                recent_decisions = self.trading_history[-5:]
                losing_streak = sum(1 for d in recent_decisions if d.profit_potential < 0)
                performance_risk = losing_streak / 5

            # Combine risk components
            total_risk = (
                volatility_risk * 0.4
                + trend_risk * 0.3
                + concentration_risk * 0.2
                + performance_risk * 0.1
            )

            return min(total_risk, 1.0)

        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default moderate risk

    def _calculate_daily_pnl(self) -> float:
        """Calculate profit/loss for the current day."""
        try:
            current_time = time.time()
            day_start = current_time - (current_time % 86400)  # Start of current day

            daily_trades = [
                decision for decision in self.trading_history if decision.timestamp >= day_start
            ]

            return sum(decision.profit_potential for decision in daily_trades)

        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0

    def _market_data_to_dict(self, market_data: MarketData) -> Dict[str, Any]:
        """Convert MarketData to dictionary for profit vectorization."""
        return {
            "symbol": market_data.symbol,
            "price": market_data.price,
            "volume": market_data.volume,
            "volatility": market_data.volatility,
            "trend_strength": market_data.trend_strength,
            "entropy_level": market_data.entropy_level,
            "timestamp": market_data.timestamp,
            **market_data.metadata,
        }

    def _update_pipeline_metrics(self, decision: TradingDecision) -> None:
        """Update pipeline performance metrics."""
        self.state.total_trades += 1
        self.state.total_profit += decision.profit_potential

        if decision.profit_potential > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1

        # Update state timestamp
        self.state.timestamp = decision.timestamp

        # Keep trading history manageable
        if len(self.trading_history) > 1000:
            self.trading_history = self.trading_history[-500:]

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance summary."""
        win_rate = self.state.winning_trades / max(1, self.state.total_trades)

        avg_profit = self.state.total_profit / max(1, self.state.total_trades)

        return {
            "state": {
                "current_capital": self.state.current_capital,
                "total_trades": self.state.total_trades,
                "winning_trades": self.state.winning_trades,
                "losing_trades": self.state.losing_trades,
                "win_rate": win_rate,
                "total_profit": self.state.total_profit,
                "average_profit_per_trade": avg_profit,
                "active_strategy": self.state.active_strategy.value,
                "market_regime": self.state.market_regime.value,
                "thermal_state": self.state.thermal_state.value,
                "bit_phase": self.state.bit_phase.value,
            },
            "math_foundation": self.math_foundation.get_metrics(),
            "profit_vectorization": self.profit_vectorization.get_performance_summary(),
            "recent_decisions": [
                {
                    "timestamp": d.timestamp,
                    "symbol": d.symbol,
                    "action": d.action.value,
                    "profit_potential": d.profit_potential,
                    "confidence": d.confidence,
                }
                for d in self.trading_history[-5:]
            ],
        }


# Convenience functions
def create_trading_pipeline(
    initial_capital: float = 100000.0,
    strategy: StrategyBranch = StrategyBranch.MEAN_REVERSION,
    vectorization: VectorizationMode = VectorizationMode.STANDARD,
) -> CleanTradingPipeline:
    """Create a new trading pipeline instance."""
    return CleanTradingPipeline(
        initial_capital=initial_capital,
        default_strategy=strategy,
        default_vectorization=vectorization,
    )


async def run_trading_simulation(
    pipeline: CleanTradingPipeline, duration_seconds: int = 3600, tick_interval: float = 1.0
) -> Dict[str, Any]:
    """Run a trading simulation for testing."""
    logger.info(f"Starting trading simulation for {duration_seconds} seconds")

    start_time = time.time()
    decisions_made = []

    try:
        while (time.time() - start_time) < duration_seconds:
            # Generate synthetic market data
            current_price = 50000 + np.random.normal(0, 1000)  # BTC price around $50k
            volume = np.random.uniform(0.1, 10.0)
            volatility = np.random.uniform(0.1, 0.9)

            market_data = MarketData(
                symbol="BTC/USD",
                price=current_price,
                volume=volume,
                timestamp=time.time(),
                volatility=volatility,
                trend_strength=np.random.uniform(0.0, 1.0),
                entropy_level=np.random.uniform(2.0, 6.0),
            )

            # Process through pipeline
            decision = await pipeline.process_market_data(market_data)
            if decision:
                decisions_made.append(decision)
                logger.info(
                    f"Decision: {decision.action.value} {decision.quantity:.2f} "
                    f"@ {decision.price:.2f}"
                )

            await asyncio.sleep(tick_interval)

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")

    # Return simulation results
    return {
        "duration": time.time() - start_time,
        "decisions_made": len(decisions_made),
        "pipeline_summary": pipeline.get_pipeline_summary(),
        "final_capital": pipeline.state.current_capital + pipeline.state.total_profit,
    }
