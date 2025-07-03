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
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        risk_params: Optional[RiskParameters] = None,
    ):
        """Initialize the trading pipeline."""
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()

        # Initialize mathematical foundation
        self.math_foundation = CleanMathFoundation()

        # Initialize profit vectorization
        self.profit_vectorizer = CleanProfitVectorization()

        # Pipeline state
        self.state = PipelineState(
            timestamp=time.time(),
            active_strategy=StrategyBranch.MOMENTUM,
            current_capital=initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0.0,
            current_risk_level=0.0,
            market_regime=MarketRegime.SIDEWAYS,
            thermal_state=ThermalState.WARM,
            bit_phase=BitPhase.EIGHT_BIT,
        )

        # Market data history for analysis
        self.market_data_history: List[MarketData] = []
        self.decision_history: List[TradingDecision] = []

        # Performance tracking
        self.performance_metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

        logger.info(f"Clean Trading Pipeline initialized for {symbol}")

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

            # 4. Update thermal state and bit phase
            self._update_thermal_state(market_data)

            # 5. Generate trading signal using active strategy
            signal = await self._generate_signal(market_data, optimal_strategy)

            if not signal:
                return None

            # 6. Apply risk management
            risk_adjusted_signal = self._apply_risk_management(signal, market_data)

            if not risk_adjusted_signal:
                return None

            # 7. Create profit vector
            profit_vector = self._create_profit_vector(risk_adjusted_signal, market_data)

            # 8. Make final trading decision
            decision = TradingDecision(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                action=TradingAction(risk_adjusted_signal["action"]),
                quantity=risk_adjusted_signal["quantity"],
                price=market_data.price,
                confidence=risk_adjusted_signal["confidence"],
                strategy_branch=optimal_strategy,
                profit_potential=risk_adjusted_signal["profit_potential"],
                risk_score=risk_adjusted_signal["risk_score"],
                thermal_state=self.state.thermal_state,
                bit_phase=self.state.bit_phase,
                profit_vector=profit_vector,
                metadata={
                    "processing_time": time.time() - start_time,
                    "market_regime": market_regime.value,
                    "signal_metadata": risk_adjusted_signal.get("metadata", {}),
                },
            )

            # 9. Update pipeline state
            self._update_pipeline_state(decision)

            # 10. Store decision in history
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

            logger.info(f"Trading decision: {decision.action.value} {decision.quantity:.4f} @ {decision.price:.2f}")

            return decision

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None

    def _analyze_market_regime(self, market_data: MarketData) -> MarketRegime:
        """Analyze current market regime using mathematical indicators."""
        if len(self.market_data_history) < 20:
            return MarketRegime.SIDEWAYS

        # Get recent price data
        recent_prices = [md.price for md in self.market_data_history[-20:]]
        recent_volumes = [md.volume for md in self.market_data_history[-20:]]

        # Calculate trend strength
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        price_std = np.std(recent_prices)
        volume_avg = np.mean(recent_volumes)

        # Volatility analysis
        volatility = market_data.volatility
        high_vol_threshold = self.risk_params.volatility_threshold

        # Regime classification logic
        if volatility > high_vol_threshold:
            return MarketRegime.VOLATILE
        elif abs(trend_slope) < price_std * 0.1:
            return MarketRegime.SIDEWAYS
        elif trend_slope > 0:
            return MarketRegime.TRENDING_UP
        elif trend_slope < 0:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.CALM

    def _determine_optimal_strategy(self, regime: MarketRegime, market_data: MarketData) -> StrategyBranch:
        """Determine optimal strategy based on market regime."""
        strategy_map = {
            MarketRegime.TRENDING_UP: StrategyBranch.MOMENTUM,
            MarketRegime.TRENDING_DOWN: StrategyBranch.MOMENTUM,
            MarketRegime.SIDEWAYS: StrategyBranch.MEAN_REVERSION,
            MarketRegime.VOLATILE: StrategyBranch.SCALPING,
            MarketRegime.CALM: StrategyBranch.GRID,
        }

        base_strategy = strategy_map.get(regime, StrategyBranch.MOMENTUM)

        # Strategy refinement based on additional factors
        if market_data.volume > 1.5 * np.mean([md.volume for md in self.market_data_history[-10:]]):
            # High volume - prefer momentum or arbitrage
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                return StrategyBranch.MOMENTUM
            else:
                return StrategyBranch.ARBITRAGE

        return base_strategy

    def _update_thermal_state(self, market_data: MarketData):
        """Update thermal state based on market conditions."""
        # Thermal state logic based on entropy and volatility
        entropy = market_data.entropy_level
        volatility = market_data.volatility

        if entropy > 6.0 or volatility > 0.8:
            self.state.thermal_state = ThermalState.HOT
            self.state.bit_phase = BitPhase.THIRTY_TWO_BIT
        elif entropy > 4.0 or volatility > 0.5:
            self.state.thermal_state = ThermalState.WARM
            self.state.bit_phase = BitPhase.SIXTEEN_BIT
        else:
            self.state.thermal_state = ThermalState.COOL
            self.state.bit_phase = BitPhase.EIGHT_BIT

    async def _generate_signal(
        self, market_data: MarketData, strategy: StrategyBranch
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on strategy."""
        signal_generators = {
            StrategyBranch.MOMENTUM: self._momentum_signal,
            StrategyBranch.MEAN_REVERSION: self._mean_reversion_signal,
            StrategyBranch.ARBITRAGE: self._arbitrage_signal,
            StrategyBranch.SCALPING: self._scalping_signal,
            StrategyBranch.SWING: self._swing_signal,
            StrategyBranch.GRID: self._grid_signal,
        }

        generator = signal_generators.get(strategy)
        if not generator:
            return None

        return generator(market_data)

    def _select_vectorization_mode(self, strategy: StrategyBranch, market_data: MarketData) -> VectorizationMode:
        """Select appropriate vectorization mode."""
        if strategy in [StrategyBranch.SCALPING, StrategyBranch.ARBITRAGE]:
            return VectorizationMode.HIGH_FREQUENCY
        elif strategy == StrategyBranch.MOMENTUM:
            return VectorizationMode.MOMENTUM_BASED
        elif strategy == StrategyBranch.MEAN_REVERSION:
            return VectorizationMode.MEAN_REVERSION
        else:
            return VectorizationMode.ADAPTIVE

    def _create_profit_vector(self, signal: Dict[str, Any], market_data: MarketData) -> ProfitVector:
        """Create profit vector for the signal."""
        mode = self._select_vectorization_mode(self.state.active_strategy, market_data)

        # Use profit vectorization system
        vector_input = {
            "price": market_data.price,
            "volume": market_data.volume,
            "volatility": market_data.volatility,
            "signal_strength": signal["confidence"],
            "quantity": signal["quantity"],
            "thermal_state": self.state.thermal_state,
            "bit_phase": self.state.bit_phase,
        }

        return self.profit_vectorizer.calculate_profit_vector(vector_input, mode)

    def _mean_reversion_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate mean reversion signal."""
        if len(self.market_data_history) < 20:
            return None

        recent_prices = [md.price for md in self.market_data_history[-20:]]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)

        current_price = market_data.price
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0

        # Mean reversion logic
        if z_score > 2.0:  # Price too high
            return {
                "action": "SELL",
                "quantity": self._calculate_position_size(market_data, "SELL"),
                "confidence": min(abs(z_score) / 3.0, 1.0),
                "profit_potential": abs(z_score) * 0.01,
                "risk_score": 0.3,
                "metadata": {"z_score": z_score, "mean_price": mean_price},
            }
        elif z_score < -2.0:  # Price too low
            return {
                "action": "BUY",
                "quantity": self._calculate_position_size(market_data, "BUY"),
                "confidence": min(abs(z_score) / 3.0, 1.0),
                "profit_potential": abs(z_score) * 0.01,
                "risk_score": 0.3,
                "metadata": {"z_score": z_score, "mean_price": mean_price},
            }

        return None

    def _momentum_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate momentum signal."""
        if len(self.market_data_history) < 10:
            return None

        recent_prices = [md.price for md in self.market_data_history[-10:]]
        short_ma = np.mean(recent_prices[-5:])
        long_ma = np.mean(recent_prices)

        momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
        volume_surge = market_data.volume / np.mean([md.volume for md in self.market_data_history[-5:]])

        # Momentum logic
        if momentum > 0.01 and volume_surge > 1.2:  # Strong upward momentum
            return {
                "action": "BUY",
                "quantity": self._calculate_position_size(market_data, "BUY"),
                "confidence": min(momentum * 10, 1.0),
                "profit_potential": momentum * 2,
                "risk_score": 0.4,
                "metadata": {"momentum": momentum, "volume_surge": volume_surge},
            }
        elif momentum < -0.01 and volume_surge > 1.2:  # Strong downward momentum
            return {
                "action": "SELL",
                "quantity": self._calculate_position_size(market_data, "SELL"),
                "confidence": min(abs(momentum) * 10, 1.0),
                "profit_potential": abs(momentum) * 2,
                "risk_score": 0.4,
                "metadata": {"momentum": momentum, "volume_surge": volume_surge},
            }

        return None

    def _arbitrage_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate arbitrage signal."""
        # Simplified arbitrage logic (would need multiple exchanges in real implementation)
        if market_data.bid and market_data.ask:
            spread = (market_data.ask - market_data.bid) / market_data.price
            
            if spread > 0.005:  # Minimum profitable spread
                return {
                    "action": "BUY",  # Buy at bid, sell at ask
                    "quantity": self._calculate_position_size(market_data, "BUY") * 0.5,
                    "confidence": min(spread * 100, 1.0),
                    "profit_potential": spread,
                    "risk_score": 0.2,
                    "metadata": {"spread": spread, "bid": market_data.bid, "ask": market_data.ask},
                }

        return None

    def _scalping_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate scalping signal."""
        if len(self.market_data_history) < 5:
            return None

        # Very short-term price movement analysis
        recent_prices = [md.price for md in self.market_data_history[-5:]]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        volatility = market_data.volatility

        # Scalping logic - capitalize on small movements
        if abs(price_change) > 0.002 and volatility > 0.3:
            action = "BUY" if price_change > 0 else "SELL"
            return {
                "action": action,
                "quantity": self._calculate_position_size(market_data, action) * 2,  # Higher frequency
                "confidence": min(abs(price_change) * 100, 1.0),
                "profit_potential": abs(price_change) * 0.5,
                "risk_score": 0.6,
                "metadata": {"price_change": price_change, "volatility": volatility},
            }

        return None

    def _swing_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate swing trading signal."""
        if len(self.market_data_history) < 50:
            return None

        # Medium-term trend analysis
        prices = [md.price for md in self.market_data_history[-50:]]
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        current_price = market_data.price
        
        # Support and resistance levels
        recent_highs = [max(prices[i:i+10]) for i in range(0, len(prices)-10, 10)]
        recent_lows = [min(prices[i:i+10]) for i in range(0, len(prices)-10, 10)]
        
        resistance = np.mean(recent_highs) if recent_highs else current_price
        support = np.mean(recent_lows) if recent_lows else current_price

        # Swing logic
        if current_price <= support * 1.02 and trend > 0:  # Near support with uptrend
            return {
                "action": "BUY",
                "quantity": self._calculate_position_size(market_data, "BUY"),
                "confidence": 0.7,
                "profit_potential": (resistance - current_price) / current_price,
                "risk_score": 0.4,
                "metadata": {"support": support, "resistance": resistance, "trend": trend},
            }
        elif current_price >= resistance * 0.98 and trend < 0:  # Near resistance with downtrend
            return {
                "action": "SELL",
                "quantity": self._calculate_position_size(market_data, "SELL"),
                "confidence": 0.7,
                "profit_potential": (current_price - support) / current_price,
                "risk_score": 0.4,
                "metadata": {"support": support, "resistance": resistance, "trend": trend},
            }

        return None

    def _grid_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate grid trading signal."""
        if len(self.market_data_history) < 20:
            return None

        # Grid trading logic
        recent_prices = [md.price for md in self.market_data_history[-20:]]
        price_range = max(recent_prices) - min(recent_prices)
        grid_size = price_range / 10  # 10 grid levels

        current_price = market_data.price
        base_price = np.mean(recent_prices)

        # Determine grid position
        grid_level = round((current_price - base_price) / grid_size)

        # Grid logic - buy low, sell high within range
        if grid_level <= -2:  # Lower grid levels
            return {
                "action": "BUY",
                "quantity": self._calculate_position_size(market_data, "BUY") * 0.8,
                "confidence": 0.6,
                "profit_potential": 0.01,
                "risk_score": 0.3,
                "metadata": {"grid_level": grid_level, "grid_size": grid_size},
            }
        elif grid_level >= 2:  # Upper grid levels
            return {
                "action": "SELL",
                "quantity": self._calculate_position_size(market_data, "SELL") * 0.8,
                "confidence": 0.6,
                "profit_potential": 0.01,
                "risk_score": 0.3,
                "metadata": {"grid_level": grid_level, "grid_size": grid_size},
            }

        return None

    def _apply_risk_management(self, signal: Dict[str, Any], market_data: MarketData) -> Optional[Dict[str, Any]]:
        """Apply risk management rules to the signal."""
        if not signal:
            return None

        # Check daily loss limit
        daily_pnl = self._calculate_daily_pnl()
        if daily_pnl < -self.risk_params.max_daily_loss * self.initial_capital:
            logger.warning("Daily loss limit reached, blocking signal")
            return None

        # Check volatility threshold
        if market_data.volatility > self.risk_params.volatility_threshold:
            signal["quantity"] *= 0.5  # Reduce position size
            signal["risk_score"] *= 1.5

        # Position sizing
        max_position = self.risk_params.max_position_size * self.state.current_capital
        signal_value = signal["quantity"] * market_data.price

        if signal_value > max_position:
            signal["quantity"] = max_position / market_data.price

        # Risk-reward ratio
        profit_potential = signal.get("profit_potential", 0)
        risk_score = signal.get("risk_score", 0.5)

        if profit_potential > 0 and risk_score > 0:
            risk_reward_ratio = profit_potential / risk_score
            if risk_reward_ratio < 1.5:  # Minimum risk-reward ratio
                return None

        # Update risk score based on current portfolio
        portfolio_risk = self._calculate_portfolio_risk()
        signal["risk_score"] = min(signal["risk_score"] + portfolio_risk, 1.0)

        return signal

    def _calculate_position_size(self, market_data: MarketData, action: str) -> float:
        """Calculate appropriate position size."""
        base_size = self.risk_params.max_position_size * self.state.current_capital
        price = market_data.price
        volatility_adjustment = 1.0 - market_data.volatility

        # Adjust for volatility
        adjusted_size = base_size * volatility_adjustment / price

        # Thermal state adjustment
        thermal_multiplier = {
            ThermalState.COOL: 0.8,
            ThermalState.WARM: 1.0,
            ThermalState.HOT: 1.2,
        }.get(self.state.thermal_state, 1.0)

        return adjusted_size * thermal_multiplier

    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk level."""
        if len(self.decision_history) < 5:
            return 0.0

        recent_decisions = self.decision_history[-5:]
        risk_scores = [d.risk_score for d in recent_decisions]
        return np.mean(risk_scores)

    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        today_start = time.time() - 24 * 3600  # 24 hours ago
        today_decisions = [d for d in self.decision_history if d.timestamp >= today_start]

        pnl = 0.0
        for decision in today_decisions:
            if decision.action == TradingAction.BUY:
                pnl -= decision.quantity * decision.price
            elif decision.action == TradingAction.SELL:
                pnl += decision.quantity * decision.price

        return pnl

    def _update_pipeline_state(self, decision: TradingDecision) -> None:
        """Update pipeline state after decision."""
        self.state.timestamp = decision.timestamp
        self.state.total_trades += 1

        # Update capital (simplified)
        if decision.action == TradingAction.BUY:
            self.state.current_capital -= decision.quantity * decision.price
        elif decision.action == TradingAction.SELL:
            self.state.current_capital += decision.quantity * decision.price

        # Update performance metrics
        self._update_pipeline_metrics()

    def _update_pipeline_metrics(self) -> None:
        """Update performance metrics."""
        if len(self.decision_history) < 2:
            return

        # Calculate basic metrics
        total_trades = len(self.decision_history)
        profitable_trades = len([d for d in self.decision_history if d.profit_potential > 0])

        self.performance_metrics["win_rate"] = profitable_trades / total_trades if total_trades > 0 else 0
        self.performance_metrics["total_return"] = (self.state.current_capital - self.initial_capital) / self.initial_capital

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        return {
            "state": {
                "symbol": self.symbol,
                "current_capital": self.state.current_capital,
                "total_trades": self.state.total_trades,
                "active_strategy": self.state.active_strategy.value,
                "market_regime": self.state.market_regime.value,
                "thermal_state": self.state.thermal_state.value,
                "bit_phase": self.state.bit_phase.value,
            },
            "performance": self.performance_metrics,
            "risk_parameters": {
                "max_position_size": self.risk_params.max_position_size,
                "stop_loss_pct": self.risk_params.stop_loss_pct,
                "take_profit_pct": self.risk_params.take_profit_pct,
            },
            "history_length": {
                "market_data": len(self.market_data_history),
                "decisions": len(self.decision_history),
            },
        }


def create_trading_pipeline(
    symbol: str = "BTCUSDT", initial_capital: float = 10000.0
) -> CleanTradingPipeline:
    """Create a new trading pipeline instance."""
    return CleanTradingPipeline(symbol=symbol, initial_capital=initial_capital)


async def run_trading_simulation(
    pipeline: CleanTradingPipeline, market_data_stream: List[MarketData]
) -> Dict[str, Any]:
    """Run a trading simulation with provided market data."""
    decisions = []
    
    for market_data in market_data_stream:
        decision = await pipeline.process_market_data(market_data)
        if decision:
            decisions.append(decision)
    
    return {
        "total_decisions": len(decisions),
        "pipeline_summary": pipeline.get_pipeline_summary(),
        "final_capital": pipeline.state.current_capital,
        "total_return": (pipeline.state.current_capital - pipeline.initial_capital) / pipeline.initial_capital,
    }
