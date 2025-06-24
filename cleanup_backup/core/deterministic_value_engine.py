#!/usr/bin/env python3
"""Deterministic Value Engine for Schwabot Trading System.

This module implements the complete deterministic mathematics for answering
the three critical questions:

1. WHEN should we make the move? (Timing determinism)
2. IF we should make the move? (Conditional determinism)
3. WHAT KIND of move should be made? (Strategy determinism)

It integrates with the randomized matrix system for portfolio substitution
(USDC ↔ XRP ↔ BTC ↔ ETH) and handles 4-bit/8-bit/42-bit phase switching
with mathematical precision and fault tolerance.

Mathematical Foundation:
- Execution Confidence: Ξ = (T·Δθ) + (ε·σ_f) + τ_p
- Entry Score: Es = H(1-Dp)LP̂
- Strategy Selection: S = argmax(Σ_i w_i · R_i)
- Portfolio Allocation: A = M_phase @ [USDC, XRP, BTC, ETH]
"""

from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of trading decisions."""

    WHEN_TO_MOVE = "when_to_move"
    IF_TO_MOVE = "if_to_move"
    WHAT_KIND_OF_MOVE = "what_kind_of_move"


class PhaseMode(Enum):
    """Bit-depth phase processing modes."""

    FOUR_BIT = 4
    EIGHT_BIT = 8
    FORTY_TWO_BIT = 42


class StrategyType(Enum):
    """Available strategy types."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    HEDGING = "hedging"
    VAULT_ACCUMULATION = "vault_accumulation"


class AssetType(Enum):
    """Supported asset types."""

    USDC = "USDC"
    XRP = "XRP"
    BTC = "BTC"
    ETH = "ETH"


@dataclass
class MarketState:
    """Complete market state for deterministic calculations."""

    # Price data
    prices: Dict[str, float] = field(default_factory=dict)
    price_deltas: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    spreads: Dict[str, float] = field(default_factory=dict)

    # Technical indicators
    momentum: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # System state
    entropy_levels: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    phase_coherence: float = 0.0
    tick_harmony: float = 0.0
    phase_drift: float = 0.0

    # Portfolio state
    positions: Dict[str, float] = field(default_factory=dict)
    available_capital: float = 0.0
    unrealized_pnl: float = 0.0

    # Timing
    timestamp: float = field(default_factory=time.time)
    last_trade_time: float = 0.0
    market_hours_active: bool = True


@dataclass
class DeterministicDecision:
    """Result of deterministic value calculation."""

    decision_type: DecisionType
    timing_score: float  # 0.0 to 1.0 - when to move
    conditional_score: float  # 0.0 to 1.0 - if to move
    strategy_weights: Dict[StrategyType, float]  # what kind of move
    asset_allocation: Dict[AssetType, float]  # portfolio allocation

    # Supporting metrics
    execution_confidence: float  # Ξ
    entry_score: float  # Es
    phase_mode: PhaseMode
    expected_return: float
    risk_adjustment: float

    # Execution parameters
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_time: Optional[float] = None

    # Metadata
    calculated_at: float = field(default_factory=time.time)
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    supporting_evidence: List[str] = field(default_factory=list)


class DeterministicValueEngine:
    """Core engine for deterministic trading value calculations."""

    def __init__(self) -> None:
        """Initialize the deterministic value engine."""
        # Mathematical parameters
        self.xi_weights = {"T_delta_theta": 0.4, "epsilon_sigma": 0.3, "tau_p": 0.3}
        self.entry_score_weights = {
            "harmony": 0.3,
            "drift": 0.2,
            "liquidity": 0.25,
            "profit": 0.25,
        }

        # Phase switching thresholds
        self.phase_thresholds = {
            PhaseMode.FOUR_BIT: {"min_entropy": 2.0, "max_complexity": 0.3},
            PhaseMode.EIGHT_BIT: {"min_entropy": 4.0, "max_complexity": 0.6},
            PhaseMode.FORTY_TWO_BIT: {"min_entropy": 6.0, "max_complexity": 1.0},
        }

        # Randomized portfolio substitution matrix
        self.portfolio_substitution_matrix = np.array(
            [
                # USDC  XRP   BTC   ETH
                [0.70, 0.15, 0.10, 0.05],  # Conservative (4-bit)
                [0.40, 0.30, 0.20, 0.10],  # Balanced (8-bit)
                [0.20, 0.25, 0.35, 0.20],  # Aggressive (42-bit)
                [0.50, 0.20, 0.20, 0.10],  # Defensive fallback
            ]
        )

        # Strategy scoring functions
        self.strategy_scorers = {
            StrategyType.MOMENTUM: self._score_momentum_strategy,
            StrategyType.MEAN_REVERSION: self._score_mean_reversion_strategy,
            StrategyType.BREAKOUT: self._score_breakout_strategy,
            StrategyType.ARBITRAGE: self._score_arbitrage_strategy,
            StrategyType.HEDGING: self._score_hedging_strategy,
            StrategyType.VAULT_ACCUMULATION: self._score_vault_strategy,
        }

        # Decision history for learning
        self.decision_history: List[DeterministicDecision] = []
        self.performance_tracker: Dict[str, List[float]] = {}

        logger.info("🎯 Deterministic Value Engine initialized")

    def calculate_deterministic_decision(
        self, market_state: MarketState
    ) -> DeterministicDecision:
        """Calculate complete deterministic trading decision."""
        start_time = time.time()

        try:
            # 1. WHEN - Calculate timing score
            timing_score = self._calculate_timing_determinism(market_state)

            # 2. IF - Calculate conditional score
            conditional_score = self._calculate_conditional_determinism(market_state)

            # 3. WHAT KIND - Calculate strategy weights
            strategy_weights = self._calculate_strategy_determinism(market_state)

            # 4. Calculate execution confidence (Ξ)
            execution_confidence = self._calculate_execution_confidence(market_state)

            # 5. Calculate entry score (Es)
            entry_score = self._calculate_entry_score(market_state)

            # 6. Determine optimal phase mode
            phase_mode = self._determine_phase_mode(market_state)

            # 7. Calculate asset allocation using randomized matrix
            asset_allocation = self._calculate_asset_allocation(
                phase_mode, strategy_weights, market_state
            )

            # 8. Calculate position sizing and risk parameters
            position_size = self._calculate_position_size(
                execution_confidence, entry_score, market_state
            )
            stop_loss, take_profit = self._calculate_risk_parameters(
                market_state, strategy_weights
            )

            # 9. Estimate expected return and risk
            expected_return = self._calculate_expected_return(
                strategy_weights, asset_allocation, market_state
            )
            risk_adjustment = self._calculate_risk_adjustment(
                market_state, asset_allocation
            )

            # Create decision object
            decision = DeterministicDecision(
                decision_type=DecisionType.WHAT_KIND_OF_MOVE,  # Composite decision
                timing_score=timing_score,
                conditional_score=conditional_score,
                strategy_weights=strategy_weights,
                asset_allocation=asset_allocation,
                execution_confidence=execution_confidence,
                entry_score=entry_score,
                phase_mode=phase_mode,
                expected_return=expected_return,
                risk_adjustment=risk_adjustment,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_hold_time=self._calculate_max_hold_time(strategy_weights),
                supporting_evidence=self._generate_supporting_evidence(
                    market_state, execution_confidence, entry_score
                ),
            )

            # Store decision for learning
            self.decision_history.append(decision)
            self._update_performance_tracking(decision)

            calculation_time = time.time() - start_time
            logger.debug(
                f"🎯 Deterministic decision calculated in {calculation_time:.4f}s"
            )

            return decision

        except Exception as e:
            logger.error(f"❌ Deterministic calculation failed: {e}")
            # Return safe fallback decision
            return self._create_fallback_decision(market_state)

    def _calculate_timing_determinism(self, market_state: MarketState) -> float:
        """Calculate WHEN to make the move (timing determinism)."""

        # Time-based factors
        time_since_last_trade = market_state.timestamp - market_state.last_trade_time
        optimal_trade_interval = 3600.0  # 1 hour base interval
        time_factor = min(time_since_last_trade / optimal_trade_interval, 1.0)

        # Market rhythm alignment
        market_cycle_phase = (market_state.timestamp % 86400) / 86400  # Daily cycle
        rhythm_alignment = np.cos(2 * np.pi * market_cycle_phase) * 0.5 + 0.5

        # Volatility timing
        avg_volatility = (
            np.mean(list(market_state.volatility.values()))
            if market_state.volatility
            else 0.5
        )
        volatility_timing = np.tanh(avg_volatility * 2) * 0.7 + 0.3  # 0.3 to 1.0 range

        # Momentum alignment
        momentum_values = list(market_state.momentum.values())
        momentum_coherence = 1.0 - np.std(momentum_values) if momentum_values else 0.5

        # Phase coherence timing
        coherence_timing = market_state.phase_coherence

        # Composite timing score
        timing_score = (
            time_factor * 0.2
            + rhythm_alignment * 0.2
            + volatility_timing * 0.25
            + momentum_coherence * 0.2
            + coherence_timing * 0.15
        )

        return np.clip(timing_score, 0.0, 1.0)

    def _calculate_conditional_determinism(self, market_state: MarketState) -> float:
        """Calculate IF we should make the move (conditional determinism)."""

        # Market conditions check
        market_conditions_score = 0.0

        # 1. Liquidity condition
        volumes = list(market_state.volumes.values())
        avg_volume = np.mean(volumes) if volumes else 0
        liquidity_score = min(avg_volume / 1000000, 1.0)  # Normalize to 1M volume

        # 2. Spread condition
        spreads = list(market_state.spreads.values())
        avg_spread = np.mean(spreads) if spreads else 0.01
        spread_score = max(0, 1.0 - avg_spread * 100)  # Penalize high spreads

        # 3. Volatility condition
        volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        volatility_score = np.exp(
            -((avg_volatility - 0.02) ** 2) / 0.001
        )  # Optimal around 2%

        # 4. Market hours condition
        market_hours_score = 1.0 if market_state.market_hours_active else 0.3

        # 5. Portfolio health condition
        portfolio_health = 1.0
        if market_state.available_capital > 0:
            utilization = (
                sum(abs(pos) for pos in market_state.positions.values())
                / market_state.available_capital
            )
            portfolio_health = max(0, 1.0 - utilization) if utilization < 1 else 0

        # 6. Risk tolerance condition
        risk_score = 1.0
        if market_state.unrealized_pnl < 0:
            drawdown = abs(market_state.unrealized_pnl) / max(
                market_state.available_capital, 1
            )
            risk_score = max(
                0, 1.0 - drawdown * 5
            )  # Reduce willingness at 20% drawdown

        # Composite conditional score
        conditional_score = (
            liquidity_score * 0.2
            + spread_score * 0.15
            + volatility_score * 0.2
            + market_hours_score * 0.1
            + portfolio_health * 0.2
            + risk_score * 0.15
        )

        return np.clip(conditional_score, 0.0, 1.0)

    def _calculate_strategy_determinism(
        self, market_state: MarketState
    ) -> Dict[StrategyType, float]:
        """Calculate WHAT KIND of move should be made (strategy determinism)."""

        strategy_scores = {}

        # Score each strategy type
        for strategy_type in StrategyType:
            if strategy_type in self.strategy_scorers:
                try:
                    score = self.strategy_scorers[strategy_type](market_state)
                    strategy_scores[strategy_type] = np.clip(score, 0.0, 1.0)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Strategy scoring failed for {strategy_type}: {e}"
                    )
                    strategy_scores[strategy_type] = 0.0
            else:
                strategy_scores[strategy_type] = 0.0

        # Normalize to sum to 1.0
        total_score = sum(strategy_scores.values())
        if total_score > 0:
            strategy_scores = {k: v / total_score for k, v in strategy_scores.items()}
        else:
            # Equal weighting fallback
            num_strategies = len(StrategyType)
            strategy_scores = {k: 1.0 / num_strategies for k in StrategyType}

        return strategy_scores

    def _calculate_execution_confidence(self, market_state: MarketState) -> float:
        """Calculate execution confidence scalar: Ξ = (T·Δθ) + (ε·σ_f) + τ_p"""

        # T·Δθ - Triplet entropy * braid angle drift
        T_entropy = (
            np.mean(list(market_state.entropy_levels.values()))
            if market_state.entropy_levels
            else 3.0
        )
        delta_theta = market_state.phase_drift
        T_delta_theta_term = T_entropy * delta_theta

        # ε·σ_f - Coherence * standard deviation of fractal loops
        epsilon = market_state.phase_coherence
        sigma_f = (
            np.std(list(market_state.price_deltas.values()))
            if market_state.price_deltas
            else 0.02
        )
        epsilon_sigma_term = epsilon * sigma_f

        # τ_p - Profit-time modifier
        time_factor = min(
            (time.time() - market_state.last_trade_time) / 3600, 1.0
        )  # Hours
        profit_factor = max(
            market_state.unrealized_pnl / max(market_state.available_capital, 1), -1.0
        )
        tau_p = np.tanh(profit_factor) * np.exp(-time_factor)

        # Weighted combination
        xi = (
            T_delta_theta_term * self.xi_weights["T_delta_theta"]
            + epsilon_sigma_term * self.xi_weights["epsilon_sigma"]
            + tau_p * self.xi_weights["tau_p"]
        )

        return xi

    def _calculate_entry_score(self, market_state: MarketState) -> float:
        """Calculate entry score: Es = H(1-Dp)LP̂"""

        # H - Tick harmony
        harmony = market_state.tick_harmony

        # Dp - Phase drift penalty
        drift_penalty = market_state.phase_drift

        # L - Liquidity score
        volumes = list(market_state.volumes.values())
        liquidity = min(np.mean(volumes) / 1000000, 1.0) if volumes else 0.5

        # P̂ - Projected profit
        price_changes = list(market_state.price_deltas.values())
        projected_profit = (
            np.mean([abs(pc) for pc in price_changes]) if price_changes else 0.01
        )

        # Composite entry score
        entry_score = harmony * (1 - drift_penalty) * liquidity * projected_profit * 100

        return np.clip(entry_score, 0.0, 1.0)

    def _determine_phase_mode(self, market_state: MarketState) -> PhaseMode:
        """Determine optimal bit-depth phase mode."""

        # Calculate system complexity
        entropy_values = list(market_state.entropy_levels.values())
        avg_entropy = np.mean(entropy_values) if entropy_values else 3.0

        price_volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(price_volatilities) if price_volatilities else 0.02

        complexity_score = (avg_entropy / 8.0) + (avg_volatility / 0.1)  # Normalized

        # Choose phase mode based on complexity
        if (
            complexity_score
            < self.phase_thresholds[PhaseMode.FOUR_BIT]["max_complexity"]
        ):
            return PhaseMode.FOUR_BIT
        elif (
            complexity_score
            < self.phase_thresholds[PhaseMode.EIGHT_BIT]["max_complexity"]
        ):
            return PhaseMode.EIGHT_BIT
        else:
            return PhaseMode.FORTY_TWO_BIT

    def _calculate_asset_allocation(
        self,
        phase_mode: PhaseMode,
        strategy_weights: Dict[StrategyType, float],
        market_state: MarketState,
    ) -> Dict[AssetType, float]:
        """Calculate asset allocation using randomized portfolio substitution matrix."""

        # Map phase mode to matrix row
        phase_to_row = {
            PhaseMode.FOUR_BIT: 0,
            PhaseMode.EIGHT_BIT: 1,
            PhaseMode.FORTY_TWO_BIT: 2,
        }

        base_row = phase_to_row.get(phase_mode, 3)  # Fallback to row 3

        # Get base allocation from matrix
        base_allocation = self.portfolio_substitution_matrix[base_row]

        # Apply strategy-specific adjustments
        strategy_adjustments = np.array([0.0, 0.0, 0.0, 0.0])  # USDC, XRP, BTC, ETH

        # Momentum strategy favors BTC/ETH
        if StrategyType.MOMENTUM in strategy_weights:
            momentum_weight = strategy_weights[StrategyType.MOMENTUM]
            strategy_adjustments += np.array([-0.1, -0.05, 0.1, 0.05]) * momentum_weight

        # Mean reversion favors stable assets
        if StrategyType.MEAN_REVERSION in strategy_weights:
            reversion_weight = strategy_weights[StrategyType.MEAN_REVERSION]
            strategy_adjustments += (
                np.array([0.15, 0.05, -0.1, -0.1]) * reversion_weight
            )

        # Vault accumulation favors USDC
        if StrategyType.VAULT_ACCUMULATION in strategy_weights:
            vault_weight = strategy_weights[StrategyType.VAULT_ACCUMULATION]
            strategy_adjustments += np.array([0.2, -0.05, -0.1, -0.05]) * vault_weight

        # Apply adjustments and normalize
        adjusted_allocation = base_allocation + strategy_adjustments
        adjusted_allocation = np.maximum(
            adjusted_allocation, 0.0
        )  # No negative allocations
        adjusted_allocation = adjusted_allocation / np.sum(
            adjusted_allocation
        )  # Normalize

        # Convert to dictionary
        asset_allocation = {
            AssetType.USDC: float(adjusted_allocation[0]),
            AssetType.XRP: float(adjusted_allocation[1]),
            AssetType.BTC: float(adjusted_allocation[2]),
            AssetType.ETH: float(adjusted_allocation[3]),
        }

        return asset_allocation

    def _calculate_position_size(
        self, execution_confidence: float, entry_score: float, market_state: MarketState
    ) -> float:
        """Calculate optimal position size based on confidence and kelly criterion."""

        # Base position size (% of available capital)
        base_size = 0.1  # 10% base allocation

        # Confidence multiplier
        confidence_multiplier = min(execution_confidence * 2, 3.0)  # Up to 3x

        # Entry score multiplier
        entry_multiplier = entry_score * 2  # Up to 2x

        # Risk adjustment based on volatility
        volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        risk_adjustment = 1.0 / (
            1.0 + avg_volatility * 10
        )  # Reduce size with volatility

        # Calculate final position size
        position_size = (
            base_size * confidence_multiplier * entry_multiplier * risk_adjustment
        )

        # Apply maximum position limit (50% of capital)
        max_position = 0.5
        position_size = min(position_size, max_position)

        return position_size

    def _calculate_risk_parameters(
        self, market_state: MarketState, strategy_weights: Dict[StrategyType, float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""

        # Get average price and volatility
        prices = list(market_state.prices.values())
        volatilities = list(market_state.volatility.values())

        if not prices or not volatilities:
            return None, None

        avg_price = np.mean(prices)
        avg_volatility = np.mean(volatilities)

        # Strategy-dependent risk parameters
        risk_multiplier = 1.0
        reward_multiplier = 2.0

        # Adjust based on dominant strategy
        max_strategy = max(strategy_weights.items(), key=lambda x: x[1])
        dominant_strategy = max_strategy[0]

        if dominant_strategy == StrategyType.MOMENTUM:
            risk_multiplier = 1.5  # Wider stops for momentum
            reward_multiplier = 3.0
        elif dominant_strategy == StrategyType.MEAN_REVERSION:
            risk_multiplier = 0.8  # Tighter stops for mean reversion
            reward_multiplier = 1.5
        elif dominant_strategy == StrategyType.BREAKOUT:
            risk_multiplier = 2.0  # Wide stops for breakouts
            reward_multiplier = 4.0

        # Calculate stop loss (below current price)
        stop_loss_distance = avg_volatility * risk_multiplier * avg_price
        stop_loss = avg_price - stop_loss_distance

        # Calculate take profit (above current price)
        take_profit_distance = avg_volatility * reward_multiplier * avg_price
        take_profit = avg_price + take_profit_distance

        return stop_loss, take_profit

    def _calculate_expected_return(
        self,
        strategy_weights: Dict[StrategyType, float],
        asset_allocation: Dict[AssetType, float],
        market_state: MarketState,
    ) -> float:
        """Calculate expected return based on strategy and allocation."""

        # Base expected returns by asset (annualized)
        base_returns = {
            AssetType.USDC: 0.03,  # 3% risk-free rate
            AssetType.XRP: 0.15,  # 15% expected return
            AssetType.BTC: 0.25,  # 25% expected return
            AssetType.ETH: 0.20,  # 20% expected return
        }

        # Strategy return multipliers
        strategy_multipliers = {
            StrategyType.MOMENTUM: 1.3,
            StrategyType.MEAN_REVERSION: 1.1,
            StrategyType.BREAKOUT: 1.5,
            StrategyType.ARBITRAGE: 1.2,
            StrategyType.HEDGING: 0.9,
            StrategyType.VAULT_ACCUMULATION: 0.8,
        }

        # Calculate weighted strategy multiplier
        weighted_multiplier = sum(
            strategy_weights[strategy] * strategy_multipliers[strategy]
            for strategy in StrategyType
        )

        # Calculate weighted asset return
        expected_asset_return = sum(
            asset_allocation[asset] * base_returns[asset] for asset in AssetType
        )

        # Apply strategy multiplier
        expected_return = expected_asset_return * weighted_multiplier

        # Adjust for market conditions
        momentum_values = list(market_state.momentum.values())
        if momentum_values:
            momentum_adjustment = np.tanh(np.mean(momentum_values)) * 0.2 + 1.0
            expected_return *= momentum_adjustment

        return expected_return

    def _calculate_risk_adjustment(
        self, market_state: MarketState, asset_allocation: Dict[AssetType, float]
    ) -> float:
        """Calculate risk adjustment factor."""

        # Volatility-based risk
        volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        volatility_risk = avg_volatility * 2  # Scale volatility

        # Concentration risk
        allocation_values = list(asset_allocation.values())
        concentration_risk = (
            max(allocation_values) - 0.25
        )  # Penalty above 25% concentration
        concentration_risk = max(concentration_risk, 0)

        # Correlation risk (simplified)
        correlation_values = list(market_state.correlations.values())
        avg_correlation = (
            np.mean([abs(c) for c in correlation_values]) if correlation_values else 0.5
        )
        correlation_risk = avg_correlation * 0.5  # High correlation increases risk

        # Total risk adjustment
        total_risk = volatility_risk + concentration_risk * 2 + correlation_risk
        risk_adjustment = np.exp(-total_risk)  # Exponential penalty

        return np.clip(risk_adjustment, 0.1, 1.0)

    def _calculate_max_hold_time(
        self, strategy_weights: Dict[StrategyType, float]
    ) -> float:
        """Calculate maximum hold time based on strategy mix."""

        # Base hold times by strategy (in hours)
        base_hold_times = {
            StrategyType.MOMENTUM: 24.0,  # 1 day
            StrategyType.MEAN_REVERSION: 4.0,  # 4 hours
            StrategyType.BREAKOUT: 48.0,  # 2 days
            StrategyType.ARBITRAGE: 1.0,  # 1 hour
            StrategyType.HEDGING: 168.0,  # 1 week
            StrategyType.VAULT_ACCUMULATION: 720.0,  # 1 month
        }

        # Calculate weighted average hold time
        weighted_hold_time = sum(
            strategy_weights[strategy] * base_hold_times[strategy]
            for strategy in StrategyType
        )

        return weighted_hold_time

    def _generate_supporting_evidence(
        self, market_state: MarketState, execution_confidence: float, entry_score: float
    ) -> List[str]:
        """Generate supporting evidence for the decision."""

        evidence = []

        # Confidence evidence
        if execution_confidence > 0.8:
            evidence.append(f"High execution confidence: {execution_confidence:.3f}")
        elif execution_confidence < 0.3:
            evidence.append(f"Low execution confidence: {execution_confidence:.3f}")

        # Entry score evidence
        if entry_score > 0.7:
            evidence.append(f"Strong entry signal: {entry_score:.3f}")
        elif entry_score < 0.3:
            evidence.append(f"Weak entry signal: {entry_score:.3f}")

        # Market condition evidence
        if market_state.tick_harmony > 0.8:
            evidence.append("Strong tick harmony detected")

        if market_state.phase_coherence > 0.7:
            evidence.append("High phase coherence")

        # Volatility evidence
        volatilities = list(market_state.volatility.values())
        if volatilities:
            avg_vol = np.mean(volatilities)
            if avg_vol > 0.05:
                evidence.append(f"Elevated volatility: {avg_vol:.3f}")
            elif avg_vol < 0.01:
                evidence.append(f"Low volatility: {avg_vol:.3f}")

        return evidence

    def _create_fallback_decision(
        self, market_state: MarketState
    ) -> DeterministicDecision:
        """Create safe fallback decision when calculation fails."""

        return DeterministicDecision(
            decision_type=DecisionType.IF_TO_MOVE,
            timing_score=0.5,
            conditional_score=0.3,  # Conservative
            strategy_weights={
                strategy: 1.0 / len(StrategyType) for strategy in StrategyType
            },
            asset_allocation={
                AssetType.USDC: 0.70,  # Safe allocation
                AssetType.XRP: 0.15,
                AssetType.BTC: 0.10,
                AssetType.ETH: 0.05,
            },
            execution_confidence=0.5,
            entry_score=0.3,
            phase_mode=PhaseMode.FOUR_BIT,  # Conservative mode
            expected_return=0.05,
            risk_adjustment=0.8,
            position_size=0.05,  # Small position
            supporting_evidence=["Fallback decision due to calculation error"],
        )

    def _update_performance_tracking(self, decision: DeterministicDecision) -> None:
        """Update performance tracking for learning."""

        decision_key = f"{decision.phase_mode.value}_{decision.decision_type.value}"

        if decision_key not in self.performance_tracker:
            self.performance_tracker[decision_key] = []

        # Track key metrics
        performance_score = (
            decision.execution_confidence * 0.4
            + decision.entry_score * 0.3
            + decision.timing_score * 0.2
            + decision.conditional_score * 0.1
        )

        self.performance_tracker[decision_key].append(performance_score)

        # Keep only recent history
        if len(self.performance_tracker[decision_key]) > 100:
            self.performance_tracker[decision_key] = self.performance_tracker[
                decision_key
            ][-100:]

    # Strategy scoring functions
    def _score_momentum_strategy(self, market_state: MarketState) -> float:
        """Score momentum strategy based on market conditions."""
        momentum_values = list(market_state.momentum.values())
        if not momentum_values:
            return 0.5

        avg_momentum = np.mean(momentum_values)
        momentum_consistency = 1.0 - np.std(momentum_values) / (
            np.mean(np.abs(momentum_values)) + 1e-8
        )

        return (np.tanh(abs(avg_momentum) * 5) + momentum_consistency) / 2

    def _score_mean_reversion_strategy(self, market_state: MarketState) -> float:
        """Score mean reversion strategy."""
        price_deltas = list(market_state.price_deltas.values())
        if not price_deltas:
            return 0.5

        # Look for overextension
        avg_delta = np.mean(price_deltas)
        volatilities = list(market_state.volatility.values())
        avg_volatility = np.mean(volatilities) if volatilities else 0.02

        overextension = abs(avg_delta) / (avg_volatility + 1e-8)
        return min(overextension / 2, 1.0)

    def _score_breakout_strategy(self, market_state: MarketState) -> float:
        """Score breakout strategy."""
        volatilities = list(market_state.volatility.values())
        volumes = list(market_state.volumes.values())

        if not volatilities or not volumes:
            return 0.5

        # Breakout = low volatility + high volume
        avg_volatility = np.mean(volatilities)
        avg_volume = np.mean(volumes)

        low_vol_score = np.exp(-avg_volatility * 50)  # Favor low volatility
        high_vol_score = np.tanh(avg_volume / 1000000)  # Favor high volume

        return (low_vol_score + high_vol_score) / 2

    def _score_arbitrage_strategy(self, market_state: MarketState) -> float:
        """Score arbitrage strategy."""
        spreads = list(market_state.spreads.values())
        if not spreads:
            return 0.3

        # Arbitrage favors wide spreads
        avg_spread = np.mean(spreads)
        return min(avg_spread * 100, 1.0)  # Scale spread to 0-1

    def _score_hedging_strategy(self, market_state: MarketState) -> float:
        """Score hedging strategy."""
        correlation_values = list(market_state.correlations.values())
        if not correlation_values:
            return 0.5

        # Hedging is valuable when correlations are unstable
        correlation_stability = 1.0 - np.std(correlation_values)
        return 1.0 - correlation_stability

    def _score_vault_strategy(self, market_state: MarketState) -> float:
        """Score vault accumulation strategy."""
        # Vault strategy is good when other strategies score low
        # and market conditions are uncertain

        uncertainty = 0.0

        # High volatility increases uncertainty
        volatilities = list(market_state.volatility.values())
        if volatilities:
            avg_volatility = np.mean(volatilities)
            uncertainty += min(avg_volatility * 10, 0.5)

        # Low liquidity increases uncertainty
        volumes = list(market_state.volumes.values())
        if volumes:
            avg_volume = np.mean(volumes)
            if avg_volume < 500000:  # Low volume threshold
                uncertainty += 0.3

        # Negative P&L increases vault appeal
        if market_state.unrealized_pnl < 0:
            uncertainty += min(
                abs(market_state.unrealized_pnl)
                / max(market_state.available_capital, 1),
                0.4,
            )

        return min(uncertainty, 1.0)


# Factory functions
def create_deterministic_value_engine() -> DeterministicValueEngine:
    """Create and configure deterministic value engine."""
    return DeterministicValueEngine()


def calculate_trading_decision(market_data: Dict[str, Any]) -> DeterministicDecision:
    """Main function to calculate deterministic trading decision."""

    # Convert market data to MarketState
    market_state = MarketState(
        prices=market_data.get("prices", {}),
        price_deltas=market_data.get("price_deltas", {}),
        volumes=market_data.get("volumes", {}),
        spreads=market_data.get("spreads", {}),
        momentum=market_data.get("momentum", {}),
        volatility=market_data.get("volatility", {}),
        correlations=market_data.get("correlations", {}),
        entropy_levels=market_data.get("entropy_levels", {}),
        confidence_scores=market_data.get("confidence_scores", {}),
        phase_coherence=market_data.get("phase_coherence", 0.5),
        tick_harmony=market_data.get("tick_harmony", 0.5),
        phase_drift=market_data.get("phase_drift", 0.1),
        positions=market_data.get("positions", {}),
        available_capital=market_data.get("available_capital", 10000.0),
        unrealized_pnl=market_data.get("unrealized_pnl", 0.0),
        timestamp=time.time(),
        last_trade_time=market_data.get("last_trade_time", 0.0),
        market_hours_active=market_data.get("market_hours_active", True),
    )

    # Create engine and calculate decision
    engine = create_deterministic_value_engine()
    return engine.calculate_deterministic_decision(market_state)


if __name__ == "__main__":
    # Example usage
    sample_market_data = {
        "prices": {"BTC": 45000.0, "ETH": 3000.0, "XRP": 0.6},
        "price_deltas": {"BTC": 0.02, "ETH": 0.015, "XRP": 0.03},
        "volumes": {"BTC": 1000000, "ETH": 800000, "XRP": 2000000},
        "spreads": {"BTC": 0.001, "ETH": 0.002, "XRP": 0.0015},
        "volatility": {"BTC": 0.03, "ETH": 0.025, "XRP": 0.04},
        "entropy_levels": {"price_entropy": 4.5, "volume_entropy": 3.8},
        "confidence_scores": {"execution_confidence": 0.75},
        "phase_coherence": 0.8,
        "tick_harmony": 0.7,
        "phase_drift": 0.15,
        "available_capital": 10000.0,
        "unrealized_pnl": 150.0,
    }

    decision = calculate_trading_decision(sample_market_data)

    print(f"🎯 Deterministic Decision:")
    print(f"   Timing Score: {decision.timing_score:.3f}")
    print(f"   Conditional Score: {decision.conditional_score:.3f}")
    print(f"   Execution Confidence: {decision.execution_confidence:.3f}")
    print(f"   Entry Score: {decision.entry_score:.3f}")
    print(f"   Phase Mode: {decision.phase_mode.value}-bit")
    print(f"   Position Size: {decision.position_size:.3f}")
    print(f"   Expected Return: {decision.expected_return:.3f}")
    print(f"   Asset Allocation: {decision.asset_allocation}")
    print(
        f"   Top Strategy: {max(decision.strategy_weights.items(), key=lambda x: x[1])}"
    )
