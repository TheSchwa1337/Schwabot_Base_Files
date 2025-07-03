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
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .clean_profit_vectorization import CleanProfitVectorization, ProfitVector, VectorizationMode
from .strategy_bit_mapper import StrategyBitMapper, ExpansionMode
from .portfolio_tracker import PortfolioTracker # Import the existing PortfolioTracker
from .ccxt_trading_executor import CCXTTradingExecutor, IntegratedTradingSignal, TradingPair # Import CCXT executor components
from .phase_bit_integration import phase_bit_integration # Import phase bit integration for dualistic state analysis

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
    FERRIS_WHEEL = "ferris_wheel" # Add Ferris Wheel as a strategy branch


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
        matrix_dir: Union[str, Path] = "data/matrices" # Added matrix_dir for StrategyBitMapper
    ):
        """Initialize the trading pipeline."""
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()

        # Initialize mathematical foundation
        self.math_foundation = CleanMathFoundation()

        # Initialize profit vectorization
        self.profit_vectorizer = CleanProfitVectorization()

        # Initialize StrategyBitMapper for Ferris Wheel and other expansions
        self.matrix_dir = Path(matrix_dir)
        self.strategy_bit_mapper = StrategyBitMapper(matrix_dir=self.matrix_dir)

        # Initialize PortfolioTracker (assuming it handles real portfolio data eventually)
        self.portfolio_tracker = PortfolioTracker() # Placeholder until actual implementation is verified

        # Initialize CCXTTradingExecutor (needs a proper config for live trading)
        self.ccxt_executor = CCXTTradingExecutor(config={
            "exchange": "binance", # Example: configure for your exchange
            "apiKey": "YOUR_API_KEY",
            "secret": "YOUR_SECRET_KEY",
        })

        # Initialize PhaseBitIntegration for dualistic state analysis
        self.phase_bit_integration = phase_bit_integration

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

            # 3. Determine optimal strategy based on market regime OR Ferris Wheel
            # For now, we'll use a simple approach to derive a base_strategy_id
            # from the total trades, and then let Ferris Wheel expand it.
            # In a real scenario, this 'base_strategy_id' might come from
            # a Soulprint match, or other complex logic.
            base_strategy_id = self.state.total_trades % (1 << 4) # Use 4 bits for base, can be adjusted

            # Always expand strategy bits with Ferris Wheel mode on every tick
            expanded_strategy_id = self.strategy_bit_mapper.expand_strategy_bits(
                base_strategy_id,
                target_bits=8, # Or 16, 256 as per your system's needs for the block
                mode=ExpansionMode.FERRIS_WHEEL
            )
            # This expanded ID now represents the "turn" of the Ferris Wheel

            # Now, select the asset and formulate the trading signal based on the expanded_strategy_id
            trading_signal, selected_asset_symbol = self._determine_asset_and_signal(
                expanded_strategy_id, market_data, self.state.active_strategy
            )

            # Update active strategy in pipeline state for tracking
            self.state.active_strategy = StrategyBranch.FERRIS_WHEEL # Indicate Ferris Wheel is driving

            if not trading_signal:
                return None

            # 6. Apply risk management (signal is adjusted in _determine_asset_and_signal)
            risk_adjusted_signal = trading_signal # Assume risk logic is now within signal determination

            if not risk_adjusted_signal:
                return None

            # 7. Create profit vector
            profit_vector = self._create_profit_vector(
                risk_adjusted_signal, market_data
            ) # Use risk_adjusted_signal data

            # 8. Execute trade via CCXT
            execution_result = await self.ccxt_executor.execute_signal(risk_adjusted_signal)

            if not execution_result.executed:
                logger.warning(f"Trade execution failed: {execution_result.error_message}")
                return None

            # 9. Make final trading decision based on execution result
            decision = TradingDecision(
                timestamp=execution_result.timestamp,
                symbol=selected_asset_symbol, # Use the symbol from the selected asset
                action=TradingAction(risk_adjusted_signal.recommended_action.upper()),
                quantity=float(execution_result.fill_amount),
                price=float(execution_result.fill_price),
                confidence=float(risk_adjusted_signal.confidence_score),
                strategy_branch=self.state.active_strategy,
                profit_potential=float(risk_adjusted_signal.profit_potential),
                risk_score=float(risk_adjusted_signal.risk_assessment.get("overall_risk", 0.5)),
                thermal_state=self.state.thermal_state,
                bit_phase=self.state.bit_phase,
                profit_vector=profit_vector,
                metadata={
                    "processing_time": time.time() - start_time,
                    "market_regime": market_regime.value,
                    "ferris_expanded_id": expanded_strategy_id,
                    "execution_status": "success",
                    "realized_profit": float(execution_result.profit_realized or 0.0),
                },
            )

            # 10. Update pipeline state
            self._update_pipeline_state(decision)

            # 11. Store decision in history
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

            logger.info(
                f"Trading decision executed: {decision.action.value} "
                f"{decision.quantity:.4f} {decision.symbol} @ {decision.price:.2f}"
                f" | Profit: {decision.metadata.get('realized_profit', 0.0):.4f}"
            )

            return decision

        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
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

    def _determine_optimal_strategy(
        self, regime: MarketRegime, market_data: MarketData
    ) -> StrategyBranch:
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

    def _select_vectorization_mode(
        self, strategy: StrategyBranch, market_data: MarketData
    ) -> VectorizationMode:
        """Select appropriate vectorization mode."""
        if strategy in [StrategyBranch.SCALPING, StrategyBranch.ARBITRAGE]:
            return VectorizationMode.HIGH_FREQUENCY
        elif strategy == StrategyBranch.MOMENTUM:
            return VectorizationMode.MOMENTUM_BASED
        elif strategy == StrategyBranch.MEAN_REVERSION:
            return VectorizationMode.MEAN_REVERSION
        else:
            return VectorizationMode.ADAPTIVE

    def _create_profit_vector(
        self, signal: Dict[str, Any], market_data: MarketData
    ) -> ProfitVector:
        """Create profit vector for the signal."""
        # This method needs to adapt to receive IntegratedTradingSignal or its data
        # For now, I'll adjust it to expect a dict compatible with previous signals.
        # If signal is IntegratedTradingSignal, convert it to dict for compatibility.
        if isinstance(signal, IntegratedTradingSignal):
            # Convert to a dictionary that _create_profit_vector can understand
            # Adjust keys to match what profit_vectorizer expects
            signal_dict = {
                "action": signal.recommended_action.upper(),
                "quantity": float(signal.quantity), # Use the actual quantity from IntegratedTradingSignal
                "confidence": float(signal.confidence_score),
                "profit_potential": float(signal.profit_potential),
                "risk_score": float(signal.risk_assessment.get("overall_risk", 0.5)),
            }
        else:
            signal_dict = signal
        
        mode = self._select_vectorization_mode(self.state.active_strategy, market_data)
        return self.profit_vectorizer.calculate_profit_vector(
            market_data.price,
            market_data.volume,
            signal_dict, # Pass the adjusted signal dict
            mode=mode,
        )

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
        else:
            return None

    def _momentum_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate momentum signal."""
        if len(self.market_data_history) < 10:
            return None

        recent_prices = [md.price for md in self.market_data_history[-10:]]
        short_ma = np.mean(recent_prices[-5:])
        long_ma = np.mean(recent_prices)

        momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
        volume_surge = market_data.volume / np.mean(
            [md.volume for md in self.market_data_history[-5:]]
        )

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
                # Higher frequency
                "quantity": self._calculate_position_size(market_data, action) * 2,
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
        recent_highs = [max(prices[i : i + 10]) for i in range(0, len(prices) - 10, 10)]
        recent_lows = [min(prices[i : i + 10]) for i in range(0, len(prices) - 10, 10)]

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

    def _apply_risk_management(
        self, signal: Dict[str, Any], market_data: MarketData
    ) -> Optional[Dict[str, Any]]:
        # This method's logic will now be largely integrated into _determine_asset_and_signal
        # so it's kept as a placeholder or for very specific, final checks if needed.
        # For now, it simply returns the signal, as risk logic is shifting.
        return signal

    # --- New method to determine asset and formulate signal based on Ferris Wheel ---
    async def _determine_asset_and_signal(
        self, expanded_strategy_id: int, market_data: MarketData, current_strategy_branch: StrategyBranch
    ) -> Tuple[Optional[IntegratedTradingSignal], Optional[str]]:
        """
        Determines the asset to trade and formulates the trading signal
        based on the expanded strategy ID from the Ferris Wheel.

        This method integrates:
        - Ferris Wheel expansion logic
        - Dualistic state machine
        - Profit vectorization
        - Real-time portfolio data
        - Advanced mathematical decision making
        - Ghost Core interaction
        """
        try:
            # 1. PHASE BIT INTEGRATION - Determine optimal bit phase for this decision
            context_hash = f"{expanded_strategy_id}_{market_data.symbol}_{market_data.timestamp}"
            phase_resolution = self.phase_bit_integration.resolve_bit_phase(
                context_hash=context_hash,
                resolution_mode="auto"
            )
            
            logger.info(f"Phase resolution: {phase_resolution.bit_phase.value} bits, "
                       f"strategy: {phase_resolution.strategy_type.value}, "
                       f"confidence: {phase_resolution.confidence:.3f}")

            # 2. REAL ASSET SELECTION - Get available assets from portfolio
            portfolio_summary = self.portfolio_tracker.get_portfolio_summary()
            available_assets = []
            
            # Extract available trading pairs from portfolio
            if "positions" in portfolio_summary and portfolio_summary["positions"]:
                available_assets = list(portfolio_summary["positions"].keys())
            else:
                # Fallback to default tradable assets if portfolio is empty
                available_assets = [
                    "BTC/USDC", "ETH/USDC", "XRP/USDC", 
                    "BTC/USDT", "ETH/USDT", "SOL/USDC"
                ]
                logger.info("Using default tradable assets (portfolio empty)")

            if not available_assets:
                logger.error("No tradable assets available")
                return None, None

            # 3. FERRIS WHEEL ASSET SELECTION - Use expanded strategy ID for deterministic selection
            asset_index = expanded_strategy_id % len(available_assets)
            selected_asset_symbol = available_assets[asset_index]
            
            # Convert to TradingPair enum if possible
            try:
                selected_trading_pair = TradingPair(selected_asset_symbol)
            except ValueError:
                # Create a custom TradingPair for unsupported symbols
                selected_trading_pair = TradingPair.BTC_USDC  # Default fallback
                logger.warning(f"Unsupported trading pair: {selected_asset_symbol}, using default")

            logger.info(f"Ferris Wheel selected: {selected_asset_symbol} (ID: {expanded_strategy_id}, "
                       f"Index: {asset_index}/{len(available_assets)})")

            # 4. DUALISTIC STATE ANALYSIS - Determine current market state and strategy alignment
            dualistic_state = self._analyze_dualistic_state(
                market_data, expanded_strategy_id, phase_resolution
            )
            
            # 5. PROFIT VECTORIZATION - Calculate expected profit vector for this asset/strategy
            profit_vector = await self._calculate_advanced_profit_vector(
                market_data, selected_asset_symbol, expanded_strategy_id, 
                phase_resolution, dualistic_state
            )

            # 6. ADVANCED DECISION LOGIC - Integrate all mathematical components
            decision_data = self._integrate_decision_components(
                market_data, profit_vector, dualistic_state, 
                expanded_strategy_id, phase_resolution
            )

            # 7. POSITION SIZING - Calculate optimal position size
            quantity = self._calculate_advanced_position_size(
                market_data, decision_data, profit_vector, dualistic_state
            )

            # 8. RISK ASSESSMENT - Comprehensive risk evaluation
            risk_assessment = self._calculate_comprehensive_risk(
                market_data, decision_data, profit_vector, quantity, dualistic_state
            )

            # 9. FINAL SIGNAL CONSTRUCTION - Create the complete trading signal
            if decision_data["action"] != "hold" and quantity > 0:
                signal = IntegratedTradingSignal(
                    signal_id=str(uuid.uuid4()),
                    recommended_action=decision_data["action"],
                    target_pair=selected_trading_pair,
                    quantity=Decimal(str(quantity)),
                    confidence_score=Decimal(str(decision_data["confidence"])),
                    profit_potential=Decimal(str(profit_vector.profit_score)),
                    risk_assessment={
                        "overall_risk": risk_assessment["overall_risk"],
                        "ferris_id": expanded_strategy_id,
                        "bit_phase": phase_resolution.bit_phase.value,
                        "dualistic_state": dualistic_state["state"],
                        "market_regime": market_data.metadata.get("market_regime", "unknown"),
                        "volatility_risk": risk_assessment["volatility_risk"],
                        "position_risk": risk_assessment["position_risk"],
                        "strategy_risk": risk_assessment["strategy_risk"]
                    },
                    ghost_route=f"ferris_wheel_{phase_resolution.strategy_type.value}",
                    timestamp=market_data.timestamp,
                )
                
                logger.info(f"Generated signal: {signal.recommended_action} {quantity} {selected_asset_symbol} "
                           f"(Confidence: {signal.confidence_score}, Profit: {signal.profit_potential})")
                
                return signal, selected_asset_symbol
            else:
                logger.info(f"No trade recommended for {selected_asset_symbol} "
                           f"(Action: {decision_data['action']}, Quantity: {quantity})")
                return None, selected_asset_symbol

        except Exception as e:
            logger.error(f"Error in _determine_asset_and_signal: {e}")
            return None, None

    def _analyze_dualistic_state(
        self, market_data: MarketData, expanded_strategy_id: int, phase_resolution: Any
    ) -> Dict[str, Any]:
        """
        Analyze dualistic state based on market data and strategy ID.
        This integrates your dualistic state machine logic.
        """
        # Create a hash-based dualistic state determination
        state_hash = hash(f"{expanded_strategy_id}_{market_data.price}_{market_data.volume}")
        
        # Determine state based on hash and market conditions
        if state_hash % 2 == 0:
            state = "positive_phase"
            confidence_multiplier = 1.2
        else:
            state = "negative_phase"
            confidence_multiplier = 0.8

        # Adjust based on market conditions
        if market_data.trend_strength > 0.7:
            state += "_trending"
        elif market_data.volatility > 0.8:
            state += "_volatile"

        return {
            "state": state,
            "confidence_multiplier": confidence_multiplier,
            "hash_value": state_hash,
            "bit_phase": phase_resolution.bit_phase.value,
            "market_alignment": market_data.trend_strength
        }

    async def _calculate_advanced_profit_vector(
        self, market_data: MarketData, asset_symbol: str, 
        expanded_strategy_id: int, phase_resolution: Any, dualistic_state: Dict[str, Any]
    ) -> ProfitVector:
        """
        Calculate advanced profit vector using your profit vectorization system.
        """
        # Create comprehensive input for profit vectorization
        vector_input = {
            "price": market_data.price,
            "volume": market_data.volume,
            "volatility": market_data.volatility,
            "trend_strength": market_data.trend_strength,
            "entropy_level": market_data.entropy_level,
            "asset_symbol": asset_symbol,
            "strategy_id": expanded_strategy_id,
            "bit_phase": phase_resolution.bit_phase.value,
            "dualistic_state": dualistic_state["state"],
            "confidence_multiplier": dualistic_state["confidence_multiplier"],
            "timestamp": market_data.timestamp,
            "metadata": market_data.metadata
        }

        # Select vectorization mode based on strategy and market conditions
        if market_data.volatility > 0.8:
            mode = VectorizationMode.HIGH_FREQUENCY
        elif market_data.trend_strength > 0.7:
            mode = VectorizationMode.MOMENTUM_BASED
        elif market_data.trend_strength < 0.3:
            mode = VectorizationMode.MEAN_REVERSION
        else:
            mode = VectorizationMode.ADAPTIVE

        # Calculate profit vector
        profit_vector = self.profit_vectorizer.calculate_profit_vector(
            vector_input, mode=mode
        )

        # Apply dualistic state adjustments
        profit_vector.profit_score *= dualistic_state["confidence_multiplier"]
        profit_vector.confidence_score *= dualistic_state["confidence_multiplier"]

        return profit_vector

    def _integrate_decision_components(
        self, market_data: MarketData, profit_vector: ProfitVector,
        dualistic_state: Dict[str, Any], expanded_strategy_id: int, phase_resolution: Any
    ) -> Dict[str, Any]:
        """
        Integrate all decision components to determine final trading action.
        """
        # Base decision logic
        action = "hold"
        confidence = 0.0
        profit_potential = 0.0

        # 1. Profit Vector Analysis
        if profit_vector.profit_score > 0.02:  # 2% minimum profit potential
            action = "buy"
            confidence = profit_vector.confidence_score
            profit_potential = profit_vector.profit_score
        elif profit_vector.profit_score < -0.02:  # Negative profit potential
            action = "sell"
            confidence = profit_vector.confidence_score
            profit_potential = abs(profit_vector.profit_score)

        # 2. Market Regime Analysis
        if market_data.trend_strength > 0.7 and action == "buy":
            confidence *= 1.2  # Boost confidence for strong trends
        elif market_data.trend_strength < 0.3 and action == "sell":
            confidence *= 1.1  # Boost confidence for strong downtrends

        # 3. Volatility Adjustment
        if market_data.volatility > 0.8:
            confidence *= 0.8  # Reduce confidence in high volatility
            profit_potential *= 1.3  # But increase profit potential

        # 4. Dualistic State Integration
        confidence *= dualistic_state["confidence_multiplier"]

        # 5. Bit Phase Precision
        if phase_resolution.bit_phase.value >= 16:
            confidence *= 1.1  # Higher precision = higher confidence
        elif phase_resolution.bit_phase.value <= 4:
            confidence *= 0.9  # Lower precision = lower confidence

        # 6. Strategy ID Pattern Analysis
        strategy_pattern = expanded_strategy_id % 256
        if strategy_pattern in [0, 64, 128, 192]:  # Special pattern values
            confidence *= 1.15  # Boost confidence for special patterns

        # Ensure confidence is within bounds
        confidence = max(0.1, min(1.0, confidence))

        return {
            "action": action,
            "confidence": confidence,
            "profit_potential": profit_potential,
            "dualistic_state": dualistic_state,
            "bit_phase": phase_resolution.bit_phase.value,
            "strategy_pattern": strategy_pattern
        }

    def _calculate_advanced_position_size(
        self, market_data: MarketData, decision_data: Dict[str, Any],
        profit_vector: ProfitVector, dualistic_state: Dict[str, Any]
    ) -> float:
        """
        Calculate advanced position size based on multiple factors.
        """
        # Base position size
        base_size = self.risk_params.max_position_size * self.state.current_capital
        
        # Price adjustment
        price = market_data.price
        if price <= 0:
            return 0.0
        
        # Volatility adjustment
        volatility_adjustment = 1.0 - (market_data.volatility * 0.5)
        
        # Confidence adjustment
        confidence_adjustment = decision_data["confidence"]
        
        # Profit potential adjustment
        profit_adjustment = min(profit_vector.profit_score * 10, 2.0)  # Cap at 2x
        
        # Dualistic state adjustment
        dualistic_adjustment = dualistic_state["confidence_multiplier"]
        
        # Thermal state adjustment
        thermal_multiplier = {
            ThermalState.COOL: 0.8,
            ThermalState.WARM: 1.0,
            ThermalState.HOT: 1.2,
        }.get(self.state.thermal_state, 1.0)
        
        # Calculate final position size
        adjusted_size = (base_size * volatility_adjustment * confidence_adjustment * 
                        profit_adjustment * dualistic_adjustment * thermal_multiplier / price)
        
        # Ensure positive and reasonable size
        return max(0.0, min(adjusted_size, base_size * 2.0))  # Cap at 2x base size

    def _calculate_comprehensive_risk(
        self, market_data: MarketData, decision_data: Dict[str, Any],
        profit_vector: ProfitVector, quantity: float, dualistic_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk assessment.
        """
        # Volatility risk
        volatility_risk = market_data.volatility * 0.8
        
        # Position risk (based on position size relative to capital)
        position_risk = min((quantity * market_data.price) / self.state.current_capital, 1.0)
        
        # Strategy risk (based on confidence and profit potential)
        strategy_risk = 1.0 - decision_data["confidence"]
        
        # Market regime risk
        if market_data.trend_strength < 0.3:
            regime_risk = 0.6
        elif market_data.trend_strength > 0.7:
            regime_risk = 0.3
        else:
            regime_risk = 0.5
        
        # Dualistic state risk
        if dualistic_state["state"].endswith("_volatile"):
            dualistic_risk = 0.7
        else:
            dualistic_risk = 0.4
        
        # Overall risk (weighted average)
        overall_risk = (
            volatility_risk * 0.3 +
            position_risk * 0.25 +
            strategy_risk * 0.2 +
            regime_risk * 0.15 +
            dualistic_risk * 0.1
        )
        
        return {
            "overall_risk": overall_risk,
            "volatility_risk": volatility_risk,
            "position_risk": position_risk,
            "strategy_risk": strategy_risk,
            "regime_risk": regime_risk,
            "dualistic_risk": dualistic_risk
        }

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

        # Ensure positive quantity
        return max(0.0, adjusted_size * thermal_multiplier)

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

        self.performance_metrics["win_rate"] = (
            profitable_trades / total_trades if total_trades > 0 else 0
        )
        self.performance_metrics["total_return"] = (
            self.state.current_capital - self.initial_capital
        ) / self.initial_capital

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
        "total_return": (pipeline.state.current_capital - pipeline.initial_capital)
        / pipeline.initial_capital,
    }
