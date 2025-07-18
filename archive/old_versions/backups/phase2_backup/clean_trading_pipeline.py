#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Trading Pipeline - Schwabot's Unified Trading Execution Layer
==================================================================

This is Schwabot's unified trading execution layer.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .ccxt_trading_executor import CCXTTradingExecutor, IntegratedTradingSignal, TradingPair
from .chrono_recursive_logic_function import (
    ChronoRecursiveLogicFunction,
    CRLFResponse,
    CRLFTriggerState,
    create_crlf,
)
from .clean_math_foundation import BitPhase, CleanMathFoundation, ThermalState
from .clean_profit_vectorization import CleanProfitVectorization, ProfitVector, VectorizationMode
from .phase_bit_integration import phase_bit_integration
from .portfolio_tracker import PortfolioTracker
from .soulprint_registry import SoulprintRegistry
from .strategy_bit_mapper import StrategyBitMapper
from .unified_market_data_pipeline import MarketDataPacket, create_unified_pipeline
from .unified_math_system import create_unified_math_system
from .zpe_zbe_core import create_zpe_zbe_core  # noqa: F401 - Used in core system initialization
from .zpe_zbe_core import (  # noqa: F401 - Used in performance monitoring and optimization
    QuantumPerformanceRegistry,
    QuantumSyncStatus,
    ZBEBalance,
    ZPEVector,
    ZPEZBEPerformanceTracker,
)

# Entropy Signal Integration
try:
    from .entropy_signal_integration import (
        EntropySignal,
        get_entropy_integrator,
        process_entropy_signal,
        should_execute_routing,
        should_execute_tick,
    )
    from .fractal_core import FractalCore

    ENTROPY_INTEGRATION_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🧠 Entropy signal integration modules loaded successfully")
except ImportError as e:
    ENTROPY_INTEGRATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Entropy integration modules not available: {e}")

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions for market execution."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyBranch(Enum):
    """
    Strategy branches for different market conditions.

    Mathematical Strategy Mapping:
    - MEAN_REVERSION: R < 0.3 (low volatility, mean-reverting markets)
    - MOMENTUM: R > 0.7 (high volatility, trending markets)
    - ARBITRAGE: |correlation| > 0.9 (high correlation opportunities)
    - SCALPING: volatility > 0.8 (high frequency opportunities)
    - SWING: 0.3 < R < 0.7 (moderate volatility, swing opportunities)
    - GRID: volatility < 0.3 (low volatility, range-bound markets)
    - FERRIS_WHEEL: cyclic patterns detected (rotational opportunities)
    """
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    GRID = "grid"
    FERRIS_WHEEL = "ferris_wheel"  # Add Ferris Wheel as a strategy branch


class MarketRegime(Enum):
    """
    Market regimes for adaptive strategy selection.

    Regime Classification:
    - TRENDING_UP: trend_strength > 0.7, volatility > 0.5
    - TRENDING_DOWN: trend_strength < -0.7, volatility > 0.5
    - SIDEWAYS: |trend_strength| < 0.3, volatility < 0.5
    - VOLATILE: volatility > 0.8 (high uncertainty)
    - CALM: volatility < 0.3 (low uncertainty)
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class MarketData:
    """
    Market data snapshot with mathematical properties.

    Mathematical Components:
    - price: Current market price (float)
    - volume: Trading volume (float)
    - volatility: Price volatility σ = std(returns) (0-1)
    - trend_strength: Trend strength T = correlation(price, time) (-1 to 1)
    - entropy_level: Market entropy H = -Σ p(x) * log2(p(x)) (bits)
    - bid/ask: Order book spread for liquidity analysis
    """
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
    """
    Trading decision output with mathematical confidence.

    Mathematical Properties:
    - confidence: Decision confidence C = f(signal_strength, market_regime, risk_score)
    - profit_potential: Expected profit P = Kelly_Criterion * position_size * price_movement
    - risk_score: Risk assessment R = volatility * position_size * leverage
    - thermal_state: Market thermal state T = sigmoid(volatility * trend_strength)
    - bit_phase: Bit phase B = (price_position, volume_flow, momentum_alignment)
    - profit_vector: Multi-dimensional profit vector V = [momentum, mean_rev, arbitrage, scalping]
    """
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
    """
    Current state of the trading pipeline with performance metrics.

    Mathematical Metrics:
    - win_rate: W = winning_trades / total_trades
    - profit_factor: PF = total_profit / abs(total_loss)
    - sharpe_ratio: SR = (return - risk_free_rate) / volatility
    - max_drawdown: MD = max(peak - current) / peak
    - current_risk_level: CR = portfolio_risk / max_allowed_risk
    """
    timestamp: float
    active_strategy: StrategyBranch
    current_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_loss: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_risk_level: float
    thermal_state: ThermalState
    bit_phase: BitPhase
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZPEZBEPipelineState:
    """Enhanced pipeline state with ZPE/ZBE integration."""
    base_state: PipelineState
    zpe_vector: ZPEVector
    zbe_balance: ZBEBalance
    quantum_sync_status: QuantumSyncStatus
    performance_tracker: ZPEZBEPerformanceTracker
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRLFEnhancedPipelineState:
    """Enhanced pipeline state with CRLF integration."""
    base_state: PipelineState
    crlf_response: CRLFResponse
    trigger_state: CRLFTriggerState
    logic_function: ChronoRecursiveLogicFunction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZPEZBEMarketData:
    """Enhanced market data with ZPE/ZBE integration."""
    base_data: MarketData
    zpe_vector: ZPEVector
    zbe_balance: ZBEBalance
    quantum_sync_status: QuantumSyncStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRLFEnhancedMarketData:
    """Enhanced market data with CRLF integration."""
    base_data: MarketData
    crlf_response: CRLFResponse
    trigger_state: CRLFTriggerState
    logic_function: ChronoRecursiveLogicFunction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRLFEnhancedTradingDecision:
    """Enhanced trading decision with CRLF integration."""
    base_decision: TradingDecision
    crlf_response: CRLFResponse
    trigger_state: CRLFTriggerState
    logic_function: ChronoRecursiveLogicFunction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZPEZBETradingDecision:
    """Enhanced trading decision with ZPE/ZBE integration."""
    base_decision: TradingDecision
    zpe_vector: ZPEVector
    zbe_balance: ZBEBalance
    quantum_sync_status: QuantumSyncStatus
    performance_tracker: ZPEZBEPerformanceTracker
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskParameters:
    """Risk management parameters for the trading pipeline."""
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown: float = 0.15
    risk_free_rate: float = 0.02
    volatility_threshold: float = 0.8
    correlation_threshold: float = 0.9
    metadata: Dict[str, Any] = field(default_factory=dict)


class CleanTradingPipeline:
    """
    Clean trading pipeline with mathematical integration.

    This pipeline integrates:
    - Clean mathematical foundation
    - Profit vectorization
    - Phase bit integration
    - Portfolio tracking
    - Strategy bit mapping
    - ZPE/ZBE core systems
    - CRLF systems
    - Entropy signal integration
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        risk_params: Optional[RiskParameters] = None,
        matrix_dir: Union[str, Path] = "data/matrices",
        registry_file: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        safe_mode: bool = False,
    ) -> None:
        """Initialize the clean trading pipeline."""
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()
        self.matrix_dir = Path(matrix_dir)
        self.registry_file = registry_file
        self.pipeline_config = pipeline_config or self._default_pipeline_config()
        self.safe_mode = safe_mode

        # Initialize core components
        self.math_foundation = CleanMathFoundation()
        self.profit_vectorization = CleanProfitVectorization()
        self.portfolio_tracker = PortfolioTracker(initial_capital)
        self.strategy_mapper = StrategyBitMapper()
        self.soulprint_registry = SoulprintRegistry(registry_file)

        # Initialize ZPE/ZBE core
        self.zpe_zbe_core = create_zpe_zbe_core()
        self.quantum_performance_registry = QuantumPerformanceRegistry()

        # Initialize CRLF system
        self.crlf_system = create_crlf()

        # Initialize unified systems
        self.unified_pipeline = create_unified_pipeline()
        self.unified_math_system = create_unified_math_system()

        # Initialize trading executor
        self.trading_executor = CCXTTradingExecutor()

        # Initialize entropy integration if available
        if ENTROPY_INTEGRATION_AVAILABLE:
            self.entropy_integrator = get_entropy_integrator()
            self.fractal_core = FractalCore()
        else:
            self.entropy_integrator = None
            self.fractal_core = None

        # Pipeline state
        self.current_state = PipelineState(
            timestamp=time.time(),
            active_strategy=StrategyBranch.SWING,
            current_capital=initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0.0,
            total_loss=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_risk_level=0.0,
            thermal_state=ThermalState.COOL,
            bit_phase=BitPhase.EIGHT_BIT,
        )

        # Performance tracking
        self.performance_history = []
        self.error_count = 0
        self.success_count = 0

        logger.info(f"Clean Trading Pipeline initialized for {symbol}")

    def _default_pipeline_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "enable_mathematical_integration": True,
            "enable_zpe_zbe_integration": True,
            "enable_crlf_integration": True,
            "enable_entropy_integration": ENTROPY_INTEGRATION_AVAILABLE,
            "update_interval": 1.0,
            "max_concurrent_trades": 5,
            "enable_circuit_breakers": True,
            "enable_performance_tracking": True,
        }

    def process_market_data(self, market_data: MarketData) -> TradingDecision:
        """
        Process market data through the complete mathematical pipeline.

        Mathematical Pipeline:
        1. Market data validation and preprocessing
        2. Thermal state calculation
        3. Bit phase determination
        4. Strategy branch selection
        5. Profit vectorization
        6. Risk assessment
        7. Decision generation
        """
        try:
            # Step 1: Validate and preprocess market data
            validated_data = self._validate_market_data(market_data)

            # Step 2: Calculate thermal state
            thermal_state = self._calculate_thermal_state(validated_data)

            # Step 3: Determine bit phase
            bit_phase = self._determine_bit_phase(validated_data)

            # Step 4: Select strategy branch
            strategy_branch = self._select_strategy_branch(validated_data, thermal_state)

            # Step 5: Calculate profit vector
            profit_vector = self._calculate_profit_vector(validated_data, strategy_branch)

            # Step 6: Assess risk
            risk_score = self._assess_risk(validated_data, profit_vector)

            # Step 7: Generate trading decision
            decision = self._generate_trading_decision(
                validated_data, strategy_branch, profit_vector, risk_score, thermal_state, bit_phase
            )

            # Update pipeline state
            self._update_pipeline_state(decision)

            return decision

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return self._create_fallback_decision(market_data)

    def _validate_market_data(self, market_data: MarketData) -> MarketData:
        """Validate and preprocess market data."""
        # Basic validation
        if market_data.price <= 0:
            raise ValueError("Invalid price")
        if market_data.volume < 0:
            raise ValueError("Invalid volume")
        if market_data.timestamp <= 0:
            raise ValueError("Invalid timestamp")

        # Normalize volatility and trend strength
        validated_data = MarketData(
            symbol=market_data.symbol,
            price=market_data.price,
            volume=market_data.volume,
            timestamp=market_data.timestamp,
            bid=market_data.bid,
            ask=market_data.ask,
            volatility=max(0.0, min(1.0, market_data.volatility)),
            trend_strength=max(-1.0, min(1.0, market_data.trend_strength)),
            entropy_level=max(0.0, market_data.entropy_level),
            metadata=market_data.metadata.copy(),
        )

        return validated_data

    def _calculate_thermal_state(self, market_data: MarketData) -> ThermalState:
        """Calculate thermal state based on market conditions."""
        # Thermal state calculation: T = sigmoid(volatility * trend_strength)
        thermal_score = market_data.volatility * abs(market_data.trend_strength)
        
        if thermal_score < 0.3:
            return ThermalState.COOL
        elif thermal_score < 0.7:
            return ThermalState.WARM
        else:
            return ThermalState.HOT

    def _determine_bit_phase(self, market_data: MarketData) -> BitPhase:
        """Determine bit phase based on market conditions."""
        # Bit phase determination based on volatility and entropy
        volatility_factor = market_data.volatility
        entropy_factor = market_data.entropy_level / 8.0  # Normalize to 0-1
        
        combined_factor = (volatility_factor + entropy_factor) / 2.0
        
        if combined_factor < 0.2:
            return BitPhase.FOUR_BIT
        elif combined_factor < 0.4:
            return BitPhase.EIGHT_BIT
        elif combined_factor < 0.6:
            return BitPhase.SIXTEEN_BIT
        elif combined_factor < 0.8:
            return BitPhase.THIRTY_TWO_BIT
        else:
            return BitPhase.FORTY_TWO_BIT

    def _select_strategy_branch(self, market_data: MarketData, thermal_state: ThermalState) -> StrategyBranch:
        """Select strategy branch based on market conditions."""
        # Strategy selection logic
        volatility = market_data.volatility
        trend_strength = abs(market_data.trend_strength)
        
        if volatility < 0.3:
            if trend_strength < 0.3:
                return StrategyBranch.GRID
            else:
                return StrategyBranch.MEAN_REVERSION
        elif volatility > 0.8:
            return StrategyBranch.SCALPING
        elif trend_strength > 0.7:
            return StrategyBranch.MOMENTUM
        elif trend_strength < 0.3:
            return StrategyBranch.SWING
        else:
            return StrategyBranch.FERRIS_WHEEL

    def _calculate_profit_vector(self, market_data: MarketData, strategy_branch: StrategyBranch) -> ProfitVector:
        """Calculate profit vector using the profit vectorization system."""
        vector_input = {
            "price": market_data.price,
            "volume": market_data.volume,
            "volatility": market_data.volatility,
            "trend_strength": market_data.trend_strength,
            "entropy_level": market_data.entropy_level,
            "strategy_branch": strategy_branch.value,
        }
        
        return self.profit_vectorization.calculate_profit_vector(vector_input)

    def _assess_risk(self, market_data: MarketData, profit_vector: ProfitVector) -> float:
        """Assess risk based on market data and profit vector."""
        # Risk assessment: R = volatility * position_size * leverage
        base_risk = market_data.volatility * profit_vector.confidence_score
        risk_score = min(1.0, base_risk)
        
        return risk_score

    def _generate_trading_decision(
        self,
        market_data: MarketData,
        strategy_branch: StrategyBranch,
        profit_vector: ProfitVector,
        risk_score: float,
        thermal_state: ThermalState,
        bit_phase: BitPhase,
    ) -> TradingDecision:
        """Generate trading decision based on all calculated factors."""
        # Decision logic based on profit vector and risk
        if profit_vector.profit_score > 0.7 and risk_score < 0.5:
            action = TradingAction.BUY
        elif profit_vector.profit_score < 0.3 and risk_score < 0.5:
            action = TradingAction.SELL
        else:
            action = TradingAction.HOLD

        # Calculate position size based on confidence and risk
        position_size = self._calculate_position_size(profit_vector.confidence_score, risk_score)

        # Calculate profit potential
        profit_potential = profit_vector.profit_score * position_size * market_data.price

        decision = TradingDecision(
            timestamp=time.time(),
            symbol=market_data.symbol,
            action=action,
            quantity=position_size,
            price=market_data.price,
            confidence=profit_vector.confidence_score,
            strategy_branch=strategy_branch,
            profit_potential=profit_potential,
            risk_score=risk_score,
            thermal_state=thermal_state,
            bit_phase=bit_phase,
            profit_vector=profit_vector,
        )

        return decision

    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate position size based on confidence and risk."""
        # Kelly Criterion inspired position sizing
        base_size = confidence * (1 - risk_score)
        max_size = self.risk_params.max_position_size
        
        return min(base_size, max_size)

    def _update_pipeline_state(self, decision: TradingDecision) -> None:
        """Update pipeline state based on decision."""
        self.current_state.timestamp = time.time()
        self.current_state.active_strategy = decision.strategy_branch
        self.current_state.thermal_state = decision.thermal_state
        self.current_state.bit_phase = decision.bit_phase

    def _create_fallback_decision(self, market_data: MarketData) -> TradingDecision:
        """Create fallback decision when processing fails."""
        fallback_vector = ProfitVector(
            vector_id=f"fallback_{int(time.time())}",
            btc_price=market_data.price,
            volume=market_data.volume,
            profit_score=0.0,
            confidence_score=0.1,
            mode="fallback",
            method="error_recovery",
            timestamp=time.time(),
        )

        return TradingDecision(
            timestamp=time.time(),
            symbol=market_data.symbol,
            action=TradingAction.HOLD,
            quantity=0.0,
            price=market_data.price,
            confidence=0.1,
            strategy_branch=StrategyBranch.SWING,
            profit_potential=0.0,
            risk_score=1.0,
            thermal_state=ThermalState.COOL,
            bit_phase=BitPhase.EIGHT_BIT,
            profit_vector=fallback_vector,
        )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "symbol": self.symbol,
            "current_state": self.current_state,
            "performance_metrics": {
                "total_trades": self.current_state.total_trades,
                "win_rate": self.current_state.win_rate,
                "profit_factor": self.current_state.profit_factor,
                "sharpe_ratio": self.current_state.sharpe_ratio,
                "max_drawdown": self.current_state.max_drawdown,
            },
            "system_status": {
                "entropy_integration_available": ENTROPY_INTEGRATION_AVAILABLE,
                "safe_mode": self.safe_mode,
                "error_count": self.error_count,
                "success_count": self.success_count,
            },
        }

    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            if self.entropy_integrator:
                self.entropy_integrator.cleanup()
            if self.fractal_core:
                self.fractal_core.cleanup()
            
            logger.info("Clean Trading Pipeline resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up pipeline: {e}")


# Global instance for easy access
clean_trading_pipeline = CleanTradingPipeline()
