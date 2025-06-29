# -*- coding: utf-8 -*-
"""
Dual Brain Architecture
=======================

Advanced dual-brain trading system that separates mining/hashing operations
from trading decision-making. One brain handles BTC mining analysis and
hashing operations, while the other handles actual trading through API bridges.

Mathematical Foundation:
    - Left Brain (Mining): H(b) = sumᵢ hash_powerᵢ * difficultyᵢ * thermal_stateᵢ
    - Right Brain (Trading): T(m) = sumⱼ market_signalⱼ * whale_alertⱼ * flip_logicⱼ
    - Brain Synchronization: S = sync(H(b), T(m)) -> unified_decision
    - Flip Logic: FL = (mining_signal ⊕ trading_signal) * thermal_multiplier
    - 32-bit Thermal Integration: All operations enhanced at HOT thermal state
"""

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import core systems
try:
    from core.dual_unicore_handler import DualUnicoreHandler
    from core.dualistic_thought_engines import DualisticThoughtEngines
    from core.exchange_plumbing import ExchangePlumbing
    from core.phase_bit_integration import BitPhase, PhaseBitIntegration
    from core.unified_math_system import UnifiedMathSystem
    from core.whale_tracker_integration import WhaleTrackerIntegration, whale_tracker

    CORE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core systems not available: {e}")
    CORE_SYSTEMS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Thermal state constants for dual brain operations
COOL = "cool"  # Low thermal state (4-bit operations)
WARM = "warm"  # Mid thermal state (8-bit operations)
HOT = "hot"  # High thermal state (32-bit operations)
CRITICAL = "critical"  # Extreme thermal state (42-bit operations)


class BrainType(Enum):
    """Types of brain processes."""

    LEFT_BRAIN = "left_brain"  # Mining/Hashing brain
    RIGHT_BRAIN = "right_brain"  # Trading/Decision brain


class FlipLogicSignal(Enum):
    """Flip logic signals for profit calculations."""

    STRONG_BUY = "strong_buy"
    MODERATE_BUY = "moderate_buy"
    HOLD = "hold"
    MODERATE_SELL = "moderate_sell"
    STRONG_SELL = "strong_sell"
    WAIT_SIGNAL = "wait_signal"


class MiningOperationType(Enum):
    """Types of mining operations."""

    HASH_GENERATION = "hash_generation"
    DIFFICULTY_ANALYSIS = "difficulty_analysis"
    BLOCK_ANALYSIS = "block_analysis"
    NONCE_OPTIMIZATION = "nonce_optimization"
    POOL_COORDINATION = "pool_coordination"
    THERMAL_MANAGEMENT = "thermal_management"


class TradingOperationType(Enum):
    """Types of trading operations."""

    MARKET_ANALYSIS = "market_analysis"
    ORDER_EXECUTION = "order_execution"
    RISK_MANAGEMENT = "risk_management"
    WHALE_MONITORING = "whale_monitoring"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    PROFIT_TAKING = "profit_taking"


@dataclass
class BrainState:
    """Represents the state of one brain."""

    brain_type: BrainType
    thermal_state: str
    active_operations: List[str]
    performance_metrics: Dict[str, float]
    last_decision: Optional[str]
    decision_confidence: float
    processing_load: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FlipLogicResult:
    """Result of flip logic calculation."""

    flip_signal: FlipLogicSignal
    confidence: float
    mining_contribution: float
    trading_contribution: float
    thermal_multiplier: float
    reasoning: str
    profit_potential: float
    risk_assessment: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DualBrainDecision:
    """Unified decision from both brains."""

    left_brain_state: BrainState
    right_brain_state: BrainState
    flip_logic_result: FlipLogicResult
    synchronized_action: str
    overall_confidence: float
    expected_profit: float
    thermal_enhancement: bool
    execution_priority: int
    timestamp: datetime = field(default_factory=datetime.now)


class LeftBrain:
    """
    Mining/Hashing Brain - Handles BTC mining analysis, hash generation,
    difficulty calculations, and blockchain analysis with 32-bit thermal integration.
    """

    def __init__(self):
        """Initialize the left brain (mining/hashing)."""
        self.brain_type = BrainType.LEFT_BRAIN
        self.math_system = UnifiedMathSystem() if CORE_SYSTEMS_AVAILABLE else None
        self.phase_integration = PhaseBitIntegration() if CORE_SYSTEMS_AVAILABLE else None

        # Mining state
        self.current_thermal_state = WARM
        self.mining_operations = []
        self.hash_rate = 0.0
        self.difficulty_target = 0.0
        self.block_analysis_results = {}

        # Performance tracking
        self.performance_metrics = {
            "hash_rate_th_s": 0.0,
            "difficulty_adjustment": 0.0,
            "block_time_average": 600.0,  # 10 minutes
            "mining_efficiency": 0.0,
            "thermal_efficiency": 0.0,
        }

        # Mining pool data
        self.mining_pools = {
            "antpool": {"hash_rate": 0.0, "block_count": 0},
            "f2pool": {"hash_rate": 0.0, "block_count": 0},
            "slushpool": {"hash_rate": 0.0, "block_count": 0},
            "viabtc": {"hash_rate": 0.0, "block_count": 0},
        }

        logger.info("🧠 Left Brain (Mining/Hashing) initialized")

    async def process_mining_operations(self) -> BrainState:
        """Process mining operations and return brain state."""
        try:
            # Update thermal state based on mining load
            self._update_thermal_state()

            # Perform mining operations
            await self._analyze_blockchain_state()
            await self._calculate_hash_difficulty()
            await self._optimize_mining_strategy()
            await self._monitor_mining_pools()

            # Calculate processing load
            processing_load = self._calculate_processing_load()

            # Generate mining decision
            mining_decision = self._generate_mining_decision()

            # Create brain state
            brain_state = BrainState(
                brain_type=self.brain_type,
                thermal_state=self.current_thermal_state,
                active_operations=[op.value for op in MiningOperationType],
                performance_metrics=self.performance_metrics.copy(),
                last_decision=mining_decision,
                decision_confidence=self._calculate_mining_confidence(),
                processing_load=processing_load,
            )

            logger.debug(f"🧠 Left Brain state: {mining_decision} (thermal: {self.current_thermal_state})")
            return brain_state

        except Exception as e:
            logger.error(f"Left brain processing error: {e}")
            return self._create_fallback_brain_state()

    def _update_thermal_state(self) -> None:
        """Update thermal state based on mining operations."""
        # Simulate thermal state based on hash rate and difficulty
        thermal_factor = (self.hash_rate / 100.0) + (self.difficulty_target / 1e12)

        if thermal_factor > 0.8:
            self.current_thermal_state = CRITICAL
        elif thermal_factor > 0.6:
            self.current_thermal_state = HOT  # 32-bit enhanced operations
        elif thermal_factor > 0.3:
            self.current_thermal_state = WARM
        else:
            self.current_thermal_state = COOL

    async def _analyze_blockchain_state(self) -> None:
        """Analyze current blockchain state."""
        try:
            # Simulate blockchain analysis
            current_block_height = 800000 + int(time.time() % 10000)
            block_time = 600 + np.random.normal(0, 60)  # ~10 minutes with variance

            self.block_analysis_results = {
                "current_block": current_block_height,
                "average_block_time": block_time,
                "mempool_size": np.random.randint(50000, 200000),
                "transaction_fees": np.random.uniform(10, 100),
            }

            # Update performance metrics
            self.performance_metrics["block_time_average"] = block_time

        except Exception as e:
            logger.error(f"Blockchain analysis error: {e}")

    async def _calculate_hash_difficulty(self) -> None:
        """Calculate current hash difficulty and targets."""
        try:
            # Simulate difficulty calculation
            base_difficulty = 50e12  # Base difficulty
            difficulty_adjustment = np.random.uniform(0.8, 1.2)

            self.difficulty_target = base_difficulty * difficulty_adjustment

            # Calculate hash rate based on thermal state
            thermal_multipliers = {COOL: 0.8, WARM: 1.0, HOT: 1.3, CRITICAL: 1.6}  # 32-bit enhanced hash rate

            base_hash_rate = 100.0  # TH/s
            thermal_mult = thermal_multipliers.get(self.current_thermal_state, 1.0)
            self.hash_rate = base_hash_rate * thermal_mult

            # Update performance metrics
            self.performance_metrics["hash_rate_th_s"] = self.hash_rate
            self.performance_metrics["difficulty_adjustment"] = difficulty_adjustment

        except Exception as e:
            logger.error(f"Hash difficulty calculation error: {e}")

    async def _optimize_mining_strategy(self) -> None:
        """Optimize mining strategy based on current conditions."""
        try:
            # Calculate mining efficiency
            if self.difficulty_target > 0:
                efficiency = self.hash_rate / (self.difficulty_target / 1e12)
                self.performance_metrics["mining_efficiency"] = min(efficiency, 1.0)

            # Calculate thermal efficiency
            thermal_efficiency = {
                COOL: 0.9,
                WARM: 1.0,
                HOT: 1.2,  # 32-bit thermal optimization
                CRITICAL: 0.8,  # Reduced efficiency at critical
            }.get(self.current_thermal_state, 1.0)

            self.performance_metrics["thermal_efficiency"] = thermal_efficiency

        except Exception as e:
            logger.error(f"Mining optimization error: {e}")

    async def _monitor_mining_pools(self) -> None:
        """Monitor mining pool performance."""
        try:
            # Simulate mining pool data
            total_network_hash = 200000.0  # TH/s

            for pool_name in self.mining_pools:
                # Distribute hash rate among pools
                pool_share = np.random.uniform(0.1, 0.25)
                self.mining_pools[pool_name]["hash_rate"] = total_network_hash * pool_share
                self.mining_pools[pool_name]["block_count"] = np.random.randint(0, 10)

        except Exception as e:
            logger.error(f"Mining pool monitoring error: {e}")

    def _calculate_processing_load(self) -> float:
        """Calculate current processing load."""
        try:
            # Base load from hash rate
            hash_load = min(self.hash_rate / 200.0, 1.0)

            # Thermal load
            thermal_loads = {COOL: 0.2, WARM: 0.5, HOT: 0.8, CRITICAL: 1.0}
            thermal_load = thermal_loads.get(self.current_thermal_state, 0.5)

            # Combined load
            return (hash_load + thermal_load) / 2.0

        except Exception:
            return 0.5

    def _generate_mining_decision(self) -> str:
        """Generate mining-based decision signal."""
        try:
            efficiency = self.performance_metrics.get("mining_efficiency", 0.5)
            thermal_efficiency = self.performance_metrics.get("thermal_efficiency", 1.0)

            combined_efficiency = efficiency * thermal_efficiency

            if combined_efficiency > 0.8:
                return "mining_bullish"
            elif combined_efficiency > 0.6:
                return "mining_positive"
            elif combined_efficiency > 0.4:
                return "mining_neutral"
            else:
                return "mining_bearish"

        except Exception:
            return "mining_neutral"

    def _calculate_mining_confidence(self) -> float:
        """Calculate confidence in mining analysis."""
        try:
            efficiency = self.performance_metrics.get("mining_efficiency", 0.5)
            thermal_efficiency = self.performance_metrics.get("thermal_efficiency", 1.0)

            # Higher confidence in HOT state (32-bit operations)
            thermal_confidence = {COOL: 0.7, WARM: 0.8, HOT: 0.95, CRITICAL: 0.9}.get(  # 32-bit enhanced confidence
                self.current_thermal_state, 0.8
            )

            base_confidence = (efficiency + thermal_efficiency) / 2.0
            return min(base_confidence * thermal_confidence, 1.0)

        except Exception:
            return 0.5

    def _create_fallback_brain_state(self) -> BrainState:
        """Create fallback brain state for error conditions."""
        return BrainState(
            brain_type=self.brain_type,
            thermal_state=WARM,
            active_operations=[],
            performance_metrics={},
            last_decision="mining_neutral",
            decision_confidence=0.5,
            processing_load=0.5,
        )


class RightBrain:
    """
    Trading/Decision Brain - Handles market analysis, whale tracking,
    order execution, and trading decisions with 32-bit thermal integration.
    """

    def __init__(self):
        """Initialize the right brain (trading/decisions)."""
        self.brain_type = BrainType.RIGHT_BRAIN
        self.math_system = UnifiedMathSystem() if CORE_SYSTEMS_AVAILABLE else None
        self.whale_tracker = whale_tracker if CORE_SYSTEMS_AVAILABLE else None
        self.exchange_plumbing = ExchangePlumbing() if CORE_SYSTEMS_AVAILABLE else None
        self.dualistic_engines = DualisticThoughtEngines() if CORE_SYSTEMS_AVAILABLE else None

        # Trading state
        self.current_thermal_state = WARM
        self.trading_operations = []
        self.market_analysis = {}
        self.portfolio_state = {}

        # Performance tracking
        self.performance_metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "thermal_trading_efficiency": 0.0,
        }

        # Market data
        self.market_data = {
            "btc_price": 65000.0,
            "volume_24h": 0.0,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "price_history": [],
            "volume_history": [],
            "rsi": 50.0,
            "macd_signal": 0.0,
            "moving_average": 65000.0,
            "sentiment_score": 0.5,
        }

        logger.info("🧠 Right Brain (Trading/Decisions) initialized")

    async def process_trading_operations(self) -> BrainState:
        """Process trading operations and return brain state."""
        try:
            # Update thermal state based on trading activity
            self._update_thermal_state()

            # Perform trading operations
            await self._analyze_market_conditions()
            await self._monitor_whale_activity()
            await self._assess_portfolio_risk()

            # Generate trading decision using the DualisticThoughtEngines
            if not self.dualistic_engines:
                raise Exception("DualisticThoughtEngines not available.")

            thought_vector = self.dualistic_engines.process_market_data(
                self.market_analysis, self.current_thermal_state
            )

            trading_decision = thought_vector.decision
            trading_confidence = thought_vector.confidence

            # Calculate processing load
            processing_load = self._calculate_processing_load()

            # Create brain state
            brain_state = BrainState(
                brain_type=self.brain_type,
                thermal_state=self.current_thermal_state,
                active_operations=[op.value for op in TradingOperationType],
                performance_metrics=self.performance_metrics.copy(),
                last_decision=trading_decision,
                decision_confidence=trading_confidence,
                processing_load=processing_load,
            )

            logger.debug(
                f"🧠 Right Brain state: {trading_decision} (thermal: {
                    self.current_thermal_state}) via Dualistic Engines"
            )
            return brain_state

        except Exception as e:
            logger.error(f"Right brain processing error: {e}")
            return self._create_fallback_brain_state()

    def _update_thermal_state(self) -> None:
        """Update thermal state based on trading activity."""
        # Base thermal state on market volatility and trading volume
        volatility = self.market_data.get("volatility", 0.0)
        volume = self.market_data.get("volume_24h", 0.0)

        thermal_factor = volatility + (volume / 50000000000.0)  # Normalize volume

        if thermal_factor > 0.8:
            self.current_thermal_state = CRITICAL
        elif thermal_factor > 0.6:
            self.current_thermal_state = HOT  # 32-bit enhanced trading
        elif thermal_factor > 0.3:
            self.current_thermal_state = WARM
        else:
            self.current_thermal_state = COOL

    async def _analyze_market_conditions(self) -> None:
        """Analyze current market conditions."""
        try:
            # Simulate market analysis
            base_price = 65000.0
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price = base_price * (1 + price_change)

            # Populate a more detailed market_analysis dictionary for the dualistic engine
            self.market_analysis = {
                "current_price": current_price,
                "volume_24h": np.random.uniform(20e9, 50e9),
                "volatility": abs(price_change),
                "trend_strength": np.random.uniform(-1.0, 1.0),
                "rsi": 50 + np.random.uniform(-20, 20),
                "macd_signal": np.random.uniform(-0.05, 0.05),
                "moving_average": current_price * (1 + np.random.uniform(-0.01, 0.01)),
                "sentiment_score": np.random.uniform(0.2, 0.8),
                "volume_change": np.random.uniform(-0.3, 0.3),
                "previous_close": base_price,
                "price_history": [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(50)],
                "volume_history": [np.random.uniform(100, 500) for _ in range(50)],
                "phase_data": [np.random.rand() for _ in range(4)],
                "actual_profit_from_last_trade": self.performance_metrics.get("last_trade_profit", 0.0),
            }

            # Update performance metrics based on thermal state
            thermal_multiplier = {COOL: 0.8, WARM: 1.0, HOT: 1.3, CRITICAL: 1.6}.get(  # 32-bit enhanced analysis
                self.current_thermal_state, 1.0
            )

            self.performance_metrics["thermal_trading_efficiency"] = thermal_multiplier

        except Exception as e:
            logger.error(f"Market analysis error: {e}")

    async def _monitor_whale_activity(self) -> None:
        """Monitor whale activity and integrate with trading decisions."""
        try:
            if self.whale_tracker:
                whale_summary = self.whale_tracker.get_whale_summary()

                # Extract whale metrics
                whale_stats = whale_summary.get("statistics", {})
                accumulation_score = whale_stats.get("whale_accumulation_score", 0.0)
                alert_level = whale_stats.get("current_alert_level", "none")

                # Update market analysis with whale data
                self.market_analysis["whale_accumulation"] = accumulation_score
                self.market_analysis["whale_alert_level"] = alert_level
                self.market_analysis["consensus_signal"] = self._get_whale_consensus(whale_summary)

        except Exception as e:
            logger.error(f"Whale monitoring error: {e}")

    def _get_whale_consensus(self, whale_summary: Dict[str, Any]) -> str:
        """Derive a consensus signal from whale activity."""
        recent_alerts = whale_summary.get("recent_alerts", [])
        if not recent_alerts:
            return "neutral"

        last_alert = recent_alerts[0]
        if "buy" in last_alert.get("thermal_recommendation", ""):
            return "buy"
        if "sell" in last_alert.get("thermal_recommendation", ""):
            return "sell"

        return "neutral"

    async def _assess_portfolio_risk(self) -> None:
        """Assess current portfolio risk."""
        try:
            # Simulate portfolio analysis
            positions = {"BTC": {"amount": 0.5, "value": 32500.0}, "USDC": {"amount": 10000.0, "value": 10000.0}}

            total_value = sum(pos["value"] for pos in positions.values())
            btc_allocation = positions["BTC"]["value"] / total_value

            # Calculate risk metrics
            portfolio_risk = btc_allocation * self.market_data.get("volatility", 0.0)

            self.portfolio_state = {
                "total_value": total_value,
                "btc_allocation": btc_allocation,
                "portfolio_risk": portfolio_risk,
                "positions": positions,
            }

            # Update performance metrics
            self.performance_metrics["max_drawdown"] = portfolio_risk * 0.5

        except Exception as e:
            logger.error(f"Portfolio risk assessment error: {e}")

    def _calculate_processing_load(self) -> float:
        """Calculate current processing load."""
        try:
            # Base load from market activity
            volatility = self.market_data.get("volatility", 0.0)
            market_load = min(volatility * 10.0, 1.0)

            # Thermal load
            thermal_loads = {COOL: 0.2, WARM: 0.5, HOT: 0.8, CRITICAL: 1.0}
            thermal_load = thermal_loads.get(self.current_thermal_state, 0.5)

            # Combined load
            return (market_load + thermal_load) / 2.0

        except Exception:
            return 0.5

    def _create_fallback_brain_state(self) -> BrainState:
        """Create fallback brain state for error conditions."""
        return BrainState(
            brain_type=self.brain_type,
            thermal_state=WARM,
            active_operations=[],
            performance_metrics={},
            last_decision="trading_neutral",
            decision_confidence=0.5,
            processing_load=0.5,
        )


class DualBrainArchitecture:
    """
    Main dual-brain architecture coordinator that synchronizes
    mining/hashing brain with trading/decision brain.
    """

    def __init__(self):
        """Initialize dual brain architecture."""
        self.left_brain = LeftBrain()  # Mining/Hashing
        self.right_brain = RightBrain()  # Trading/Decisions

        # Synchronization state
        self.last_synchronization = None
        self.decision_history = []
        self.flip_logic_results = []

        # Performance tracking
        self.architecture_metrics = {
            "synchronization_rate": 0.0,
            "decision_accuracy": 0.0,
            "profit_realization": 0.0,
            "thermal_optimization": 0.0,
            "last_trade_profit": 0.0,  # Track profit for historical consultation
        }

        logger.info("🧠🧠 Dual Brain Architecture initialized")

    async def run_dual_brain_cycle(self) -> DualBrainDecision:
        """Run a complete dual brain processing cycle."""
        try:
            # Process both brains in parallel
            left_task = self.left_brain.process_mining_operations()
            right_task = self.right_brain.process_trading_operations()

            left_state, right_state = await asyncio.gather(left_task, right_task)

            # Synchronize brains and apply flip logic
            flip_result = self._apply_flip_logic(left_state, right_state)

            # Generate unified decision
            unified_decision = self._generate_unified_decision(left_state, right_state, flip_result)

            # Update architecture metrics and track profit for learning
            self._update_architecture_metrics(unified_decision)

            # Store in history
            self.decision_history.append(unified_decision)
            self.flip_logic_results.append(flip_result)

            # Keep history manageable
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
                self.flip_logic_results = self.flip_logic_results[-500:]

            self.last_synchronization = datetime.now()

            logger.info(
                f"🧠🧠 Dual brain decision: {unified_decision.synchronized_action} "
                f"(confidence: {unified_decision.overall_confidence:.3f})"
            )

            return unified_decision

        except Exception as e:
            logger.error(f"Dual brain cycle error: {e}")
            return self._create_fallback_decision()

    def _apply_flip_logic(self, left_state: BrainState, right_state: BrainState) -> FlipLogicResult:
        """Apply flip logic to combine mining and trading signals."""
        try:
            # Extract decision signals
            mining_signal = left_state.last_decision or "mining_neutral"
            trading_signal = right_state.last_decision or "trading_neutral"

            # Calculate signal strengths
            mining_strength = self._calculate_signal_strength(mining_signal, "mining")
            trading_strength = self._calculate_signal_strength(trading_signal, "trading")

            # Determine thermal state (use the higher thermal state)
            thermal_states = [COOL, WARM, HOT, CRITICAL]
            left_thermal_idx = thermal_states.index(left_state.thermal_state)
            right_thermal_idx = thermal_states.index(right_state.thermal_state)
            dominant_thermal = thermal_states[max(left_thermal_idx, right_thermal_idx)]

            # Calculate thermal multiplier
            thermal_multipliers = {COOL: 0.8, WARM: 1.0, HOT: 1.4, CRITICAL: 1.8}  # 32-bit enhanced flip logic
            thermal_multiplier = thermal_multipliers.get(dominant_thermal, 1.0)

            # Apply flip logic (XOR-like operation for signal combination)
            flip_strength = self._calculate_flip_strength(mining_strength, trading_strength, thermal_multiplier)

            # Determine flip signal
            flip_signal = self._determine_flip_signal(flip_strength)

            # Calculate confidence
            confidence = self._calculate_flip_confidence(
                left_state.decision_confidence, right_state.decision_confidence, thermal_multiplier
            )

            # Calculate profit potential
            profit_potential = self._calculate_profit_potential(mining_strength, trading_strength, thermal_multiplier)

            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(left_state.processing_load, right_state.processing_load)

            # Generate reasoning
            reasoning = self._generate_flip_reasoning(mining_signal, trading_signal, dominant_thermal, flip_signal)

            return FlipLogicResult(
                flip_signal=flip_signal,
                confidence=confidence,
                mining_contribution=mining_strength,
                trading_contribution=trading_strength,
                thermal_multiplier=thermal_multiplier,
                reasoning=reasoning,
                profit_potential=profit_potential,
                risk_assessment=risk_assessment,
            )

        except Exception as e:
            logger.error(f"Flip logic error: {e}")
            return self._create_fallback_flip_result()

    def _calculate_signal_strength(self, signal: str, signal_type: str) -> float:
        """Calculate numerical strength of a signal."""
        if signal_type == "mining":
            signal_map = {"mining_bullish": 0.8, "mining_positive": 0.6, "mining_neutral": 0.0, "mining_bearish": -0.8}
        else:  # trading
            signal_map = {
                "trading_very_bullish": 1.0,
                "trading_bullish": 0.8,
                "trading_positive": 0.6,
                "trading_neutral": 0.0,
                "trading_negative": -0.6,
                "trading_bearish": -0.8,
                "trading_very_bearish": -1.0,
            }

        return signal_map.get(signal, 0.0)

    def _calculate_flip_strength(
        self, mining_strength: float, trading_strength: float, thermal_multiplier: float
    ) -> float:
        """Calculate combined flip strength using mathematical operations."""
        try:
            # XOR-like operation: strong when signals agree or when one is very strong
            if (mining_strength > 0 and trading_strength > 0) or (mining_strength < 0 and trading_strength < 0):
                # Signals agree - amplify
                flip_strength = (mining_strength + trading_strength) / 2.0
            else:
                # Signals disagree - take the stronger signal but reduce it
                if abs(mining_strength) > abs(trading_strength):
                    flip_strength = mining_strength * 0.7
                else:
                    flip_strength = trading_strength * 0.7

            # Apply thermal multiplier
            return flip_strength * thermal_multiplier

        except Exception:
            return 0.0

    def _determine_flip_signal(self, flip_strength: float) -> FlipLogicSignal:
        """Determine flip logic signal from strength."""
        if flip_strength > 0.7:
            return FlipLogicSignal.STRONG_BUY
        elif flip_strength > 0.3:
            return FlipLogicSignal.MODERATE_BUY
        elif flip_strength > -0.3:
            return FlipLogicSignal.HOLD
        elif flip_strength > -0.7:
            return FlipLogicSignal.MODERATE_SELL
        else:
            return FlipLogicSignal.STRONG_SELL

    def _calculate_flip_confidence(self, left_conf: float, right_conf: float, thermal_mult: float) -> float:
        """Calculate flip logic confidence."""
        # Combined confidence with thermal enhancement
        base_confidence = (left_conf + right_conf) / 2.0

        # Thermal enhancement for confidence
        if thermal_mult > 1.3:  # HOT or CRITICAL state
            thermal_conf_boost = min(thermal_mult - 1.0, 0.3)
            return min(base_confidence + thermal_conf_boost, 1.0)

        return base_confidence

    def _calculate_profit_potential(
        self, mining_strength: float, trading_strength: float, thermal_mult: float
    ) -> float:
        """Calculate profit potential from flip logic."""
        # Profit potential is higher when signals are strong and aligned
        signal_alignment = 1.0 - abs(mining_strength - trading_strength) / 2.0
        signal_magnitude = (abs(mining_strength) + abs(trading_strength)) / 2.0

        base_profit = signal_alignment * signal_magnitude * 0.5

        # Thermal enhancement
        return base_profit * thermal_mult

    def _calculate_risk_assessment(self, left_load: float, right_load: float) -> float:
        """Calculate risk assessment from processing loads."""
        # Higher processing loads indicate higher risk
        avg_load = (left_load + right_load) / 2.0
        return min(avg_load * 1.5, 1.0)

    def _generate_flip_reasoning(
        self, mining_signal: str, trading_signal: str, thermal_state: str, flip_signal: FlipLogicSignal
    ) -> str:
        """Generate human-readable reasoning for flip logic."""
        return (
            f"Mining: {mining_signal}, Trading: {trading_signal}, " f"Thermal: {thermal_state} -> {flip_signal.value}"
        )

    def _generate_unified_decision(
        self, left_state: BrainState, right_state: BrainState, flip_result: FlipLogicResult
    ) -> DualBrainDecision:
        """Generate unified decision from both brains and flip logic."""
        try:
            # Map flip signal to action
            action_map = {
                FlipLogicSignal.STRONG_BUY: "execute_aggressive_buy",
                FlipLogicSignal.MODERATE_BUY: "execute_moderate_buy",
                FlipLogicSignal.HOLD: "maintain_position",
                FlipLogicSignal.MODERATE_SELL: "execute_moderate_sell",
                FlipLogicSignal.STRONG_SELL: "execute_aggressive_sell",
                FlipLogicSignal.WAIT_SIGNAL: "wait_for_signal",
            }

            synchronized_action = action_map.get(flip_result.flip_signal, "maintain_position")

            # Calculate overall confidence
            overall_confidence = flip_result.confidence

            # Calculate expected profit
            expected_profit = flip_result.profit_potential * 1000.0  # Scale to USD

            # Determine if thermal enhancement is active
            thermal_enhancement = flip_result.thermal_multiplier > 1.2

            # Calculate execution priority
            priority_map = {
                FlipLogicSignal.STRONG_BUY: 1,
                FlipLogicSignal.STRONG_SELL: 1,
                FlipLogicSignal.MODERATE_BUY: 2,
                FlipLogicSignal.MODERATE_SELL: 2,
                FlipLogicSignal.HOLD: 3,
                FlipLogicSignal.WAIT_SIGNAL: 4,
            }
            execution_priority = priority_map.get(flip_result.flip_signal, 3)

            return DualBrainDecision(
                left_brain_state=left_state,
                right_brain_state=right_state,
                flip_logic_result=flip_result,
                synchronized_action=synchronized_action,
                overall_confidence=overall_confidence,
                expected_profit=expected_profit,
                thermal_enhancement=thermal_enhancement,
                execution_priority=execution_priority,
            )

        except Exception as e:
            logger.error(f"Unified decision generation error: {e}")
            return self._create_fallback_decision()

    def _update_architecture_metrics(self, decision: DualBrainDecision) -> None:
        """Update architecture performance metrics."""
        try:
            # Update synchronization rate
            if self.last_synchronization:
                time_since_sync = (datetime.now() - self.last_synchronization).total_seconds()
                self.architecture_metrics["synchronization_rate"] = 1.0 / max(time_since_sync, 1.0)

            # Update thermal optimization metric
            if decision.thermal_enhancement:
                current_thermal = self.architecture_metrics["thermal_optimization"]
                self.architecture_metrics["thermal_optimization"] = min(current_thermal + 0.1, 1.0)

            # Simulate profit realization and store for next cycle's historical consultation
            # In a real system, this would come from filled order execution data.
            if "buy" in decision.synchronized_action and decision.flip_logic_result.profit_potential > 0:
                realized_profit = decision.expected_profit * np.random.uniform(0.5, 1.2)
            elif "sell" in decision.synchronized_action and decision.flip_logic_result.profit_potential < 0:
                realized_profit = abs(decision.expected_profit) * np.random.uniform(0.5, 1.2)
            else:
                realized_profit = decision.expected_profit * np.random.uniform(-0.5, 0.5)

            self.architecture_metrics["profit_realization"] += realized_profit
            self.architecture_metrics["last_trade_profit"] = realized_profit

        except Exception as e:
            logger.error(f"Architecture metrics update error: {e}")

    def _create_fallback_flip_result(self) -> FlipLogicResult:
        """Create fallback flip logic result."""
        return FlipLogicResult(
            flip_signal=FlipLogicSignal.HOLD,
            confidence=0.5,
            mining_contribution=0.0,
            trading_contribution=0.0,
            thermal_multiplier=1.0,
            reasoning="Fallback due to error",
            profit_potential=0.0,
            risk_assessment=0.5,
        )

    def _create_fallback_decision(self) -> DualBrainDecision:
        """Create fallback decision for error conditions."""
        fallback_state = BrainState(
            brain_type=BrainType.LEFT_BRAIN,
            thermal_state=WARM,
            active_operations=[],
            performance_metrics={},
            last_decision="neutral",
            decision_confidence=0.5,
            processing_load=0.5,
        )

        fallback_flip = self._create_fallback_flip_result()

        return DualBrainDecision(
            left_brain_state=fallback_state,
            right_brain_state=fallback_state,
            flip_logic_result=fallback_flip,
            synchronized_action="maintain_position",
            overall_confidence=0.5,
            expected_profit=0.0,
            thermal_enhancement=False,
            execution_priority=3,
        )

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get comprehensive architecture summary."""
        return {
            "architecture_metrics": self.architecture_metrics,
            "left_brain_metrics": self.left_brain.performance_metrics,
            "right_brain_metrics": self.right_brain.performance_metrics,
            "recent_decisions": [
                {
                    "action": decision.synchronized_action,
                    "confidence": decision.overall_confidence,
                    "thermal_enhancement": decision.thermal_enhancement,
                    "expected_profit": decision.expected_profit,
                    "timestamp": decision.timestamp.isoformat(),
                }
                for decision in self.decision_history[-10:]
            ],
            "current_thermal_states": {
                "left_brain": self.left_brain.current_thermal_state,
                "right_brain": self.right_brain.current_thermal_state,
            },
            "last_synchronization": self.last_synchronization.isoformat() if self.last_synchronization else None,
        }


# Global instance for easy access
dual_brain = DualBrainArchitecture()


# Example usage and testing
if __name__ == "__main__":
    print("🧠🧠 Dual Brain Architecture System")
    print("=" * 50)

    async def demo_dual_brain():
        """Demonstrate dual brain functionality."""
        architecture = DualBrainArchitecture()

        # Run a few cycles
        for i in range(3):
            print(f"\n--- Cycle {i + 1} ---")
            decision = await architecture.run_dual_brain_cycle()

            print(f"🧠 Left Brain (Mining): {decision.left_brain_state.last_decision}")
            print(f"   Thermal: {decision.left_brain_state.thermal_state}")
            print(f"   Confidence: {decision.left_brain_state.decision_confidence:.3f}")

            print(f"🧠 Right Brain (Trading): {decision.right_brain_state.last_decision}")
            print(f"   Thermal: {decision.right_brain_state.thermal_state}")
            print(f"   Confidence: {decision.right_brain_state.decision_confidence:.3f}")

            print(f"⚡ Flip Logic: {decision.flip_logic_result.flip_signal.value}")
            print(f"   Reasoning: {decision.flip_logic_result.reasoning}")
            print(f"   Thermal Multiplier: {decision.flip_logic_result.thermal_multiplier:.2f}")

            print(f"🎯 Unified Decision: {decision.synchronized_action}")
            print(f"   Overall Confidence: {decision.overall_confidence:.3f}")
            print(f"   Expected Profit: ${decision.expected_profit:.2f}")
            print(f"   Thermal Enhancement: {decision.thermal_enhancement}")

            # Wait between cycles
            await asyncio.sleep(1)

        # Show summary
        summary = architecture.get_architecture_summary()
        print(f"\n📊 Architecture Summary:")
        print(f"   Architecture Metrics: {summary['architecture_metrics']}")
        print(f"   Current Thermal States: {summary['current_thermal_states']}")

    # Run demo
    import asyncio

    asyncio.run(demo_dual_brain())
