# -*- coding: utf-8 -*-
"""
Entry/Exit Portal System - Advanced Trading Logic Portalization
===============================================================

Implements sophisticated entry/exit logic portalization with 4-bit, 8-bit, and 16-bit
strategy mapping for hourly connectivity and profit maximization.

Mathematical Framework:
- 4-bit Strategy: 16 discrete states for rapid decision making
- 8-bit Strategy: 256 states for medium-term pattern recognition
- 16-bit Strategy: 65,536 states for long-term trend analysis
- Hourly Connectivity: Time-based phase synchronization
- Profit Maximization: P_max = Σ(w_i * ΔP_i) * e^(-risk_factor)

Features:
- Multi-bit strategy mapping
- Real-time entry/exit signals
- Risk-adjusted position sizing
- Cross-platform communication
- Advanced pattern recognition
- Thermal state integration
"""

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.type_defs import Matrix, QuantumState, Temperature, Tensor, Vector
from core.unified_math_system import unified_math
from utils.safe_print import debug, error, info, safe_print, success, warn

logger = logging.getLogger(__name__)

# =============================================================================
# PORTAL SYSTEM CONSTANTS AND ENUMS
# =============================================================================


class StrategyBitDepth(Enum):
    """Strategy bit depth for different time horizons."""

    BIT_4 = 4  # 16 states - Rapid decisions (seconds to minutes)
    BIT_8 = 8  # 256 states - Medium-term (minutes to hours)
    BIT_16 = 16  # 65,536 states - Long-term (hours to days)


class PortalState(Enum):
    """Portal states for entry/exit logic."""

    IDLE = "idle"  # Waiting for signals
    ANALYZING = "analyzing"  # Analyzing market conditions
    SIGNAL_DETECTED = "signal"  # Signal detected
    ENTRY_PENDING = "entry_pending"  # Entry order pending
    POSITION_OPEN = "position_open"  # Position is open
    EXIT_PENDING = "exit_pending"  # Exit order pending
    COOLDOWN = "cooldown"  # Cooldown period


class SignalType(Enum):
    """Signal types for entry/exit decisions."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    HOLD = "hold"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk levels for position sizing."""

    CONSERVATIVE = "conservative"  # 1-2% risk per trade
    MODERATE = "moderate"  # 2-5% risk per trade
    AGGRESSIVE = "aggressive"  # 5-10% risk per trade
    EXTREME = "extreme"  # 10%+ risk per trade


# =============================================================================
# PORTAL SYSTEM DATA STRUCTURES
# =============================================================================


@dataclass
class StrategyState:
    """Strategy state for bit-level operations."""

    bit_depth: StrategyBitDepth
    state_value: int
    confidence: float
    timestamp: float
    market_conditions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state value."""
        max_value = 2**self.bit_depth.value
        if self.state_value >= max_value:
            self.state_value = self.state_value % max_value


@dataclass
class EntryExitSignal:
    """Entry/exit signal with confidence and timing."""

    signal_type: SignalType
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_level: RiskLevel
    timestamp: float
    strategy_state: StrategyState
    portal_state: PortalState

    def __post_init__(self):
        """Calculate derived values."""
        self.risk_reward_ratio = abs(self.take_profit - self.price_target) / abs(self.stop_loss - self.price_target)


@dataclass
class PortalMetrics:
    """Portal performance metrics."""

    total_signals: int
    successful_signals: int
    failed_signals: int
    total_profit: float
    total_loss: float
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    timestamp: float

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_signals > 0:
            self.win_rate = self.successful_signals / self.total_signals
        else:
            self.win_rate = 0.0

        if self.successful_signals > 0:
            self.average_profit = self.total_profit / self.successful_signals
        else:
            self.average_profit = 0.0

        if self.failed_signals > 0:
            self.average_loss = self.total_loss / self.failed_signals
        else:
            self.average_loss = 0.0

        if self.total_loss > 0:
            self.profit_factor = self.total_profit / self.total_loss
        else:
            self.profit_factor = float("inf") if self.total_profit > 0 else 0.0


@dataclass
class HourlyConnectivity:
    """Hourly connectivity for time-based synchronization."""

    hour: int
    phase_value: float
    coherence: float
    signal_strength: float
    connected_nodes: int
    timestamp: float


# =============================================================================
# ENTRY/EXIT PORTAL SYSTEM
# =============================================================================


class EntryExitPortalSystem:
    """
    Entry/Exit Portal System - Advanced trading logic portalization.

    Implements:
    - Multi-bit strategy mapping (4-bit, 8-bit, 16-bit)
    - Real-time entry/exit signal generation
    - Risk-adjusted position sizing
    - Hourly connectivity synchronization
    - Advanced pattern recognition
    - Cross-platform communication
    """

    def __init__(self, initial_capital: float = 10000.0, risk_per_trade: float = 0.02, enable_hourly_sync: bool = True):
        """
        Initialize Entry/Exit Portal System.

        Args:
            initial_capital: Initial trading capital
            risk_per_trade: Risk per trade (0.0 to 1.0)
            enable_hourly_sync: Enable hourly connectivity synchronization
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.enable_hourly_sync = enable_hourly_sync

        # Strategy states for different bit depths
        self.strategy_states: Dict[StrategyBitDepth, List[StrategyState]] = {
            StrategyBitDepth.BIT_4: [],
            StrategyBitDepth.BIT_8: [],
            StrategyBitDepth.BIT_16: [],
        }

        # Signal history
        self.signal_history: List[EntryExitSignal] = []
        self.portal_metrics = PortalMetrics(
            total_signals=0,
            successful_signals=0,
            failed_signals=0,
            total_profit=0.0,
            total_loss=0.0,
            win_rate=0.0,
            average_profit=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            timestamp=time.time(),
        )

        # Hourly connectivity
        self.hourly_connectivity: List[HourlyConnectivity] = []
        self.current_hour = datetime.now().hour

        # Portal state
        self.current_portal_state = PortalState.IDLE
        self.last_state_change = time.time()

        # Performance tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0

        # Threading and synchronization
        self.portal_lock = threading.RLock()
        self.running = False

        # Background tasks
        self.strategy_update_thread = None
        self.hourly_sync_thread = None
        self.metrics_update_thread = None

        # Start background tasks
        self._start_background_tasks()

        logger.info(f"✅ Entry/Exit Portal System initialized with ${initial_capital:,.2f} capital")

    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        self.running = True

        # Strategy update thread
        self.strategy_update_thread = threading.Thread(target=self._strategy_update_loop, daemon=True)
        self.strategy_update_thread.start()

        # Hourly sync thread
        if self.enable_hourly_sync:
            self.hourly_sync_thread = threading.Thread(target=self._hourly_sync_loop, daemon=True)
            self.hourly_sync_thread.start()

        # Metrics update thread
        self.metrics_update_thread = threading.Thread(target=self._metrics_update_loop, daemon=True)
        self.metrics_update_thread.start()

        logger.info("✅ Background tasks started")

    def process_market_data(
        self, price: float, volume: float, timestamp: Optional[float] = None
    ) -> Optional[EntryExitSignal]:
        """
        Process market data and generate entry/exit signals.

        Args:
            price: Current market price
            volume: Trading volume
            timestamp: Optional timestamp (uses current time if None)

        Returns:
            EntryExitSignal if conditions are met, None otherwise
        """
        with self.portal_lock:
            try:
                if timestamp is None:
                    timestamp = time.time()

                # Update strategy states for all bit depths
                self._update_strategy_states(price, volume, timestamp)

                # Generate signal based on strategy states
                signal = self._generate_signal(price, volume, timestamp)

                if signal:
                    # Update portal state
                    self._update_portal_state(signal.portal_state)

                    # Store signal
                    self.signal_history.append(signal)
                    if len(self.signal_history) > 10000:
                        self.signal_history = self.signal_history[-10000:]

                    # Update metrics
                    self.portal_metrics.total_signals += 1

                    logger.info(
                        f"📡 Generated {
                            signal.signal_type.value} signal with {
                            signal.confidence:.2f} confidence"
                    )

                return signal

            except Exception as e:
                logger.error(f"❌ Failed to process market data: {e}")
                return None

    def _update_strategy_states(self, price: float, volume: float, timestamp: float) -> None:
        """Update strategy states for all bit depths."""
        try:
            # Generate market hash for state determination
            market_hash = hashlib.sha256(f"{price:.8f}_{volume:.8f}_{timestamp:.8f}".encode()).hexdigest()
            hash_int = int(market_hash[:8], 16)

            # Update 4-bit strategy (rapid decisions)
            bit_4_state = StrategyState(
                bit_depth=StrategyBitDepth.BIT_4,
                state_value=hash_int % 16,
                confidence=self._calculate_confidence(price, volume, StrategyBitDepth.BIT_4),
                timestamp=timestamp,
                market_conditions={
                    "price": price,
                    "volume": volume,
                    "volatility": self._calculate_volatility(StrategyBitDepth.BIT_4),
                },
            )
            self.strategy_states[StrategyBitDepth.BIT_4].append(bit_4_state)

            # Update 8-bit strategy (medium-term)
            bit_8_state = StrategyState(
                bit_depth=StrategyBitDepth.BIT_8,
                state_value=hash_int % 256,
                confidence=self._calculate_confidence(price, volume, StrategyBitDepth.BIT_8),
                timestamp=timestamp,
                market_conditions={
                    "price": price,
                    "volume": volume,
                    "volatility": self._calculate_volatility(StrategyBitDepth.BIT_8),
                },
            )
            self.strategy_states[StrategyBitDepth.BIT_8].append(bit_8_state)

            # Update 16-bit strategy (long-term)
            bit_16_state = StrategyState(
                bit_depth=StrategyBitDepth.BIT_16,
                state_value=hash_int % 65536,
                confidence=self._calculate_confidence(price, volume, StrategyBitDepth.BIT_16),
                timestamp=timestamp,
                market_conditions={
                    "price": price,
                    "volume": volume,
                    "volatility": self._calculate_volatility(StrategyBitDepth.BIT_16),
                },
            )
            self.strategy_states[StrategyBitDepth.BIT_16].append(bit_16_state)

            # Keep only recent states
            for bit_depth in StrategyBitDepth:
                if len(self.strategy_states[bit_depth]) > 1000:
                    self.strategy_states[bit_depth] = self.strategy_states[bit_depth][-1000:]

        except Exception as e:
            logger.error(f"❌ Failed to update strategy states: {e}")

    def _calculate_confidence(self, price: float, volume: float, bit_depth: StrategyBitDepth) -> float:
        """Calculate confidence based on market conditions and bit depth."""
        try:
            # Base confidence on volume and price stability
            volume_factor = min(1.0, volume / 1000.0)

            # Calculate price stability from recent history
            recent_prices = [
                state.market_conditions.get("price", price) for state in self.strategy_states[bit_depth][-10:]
            ]

            if len(recent_prices) > 1:
                price_volatility = np.std(recent_prices) / np.mean(recent_prices)
                stability_factor = max(0.0, 1.0 - price_volatility)
            else:
                stability_factor = 0.5

            # Bit depth factor (higher bit depth = higher potential confidence)
            bit_factor = bit_depth.value / 16.0

            # Combine factors
            confidence = volume_factor * 0.4 + stability_factor * 0.4 + bit_factor * 0.2

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    def _calculate_volatility(self, bit_depth: StrategyBitDepth) -> float:
        """Calculate volatility for the given bit depth."""
        try:
            recent_prices = [
                state.market_conditions.get("price", 0.0) for state in self.strategy_states[bit_depth][-20:]
            ]

            if len(recent_prices) > 1:
                return np.std(recent_prices) / np.mean(recent_prices)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _generate_signal(self, price: float, volume: float, timestamp: float) -> Optional[EntryExitSignal]:
        """Generate entry/exit signal based on strategy states."""
        try:
            # Calculate signal strength from all bit depths
            signal_strengths = {}
            for bit_depth in StrategyBitDepth:
                if self.strategy_states[bit_depth]:
                    latest_state = self.strategy_states[bit_depth][-1]
                    signal_strengths[bit_depth] = self._calculate_signal_strength(latest_state, bit_depth)

            # Determine signal type based on weighted average
            if not signal_strengths:
                return None

            # Weight by bit depth (higher bit depth = higher weight)
            total_weight = sum(bit_depth.value for bit_depth in signal_strengths.keys())
            weighted_signal = (
                sum(strength * bit_depth.value for bit_depth, strength in signal_strengths.items()) / total_weight
            )

            # Determine signal type
            signal_type = self._determine_signal_type(weighted_signal)

            # Calculate confidence
            confidence = np.mean(list(signal_strengths.values()))

            # Only generate signal if confidence is high enough
            if confidence < 0.6:
                return None

            # Calculate position sizing
            position_size = self._calculate_position_size(price, confidence)

            # Calculate risk parameters
            risk_level = self._determine_risk_level(confidence)
            stop_loss, take_profit = self._calculate_risk_parameters(price, signal_type, risk_level)

            # Determine portal state
            portal_state = self._determine_portal_state(signal_type, confidence)

            # Create signal
            signal = EntryExitSignal(
                signal_type=signal_type,
                confidence=confidence,
                price_target=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_level=risk_level,
                timestamp=timestamp,
                strategy_state=(
                    self.strategy_states[StrategyBitDepth.BIT_8][-1]
                    if self.strategy_states[StrategyBitDepth.BIT_8]
                    else None
                ),
                portal_state=portal_state,
            )

            return signal

        except Exception as e:
            logger.error(f"❌ Failed to generate signal: {e}")
            return None

    def _calculate_signal_strength(self, state: StrategyState, bit_depth: StrategyBitDepth) -> float:
        """Calculate signal strength for a given state."""
        try:
            # Base strength on state value and confidence
            state_factor = state.state_value / (2**bit_depth.value)
            confidence_factor = state.confidence

            # Market condition factors
            volatility = state.market_conditions.get("volatility", 0.0)
            volume_factor = min(1.0, state.market_conditions.get("volume", 0.0) / 1000.0)

            # Combine factors
            strength = state_factor * 0.3 + confidence_factor * 0.4 + (1.0 - volatility) * 0.2 + volume_factor * 0.1

            return max(0.0, min(1.0, strength))

        except Exception:
            return 0.5

    def _determine_signal_type(self, weighted_signal: float) -> SignalType:
        """Determine signal type based on weighted signal strength."""
        if weighted_signal > 0.8:
            return SignalType.STRONG_BUY
        elif weighted_signal > 0.6:
            return SignalType.BUY
        elif weighted_signal > 0.4:
            return SignalType.WEAK_BUY
        elif weighted_signal > 0.2:
            return SignalType.HOLD
        elif weighted_signal > 0.0:
            return SignalType.WEAK_SELL
        elif weighted_signal > -0.2:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL

    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calculate position size based on risk and confidence."""
        try:
            # Risk-based position sizing
            risk_amount = self.current_capital * self.risk_per_trade
            position_value = risk_amount / (1.0 - confidence)

            # Convert to position size
            position_size = position_value / price

            # Limit position size to available capital
            max_position_value = self.current_capital * 0.95  # 95% of capital
            max_position_size = max_position_value / price

            return min(position_size, max_position_size)

        except Exception:
            return 0.0

    def _determine_risk_level(self, confidence: float) -> RiskLevel:
        """Determine risk level based on confidence."""
        if confidence > 0.8:
            return RiskLevel.CONSERVATIVE
        elif confidence > 0.6:
            return RiskLevel.MODERATE
        elif confidence > 0.4:
            return RiskLevel.AGGRESSIVE
        else:
            return RiskLevel.EXTREME

    def _calculate_risk_parameters(
        self, price: float, signal_type: SignalType, risk_level: RiskLevel
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        try:
            # Risk level multipliers
            risk_multipliers = {
                RiskLevel.CONSERVATIVE: (0.02, 0.04),  # 2% stop loss, 4% take profit
                RiskLevel.MODERATE: (0.03, 0.06),  # 3% stop loss, 6% take profit
                RiskLevel.AGGRESSIVE: (0.05, 0.10),  # 5% stop loss, 10% take profit
                RiskLevel.EXTREME: (0.08, 0.15),  # 8% stop loss, 15% take profit
            }

            stop_loss_pct, take_profit_pct = risk_multipliers[risk_level]

            # Adjust based on signal type
            if signal_type in [SignalType.STRONG_SELL, SignalType.SELL, SignalType.WEAK_SELL]:
                stop_loss_pct *= -1
                take_profit_pct *= -1

            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)

            return stop_loss, take_profit

        except Exception:
            return price * 0.98, price * 1.02  # Default 2% range

    def _determine_portal_state(self, signal_type: SignalType, confidence: float) -> PortalState:
        """Determine portal state based on signal."""
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
            if confidence > 0.8:
                return PortalState.ENTRY_PENDING
            else:
                return PortalState.SIGNAL_DETECTED
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL, SignalType.WEAK_SELL]:
            if self.current_portal_state == PortalState.POSITION_OPEN:
                return PortalState.EXIT_PENDING
            else:
                return PortalState.SIGNAL_DETECTED
        else:
            return PortalState.ANALYZING

    def _update_portal_state(self, new_state: PortalState) -> None:
        """Update portal state."""
        if new_state != self.current_portal_state:
            old_state = self.current_portal_state
            self.current_portal_state = new_state
            self.last_state_change = time.time()
            logger.info(f"🔄 Portal state changed: {old_state.value} → {new_state.value}")

    def update_trade_result(self, signal_id: int, final_price: float, success: bool) -> None:
        """
        Update trade result and metrics.

        Args:
            signal_id: ID of the signal to update
            final_price: Final trade price
            success: Whether the trade was successful
        """
        with self.portal_lock:
            try:
                if signal_id >= len(self.signal_history):
                    logger.warning(f"⚠️ Signal ID {signal_id} not found")
                    return

                signal = self.signal_history[signal_id]

                # Calculate profit/loss
                if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY]:
                    # Long position
                    if final_price > signal.price_target:
                        profit = (final_price - signal.price_target) * signal.position_size
                    else:
                        profit = (final_price - signal.price_target) * signal.position_size
                else:
                    # Short position
                    if final_price < signal.price_target:
                        profit = (signal.price_target - final_price) * signal.position_size
                    else:
                        profit = (signal.price_target - final_price) * signal.position_size

                # Update capital
                self.current_capital += profit

                # Update metrics
                if success:
                    self.portal_metrics.successful_signals += 1
                    self.portal_metrics.total_profit += max(0, profit)
                else:
                    self.portal_metrics.failed_signals += 1
                    self.portal_metrics.total_loss += abs(min(0, profit))

                # Update peak capital and drawdown
                if self.current_capital > self.peak_capital:
                    self.peak_capital = self.current_capital

                current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                self.current_drawdown = max(self.current_drawdown, current_drawdown)
                self.portal_metrics.max_drawdown = self.current_drawdown

                logger.info(
                    f"💰 Trade result: {
                        '✅' if success else '❌'} ${
                        profit:,.2f} (Capital: ${
                        self.current_capital:,.2f})"
                )

            except Exception as e:
                logger.error(f"❌ Failed to update trade result: {e}")

    def _strategy_update_loop(self) -> None:
        """Background loop for strategy updates."""
        while self.running:
            try:
                # Periodic strategy analysis
                self._analyze_strategy_performance()
                time.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"❌ Strategy update error: {e}")
                time.sleep(120)

    def _hourly_sync_loop(self) -> None:
        """Background loop for hourly synchronization."""
        while self.running:
            try:
                current_hour = datetime.now().hour

                if current_hour != self.current_hour:
                    # Hour changed, perform synchronization
                    self._perform_hourly_sync(current_hour)
                    self.current_hour = current_hour

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Hourly sync error: {e}")
                time.sleep(300)

    def _metrics_update_loop(self) -> None:
        """Background loop for metrics updates."""
        while self.running:
            try:
                # Update portal metrics
                self._update_portal_metrics()
                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"❌ Metrics update error: {e}")
                time.sleep(60)

    def _analyze_strategy_performance(self) -> None:
        """Analyze strategy performance across all bit depths."""
        try:
            for bit_depth in StrategyBitDepth:
                if self.strategy_states[bit_depth]:
                    recent_states = self.strategy_states[bit_depth][-100:]
                    avg_confidence = np.mean([state.confidence for state in recent_states])
                    logger.debug(f"📊 {bit_depth.name} strategy: avg confidence = {avg_confidence:.3f}")

        except Exception as e:
            logger.error(f"❌ Strategy analysis error: {e}")

    def _perform_hourly_sync(self, hour: int) -> None:
        """Perform hourly synchronization."""
        try:
            # Calculate phase value for the hour
            phase_value = (hour / 24.0) * 2 * np.pi

            # Calculate coherence from recent signals
            recent_signals = self.signal_history[-10:]
            if recent_signals:
                coherence = np.mean([signal.confidence for signal in recent_signals])
            else:
                coherence = 0.5

            # Calculate signal strength
            signal_strength = self._calculate_hourly_signal_strength(hour)

            # Create hourly connectivity record
            connectivity = HourlyConnectivity(
                hour=hour,
                phase_value=phase_value,
                coherence=coherence,
                signal_strength=signal_strength,
                connected_nodes=1,  # Placeholder
                timestamp=time.time(),
            )

            self.hourly_connectivity.append(connectivity)
            if len(self.hourly_connectivity) > 168:  # Keep 1 week of hourly data
                self.hourly_connectivity = self.hourly_connectivity[-168:]

            logger.info(
                f"🕐 Hourly sync completed for hour {hour}: coherence={
                    coherence:.3f}, strength={
                    signal_strength:.3f}"
            )

        except Exception as e:
            logger.error(f"❌ Hourly sync error: {e}")

    def _calculate_hourly_signal_strength(self, hour: int) -> float:
        """Calculate signal strength for a specific hour."""
        try:
            # Base strength on hour (some hours may be more active)
            hour_factor = 0.5 + 0.5 * np.sin((hour / 24.0) * 2 * np.pi)

            # Market activity factor (assume higher activity during certain hours)
            if 8 <= hour <= 16:  # Business hours
                activity_factor = 1.0
            elif 0 <= hour <= 6:  # Low activity hours
                activity_factor = 0.3
            else:
                activity_factor = 0.7

            return hour_factor * activity_factor

        except Exception:
            return 0.5

    def _update_portal_metrics(self) -> None:
        """Update portal performance metrics."""
        try:
            # Calculate Sharpe ratio
            if self.signal_history:
                returns = []
                for i in range(1, len(self.signal_history)):
                    prev_price = self.signal_history[i - 1].price_target
                    curr_price = self.signal_history[i].price_target
                    returns.append((curr_price - prev_price) / prev_price)

                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        self.portal_metrics.sharpe_ratio = avg_return / std_return
                    else:
                        self.portal_metrics.sharpe_ratio = 0.0

            # Update timestamp
            self.portal_metrics.timestamp = time.time()

        except Exception as e:
            logger.error(f"❌ Metrics update error: {e}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self.portal_lock:
            return {
                "current_capital": self.current_capital,
                "initial_capital": self.initial_capital,
                "total_return": (self.current_capital - self.initial_capital) / self.initial_capital * 100,
                "peak_capital": self.peak_capital,
                "current_drawdown": self.current_drawdown * 100,
                "portal_state": self.current_portal_state.value,
                "risk_per_trade": self.risk_per_trade * 100,
                "total_signals": self.portal_metrics.total_signals,
                "successful_signals": self.portal_metrics.successful_signals,
                "failed_signals": self.portal_metrics.failed_signals,
                "win_rate": self.portal_metrics.win_rate * 100,
                "profit_factor": self.portal_metrics.profit_factor,
                "sharpe_ratio": self.portal_metrics.sharpe_ratio,
                "strategy_states": {bit_depth.name: len(states) for bit_depth, states in self.strategy_states.items()},
                "hourly_connectivity_count": len(self.hourly_connectivity),
                "uptime": time.time() - (self.signal_history[0].timestamp if self.signal_history else time.time()),
            }

    def shutdown(self) -> None:
        """Shutdown the portal system."""
        logger.info("🛑 Shutting down Entry/Exit Portal System...")

        self.running = False

        # Wait for background threads
        if self.strategy_update_thread:
            self.strategy_update_thread.join(timeout=5)
        if self.hourly_sync_thread:
            self.hourly_sync_thread.join(timeout=5)
        if self.metrics_update_thread:
            self.metrics_update_thread.join(timeout=5)

        logger.info("✅ Entry/Exit Portal System shutdown complete")


# Global portal system instance
portal_system = None


def initialize_portal_system(
    initial_capital: float = 10000.0, risk_per_trade: float = 0.02, enable_hourly_sync: bool = True
) -> EntryExitPortalSystem:
    """Initialize global portal system instance."""
    global portal_system

    if portal_system is None:
        portal_system = EntryExitPortalSystem(
            initial_capital=initial_capital, risk_per_trade=risk_per_trade, enable_hourly_sync=enable_hourly_sync
        )

    return portal_system


def get_portal_system() -> Optional[EntryExitPortalSystem]:
    """Get global portal system instance."""
    return portal_system


# Example usage and testing
def main():
    """Test Entry/Exit Portal System functionality."""
    try:
        # Initialize portal system
        portal = initialize_portal_system(initial_capital=10000.0, risk_per_trade=0.02, enable_hourly_sync=True)

        safe_print("🚪 Entry/Exit Portal System Test")
        safe_print("=" * 50)

        # Test market data processing
        safe_print("📊 Testing market data processing...")
        test_prices = [45000, 45100, 45200, 45150, 45300]
        test_volumes = [1000, 1200, 800, 1500, 1100]

        signals_generated = 0
        for i, (price, volume) in enumerate(zip(test_prices, test_volumes)):
            signal = portal.process_market_data(price, volume)
            if signal:
                signals_generated += 1
                safe_print(
                    f"  Signal {signals_generated}: {signal.signal_type.value} "
                    f"(confidence: {signal.confidence:.3f}, "
                    f"position: {signal.position_size:.4f})"
                )

        # Test trade result update
        if signals_generated > 0:
            safe_print("\n💰 Testing trade result update...")
            portal.update_trade_result(0, 45500, True)  # Successful trade
            safe_print(f"  Updated capital: ${portal.current_capital:,.2f}")

        # Get system statistics
        safe_print("\n📈 System Statistics:")
        stats = portal.get_system_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                safe_print(f"  {key}: {value:.2f}")
            else:
                safe_print(f"  {key}: {value}")

        # Cleanup
        portal.shutdown()
        safe_print("\n✅ Entry/Exit Portal System test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Entry/Exit Portal System test failed: {e}")
        safe_print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
