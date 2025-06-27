from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Comprehensive Anomaly Filter System for Schwabot.

This module addresses ALL potential filter anomaly exposures that could
cause system failures during market swings, data feed issues, or mathematical
edge cases. It implements multiple layers of protection beyond the basic
GAN filter to ensure bulletproof operation.

Key Protections:
- Market regime change detection (bull/bear/sideways transitions)
- Data feed anomaly detection (stale/corrupted/missing data)
- Mathematical edge case handling (singularities, overflows, NaN)
- Execution timing anomalies (latency spikes, order rejections)
- Portfolio state anomalies (margin calls, liquidity crunches)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies the system can detect."""

    MARKET_REGIME_SHIFT = "market_regime_shift"
    DATA_FEED_CORRUPTION = "data_feed_corruption"
    MATHEMATICAL_SINGULARITY = "mathematical_singularity"
    EXECUTION_TIMING = "execution_timing"
    PORTFOLIO_STATE = "portfolio_state"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_EXPLOSION = "volatility_explosion"


@dataclass
class AnomalySignal:
    """Container for anomaly detection signals."""

    anomaly_type: AnomalyType
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0.0 to 1.0
    detected_at: float
    description: str
    affected_components: List[str] = field(default_factory=list)
    recommended_action: str = ""
    time_to_resolution: Optional[float] = None


@dataclass
class SystemState:
    """Current system state for anomaly detection."""

    # Market data state
    prices: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    spreads: Dict[str, float] = field(default_factory=dict)
    last_update: float = 0.0

    # Mathematical state
    matrix_conditions: Dict[str, float] = field(default_factory=dict)
    entropy_levels: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Execution state
    pending_orders: int = 0
    execution_latencies: List[float] = field(default_factory=list)
    rejection_rate: float = 0.0

    # Portfolio state
    positions: Dict[str, float] = field(default_factory=dict)
    available_margin: float = 0.0
    unrealized_pnl: float = 0.0


class ComprehensiveAnomalyFilter:
    """Multi-layered anomaly detection and filtering system."""

    def __init__(self) -> None:
        """Initialize the comprehensive anomaly filter."""
        self.detection_history: List[AnomalySignal] = []
        self.system_baseline: Optional[SystemState] = None
        self.calibration_data: Dict[str, List[float]] = {}
        self.regime_detector = MarketRegimeDetector()
        self.data_validator = DataFeedValidator()
        self.math_protector = MathematicalSafetyProtector()
        self.execution_monitor = ExecutionAnomalyMonitor()
        self.portfolio_guardian = PortfolioStateGuardian()

        # Anomaly thresholds
        self.thresholds = {
            "price_jump_sigma": 5.0,  # 5-sigma price jump threshold
            "volume_spike_multiplier": 10.0,  # 10x volume spike
            "correlation_breakdown": 0.3,  # Correlation drops below 0.3
            "latency_spike_ms": 1000.0,  # 1 second latency spike
            "matrix_condition_limit": 1e12,  # Matrix conditioning limit
            "entropy_explosion": 8.0,  # Entropy spike threshold
            "margin_cushion": 0.20,  # 20% margin cushion required
        }

        logger.info("\\u1f6e1\\ufe0f Comprehensive Anomaly Filter initialized")

    def detect_all_anomalies(self, current_state: SystemState) -> List[AnomalySignal]:
        """Run comprehensive anomaly detection across all systems."""
        start_time = time.time()
        all_anomalies = []

        try:
            # 1. Market regime anomalies
            regime_anomalies = self.regime_detector.detect_regime_shifts(current_state)
            all_anomalies.extend(regime_anomalies)

            # 2. Data feed anomalies
            data_anomalies = self.data_validator.validate_data_integrity(current_state)
            all_anomalies.extend(data_anomalies)

            # 3. Mathematical anomalies
            math_anomalies = self.math_protector.check_mathematical_safety(
                current_state
            )
            all_anomalies.extend(math_anomalies)

            # 4. Execution anomalies
            exec_anomalies = self.execution_monitor.detect_execution_issues(
                current_state
            )
            all_anomalies.extend(exec_anomalies)

            # 5. Portfolio state anomalies
            portfolio_anomalies = self.portfolio_guardian.check_portfolio_health(
                current_state
            )
            all_anomalies.extend(portfolio_anomalies)

            # Store detection results
            self.detection_history.extend(all_anomalies)

            # Keep only recent history
            cutoff_time = time.time() - 3600  # Keep 1 hour of history
            self.detection_history = [
                signal
                for signal in self.detection_history
                if signal.detected_at > cutoff_time
            ]

            detection_time = time.time() - start_time
            logger.debug(
                f"\\u1f50d Anomaly detection completed in {detection_time:.4f}s, found {len(all_anomalies)} anomalies"
            )

            return all_anomalies

        except Exception as e:
            logger.error(f"\\u274c Anomaly detection failed: {e}")
            # Return critical system anomaly
            return [
                AnomalySignal(
                    anomaly_type=AnomalyType.MATHEMATICAL_SINGULARITY,
                    severity="critical",
                    confidence=1.0,
                    detected_at=time.time(),
                    description=f"Anomaly detection system failure: {str(e)}",
                    recommended_action="EMERGENCY_STOP",
                )
            ]

    def should_block_execution(
        self, anomalies: List[AnomalySignal]
    ) -> Tuple[bool, str]:
        """Determine if execution should be blocked based on detected anomalies."""
        if not anomalies:
            return False, "No anomalies detected"

        # Check for critical anomalies
        critical_anomalies = [a for a in anomalies if a.severity == "critical"]
        if critical_anomalies:
            return (
                True,
                f"Critical anomalies detected: {[a.anomaly_type.value for a in critical_anomalies]}",
            )

        # Check for high severity anomalies
        high_severity_count = len([a for a in anomalies if a.severity == "high"])
        if high_severity_count >= 3:
            return (
                True,
                f"Multiple high-severity anomalies detected: {high_severity_count}",
            )

        # Check for specific blocking conditions
        blocking_types = {
            AnomalyType.MATHEMATICAL_SINGULARITY,
            AnomalyType.LIQUIDITY_CRUNCH,
            AnomalyType.DATA_FEED_CORRUPTION,
        }

        for anomaly in anomalies:
            if anomaly.anomaly_type in blocking_types and anomaly.confidence > 0.8:
                return True, f"Blocking anomaly: {anomaly.anomaly_type.value}"

        return False, "Anomalies present but not blocking"

    def get_recommended_actions(self, anomalies: List[AnomalySignal]) -> List[str]:
        """Get recommended actions based on detected anomalies."""
        actions = []

        for anomaly in anomalies:
            if anomaly.recommended_action:
                actions.append(
                    f"{anomaly.anomaly_type.value}: {anomaly.recommended_action}"
                )

        # Deduplicate actions
        return list(set(actions))


class MarketRegimeDetector:
    """Detect market regime changes and transitions."""

    def __init__(self) -> None:
        """Initialize market regime detector."""
        self.price_history: Dict[str, List[float]] = {}
        self.volatility_history: Dict[str, List[float]] = {}
        self.regime_states = {"bull", "bear", "sideways", "transitioning"}
        self.current_regime = "sideways"

    def detect_regime_shifts(self, state: SystemState) -> List[AnomalySignal]:
        """Detect market regime shifts that could affect strategy performance."""
        anomalies = []

        for symbol, price in state.prices.items():
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append(price)

            # Keep only recent history (last 100 prices)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]

            # Need at least 20 data points for regime detection
            if len(self.price_history[symbol]) < 20:
                continue

            try:
                # Calculate returns and volatility
                prices = np.array(self.price_history[symbol])
                returns = np.diff(prices) / prices[:-1]

                # Detect volatility regime shift
                recent_vol = unified_math.unified_math.std(returns[-10:])  # Last 10 periods
                historical_vol = unified_math.unified_math.std(returns[:-10])  # Earlier periods

                if recent_vol > historical_vol * 3:  # 3x volatility increase
                    anomalies.append(
                        AnomalySignal(
                            anomaly_type=AnomalyType.VOLATILITY_EXPLOSION,
                            severity="high",
                            confidence=unified_math.min(recent_vol / historical_vol / 3, 1.0),
                            detected_at=time.time(),
                            description=f"Volatility explosion detected in {symbol}: {recent_vol:.4f} vs {historical_vol:.4f}",
                            affected_components=[symbol],
                            recommended_action="REDUCE_POSITION_SIZE",
                        )
                    )

                # Detect trend regime shift using rolling correlation
                if len(returns) >= 40:
                    early_trend = unified_math.unified_math.correlation(np.arange(20), returns[:20])[0, 1]
                    recent_trend = unified_math.unified_math.correlation(np.arange(20), returns[-20:])[0, 1]

                    if unified_math.abs(early_trend - recent_trend) > 0.8:  # Strong trend change
                        anomalies.append(
                            AnomalySignal(
                                anomaly_type=AnomalyType.MARKET_REGIME_SHIFT,
                                severity="medium",
                                confidence=unified_math.abs(early_trend - recent_trend) / 2,
                                detected_at=time.time(),
                                description=f"Market regime shift detected in {symbol}: trend correlation changed from {early_trend:.3f} to {recent_trend:.3f}",
                                affected_components=[symbol],
                                recommended_action="REASSESS_STRATEGY_WEIGHTS",
                            )
                        )

            except Exception as e:
                logger.warning(f"\\u26a0\\ufe0f Regime detection failed for {symbol}: {e}")

        return anomalies


class DataFeedValidator:
    """Validate data feed integrity and detect corruption."""

    def __init__(self) -> None:
        """Initialize data feed validator."""
        self.last_valid_state: Optional[SystemState] = None
        self.stale_data_threshold = 60.0  # 60 seconds
        self.price_jump_threshold = 0.10  # 10% price jump threshold

    def validate_data_integrity(self, state: SystemState) -> List[AnomalySignal]:
        """Validate data feed integrity and detect anomalies."""
        anomalies = []
        current_time = time.time()

        # Check for stale data
        if current_time - state.last_update > self.stale_data_threshold:
            anomalies.append(
                AnomalySignal(
                    anomaly_type=AnomalyType.DATA_FEED_CORRUPTION,
                    severity="high",
                    confidence=1.0,
                    detected_at=current_time,
                    description=f"Stale data detected: last update {current_time - state.last_update:.1f}s ago",
                    affected_components=["data_feed"],
                    recommended_action="SWITCH_TO_BACKUP_FEED",
                )
            )

        # Check for impossible price jumps
        if self.last_valid_state:
            for symbol, price in state.prices.items():
                if symbol in self.last_valid_state.prices:
                    last_price = self.last_valid_state.prices[symbol]
                    if last_price > 0:
                        price_change = unified_math.abs(price - last_price) / last_price

                        if price_change > self.price_jump_threshold:
                            anomalies.append(
                                AnomalySignal(
                                    anomaly_type=AnomalyType.DATA_FEED_CORRUPTION,
                                    severity="high",
                                    confidence=min(
                                        price_change / self.price_jump_threshold, 1.0
                                    ),
                                    detected_at=current_time,
                                    description=f"Impossible price jump in {symbol}: {price_change:.2%} change",
                                    affected_components=[symbol],
                                    recommended_action="VALIDATE_PRICE_DATA",
                                )
                            )

        # Check for missing critical data
        required_fields = ["prices", "volumes"]
        for field_name in required_fields:
            if not getattr(state, field_name):
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.DATA_FEED_CORRUPTION,
                        severity="critical",
                        confidence=1.0,
                        detected_at=current_time,
                        description=f"Missing critical data field: {field_name}",
                        affected_components=["data_feed"],
                        recommended_action="EMERGENCY_DATA_RECOVERY",
                    )
                )

        # Update last valid state if current state is clean
        if not anomalies:
            self.last_valid_state = state

        return anomalies


class MathematicalSafetyProtector:
    """Protect against mathematical edge cases and singularities."""

    def __init__(self) -> None:
        """Initialize mathematical safety protector."""
        self.matrix_condition_limit = 1e12
        self.entropy_explosion_threshold = 10.0
        self.confidence_validation_range = (0.0, 1.0)

    def check_mathematical_safety(self, state: SystemState) -> List[AnomalySignal]:
        """Check for mathematical anomalies and edge cases."""
        anomalies = []
        current_time = time.time()

        # Check matrix conditioning
        for matrix_name, condition_number in state.matrix_conditions.items():
            if condition_number > self.matrix_condition_limit:
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.MATHEMATICAL_SINGULARITY,
                        severity="high",
                        confidence=min(
                            condition_number / self.matrix_condition_limit, 1.0
                        ),
                        detected_at=current_time,
                        description=f"Matrix {matrix_name} is poorly conditioned: {condition_number:.2e}",
                        affected_components=[matrix_name],
                        recommended_action="APPLY_REGULARIZATION",
                    )
                )

        # Check entropy explosion
        for component, entropy in state.entropy_levels.items():
            if entropy > self.entropy_explosion_threshold:
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.MATHEMATICAL_SINGULARITY,
                        severity="medium",
                        confidence=unified_math.min(entropy / self.entropy_explosion_threshold, 1.0),
                        detected_at=current_time,
                        description=f"Entropy explosion in {component}: {entropy:.2f}",
                        affected_components=[component],
                        recommended_action="INCREASE_REGULARIZATION",
                    )
                )

        # Check confidence score validity
        for component, confidence in state.confidence_scores.items():
            if not (
                self.confidence_validation_range[0]
                <= confidence
                <= self.confidence_validation_range[1]
            ):
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.MATHEMATICAL_SINGULARITY,
                        severity="medium",
                        confidence=1.0,
                        detected_at=current_time,
                        description=f"Invalid confidence score in {component}: {confidence}",
                        affected_components=[component],
                        recommended_action="RESET_CONFIDENCE_CALCULATION",
                    )
                )

        # Check for NaN or infinite values
        all_values = (
            list(state.prices.values())
            + list(state.volumes.values())
            + list(state.spreads.values())
            + list(state.matrix_conditions.values())
            + list(state.entropy_levels.values())
            + list(state.confidence_scores.values())
        )

        for i, value in enumerate(all_values):
            if np.isnan(value) or np.isinf(value):
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.MATHEMATICAL_SINGULARITY,
                        severity="critical",
                        confidence=1.0,
                        detected_at=current_time,
                        description=f"NaN or infinite value detected: {value}",
                        affected_components=["numerical_computation"],
                        recommended_action="EMERGENCY_NUMERICAL_RESET",
                    )
                )

        return anomalies


class ExecutionAnomalyMonitor:
    """Monitor execution timing and order flow anomalies."""

    def __init__(self) -> None:
        """Initialize execution anomaly monitor."""
        self.normal_latency_ms = 100.0  # Expected normal latency
        self.latency_spike_threshold = 5.0  # 5x normal latency
        self.rejection_rate_threshold = 0.20  # 20% rejection rate threshold

    def detect_execution_issues(self, state: SystemState) -> List[AnomalySignal]:
        """Detect execution timing and order flow anomalies."""
        anomalies = []
        current_time = time.time()

        # Check execution latency spikes
        if state.execution_latencies:
            recent_latency = (
                unified_math.unified_math.mean(state.execution_latencies[-10:]) * 1000
            )  # Convert to ms

            if recent_latency > self.normal_latency_ms * self.latency_spike_threshold:
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.EXECUTION_TIMING,
                        severity="medium",
                        confidence=min(
                            recent_latency
                            / (self.normal_latency_ms * self.latency_spike_threshold),
                            1.0,
                        ),
                        detected_at=current_time,
                        description=f"Execution latency spike: {recent_latency:.1f}ms vs normal {self.normal_latency_ms}ms",
                        affected_components=["execution_engine"],
                        recommended_action="SWITCH_TO_BACKUP_BROKER",
                    )
                )

        # Check high order rejection rate
        if state.rejection_rate > self.rejection_rate_threshold:
            anomalies.append(
                AnomalySignal(
                    anomaly_type=AnomalyType.EXECUTION_TIMING,
                    severity="high",
                    confidence=min(
                        state.rejection_rate / self.rejection_rate_threshold, 1.0
                    ),
                    detected_at=current_time,
                    description=f"High order rejection rate: {state.rejection_rate:.1%}",
                    affected_components=["execution_engine"],
                    recommended_action="REDUCE_ORDER_FREQUENCY",
                )
            )

        # Check for excessive pending orders
        if state.pending_orders > 50:  # Arbitrary threshold
            anomalies.append(
                AnomalySignal(
                    anomaly_type=AnomalyType.EXECUTION_TIMING,
                    severity="medium",
                    confidence=unified_math.min(state.pending_orders / 50, 1.0),
                    detected_at=current_time,
                    description=f"Excessive pending orders: {state.pending_orders}",
                    affected_components=["order_management"],
                    recommended_action="CANCEL_STALE_ORDERS",
                )
            )

        return anomalies


class PortfolioStateGuardian:
    """Monitor portfolio state and margin health."""

    def __init__(self) -> None:
        """Initialize portfolio state guardian."""
        self.margin_cushion_threshold = 0.20  # 20% margin cushion required
        self.concentration_limit = 0.50  # 50% max concentration in single asset
        self.drawdown_alert_threshold = 0.10  # 10% drawdown alert

    def check_portfolio_health(self, state: SystemState) -> List[AnomalySignal]:
        """Check portfolio health and margin conditions."""
        anomalies = []
        current_time = time.time()

        # Check margin health
        if state.available_margin > 0:
            total_portfolio_value = sum(unified_math.abs(pos) for pos in state.positions.values())

            if total_portfolio_value > 0:
                margin_ratio = state.available_margin / total_portfolio_value

                if margin_ratio < self.margin_cushion_threshold:
                    anomalies.append(
                        AnomalySignal(
                            anomaly_type=AnomalyType.PORTFOLIO_STATE,
                            severity="high",
                            confidence=1.0
                            - (margin_ratio / self.margin_cushion_threshold),
                            detected_at=current_time,
                            description=f"Low margin cushion: {margin_ratio:.1%}",
                            affected_components=["portfolio_management"],
                            recommended_action="REDUCE_POSITION_SIZES",
                        )
                    )

        # Check position concentration
        if state.positions:
            total_exposure = sum(unified_math.abs(pos) for pos in state.positions.values())

            if total_exposure > 0:
                for symbol, position in state.positions.items():
                    concentration = unified_math.abs(position) / total_exposure

                    if concentration > self.concentration_limit:
                        anomalies.append(
                            AnomalySignal(
                                anomaly_type=AnomalyType.PORTFOLIO_STATE,
                                severity="medium",
                                confidence=concentration / self.concentration_limit - 1,
                                detected_at=current_time,
                                description=f"High concentration in {symbol}: {concentration:.1%}",
                                affected_components=["portfolio_management"],
                                recommended_action="DIVERSIFY_HOLDINGS",
                            )
                        )

        # Check unrealized P&L drawdown
        if state.unrealized_pnl < 0:
            drawdown_pct = unified_math.abs(state.unrealized_pnl) / max(
                sum(unified_math.abs(pos) for pos in state.positions.values()), 1
            )

            if drawdown_pct > self.drawdown_alert_threshold:
                anomalies.append(
                    AnomalySignal(
                        anomaly_type=AnomalyType.PORTFOLIO_STATE,
                        severity="high",
                        confidence=min(
                            drawdown_pct / self.drawdown_alert_threshold, 1.0
                        ),
                        detected_at=current_time,
                        description=f"Portfolio drawdown: {drawdown_pct:.1%}",
                        affected_components=["risk_management"],
                        recommended_action="CONSIDER_STOP_LOSS",
                    )
                )

        return anomalies


# Main functions for integration with existing system
def create_comprehensive_anomaly_filter() -> ComprehensiveAnomalyFilter:
    """Create and configure the comprehensive anomaly filter."""
    return ComprehensiveAnomalyFilter()


def validate_system_safety(
    current_state: SystemState,
) -> Tuple[bool, List[AnomalySignal], List[str]]:
    """Validate system safety and return execution decision."""
    filter_system = create_comprehensive_anomaly_filter()

    # Detect all anomalies
    anomalies = filter_system.detect_all_anomalies(current_state)

    # Determine if execution should proceed
    should_block, block_reason = filter_system.should_block_execution(anomalies)

    # Get recommended actions
    actions = filter_system.get_recommended_actions(anomalies)

    return not should_block, anomalies, actions


if __name__ == "__main__":
    # Example usage
    test_state = SystemState(
        prices={"BTC": 45000.0, "ETH": 3000.0},
        volumes={"BTC": 1000000, "ETH": 500000},
        last_update=time.time(),
        matrix_conditions={"correlation_matrix": 1e8},
        entropy_levels={"price_entropy": 4.5},
        confidence_scores={"execution_confidence": 0.85},
    )

    safe_to_execute, detected_anomalies, recommended_actions = validate_system_safety(
        test_state
    )

    safe_print(f"\\u2705 Safe to execute: {safe_to_execute}")
    safe_print(f"\\u1f50d Anomalies detected: {len(detected_anomalies)}")
    safe_print(f"\\u1f4cb Recommended actions: {len(recommended_actions)}")

"""