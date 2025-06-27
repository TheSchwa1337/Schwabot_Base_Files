from __future__ import annotations

from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""BTC Investment Ratio Controller - Logical Sequencing for BTC Trading.

This module implements the logical sequencing for BTC investment ratio decisions
based on price analysis, hash correlation, network strength, and multi-bit
signal processing. Integrates all mathematical components into coherent
trading logic.

Mathematical Foundation:
- Execution Confidence: \\u039e = (T \\u00b7 \\u0394\\u03b8) + (\\u03b5 \\u00d7 \\u03c3_f) + \\u03c4_p
- Entry Score: \\u1d4d4\\u209b = \\u1d4d7 \\u00d7 (1 \\u2212 \\u1d4d3\\u209a) \\u00d7 \\u1d4db \\u00d7 P\\u0302
- BTC Investment Ratio: R_btc = f(\\u039e, \\u1d4d4\\u209b, Xi_btc, network_strength)

Logical Sequencing:
1. Collect unified signals from all engines
2. Calculate execution confidence and entry scores
3. Evaluate BTC-specific metrics (hash, network, volume)
4. Determine investment ratio based on multi-factor analysis
5. Apply risk management and position sizing
6. Route profits according to phase-based allocation

Windows CLI compatible with comprehensive error handling.
"""


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.unified_math_system import unified_math

from core.unified_signal_metrics import (
    BTCInvestmentSignals,
    TradingSignalMetrics,
    collect_unified_signals,
)

logger = logging.getLogger(__name__)


class InvestmentDecision(Enum):
    """Investment decision types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_ACTION = "no_action"


class RiskLevel(Enum):
    """Risk assessment levels."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class InvestmentRatioResult:
    """Result of investment ratio analysis."""

    decision: InvestmentDecision
    confidence: float  # Overall decision confidence [0, 1]
    btc_allocation_ratio: float  # Recommended BTC allocation [0, 1]
    position_size_multiplier: float  # Position sizing factor [0.1, 3.0]
    risk_level: RiskLevel
    execution_priority: int  # 1=highest, 5=lowest
    reasoning: str  # Human-readable explanation
    signal_breakdown: Dict[str, float]  # Individual signal contributions
    timestamp: float


@dataclass
class BTCInvestmentRatioController:
    """Main controller for BTC investment ratio decisions."""

    # Configuration parameters
    confidence_threshold_high: float = 1.15
    confidence_threshold_low: float = 0.85
    entry_score_threshold_high: float = 0.90
    entry_score_threshold_low: float = 0.70
    btc_xi_threshold: float = 0.75
    network_strength_threshold: float = 0.60
    max_btc_allocation: float = 0.80
    min_btc_allocation: float = 0.10

    # State tracking
    decision_history: List[InvestmentRatioResult] = field(default_factory=list)
    last_decision_time: float = 0.0
    cooldown_period: float = 60.0  # Minimum seconds between decisions

    def analyze_investment_ratio(
        self,
        cursor_state: Optional[Dict] = None,
        fractal_state: Optional[Dict] = None,
        collapse_state: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        btc_data: Optional[Dict] = None,
        volume_data: Optional[Dict] = None,
        network_data: Optional[Dict] = None,
    ) -> InvestmentRatioResult:
        """Analyze BTC investment ratio based on all available signals.

        Parameters
        ----------
        cursor_state : Dict, optional
            Cursor engine state
        fractal_state : Dict, optional
            Fractal engine state
        collapse_state : Dict, optional
            Collapse engine state
        market_data : Dict, optional
            Current market data
        btc_data : Dict, optional
            BTC-specific data
        volume_data : Dict, optional
            Volume profile data
        network_data : Dict, optional
            BTC network data

        Returns
        -------
        InvestmentRatioResult
            Complete investment ratio analysis
        """
        try:
            current_time = time.time()

            # Check cooldown period
            if current_time - self.last_decision_time < self.cooldown_period:
                return self._create_no_action_result("Cooldown period active")

            # Step 1: Collect unified signals
            core_signals, btc_signals = collect_unified_signals(
                cursor_state,
                fractal_state,
                collapse_state,
                market_data,
                btc_data,
                volume_data,
                network_data,
            )

            # Step 2: Calculate execution confidence (\\u039e)
            execution_confidence = self._calculate_execution_confidence(core_signals)

            # Step 3: Calculate entry score (\\u1d4d4\\u209b)
            entry_score = self._calculate_entry_score(core_signals)

            # Step 4: Evaluate BTC-specific metrics
            btc_strength = self._evaluate_btc_strength(btc_signals)

            # Step 5: Determine investment decision
            decision, reasoning = self._determine_investment_decision(
                execution_confidence, entry_score, btc_strength, btc_signals
            )

            # Step 6: Calculate allocation ratio
            btc_allocation = self._calculate_btc_allocation_ratio(
                execution_confidence, entry_score, btc_strength, btc_signals
            )

            # Step 7: Determine position sizing
            position_multiplier = self._calculate_position_multiplier(
                execution_confidence, entry_score, btc_strength
            )

            # Step 8: Assess risk level
            risk_level = self._assess_risk_level(
                core_signals, btc_signals, execution_confidence
            )

            # Step 9: Set execution priority
            execution_priority = self._determine_execution_priority(
                decision, execution_confidence, entry_score
            )

            # Step 10: Create result
            result = InvestmentRatioResult(
                decision=decision,
                confidence=execution_confidence,
                btc_allocation_ratio=btc_allocation,
                position_size_multiplier=position_multiplier,
                risk_level=risk_level,
                execution_priority=execution_priority,
                reasoning=reasoning,
                signal_breakdown=self._create_signal_breakdown(
                    core_signals, btc_signals, execution_confidence, entry_score
                ),
                timestamp=current_time,
            )

            # Store in history
            self.decision_history.append(result)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]

            self.last_decision_time = current_time

            logger.info(
                f"Investment decision: {decision.value}, "
                f"BTC allocation: {btc_allocation:.2%}, "
                f"confidence: {execution_confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in investment ratio analysis: {e}")
            return self._create_error_result(str(e))

    def _calculate_execution_confidence(self, signals: TradingSignalMetrics) -> float:
        """Calculate execution confidence scalar \\u039e."""
        try:
            from core.entry_gate import execution_confidence

            return execution_confidence(
                signals.triplet_entropy,
                signals.theta_drift,
                signals.coherence,
                signals.loop_volatility,
                signals.profit_decay,
            )
        except Exception as e:
            logger.warning(f"Error calculating execution confidence: {e}")
            # Fallback calculation
            return (
                (signals.triplet_entropy * signals.theta_drift)
                + (signals.coherence * signals.loop_volatility)
                + signals.profit_decay
            )

    def _calculate_entry_score(self, signals: TradingSignalMetrics) -> float:
        """Calculate entropy-weighted entry score \\u1d4d4\\u209b."""
        try:
            from core.entry_gate import entry_score

            return entry_score(
                signals.harmony,
                signals.drift_penalty,
                signals.liquidity_score,
                signals.projected_profit,
            )
        except Exception as e:
            logger.warning(f"Error calculating entry score: {e}")
            # Fallback calculation
            return (
                signals.harmony
                * (1.0 - signals.drift_penalty)
                * signals.liquidity_score
                * signals.projected_profit
            )

    def _evaluate_btc_strength(self, btc_signals: BTCInvestmentSignals) -> float:
        """Evaluate overall BTC strength from network and price metrics."""
        # Weighted combination of BTC-specific signals
        strength = (
            btc_signals.xi_btc * 0.30  # BTC confidence
            + btc_signals.network_strength * 0.25  # Network health
            + btc_signals.hash_correlation * 0.20  # Hash rate correlation
            + btc_signals.volume_profile * 0.15  # Volume distribution
            + btc_signals.price_pressure * 0.10  # Price pressure
        )

        return unified_math.max(0.0, unified_math.min(1.0, strength))

    def _determine_investment_decision(
        self,
        execution_confidence: float,
        entry_score: float,
        btc_strength: float,
        btc_signals: BTCInvestmentSignals,
    ) -> Tuple[InvestmentDecision, str]:
        """Determine investment decision based on all factors."""

        # Primary gates
        high_confidence = execution_confidence > self.confidence_threshold_high
        high_entry = entry_score > self.entry_score_threshold_high
        strong_btc = btc_strength > self.btc_xi_threshold
        strong_network = btc_signals.network_strength > self.network_strength_threshold

        # Decision logic with reasoning
        if high_confidence and high_entry and strong_btc and strong_network:
            return InvestmentDecision.STRONG_BUY, (
                f"All signals positive: confidence={execution_confidence:.3f}, "
                f"entry={entry_score:.3f}, BTC strength={btc_strength:.3f}"
            )

        elif high_confidence and high_entry and (strong_btc or strong_network):
            return InvestmentDecision.BUY, (
                f"Strong core signals with good BTC metrics: "
                f"confidence={execution_confidence:.3f}, entry={entry_score:.3f}"
            )

        elif (high_confidence or high_entry) and btc_strength > 0.5:
            return InvestmentDecision.HOLD, (
                f"Mixed signals suggest holding: "
                f"confidence={execution_confidence:.3f}, entry={entry_score:.3f}"
            )

        elif (
            execution_confidence < self.confidence_threshold_low
            or entry_score < self.entry_score_threshold_low
        ):
            if btc_strength < 0.3:
                return InvestmentDecision.STRONG_SELL, (
                    f"Weak signals across all metrics: "
                    f"confidence={execution_confidence:.3f}, entry={entry_score:.3f}"
                )
            else:
                return InvestmentDecision.SELL, (
                    f"Low confidence/entry but BTC showing some strength: "
                    f"confidence={execution_confidence:.3f}, entry={entry_score:.3f}"
                )

        else:
            return InvestmentDecision.NO_ACTION, (
                f"Insufficient signal strength for clear decision: "
                f"confidence={execution_confidence:.3f}, entry={entry_score:.3f}"
            )

    def _calculate_btc_allocation_ratio(
        self,
        execution_confidence: float,
        entry_score: float,
        btc_strength: float,
        btc_signals: BTCInvestmentSignals,
    ) -> float:
        """Calculate recommended BTC allocation ratio."""

        # Base allocation from signal strength
        base_allocation = (
            execution_confidence * 0.30
            + entry_score * 0.25
            + btc_strength * 0.25
            + btc_signals.network_strength * 0.20
        )

        # Apply bounds
        allocation = max(
            self.min_btc_allocation, unified_math.min(self.max_btc_allocation, base_allocation)
        )

        # Adjust based on risk factors
        if btc_signals.price_pressure < 0.3:  # Low pressure = reduce allocation
            allocation *= 0.8
        elif btc_signals.price_pressure > 0.7:  # High pressure = increase allocation
            allocation *= 1.2

        # Network strength adjustment
        if btc_signals.network_strength < 0.4:
            allocation *= 0.7  # Reduce if network is weak

        # Final bounds check
        return unified_math.max(self.min_btc_allocation, unified_math.min(self.max_btc_allocation, allocation))

    def _calculate_position_multiplier(
        self,
        execution_confidence: float,
        entry_score: float,
        btc_strength: float,
    ) -> float:
        """Calculate position size multiplier."""
        try:
            from core.auto_scaler import scale_position

            # Use projected profit from entry score as proxy
            projected_profit = entry_score * 0.05  # Scale to reasonable profit range

            return scale_position(
                execution_confidence,
                projected_profit,
                base_scale=1.0,
                min_scale=0.1,
                max_scale=3.0,
            )
        except Exception as e:
            logger.warning(f"Error calculating position multiplier: {e}")
            # Fallback calculation
            multiplier = 1.0 + (execution_confidence - 1.0) * 0.5 + btc_strength * 0.3
            return unified_math.max(0.1, unified_math.min(3.0, multiplier))

    def _assess_risk_level(
        self,
        core_signals: TradingSignalMetrics,
        btc_signals: BTCInvestmentSignals,
        execution_confidence: float,
    ) -> RiskLevel:
        """Assess overall risk level."""

        # Risk factors
        volatility_risk = core_signals.loop_volatility
        liquidity_risk = 1.0 - core_signals.liquidity_score
        network_risk = 1.0 - btc_signals.network_strength
        confidence_risk = unified_math.max(0, 1.0 - execution_confidence)

        # Combined risk score
        risk_score = (
            volatility_risk * 0.30
            + liquidity_risk * 0.25
            + network_risk * 0.25
            + confidence_risk * 0.20
        )

        # Map to risk levels
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _determine_execution_priority(
        self,
        decision: InvestmentDecision,
        execution_confidence: float,
        entry_score: float,
    ) -> int:
        """Determine execution priority (1=highest, 5=lowest)."""

        if decision == InvestmentDecision.STRONG_BUY:
            return 1
        elif decision == InvestmentDecision.BUY and execution_confidence > 1.3:
            return 1
        elif decision == InvestmentDecision.BUY:
            return 2
        elif decision == InvestmentDecision.STRONG_SELL:
            return 2
        elif decision == InvestmentDecision.SELL:
            return 3
        elif decision == InvestmentDecision.HOLD:
            return 4
        else:
            return 5

    def _create_signal_breakdown(
        self,
        core_signals: TradingSignalMetrics,
        btc_signals: BTCInvestmentSignals,
        execution_confidence: float,
        entry_score: float,
    ) -> Dict[str, float]:
        """Create detailed signal breakdown for analysis."""
        return {
            "execution_confidence": execution_confidence,
            "entry_score": entry_score,
            "triplet_entropy": core_signals.triplet_entropy,
            "theta_drift": core_signals.theta_drift,
            "coherence": core_signals.coherence,
            "loop_volatility": core_signals.loop_volatility,
            "harmony": core_signals.harmony,
            "drift_penalty": core_signals.drift_penalty,
            "liquidity_score": core_signals.liquidity_score,
            "projected_profit": core_signals.projected_profit,
            "v_btc": btc_signals.v_btc,
            "eta_btc": btc_signals.eta_btc,
            "xi_btc": btc_signals.xi_btc,
            "price_pressure": btc_signals.price_pressure,
            "volume_profile": btc_signals.volume_profile,
            "hash_correlation": btc_signals.hash_correlation,
            "network_strength": btc_signals.network_strength,
        }

    def _create_no_action_result(self, reason: str) -> InvestmentRatioResult:
        """Create a no-action result."""
        return InvestmentRatioResult(
            decision=InvestmentDecision.NO_ACTION,
            confidence=0.0,
            btc_allocation_ratio=0.5,  # Neutral allocation
            position_size_multiplier=1.0,
            risk_level=RiskLevel.MODERATE,
            execution_priority=5,
            reasoning=reason,
            signal_breakdown={},
            timestamp=time.time(),
        )

    def _create_error_result(self, error_msg: str) -> InvestmentRatioResult:
        """Create an error result."""
        return InvestmentRatioResult(
            decision=InvestmentDecision.NO_ACTION,
            confidence=0.0,
            btc_allocation_ratio=self.min_btc_allocation,
            position_size_multiplier=0.1,
            risk_level=RiskLevel.VERY_HIGH,
            execution_priority=5,
            reasoning=f"Error in analysis: {error_msg}",
            signal_breakdown={},
            timestamp=time.time(),
        )

    def get_decision_history(self, limit: int = 10) -> List[InvestmentRatioResult]:
        """Get recent decision history."""
        return self.decision_history[-limit:] if self.decision_history else []

    def get_performance_summary(self) -> Dict:
        """Get performance summary of recent decisions."""
        if not self.decision_history:
            return {"error": "No decision history available"}

        recent_decisions = self.decision_history[-50:]  # Last 50 decisions

        decision_counts = {}
        for result in recent_decisions:
            decision = result.decision.value
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        avg_confidence = unified_math.mean([r.confidence for r in recent_decisions])
        avg_btc_allocation = unified_math.mean([r.btc_allocation_ratio for r in recent_decisions])

        return {
            "total_decisions": len(recent_decisions),
            "decision_distribution": decision_counts,
            "average_confidence": avg_confidence,
            "average_btc_allocation": avg_btc_allocation,
            "risk_level_distribution": {
                level.value: sum(1 for r in recent_decisions if r.risk_level == level)
                for level in RiskLevel
            },
            "latest_decision": (
                recent_decisions[-1].decision.value if recent_decisions else None
            ),
        }


def main() -> None:
    """Demo function for testing BTC investment ratio controller."""
    safe_print("BTC Investment Ratio Controller Demo")
    safe_print("=" * 50)

    controller = BTCInvestmentRatioController()

    # Mock comprehensive data
    mock_cursor_state = {
        "triplet_entropy": 0.82,
        "braid_angle_drift": 0.15,
    }

    mock_fractal_state = {
        "coherence_score": 0.91,
    }

    mock_collapse_state = {
        "loop_sum_volatility": 0.12,
        "profit_time_decay": 0.04,
    }

    mock_market_data = {
        "tick_deltas": [0.11, 0.13, 0.12, 0.125, 0.14],
        "target_phase": 0.125,
        "order_book": {
            "bids": [[52000, 2.5], [51950, 3.0]],
            "asks": [[52050, 2.0], [52100, 2.8]],
        },
        "recent_prices": [52000, 52025, 51975, 52050, 52100],
    }

    mock_btc_data = {
        "exit_prices": [52100, 52200, 52150],
        "entry_prices": [52000, 52050, 52075],
        "volume_weights": [1.2, 1.8, 1.0],
        "price_delta": 150.0,
        "time_delta": 60.0,
        "normalized_price_change": 0.003,
        "volatility_measure": 0.018,
    }

    mock_network_data = {
        "hash_rate": 4.5e17,  # 450 EH/s
        "difficulty": 6.2e13,
        "price": 52000,
        "mempool_size": 80000,
    }

    # Analyze investment ratio
    result = controller.analyze_investment_ratio(
        cursor_state=mock_cursor_state,
        fractal_state=mock_fractal_state,
        collapse_state=mock_collapse_state,
        market_data=mock_market_data,
        btc_data=mock_btc_data,
        network_data=mock_network_data,
    )

    safe_print(f"Investment Decision: {result.decision.value}")
    safe_print(f"Confidence: {result.confidence:.3f}")
    safe_print(f"BTC Allocation: {result.btc_allocation_ratio:.1%}")
    safe_print(f"Position Multiplier: {result.position_size_multiplier:.2f}x")
    safe_print(f"Risk Level: {result.risk_level.value}")
    safe_print(f"Execution Priority: {result.execution_priority}")
    safe_print(f"Reasoning: {result.reasoning}")

    safe_print("\\nKey Signal Breakdown:")
    breakdown = result.signal_breakdown
    safe_print(f"  Execution Confidence: {breakdown.get('execution_confidence', 0):.3f}")
    safe_print(f"  Entry Score: {breakdown.get('entry_score', 0):.3f}")
    safe_print(f"  BTC Xi: {breakdown.get('xi_btc', 0):.3f}")
    safe_print(f"  Network Strength: {breakdown.get('network_strength', 0):.3f}")
    safe_print(f"  Price Pressure: {breakdown.get('price_pressure', 0):.3f}")

    # Test multiple scenarios
    safe_print("\n" + "=" * 50)
    safe_print("Testing Multiple Scenarios:")

    scenarios = [
        ("Bull Market", {"triplet_entropy": 0.95, "coherence_score": 0.88}),
        ("Bear Market", {"triplet_entropy": 0.35, "coherence_score": 0.42}),
        ("Sideways", {"triplet_entropy": 0.65, "coherence_score": 0.70}),
    ]

    for scenario_name, overrides in scenarios:
        # Apply overrides
        test_cursor = {**mock_cursor_state, **overrides}
        test_fractal = {**mock_fractal_state, **overrides}

        result = controller.analyze_investment_ratio(
            cursor_state=test_cursor,
            fractal_state=test_fractal,
            collapse_state=mock_collapse_state,
            market_data=mock_market_data,
            btc_data=mock_btc_data,
            network_data=mock_network_data,
        )

        safe_print(f"\\n{scenario_name}:")
        safe_print(f"  Decision: {result.decision.value}")
        safe_print(f"  BTC Allocation: {result.btc_allocation_ratio:.1%}")
        safe_print(f"  Risk: {result.risk_level.value}")

    # Performance summary
    summary = controller.get_performance_summary()
    safe_print(f"\\nPerformance Summary: {summary}")


if __name__ == "__main__":
    main()
