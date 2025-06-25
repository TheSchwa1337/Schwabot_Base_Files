# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Collapse Confidence - Market Collapse Detection and Analysis.

This module provides advanced algorithms for:
- Market collapse pattern detection
- Confidence decay modeling
- Risk assessment during stress conditions
- Volatility spike analysis
- Liquidity crisis detection

Mathematical Foundation:
- Exponential decay models for confidence
- Volatility clustering algorithms
- Liquidity stress indicators
- Market microstructure analysis
- Confidence collapse prediction models
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from core.unified_math_system import unified_math
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceState:
    """Represents a confidence state at a point in time."""
    timestamp: datetime
    confidence_level: float  # 0.0 to 1.0
    volatility: float
    volume: float
    price_change: float
    liquidity_score: float
    stress_indicator: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollapseEvent:
    """Represents a detected collapse event."""
    event_id: str
    start_time: datetime
    end_time: Optional[datetime]
    confidence_drop: float
    volatility_spike: float
    volume_surge: float
    price_crash: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    recovery_time: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollapseAnalysis:
    """Result of collapse confidence analysis."""
    current_confidence: float
    confidence_trend: float
    collapse_risk: float
    detected_events: List[CollapseEvent]
    stress_level: str
    recovery_probability: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class CollapseConfidence:
    """
    Advanced market collapse detection and confidence analysis system.

    Provides mathematical models for:
    - Real-time confidence monitoring
    - Collapse pattern recognition
    - Risk assessment during stress
    - Recovery prediction
    """

    def __init__(self):
        """Initialize collapse confidence analyzer."""
        self.confidence_history: List[ConfidenceState] = []
        self.collapse_events: List[CollapseEvent] = []
        self.max_history = 10000

        # Thresholds for collapse detection
        self.confidence_threshold = 0.3
        self.volatility_threshold = 0.05
        self.volume_threshold = 2.0
        self.price_threshold = -0.1

        # Decay parameters
        self.confidence_decay_rate = 0.95
        self.recovery_rate = 0.02

        logger.info("CollapseConfidence initialized")

    def update_confidence_state(
        self,
        confidence_level: float,
        volatility: float,
        volume: float,
        price_change: float,
        liquidity_score: Optional[float] = None
    ) -> ConfidenceState:
        """
        Update confidence state with new market data.

        Parameters:
        -----------
        confidence_level : float
            Current confidence level (0.0 to 1.0)
        volatility : float
            Current volatility measure
        volume : float
            Current trading volume
        price_change : float
            Price change percentage
        liquidity_score : Optional[float]
            Liquidity score (0.0 to 1.0)

        Returns:
        --------
        ConfidenceState
            Updated confidence state
        """
        try:
            # Calculate stress indicator
            stress_indicator = self._calculate_stress_indicator(
                confidence_level, volatility, volume, price_change
            )

            # Use default liquidity score if not provided
            if liquidity_score is None:
                liquidity_score = self._estimate_liquidity_score(volume, volatility)

            state = ConfidenceState(
                timestamp=datetime.now(),
                confidence_level=unified_math.max(0.0, unified_math.min(1.0, confidence_level)),
                volatility=unified_math.max(0.0, volatility),
                volume=unified_math.max(0.0, volume),
                price_change=price_change,
                liquidity_score=unified_math.max(0.0, unified_math.min(1.0, liquidity_score)),
                stress_indicator=unified_math.max(0.0, unified_math.min(1.0, stress_indicator))
            )

            # Store in history
            self.confidence_history.append(state)
            if len(self.confidence_history) > self.max_history:
                self.confidence_history.pop(0)

            return state

        except Exception as e:
            logger.error(f"Error updating confidence state: {e}")
            raise

    def _calculate_stress_indicator(
        self,
        confidence: float,
        volatility: float,
        volume: float,
        price_change: float
    ) -> float:
        """Calculate market stress indicator."""
        try:
            # Normalize inputs
            vol_norm = unified_math.min(1.0, volatility / 0.1)  # Normalize to 10% volatility
            vol_norm = unified_math.min(1.0, volume / 1000000)  # Normalize to 1M volume
            price_norm = unified_math.abs(price_change) / 0.1  # Normalize to 10% price change

            # Weighted stress calculation
            stress = (
                0.3 * (1.0 - confidence) +  # Low confidence = high stress
                0.3 * vol_norm +            # High volatility = high stress
                0.2 * vol_norm +            # High volume = moderate stress
                0.2 * price_norm            # Large price moves = stress
            )

            return unified_math.max(0.0, unified_math.min(1.0, stress))

        except Exception as e:
            logger.error(f"Error calculating stress indicator: {e}")
            return 0.5

    def _estimate_liquidity_score(self, volume: float, volatility: float) -> float:
        """Estimate liquidity score from volume and volatility."""
        try:
            # Higher volume and lower volatility = higher liquidity
            volume_score = unified_math.min(1.0, volume / 1000000)  # Normalize to 1M
            volatility_penalty = unified_math.min(1.0, volatility / 0.1)  # Normalize to 10%

            liquidity = volume_score * (1.0 - volatility_penalty)
            return unified_math.max(0.0, unified_math.min(1.0, liquidity))

        except Exception as e:
            logger.error(f"Error estimating liquidity score: {e}")
            return 0.5

    def detect_collapse_events(self) -> List[CollapseEvent]:
        """
        Detect collapse events from confidence history.

        Returns:
        --------
        List[CollapseEvent]
            List of detected collapse events
        """
        try:
            if len(self.confidence_history) < 10:
                return []

            events = []
            recent_states = self.confidence_history[-100:]  # Last 100 states

            # Find periods of significant confidence decline
            for i in range(1, len(recent_states)):
                prev_state = recent_states[i-1]
                curr_state = recent_states[i]

                # Check for collapse conditions
                confidence_drop = prev_state.confidence_level - curr_state.confidence_level
                volatility_spike = curr_state.volatility - prev_state.volatility
                volume_surge = curr_state.volume / unified_math.max(prev_state.volume, 1.0)
                price_crash = curr_state.price_change

                # Detect collapse event
                if (confidence_drop > self.confidence_threshold or
                    volatility_spike > self.volatility_threshold or
                    volume_surge > self.volume_threshold or
                    price_crash < self.price_threshold):

                    # Determine severity
                    severity = self._determine_collapse_severity(
                        confidence_drop, volatility_spike, volume_surge, price_crash
                    )

                    # Check if this is a new event or continuation
                    if not self._is_continuation_of_existing_event(curr_state.timestamp, events):
                        event = CollapseEvent(
                            event_id=f"collapse_{len(events)}_{int(time.time())}",
                            start_time=curr_state.timestamp,
                            end_time=None,
                            confidence_drop=confidence_drop,
                            volatility_spike=volatility_spike,
                            volume_surge=volume_surge,
                            price_crash=price_crash,
                            severity=severity,
                            recovery_time=None
                        )
                        events.append(event)

            # Update existing events
            self._update_collapse_events(events)

            return events

        except Exception as e:
            logger.error(f"Error detecting collapse events: {e}")
            return []

    def _determine_collapse_severity(
        self,
        confidence_drop: float,
        volatility_spike: float,
        volume_surge: float,
        price_crash: float
    ) -> str:
        """Determine severity of collapse event."""
        try:
            # Calculate severity score
            severity_score = (
                confidence_drop * 0.4 +
                volatility_spike * 0.3 +
                (volume_surge - 1.0) * 0.2 +
                unified_math.abs(price_crash) * 0.1
            )

            if severity_score > 0.7:
                return "critical"
            elif severity_score > 0.5:
                return "high"
            elif severity_score > 0.3:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Error determining collapse severity: {e}")
            return "medium"

    def _is_continuation_of_existing_event(
        self,
        timestamp: datetime,
        events: List[CollapseEvent]
    ) -> bool:
        """Check if timestamp is continuation of existing collapse event."""
        try:
            for event in events:
                if event.end_time is None:  # Active event
                    time_diff = (timestamp - event.start_time).total_seconds()
                    if time_diff < 3600:  # Within 1 hour
                        return True
            return False

        except Exception as e:
            logger.error(f"Error checking event continuation: {e}")
            return False

    def _update_collapse_events(self, events: List[CollapseEvent]) -> None:
        """Update collapse events with recovery information."""
        try:
            for event in events:
                if event.end_time is None:  # Active event
                    # Check for recovery
                    if self._has_recovered(event.start_time):
                        event.end_time = datetime.now()
                        event.recovery_time = (
                            event.end_time - event.start_time
                        ).total_seconds()

            # Store events
            self.collapse_events.extend(events)

        except Exception as e:
            logger.error(f"Error updating collapse events: {e}")

    def _has_recovered(self, collapse_start: datetime) -> bool:
        """Check if market has recovered from collapse."""
        try:
            # Get recent confidence states
            recent_states = [
                state for state in self.confidence_history
                if state.timestamp > collapse_start
            ]

            if len(recent_states) < 5:
                return False

            # Check if confidence has recovered
            recent_confidence = [state.confidence_level for state in recent_states[-5:]]
            avg_confidence = unified_math.unified_math.mean(recent_confidence)

            return avg_confidence > 0.6  # Recovery threshold

        except Exception as e:
            logger.error(f"Error checking recovery: {e}")
            return False

    def analyze_collapse_confidence(self) -> CollapseAnalysis:
        """
        Perform comprehensive collapse confidence analysis.

        Returns:
        --------
        CollapseAnalysis
            Complete collapse analysis result
        """
        try:
            if not self.confidence_history:
                return self._create_empty_analysis()

            # Get current state
            current_state = self.confidence_history[-1]

            # Calculate confidence trend
            confidence_trend = self._calculate_confidence_trend()

            # Detect collapse events
            detected_events = self.detect_collapse_events()

            # Calculate collapse risk
            collapse_risk = self._calculate_collapse_risk()

            # Determine stress level
            stress_level = self._determine_stress_level(current_state.stress_indicator)

            # Calculate recovery probability
            recovery_probability = self._calculate_recovery_probability()

            # Generate recommendations
            recommendations = self._generate_recommendations(
                current_state, collapse_risk, stress_level
            )

            return CollapseAnalysis(
                current_confidence=current_state.confidence_level,
                confidence_trend=confidence_trend,
                collapse_risk=collapse_risk,
                detected_events=detected_events,
                stress_level=stress_level,
                recovery_probability=recovery_probability,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Error in collapse confidence analysis: {e}")
            return self._create_empty_analysis()

    def _calculate_confidence_trend(self) -> float:
        """Calculate confidence trend over recent history."""
        try:
            if len(self.confidence_history) < 10:
                return 0.0

            recent_states = self.confidence_history[-10:]
            confidence_values = [state.confidence_level for state in recent_states]

            # Linear regression for trend
            x = np.arange(len(confidence_values))
            slope, _, _, _, _ = stats.linregress(x, confidence_values)

            return slope

        except Exception as e:
            logger.error(f"Error calculating confidence trend: {e}")
            return 0.0

    def _calculate_collapse_risk(self) -> float:
        """Calculate current collapse risk."""
        try:
            if not self.confidence_history:
                return 0.0

            current_state = self.confidence_history[-1]

            # Risk factors
            low_confidence_risk = 1.0 - current_state.confidence_level
            high_volatility_risk = unified_math.min(1.0, current_state.volatility / 0.1)
            low_liquidity_risk = 1.0 - current_state.liquidity_score
            stress_risk = current_state.stress_indicator

            # Weighted risk calculation
            total_risk = (
                0.4 * low_confidence_risk +
                0.3 * high_volatility_risk +
                0.2 * low_liquidity_risk +
                0.1 * stress_risk
            )

            return unified_math.max(0.0, unified_math.min(1.0, total_risk))

        except Exception as e:
            logger.error(f"Error calculating collapse risk: {e}")
            return 0.5

    def _determine_stress_level(self, stress_indicator: float) -> str:
        """Determine current stress level."""
        try:
            if stress_indicator > 0.8:
                return "critical"
            elif stress_indicator > 0.6:
                return "high"
            elif stress_indicator > 0.4:
                return "medium"
            elif stress_indicator > 0.2:
                return "low"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error determining stress level: {e}")
            return "medium"

    def _calculate_recovery_probability(self) -> float:
        """Calculate probability of market recovery."""
        try:
            if not self.confidence_history:
                return 0.5

            current_state = self.confidence_history[-1]

            # Recovery factors
            confidence_factor = current_state.confidence_level
            liquidity_factor = current_state.liquidity_score
            volatility_factor = 1.0 - unified_math.min(1.0, current_state.volatility / 0.1)

            # Weighted recovery probability
            recovery_prob = (
                0.5 * confidence_factor +
                0.3 * liquidity_factor +
                0.2 * volatility_factor
            )

            return unified_math.max(0.0, unified_math.min(1.0, recovery_prob))

        except Exception as e:
            logger.error(f"Error calculating recovery probability: {e}")
            return 0.5

    def _generate_recommendations(
        self,
        current_state: ConfidenceState,
        collapse_risk: float,
        stress_level: str
    ) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []

        try:
            if collapse_risk > 0.7:
                recommendations.append("High collapse risk detected - consider reducing position sizes")
                recommendations.append("Monitor liquidity conditions closely")

            if stress_level in ["high", "critical"]:
                recommendations.append("Market stress levels elevated - increase risk monitoring")
                recommendations.append("Consider defensive positioning")

            if current_state.confidence_level < 0.3:
                recommendations.append("Low market confidence - review trading strategy")
                recommendations.append("Consider waiting for confidence recovery")

            if current_state.liquidity_score < 0.3:
                recommendations.append("Low liquidity detected - reduce trade sizes")
                recommendations.append("Monitor bid-ask spreads")

            if not recommendations:
                recommendations.append("Market conditions appear stable")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    def _create_empty_analysis(self) -> CollapseAnalysis:
        """Create empty analysis when no data available."""
        return CollapseAnalysis(
            current_confidence=0.5,
            confidence_trend=0.0,
            collapse_risk=0.5,
            detected_events=[],
            stress_level="medium",
            recovery_probability=0.5,
            recommendations=["Insufficient data for analysis"]
        )

    def get_collapse_statistics(self) -> Dict[str, Any]:
        """Get collapse confidence statistics."""
        if not self.confidence_history:
            return {"error": "No confidence history available"}

        total_events = len(self.collapse_events)
        active_events = sum(1 for event in self.collapse_events if event.end_time is None)
        recovered_events = total_events - active_events

        # Event severity distribution
        severity_counts = {}
        for event in self.collapse_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

        # Average recovery time
        recovery_times = [
            event.recovery_time for event in self.collapse_events
            if event.recovery_time is not None
        ]
        avg_recovery_time = unified_math.unified_math.mean(recovery_times) if recovery_times else 0.0

        return {
            "total_events": total_events,
            "active_events": active_events,
            "recovered_events": recovered_events,
            "severity_distribution": severity_counts,
            "average_recovery_time": avg_recovery_time,
            "current_confidence": self.confidence_history[-1].confidence_level if self.confidence_history else 0.5,
            "confidence_history_length": len(self.confidence_history)
        }


def main() -> None:
    """Test function for CollapseConfidence."""
    safe_print("📉 Testing Collapse Confidence...")

    analyzer = CollapseConfidence()

    # Simulate market data
    for i in range(100):
        # Simulate normal market conditions
        if i < 80:
            confidence = 0.7 + np.random.normal(0, 0.1)
            volatility = 0.02 + np.random.normal(0, 0.005)
            volume = 500000 + np.random.normal(0, 100000)
            price_change = np.random.normal(0, 0.01)
        else:
            # Simulate collapse conditions
            confidence = 0.2 + np.random.normal(0, 0.1)
            volatility = 0.08 + np.random.normal(0, 0.02)
            volume = 2000000 + np.random.normal(0, 500000)
            price_change = -0.05 + np.random.normal(0, 0.02)

        state = analyzer.update_confidence_state(
            confidence_level=confidence,
            volatility=volatility,
            volume=volume,
            price_change=price_change
        )

    # Perform analysis
    analysis = analyzer.analyze_collapse_confidence()
    safe_print(f"✅ Collapse analysis completed:")
    safe_print(f"   Current confidence: {analysis.current_confidence:.3f}")
    safe_print(f"   Confidence trend: {analysis.confidence_trend:.3f}")
    safe_print(f"   Collapse risk: {analysis.collapse_risk:.3f}")
    safe_print(f"   Stress level: {analysis.stress_level}")
    safe_print(f"   Recovery probability: {analysis.recovery_probability:.3f}")
    safe_print(f"   Detected events: {len(analysis.detected_events)}")
    safe_print(f"   Recommendations: {analysis.recommendations}")

    # Get statistics
    stats = analyzer.get_collapse_statistics()
    safe_print(f"📊 Collapse statistics: {stats}")

    return 0

if __name__ == "__main__":
    exit(main())
