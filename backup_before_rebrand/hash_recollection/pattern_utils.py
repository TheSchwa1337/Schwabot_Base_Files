import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from core.unified_math_system import unified_math

#!/usr/bin/env python3
"""
Pattern Utilities Module
========================

Provides pattern recognition and analysis utilities for trading data,
including technical patterns, trend analysis, and signal generation.
Integrates with the unified math system and provides API endpoints.
"""


# Import unified math system
try:

    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

    # Fallback math functions
    def unified_math(operation: str, *args, **kwargs):
        """Fallback unified math function."""
        if operation == "pattern":
            return np.random.random()  # Placeholder
        return 0.0


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Pattern type enumeration."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"


@dataclass
class PatternMatch:
    """Pattern match result."""

    pattern_id: str
    pattern_type: PatternType
    confidence: float
    start_index: int
    end_index: int
    price_data: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis result."""

    trend_direction: str
    strength: float
    duration: int
    slope: float
    r_squared: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternUtils:
    """
    Pattern Utilities for trading data analysis.

    Provides pattern recognition, trend analysis, and signal generation
    for trading decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pattern utilities."""
        self.config = config or self._default_config()

        # Pattern tracking
        self.pattern_history: List[PatternMatch] = []
        self.max_patterns = self.config.get("max_patterns", 500)

        # Performance tracking
        self.total_patterns_analyzed = 0
        self.total_trends_analyzed = 0

        logger.info("📊 Pattern Utils initialized")

    def _default_config():-> Dict[str, Any]:
        """Default configuration."""
        return {
            "max_patterns": 500,
            "min_pattern_length": 5,
            "trend_confidence_threshold": 0.6,
            "pattern_confidence_threshold": 0.5,
            "trend_window": 20,
            "breakout_threshold": 0.02,
        }

    def analyze_trend():-> TrendAnalysis:
        """
        Analyze trend in price data.

        Args:
            price_data: List of price values

        Returns:
            TrendAnalysis object with trend information
        """
        if len(price_data) < self.config["min_pattern_length"]:
            return self._create_default_trend()

        try:
            # Calculate linear regression
            x = np.arange(len(price_data))
            slope, intercept = np.polyfit(x, price_data, 1)

            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((price_data - y_pred) ** 2)
            ss_tot = np.sum((price_data - np.mean(price_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Determine trend direction and strength
            if slope > 0:
                trend_direction = "up"
                strength = min(abs(slope) * 100, 1.0)
            elif slope < 0:
                trend_direction = "down"
                strength = min(abs(slope) * 100, 1.0)
            else:
                trend_direction = "sideways"
                strength = 0.0

            # Adjust strength based on R-squared
            strength *= r_squared

            trend = TrendAnalysis(
                trend_direction=trend_direction,
                strength=strength,
                duration=len(price_data),
                slope=slope,
                r_squared=r_squared,
            )

            self.total_trends_analyzed += 1
            return trend

        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return self._create_default_trend()

    def _create_default_trend():-> TrendAnalysis:
        """Create default trend analysis."""
        return TrendAnalysis(
            trend_direction="sideways",
            strength=0.0,
            duration=0,
            slope=0.0,
            r_squared=0.0,
        )

    def detect_patterns():-> List[PatternMatch]:
        """
        Detect patterns in price data.

        Args:
            price_data: List of price values

        Returns:
            List of detected patterns
        """
        patterns = []

        try:
            if len(price_data) < self.config["min_pattern_length"]:
                return patterns

            # Detect different pattern types
            patterns.extend(self._detect_trend_patterns(price_data))
            patterns.extend(self._detect_breakout_patterns(price_data))
            patterns.extend(self._detect_reversal_patterns(price_data))
            patterns.extend(self._detect_consolidation_patterns(price_data))

            # Update pattern history
            for pattern in patterns:
                self.pattern_history.append(pattern)
                if len(self.pattern_history) > self.max_patterns:
                    self.pattern_history.pop(0)

            self.total_patterns_analyzed += len(patterns)

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    def _detect_trend_patterns():-> List[PatternMatch]:
        """Detect trend patterns."""
        patterns = []

        # Analyze overall trend
        trend = self.analyze_trend(price_data)

        if trend.strength > self.config["trend_confidence_threshold"]:
            pattern_type = (
                PatternType.TREND_UP
                if trend.trend_direction == "up"
                else PatternType.TREND_DOWN
            )

            patterns.append(
                PatternMatch(
                    pattern_id=f"trend_{int(time.time() * 1000)}",
                    pattern_type=pattern_type,
                    confidence=trend.strength,
                    start_index=0,
                    end_index=len(price_data) - 1,
                    price_data=price_data,
                )
            )

        return patterns

    def _detect_breakout_patterns():-> List[PatternMatch]:
        """Detect breakout patterns."""
        patterns = []

        if len(price_data) < 10:
            return patterns

        # Calculate moving averages
        long_ma = np.mean(price_data[-10:])

        # Detect breakouts
        current_price = price_data[-1]
        threshold = self.config["breakout_threshold"]

        if current_price > long_ma * (1 + threshold):
            # Upside breakout
            patterns.append(
                PatternMatch(
                    pattern_id=f"breakout_up_{int(time.time() * 1000)}",
                    pattern_type=PatternType.BREAKOUT,
                    confidence=0.7,
                    start_index=len(price_data) - 10,
                    end_index=len(price_data) - 1,
                    price_data=price_data[-10:],
                )
            )
        elif current_price < long_ma * (1 - threshold):
            # Downside breakout
            patterns.append(
                PatternMatch(
                    pattern_id=f"breakout_down_{int(time.time() * 1000)}",
                    pattern_type=PatternType.BREAKOUT,
                    confidence=0.7,
                    start_index=len(price_data) - 10,
                    end_index=len(price_data) - 1,
                    price_data=price_data[-10:],
                )
            )

        return patterns

    def _detect_reversal_patterns():-> List[PatternMatch]:
        """Detect reversal patterns."""
        patterns = []

        if len(price_data) < 15:
            return patterns

        # Look for double tops/bottoms
        recent_data = price_data[-15:]

        # Find local maxima and minima
        peaks = []
        troughs = []

        for i in range(1, len(recent_data) - 1):
            if (
                recent_data[i] > recent_data[i - 1]
                and recent_data[i] > recent_data[i + 1]
            ):
                peaks.append((i, recent_data[i]))
            elif (
                recent_data[i] < recent_data[i - 1]
                and recent_data[i] < recent_data[i + 1]
            ):
                troughs.append((i, recent_data[i]))

        # Check for double top
        if len(peaks) >= 2:
            peak1, peak2 = peaks[-2:]
            if abs(peak1[1] - peak2[1]) / peak1[1] < 0.05:  # Within 5%
                patterns.append(
                    PatternMatch(
                        pattern_id=f"double_top_{int(time.time() * 1000)}",
                        pattern_type=PatternType.REVERSAL,
                        confidence=0.6,
                        start_index=len(price_data) - 15 + peak1[0],
                        end_index=len(price_data) - 15 + peak2[0],
                        price_data=recent_data[peak1[0] : peak2[0] + 1],
                    )
                )

        # Check for double bottom
        if len(troughs) >= 2:
            trough1, trough2 = troughs[-2:]
            if abs(trough1[1] - trough2[1]) / trough1[1] < 0.05:  # Within 5%
                patterns.append(
                    PatternMatch(
                        pattern_id=f"double_bottom_{int(time.time() * 1000)}",
                        pattern_type=PatternType.REVERSAL,
                        confidence=0.6,
                        start_index=len(price_data) - 15 + trough1[0],
                        end_index=len(price_data) - 15 + trough2[0],
                        price_data=recent_data[trough1[0] : trough2[0] + 1],
                    )
                )

        return patterns

    def _detect_consolidation_patterns():-> List[PatternMatch]:
        """Detect consolidation patterns."""
        patterns = []

        if len(price_data) < 10:
            return patterns

        # Calculate price range and volatility
        recent_data = price_data[-10:]
        price_range = max(recent_data) - min(recent_data)
        avg_price = np.mean(recent_data)
        volatility = price_range / avg_price

        # Low volatility suggests consolidation
        if volatility < 0.05:  # Less than 5% range
            trend = self.analyze_trend(recent_data)

            if trend.strength < 0.3:  # Weak trend
                patterns.append(
                    PatternMatch(
                        pattern_id=f"consolidation_{int(time.time() * 1000)}",
                        pattern_type=PatternType.CONSOLIDATION,
                        confidence=0.8,
                        start_index=len(price_data) - 10,
                        end_index=len(price_data) - 1,
                        price_data=recent_data,
                    )
                )

        return patterns

    def get_pattern_summary():-> Dict[str, Any]:
        """Get summary of pattern analysis."""
        if not self.pattern_history:
            return {"status": "no_data"}

        # Count pattern types
        pattern_counts = {}
        for pattern in self.pattern_history:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

        return {
            "total_patterns_analyzed": self.total_patterns_analyzed,
            "total_trends_analyzed": self.total_trends_analyzed,
            "pattern_history_size": len(self.pattern_history),
            "pattern_type_counts": pattern_counts,
            "recent_patterns": len(
                [
                    p
                    for p in self.pattern_history
                    if time.time() - p.metadata.get("timestamp", 0) < 3600
                ]
            ),
        }

    def get_recent_patterns():-> List[Dict[str, Any]]:
        """Get recent detected patterns."""
        recent_patterns = self.pattern_history[-count:]
        return [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "confidence": pattern.confidence,
                "start_index": pattern.start_index,
                "end_index": pattern.end_index,
                "price_range": len(pattern.price_data),
            }
            for pattern in recent_patterns
        ]


# API Integration Functions
def create_pattern_utils_api_endpoints(app):
    """Create FastAPI endpoints for pattern utilities."""
    if not hasattr(app, "pattern_utils"):
        app.pattern_utils = PatternUtils()

    @app.post("/pattern/analyze_trend")
    async def analyze_trend_endpoint(price_data: List[float]):
        """Analyze trend in price data."""
        try:
            trend = app.pattern_utils.analyze_trend(price_data)
            return {
                "success": True,
                "trend_direction": trend.trend_direction,
                "strength": trend.strength,
                "duration": trend.duration,
                "slope": trend.slope,
                "r_squared": trend.r_squared,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/pattern/detect")
    async def detect_patterns_endpoint(price_data: List[float]):
        """Detect patterns in price data."""
        try:
            patterns = app.pattern_utils.detect_patterns(price_data)
            return {
                "success": True,
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type.value,
                        "confidence": p.confidence,
                        "start_index": p.start_index,
                        "end_index": p.end_index,
                        "price_range": len(p.price_data),
                    }
                    for p in patterns
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/pattern/summary")
    async def get_pattern_summary_endpoint():
        """Get pattern analysis summary."""
        try:
            return {"success": True, "summary": app.pattern_utils.get_pattern_summary()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/pattern/recent")
    async def get_recent_patterns_endpoint(count: int = 10):
        """Get recent detected patterns."""
        try:
            return {
                "success": True,
                "patterns": app.pattern_utils.get_recent_patterns(count),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return app
