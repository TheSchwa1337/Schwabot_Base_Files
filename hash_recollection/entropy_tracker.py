import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from core.unified_math_system import unified_math

#!/usr/bin/env python3
"""
Entropy Tracker Module
======================

Tracks entropy patterns in trading data and provides entropy-based
signals for trading decisions. Integrates with the unified math system
and provides API endpoints for entropy analysis.
"""


# Import unified math system
try:

    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

    # Fallback math functions
    def unified_math(operation: str, *args, **kwargs):
        """Fallback unified math function."""
        if operation == "entropy":
            return np.random.random()  # Placeholder
        return 0.0


logger = logging.getLogger(__name__)


class EntropyState(Enum):
    """Entropy state enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class EntropyMetrics:
    """Entropy metrics container."""

    timestamp: float
    entropy_value: float
    state: EntropyState
    confidence: float
    trend_direction: str
    volatility_factor: float
    pattern_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntropySignal:
    """Entropy-based trading signal."""

    signal_id: str
    timestamp: float
    signal_type: str  # "buy", "sell", "hold"
    strength: float
    confidence: float
    entropy_context: EntropyMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntropyTracker:
    """
    Entropy Tracker for trading pattern analysis.

    Tracks entropy patterns in price movements and provides
    entropy-based signals for trading decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the entropy tracker."""
        self.config = config or self._default_config()

        # Entropy tracking
        self.entropy_history: List[EntropyMetrics] = []
        self.max_history_size = self.config.get("max_history_size", 1000)

        # Signal generation
        self.signal_history: List[EntropySignal] = []
        self.max_signals = self.config.get("max_signals", 500)

        # Performance tracking
        self.total_entropy_calculations = 0
        self.total_signals_generated = 0

        # State management
        self.current_state = EntropyState.MEDIUM
        self.last_update = time.time()

        logger.info("🔍 Entropy Tracker initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "max_history_size": 1000,
            "max_signals": 500,
            "entropy_thresholds": {"low": 0.3, "medium": 0.6, "high": 0.8},
            "signal_confidence_threshold": 0.7,
            "volatility_window": 20,
            "pattern_detection_window": 50,
        }

    def calculate_entropy(self, price_data: List[float]) -> EntropyMetrics:
        """Calculate entropy from price data."""
        if len(price_data) < 2:
            return self._create_default_metrics()
        try:
            price_changes = np.diff(price_data)
            if UNIFIED_MATH_AVAILABLE:
                entropy_value = unified_math("entropy", price_changes)
            else:
                hist, _ = np.histogram(price_changes, bins=20, density=True)
                hist = hist[hist > 0]
                entropy_value = -np.sum(hist * np.log2(hist))
            entropy_value = np.clip(entropy_value / 4.0, 0.0, 1.0)
            thresholds = self.config["entropy_thresholds"]
            if entropy_value < thresholds["low"]:
                state = EntropyState.LOW
            elif entropy_value < thresholds["medium"]:
                state = EntropyState.MEDIUM
            elif entropy_value < thresholds["high"]:
                state = EntropyState.HIGH
            else:
                state = EntropyState.EXTREME
            volatility_factor = np.std(price_changes) / np.mean(np.abs(price_changes))
            pattern_strength = self._calculate_pattern_strength(price_data)
            trend_direction = self._determine_trend_direction(price_data)
            confidence = self._calculate_confidence(entropy_value, volatility_factor)
            metrics = EntropyMetrics(
                timestamp=time.time(),
                entropy_value=entropy_value,
                state=state,
                confidence=confidence,
                trend_direction=trend_direction,
                volatility_factor=volatility_factor,
                pattern_strength=pattern_strength,
            )
            self._update_history(metrics)
            self.total_entropy_calculations += 1
            return metrics
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return self._create_default_metrics()

    def _create_default_metrics(self) -> EntropyMetrics:
        """Create default entropy metrics."""
        return EntropyMetrics(
            timestamp=time.time(),
            entropy_value=0.5,
            state=EntropyState.MEDIUM,
            confidence=0.5,
            trend_direction="neutral",
            volatility_factor=1.0,
            pattern_strength=0.5,
        )

    def _calculate_pattern_strength(self, price_data: List[float]) -> float:
        """Calculate pattern strength in price data."""
        if len(price_data) < 10:
            return 0.5
        changes = np.diff(price_data)
        autocorr = np.correlate(changes, changes, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        pattern_strength = np.max(autocorr[1:10]) / autocorr[0]
        return np.clip(pattern_strength, 0.0, 1.0)

    def _determine_trend_direction(self, price_data: List[float]) -> str:
        """Determine trend direction from price data."""
        if len(price_data) < 5:
            return "neutral"
        recent_prices = price_data[-5:]
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        if slope > 0.001:
            return "up"
        elif slope < -0.001:
            return "down"
        else:
            return "neutral"

    def _calculate_confidence(self, entropy_value: float, volatility_factor: float) -> float:
        """Calculate confidence in entropy calculation."""
        entropy_confidence = 1.0 - abs(entropy_value - 0.5) * 2
        volatility_confidence = 1.0 - min(volatility_factor, 1.0)
        return (entropy_confidence + volatility_confidence) / 2

    def generate_signal(self, price_data: List[float]) -> Optional[EntropySignal]:
        """Generate trading signal based on entropy analysis."""
        metrics = self.calculate_entropy(price_data)
        if metrics.confidence < self.config["signal_confidence_threshold"]:
            return None
        signal_type = self._determine_signal_type(metrics)
        signal_strength = self._calculate_signal_strength(metrics)
        if signal_strength < 0.3:
            return None
        signal = EntropySignal(
            signal_id=f"entropy_{int(time.time() * 1000)}",
            timestamp=time.time(),
            signal_type=signal_type,
            strength=signal_strength,
            confidence=metrics.confidence,
            entropy_context=metrics,
        )
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_signals:
            self.signal_history.pop(0)
        self.total_signals_generated += 1
        return signal

    def _determine_signal_type(self, metrics: EntropyMetrics) -> str:
        """Determine signal type based on entropy metrics."""
        if metrics.state == EntropyState.LOW:
            return "buy" if metrics.trend_direction == "up" else "hold"
        elif metrics.state == EntropyState.HIGH:
            return "sell" if metrics.trend_direction == "down" else "hold"
        elif metrics.state == EntropyState.EXTREME:
            return "hold"
        else:
            return "hold"

    def _calculate_signal_strength(self, metrics: EntropyMetrics) -> float:
        """Calculate signal strength based on entropy metrics."""
        base_strength = metrics.pattern_strength * metrics.confidence
        if metrics.state == EntropyState.LOW:
            base_strength *= 1.2
        elif metrics.state == EntropyState.HIGH:
            base_strength *= 1.1
        elif metrics.state == EntropyState.EXTREME:
            base_strength *= 0.5
        return np.clip(base_strength, 0.0, 1.0)

    def get_entropy_summary(self) -> Dict[str, Any]:
        """Get summary of entropy tracking."""
        if not self.entropy_history:
            return {"status": "no_data"}
        recent_metrics = self.entropy_history[-10:]
        return {
            "current_state": self.current_state.value,
            "last_update": self.last_update,
            "total_calculations": self.total_entropy_calculations,
            "total_signals": self.total_signals_generated,
            "recent_entropy_avg": np.mean([m.entropy_value for m in recent_metrics]),
            "recent_confidence_avg": np.mean([m.confidence for m in recent_metrics]),
            "recent_volatility_avg": np.mean([m.volatility_factor for m in recent_metrics]),
            "history_size": len(self.entropy_history),
            "signal_history_size": len(self.signal_history),
        }

    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals."""
        recent_signals = self.signal_history[-count:]
        return [
            {
                "signal_id": signal.signal_id,
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "entropy_value": signal.entropy_context.entropy_value,
                "entropy_state": signal.entropy_context.state.value,
            }
            for signal in recent_signals
        ]


# API Integration Functions
def create_entropy_api_endpoints(app):
    """Create FastAPI endpoints for entropy tracking."""
    if not hasattr(app, "entropy_tracker"):
        app.entropy_tracker = EntropyTracker()

    @app.post("/entropy/calculate")
    async def calculate_entropy_endpoint(price_data: List[float]):
        """Calculate entropy from price data."""
        try:
            metrics = app.entropy_tracker.calculate_entropy(price_data)
            return {
                "success": True,
                "entropy_value": metrics.entropy_value,
                "state": metrics.state.value,
                "confidence": metrics.confidence,
                "trend_direction": metrics.trend_direction,
                "volatility_factor": metrics.volatility_factor,
                "pattern_strength": metrics.pattern_strength,
                "timestamp": metrics.timestamp,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/entropy/signal")
    async def generate_signal_endpoint(price_data: List[float]):
        """Generate trading signal based on entropy."""
        try:
            signal = app.entropy_tracker.generate_signal(price_data)
            if signal:
                return {
                    "success": True,
                    "signal_id": signal.signal_id,
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "entropy_context": {
                        "entropy_value": signal.entropy_context.entropy_value,
                        "state": signal.entropy_context.state.value,
                    },
                }
            else:
                return {"success": False, "message": "No signal generated"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/entropy/summary")
    async def get_entropy_summary_endpoint():
        """Get entropy tracking summary."""
        try:
            return {
                "success": True,
                "summary": app.entropy_tracker.get_entropy_summary(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/entropy/signals")
    async def get_recent_signals_endpoint(count: int = 10):
        """Get recent trading signals."""
        try:
            return {
                "success": True,
                "signals": app.entropy_tracker.get_recent_signals(count),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return app
