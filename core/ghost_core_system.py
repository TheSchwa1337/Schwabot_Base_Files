#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👻 GHOST CORE SYSTEM - ADVANCED SIGNAL PROCESSING ENGINE
=======================================================

Advanced ghost signal processing and pattern recognition system for the Schwabot trading engine.

Implements latent signal evaluation logic for "ghost trades" triggered by
registry echoes of past high-yield delta patterns. Operates in memory, not immediate execution.

Features:
- Ghost signal detection and processing
- Pattern recognition and analysis (trending, oscillating, breakout)
- Predictive modeling and activity forecasting
- Signal strength validation and filtering
- Performance tracking and optimization
- Real-time signal processing with memory management

CUDA Integration:
- GPU-accelerated signal processing with automatic CPU fallback
- Performance monitoring and optimization
- Cross-platform compatibility (Windows, macOS, Linux)

Mathematical Operations:
- Linear regression for trend detection
- Oscillation analysis with zero-crossing detection
- Breakout pattern recognition with moving averages
- Statistical analysis and confidence scoring
- Predictive modeling with weighted pattern analysis
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# CUDA Integration with Fallback
try:
    import cupy as cp
    USING_CUDA = True
    _backend = "cupy (GPU)"
    xp = cp
except ImportError:
    USING_CUDA = False
    _backend = "numpy (CPU)"
    xp = np

# Import existing Schwabot components
try:
    from .advanced_tensor_algebra import AdvancedTensorAlgebra
    from .entropy_math import EntropyMathSystem
    SCHWABOT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some Schwabot components not available: {e}")
    SCHWABOT_COMPONENTS_AVAILABLE = False

# Try to import DistributedMathematicalProcessor separately to avoid circular imports
try:
    from .distributed_mathematical_processor import DistributedMathematicalProcessor
    DISTRIBUTED_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Distributed Mathematical Processor not available: {e}")
    DISTRIBUTED_PROCESSOR_AVAILABLE = False

logger = logging.getLogger(__name__)
if USING_CUDA:
    logger.info(f"⚡ Ghost Core System using GPU acceleration: {_backend}")
else:
    logger.info(f"🔄 Ghost Core System using CPU fallback: {_backend}")

__all__ = [
    "GhostCoreSystem",
    "GhostSignal",
    "GhostPattern",
    "GhostSignalProcessor",
    "PatternAnalyzer",
    "PredictionEngine",
    "create_ghost_core_system",
]


@dataclass
class GhostSignal:
    """Ghost signal with metadata and processing information."""
    signal_type: str
    strength: float
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    confidence: float = 0.0


@dataclass
class GhostPattern:
    """Ghost pattern with characteristics and analysis results."""
    pattern_type: str
    confidence: float
    duration: float
    signals: List[GhostSignal]
    metadata: Dict[str, Any] = field(default_factory=dict)
    prediction_value: float = 0.0
    risk_score: float = 0.0


class GhostSignalProcessor:
    """Advanced signal processing for ghost signals."""
    
    def __init__(self, min_strength: float = 0.05, noise_threshold: float = 0.02):
        self.min_strength = min_strength
        self.noise_threshold = noise_threshold
        self.signal_cache: Dict[str, List[GhostSignal]] = {}
        
    def process_signal(self, signal_data: Dict[str, Any], source: str = "unknown") -> Optional[GhostSignal]:
        """Process and validate incoming signal data."""
        try:
            # Extract signal parameters
            signal_type = signal_data.get("type", "unknown")
            strength = signal_data.get("strength", 0.0)
            timestamp = signal_data.get("timestamp", time.time())
            
            # Apply noise filtering
            if strength < self.noise_threshold:
                logger.debug(f"Signal filtered as noise: {strength}")
                return None
                
            # Validate signal strength
            if strength < self.min_strength:
                logger.debug(f"Signal strength too low: {strength}")
                return None
                
            # Calculate confidence based on signal characteristics
            confidence = self._calculate_signal_confidence(signal_data)
            
            # Create ghost signal
            ghost_signal = GhostSignal(
                signal_type=signal_type,
                strength=strength,
                timestamp=timestamp,
                source=source,
                metadata=signal_data,
                confidence=confidence
            )
            
            # Cache by source
            if source not in self.signal_cache:
                self.signal_cache[source] = []
            self.signal_cache[source].append(ghost_signal)
            
            logger.debug(f"Processed ghost signal: type={signal_type}, strength={strength:.3f}, confidence={confidence:.3f}")
            return ghost_signal
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
            
    def _calculate_signal_confidence(self, signal_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a signal."""
        try:
            strength = signal_data.get("strength", 0.0)
            signal_type = signal_data.get("type", "unknown")
            
            # Base confidence on strength
            base_confidence = min(1.0, strength)
            
            # Adjust based on signal type
            type_multipliers = {
                "trend": 1.2,
                "breakout": 1.1,
                "oscillation": 0.9,
                "reversal": 1.0
            }
            
            multiplier = type_multipliers.get(signal_type, 1.0)
            confidence = base_confidence * multiplier
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.0


class PatternAnalyzer:
    """Advanced pattern analysis for ghost signals."""
    
    def __init__(self, confidence_threshold: float = 0.7, min_signals: int = 3):
        self.confidence_threshold = confidence_threshold
        self.min_signals = min_signals
        self.pattern_cache: Dict[str, List[GhostPattern]] = {}
        
    def analyze_patterns(self, signals: List[GhostSignal], lookback_period: Optional[int] = None) -> List[GhostPattern]:
        """Analyze signal history for ghost patterns."""
        try:
            if len(signals) < self.min_signals:
                return []
                
            lookback_period = lookback_period or len(signals)
            recent_signals = signals[-lookback_period:]
            
            patterns = []
            
            # Detect different pattern types
            trending_pattern = self._detect_trending_pattern(recent_signals)
            if trending_pattern:
                patterns.append(trending_pattern)
                
            oscillating_pattern = self._detect_oscillating_pattern(recent_signals)
            if oscillating_pattern:
                patterns.append(oscillating_pattern)
                
            breakout_pattern = self._detect_breakout_pattern(recent_signals)
            if breakout_pattern:
                patterns.append(breakout_pattern)
                
            reversal_pattern = self._detect_reversal_pattern(recent_signals)
            if reversal_pattern:
                patterns.append(reversal_pattern)
                
            # Filter by confidence threshold
            valid_patterns = [p for p in patterns if p.confidence >= self.confidence_threshold]
            
            # Calculate additional metrics
            for pattern in valid_patterns:
                pattern.prediction_value = self._calculate_prediction_value(pattern)
                pattern.risk_score = self._calculate_risk_score(pattern)
                
            logger.info(f"Detected {len(valid_patterns)} ghost patterns")
            return valid_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []
            
    def _detect_trending_pattern(self, signals: List[GhostSignal]) -> Optional[GhostPattern]:
        """Detect trending pattern in signal strength."""
        if len(signals) < 5:
            return None
            
        try:
            # Extract signal strengths and timestamps
            strengths = [s.strength for s in signals]
            timestamps = [s.timestamp for s in signals]
            
            # Calculate trend using linear regression
            x = xp.array(timestamps)
            y = xp.array(strengths)
            
            # Simple linear regression
            slope = xp.corrcoef(x, y)[0, 1] * xp.std(y) / xp.std(x)
            confidence = abs(xp.corrcoef(x, y)[0, 1])
            
            # Determine pattern type based on slope
            if slope > 0.01:
                pattern_type = "uptrend"
            elif slope < -0.01:
                pattern_type = "downtrend"
            else:
                pattern_type = "sideways"
                
            if confidence >= self.confidence_threshold:
                return GhostPattern(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    duration=timestamps[-1] - timestamps[0],
                    signals=signals,
                    metadata={"slope": float(slope), "trend_type": pattern_type}
                )
                
        except Exception as e:
            logger.error(f"Error detecting trending pattern: {e}")
            
        return None
        
    def _detect_oscillating_pattern(self, signals: List[GhostSignal]) -> Optional[GhostPattern]:
        """Detect oscillating pattern in signal strength."""
        if len(signals) < 6:
            return None
            
        try:
            # Extract signal strengths
            strengths = [s.strength for s in signals]
            
            # Calculate oscillation metrics
            mean_strength = xp.mean(strengths)
            std_strength = xp.std(strengths)
            
            # Count zero crossings (oscillations)
            zero_crossings = 0
            for i in range(1, len(strengths)):
                if (strengths[i] - mean_strength) * (strengths[i-1] - mean_strength) < 0:
                    zero_crossings += 1
                    
            # Calculate oscillation confidence
            oscillation_ratio = zero_crossings / (len(strengths) - 1)
            confidence = min(1.0, oscillation_ratio * 2)  # Scale to 0-1
            
            if confidence >= self.confidence_threshold:
                return GhostPattern(
                    pattern_type="oscillating",
                    confidence=confidence,
                    duration=signals[-1].timestamp - signals[0].timestamp,
                    signals=signals,
                    metadata={
                        "zero_crossings": zero_crossings,
                        "oscillation_ratio": float(oscillation_ratio),
                        "mean_strength": float(mean_strength),
                        "std_strength": float(std_strength)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error detecting oscillating pattern: {e}")
            
        return None
        
    def _detect_breakout_pattern(self, signals: List[GhostSignal]) -> Optional[GhostPattern]:
        """Detect breakout pattern in signal strength."""
        if len(signals) < 10:
            return None
            
        try:
            # Extract signal strengths
            strengths = [s.strength for s in signals]
            
            # Calculate moving average
            window = min(5, len(strengths) // 2)
            moving_avg = []
            for i in range(window, len(strengths)):
                avg = xp.mean(strengths[i-window:i])
                moving_avg.append(avg)
                
            # Detect breakout
            current_strength = strengths[-1]
            avg_strength = moving_avg[-1] if moving_avg else xp.mean(strengths[:-window])
            
            # Calculate breakout threshold
            threshold = avg_strength + 2 * xp.std(strengths[:-window]) if len(strengths) > window else avg_strength * 1.5
            
            if current_strength > threshold:
                confidence = min(1.0, (current_strength - avg_strength) / threshold)
                
                return GhostPattern(
                    pattern_type="breakout",
                    confidence=confidence,
                    duration=signals[-1].timestamp - signals[0].timestamp,
                    signals=signals,
                    metadata={
                        "breakout_strength": float(current_strength),
                        "threshold": float(threshold),
                        "avg_strength": float(avg_strength)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error detecting breakout pattern: {e}")
            
        return None
        
    def _detect_reversal_pattern(self, signals: List[GhostSignal]) -> Optional[GhostPattern]:
        """Detect reversal pattern in signal strength."""
        if len(signals) < 8:
            return None
            
        try:
            # Extract signal strengths
            strengths = [s.strength for s in signals]
            
            # Split into two halves
            mid_point = len(strengths) // 2
            first_half = strengths[:mid_point]
            second_half = strengths[mid_point:]
            
            # Calculate trends for each half
            first_trend = xp.mean(first_half[-3:]) - xp.mean(first_half[:3]) if len(first_half) >= 6 else 0
            second_trend = xp.mean(second_half[-3:]) - xp.mean(second_half[:3]) if len(second_half) >= 6 else 0
            
            # Check for reversal (opposite trends)
            if first_trend * second_trend < -0.01:  # Opposite trends
                confidence = min(1.0, abs(first_trend - second_trend) / max(abs(first_trend), abs(second_trend)))
                
                if confidence >= self.confidence_threshold:
                    return GhostPattern(
                        pattern_type="reversal",
                        confidence=confidence,
                        duration=signals[-1].timestamp - signals[0].timestamp,
                        signals=signals,
                        metadata={
                            "first_trend": float(first_trend),
                            "second_trend": float(second_trend),
                            "reversal_strength": float(abs(first_trend - second_trend))
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting reversal pattern: {e}")
            
        return None
        
    def _calculate_prediction_value(self, pattern: GhostPattern) -> float:
        """Calculate prediction value for a pattern."""
        try:
            base_value = pattern.confidence
            
            # Adjust based on pattern type
            type_multipliers = {
                "uptrend": 1.2,
                "downtrend": 0.8,
                "breakout": 1.3,
                "reversal": 1.1,
                "oscillating": 0.9
            }
            
            multiplier = type_multipliers.get(pattern.pattern_type, 1.0)
            return min(1.0, base_value * multiplier)
            
        except Exception as e:
            logger.error(f"Error calculating prediction value: {e}")
            return 0.0
            
    def _calculate_risk_score(self, pattern: GhostPattern) -> float:
        """Calculate risk score for a pattern."""
        try:
            # Base risk on pattern type
            base_risk = {
                "uptrend": 0.3,
                "downtrend": 0.7,
                "breakout": 0.5,
                "reversal": 0.6,
                "oscillating": 0.4
            }.get(pattern.pattern_type, 0.5)
            
            # Adjust based on confidence (higher confidence = lower risk)
            confidence_adjustment = 1.0 - pattern.confidence
            risk_score = base_risk + confidence_adjustment * 0.3
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5


class PredictionEngine:
    """Advanced prediction engine for ghost activity."""
    
    def __init__(self, prediction_horizon: int = 10, confidence_decay: float = 0.95):
        self.prediction_horizon = prediction_horizon
        self.confidence_decay = confidence_decay
        self.prediction_history: List[Dict[str, Any]] = []
        
    def predict_ghost_activity(self, patterns: List[GhostPattern], horizon: Optional[int] = None) -> Dict[str, Any]:
        """Predict future ghost activity based on current patterns."""
        try:
            if not patterns:
                return {"prediction": "no_data", "confidence": 0.0, "risk_level": "unknown"}
                
            horizon = horizon or self.prediction_horizon
            
            # Analyze recent patterns
            recent_patterns = patterns[-5:] if len(patterns) >= 5 else patterns
            
            # Calculate weighted prediction
            prediction_result = self._calculate_weighted_prediction(recent_patterns, horizon)
            
            # Add to prediction history
            prediction_result["timestamp"] = time.time()
            self.prediction_history.append(prediction_result)
            
            # Keep history manageable
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-50:]
                
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting ghost activity: {e}")
            return {"prediction": "error", "confidence": 0.0, "risk_level": "unknown"}
            
    def _calculate_weighted_prediction(self, patterns: List[GhostPattern], horizon: int) -> Dict[str, Any]:
        """Calculate weighted prediction based on pattern analysis."""
        try:
            # Weight patterns by confidence and recency
            weighted_predictions = []
            total_weight = 0.0
            
            for i, pattern in enumerate(patterns):
                # Weight decreases with age
                age_weight = self.confidence_decay ** i
                confidence_weight = pattern.confidence
                total_weight += age_weight * confidence_weight
                
                weighted_predictions.append({
                    "pattern_type": pattern.pattern_type,
                    "weight": age_weight * confidence_weight,
                    "prediction_value": pattern.prediction_value,
                    "risk_score": pattern.risk_score
                })
                
            if total_weight == 0:
                return {"prediction": "stable", "confidence": 0.0, "risk_level": "medium"}
                
            # Calculate weighted averages
            avg_prediction_value = sum(wp["prediction_value"] * wp["weight"] for wp in weighted_predictions) / total_weight
            avg_risk_score = sum(wp["risk_score"] * wp["weight"] for wp in weighted_predictions) / total_weight
            
            # Determine prediction direction
            if avg_prediction_value > 0.7:
                prediction = "increasing"
            elif avg_prediction_value < 0.3:
                prediction = "decreasing"
            else:
                prediction = "stable"
                
            # Determine risk level
            if avg_risk_score > 0.7:
                risk_level = "high"
            elif avg_risk_score < 0.3:
                risk_level = "low"
            else:
                risk_level = "medium"
                
            # Calculate overall confidence
            confidence = min(1.0, avg_prediction_value * 0.8 + (1.0 - avg_risk_score) * 0.2)
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "risk_level": risk_level,
                "prediction_value": float(avg_prediction_value),
                "risk_score": float(avg_risk_score),
                "horizon": horizon,
                "patterns_analyzed": len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating weighted prediction: {e}")
            return {"prediction": "error", "confidence": 0.0, "risk_level": "unknown"}


class GhostCoreSystem:
    """
    Ghost Core System for advanced signal processing and pattern recognition.

    Implements ghost signal detection, pattern analysis, and predictive modeling
    for trading system enhancement with GPU acceleration and advanced mathematical processing.
    """

    def __init__(self, sensitivity: float = 0.1, max_patterns: int = 100, max_signals: int = 1000):
        """Initialize the Ghost Core System."""
        self.sensitivity = sensitivity
        self.max_patterns = max_patterns
        self.max_signals = max_signals
        
        # Initialize components
        self.signal_processor = GhostSignalProcessor(min_strength=sensitivity * 0.5)
        self.pattern_analyzer = PatternAnalyzer(confidence_threshold=0.7)
        self.prediction_engine = PredictionEngine()
        
        # Data storage
        self.signal_history: List[GhostSignal] = []
        self.pattern_history: List[GhostPattern] = []
        self.analysis_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.total_signals = 0
        self.total_patterns = 0
        self.detection_accuracy = 0.0
        self.processing_times: List[float] = []
        
        # Initialize mathematical components if available
        self.distributed_processor = None
        self.tensor_algebra = None
        self.entropy_system = None
        
        if SCHWABOT_COMPONENTS_AVAILABLE:
            try:
                self.tensor_algebra = AdvancedTensorAlgebra()
                self.entropy_system = EntropyMathSystem()
                logger.info("✅ Ghost Core System integrated with Schwabot mathematical components")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize some mathematical components: {e}")
                
        if DISTRIBUTED_PROCESSOR_AVAILABLE:
            try:
                self.distributed_processor = DistributedMathematicalProcessor()
                logger.info("✅ Ghost Core System integrated with Distributed Mathematical Processor")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Distributed Mathematical Processor: {e}")
        
        logger.info(f"👻 Ghost Core System initialized with sensitivity {sensitivity}")

    def process_signal(self, signal_data: Dict[str, Any], source: str = "unknown") -> Optional[GhostSignal]:
        """
        Process incoming signal data and create a ghost signal.

        Args:
            signal_data: Raw signal data dictionary
            source: Signal source identifier

        Returns:
            GhostSignal with processed information or None if filtered
        """
        start_time = time.time()
        
        try:
            # Process signal
            ghost_signal = self.signal_processor.process_signal(signal_data, source)
            
            if ghost_signal:
                # Add to history
                self.signal_history.append(ghost_signal)
                self.total_signals += 1
                
                # Keep history manageable
                if len(self.signal_history) > self.max_signals:
                    self.signal_history = self.signal_history[-self.max_signals:]
                    
                # Mark as processed
                ghost_signal.processed = True
                
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep processing times manageable
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-50:]
                
            return ghost_signal
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None

    def analyze_patterns(self, lookback_period: Optional[int] = None) -> List[GhostPattern]:
        """
        Analyze signal history for ghost patterns.

        Args:
            lookback_period: Number of signals to analyze (defaults to all)

        Returns:
            List of detected GhostPattern objects
        """
        try:
            lookback_period = lookback_period or len(self.signal_history)
            
            if len(self.signal_history) < 3:
                return []
                
            # Get recent signals
            recent_signals = self.signal_history[-lookback_period:]
            
            # Analyze patterns
            patterns = self.pattern_analyzer.analyze_patterns(recent_signals, lookback_period)
            
            # Add to pattern history
            self.pattern_history.extend(patterns)
            self.total_patterns += len(patterns)
            
            # Keep pattern history manageable
            if len(self.pattern_history) > self.max_patterns:
                self.pattern_history = self.pattern_history[-self.max_patterns:]
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []

    def predict_ghost_activity(self, horizon: int = 10) -> Dict[str, Any]:
        """
        Predict future ghost activity based on current patterns.

        Args:
            horizon: Prediction horizon in time steps

        Returns:
            Dictionary with prediction results
        """
        try:
            return self.prediction_engine.predict_ghost_activity(self.pattern_history, horizon)
        except Exception as e:
            logger.error(f"Error predicting ghost activity: {e}")
            return {"prediction": "error", "confidence": 0.0, "risk_level": "unknown"}

    def get_ghost_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ghost system statistics."""
        try:
            if not self.signal_history:
                return {"status": "no_data"}
                
            # Calculate signal statistics
            signal_strengths = [s.strength for s in self.signal_history]
            signal_types = [s.signal_type for s in self.signal_history]
            signal_confidences = [s.confidence for s in self.signal_history]
            
            # Pattern statistics
            pattern_types = [p.pattern_type for p in self.pattern_history]
            pattern_confidences = [p.confidence for p in self.pattern_history]
            pattern_prediction_values = [p.prediction_value for p in self.pattern_history]
            pattern_risk_scores = [p.risk_score for p in self.pattern_history]
            
            # Performance statistics
            avg_processing_time = xp.mean(self.processing_times) if self.processing_times else 0.0
            
            return {
                "total_signals": self.total_signals,
                "total_patterns": self.total_patterns,
                "current_signals": len(self.signal_history),
                "current_patterns": len(self.pattern_history),
                "avg_signal_strength": float(xp.mean(signal_strengths)) if signal_strengths else 0.0,
                "max_signal_strength": float(xp.max(signal_strengths)) if signal_strengths else 0.0,
                "avg_signal_confidence": float(xp.mean(signal_confidences)) if signal_confidences else 0.0,
                "avg_pattern_confidence": float(xp.mean(pattern_confidences)) if pattern_confidences else 0.0,
                "avg_prediction_value": float(xp.mean(pattern_prediction_values)) if pattern_prediction_values else 0.0,
                "avg_risk_score": float(xp.mean(pattern_risk_scores)) if pattern_risk_scores else 0.0,
                "signal_types": list(set(signal_types)),
                "pattern_types": list(set(pattern_types)),
                "detection_accuracy": self.detection_accuracy,
                "sensitivity": self.sensitivity,
                "avg_processing_time": float(avg_processing_time),
                "backend": _backend,
                "schwabot_components_available": SCHWABOT_COMPONENTS_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Error getting ghost statistics: {e}")
            return {"status": "error", "error": str(e)}

    def clear_cache(self) -> None:
        """Clear the analysis cache and processing times."""
        self.analysis_cache.clear()
        self.processing_times.clear()
        logger.debug("Ghost Core System cache cleared")

    def cleanup_old_signals(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old signals from history.

        Args:
            max_age_hours: Maximum age of signals to keep

        Returns:
            Number of signals removed
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Filter out old signals
            old_count = len(self.signal_history)
            self.signal_history = [
                s for s in self.signal_history 
                if current_time - s.timestamp < max_age_seconds
            ]
            new_count = len(self.signal_history)
            removed_count = old_count - new_count
            
            # Also clean up old patterns
            old_pattern_count = len(self.pattern_history)
            self.pattern_history = [
                p for p in self.pattern_history 
                if current_time - p.signals[-1].timestamp < max_age_seconds
            ]
            removed_patterns = old_pattern_count - len(self.pattern_history)
            
            logger.info(f"Cleaned up {removed_count} old signals and {removed_patterns} old patterns")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
            return 0

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics."""
        try:
            stats = self.get_ghost_statistics()
            
            # Calculate health indicators
            signal_health = min(1.0, len(self.signal_history) / self.max_signals) if self.max_signals > 0 else 0.0
            pattern_health = min(1.0, len(self.pattern_history) / self.max_patterns) if self.max_patterns > 0 else 0.0
            processing_health = 1.0 - min(1.0, stats.get("avg_processing_time", 0.0) / 0.1)  # Normalize to 100ms
            
            overall_health = (signal_health + pattern_health + processing_health) / 3
            
            return {
                "overall_health": float(overall_health),
                "signal_health": float(signal_health),
                "pattern_health": float(pattern_health),
                "processing_health": float(processing_health),
                "backend_status": _backend,
                "components_available": SCHWABOT_COMPONENTS_AVAILABLE,
                "cache_size": len(self.analysis_cache),
                "processing_times_count": len(self.processing_times)
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"overall_health": 0.0, "error": str(e)}


def create_ghost_core_system(sensitivity: float = 0.1, max_patterns: int = 100, max_signals: int = 1000) -> GhostCoreSystem:
    """Create a new Ghost Core System instance."""
    return GhostCoreSystem(sensitivity=sensitivity, max_patterns=max_patterns, max_signals=max_signals)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Ghost Core System
    ghost_system = create_ghost_core_system(sensitivity=0.05, max_patterns=50, max_signals=500)
    
    print("=== Testing Ghost Core System ===")
    
    # Simulate some ghost signals
    import random
    
    for i in range(20):
        signal_data = {
            "type": random.choice(["trend", "oscillation", "breakout", "reversal"]),
            "strength": random.uniform(0.1, 1.0),
            "timestamp": time.time() + i,
        }
        ghost_system.process_signal(signal_data, "test_source")
    
    # Analyze patterns
    patterns = ghost_system.analyze_patterns()
    print(f"Detected patterns: {len(patterns)}")
    
    for pattern in patterns:
        print(
            f"Pattern: {pattern.pattern_type}, Confidence: {pattern.confidence:.3f}, "
            f"Duration: {pattern.duration:.2f}, Prediction Value: {pattern.prediction_value:.3f}, "
            f"Risk Score: {pattern.risk_score:.3f}"
        )
    
    # Get prediction
    prediction = ghost_system.predict_ghost_activity()
    print(f"Prediction: {prediction}")
    
    # Get statistics
    stats = ghost_system.get_ghost_statistics()
    print(f"Statistics: {stats}")
    
    # Get system health
    health = ghost_system.get_system_health()
    print(f"System Health: {health}") 