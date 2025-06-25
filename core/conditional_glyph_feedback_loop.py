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
"""Conditional Glyph Feedback Loop - Pattern Recognition and Conditional Logic.

This module provides advanced algorithms for:
- Glyph pattern recognition and classification
- Conditional logic processing
- Feedback loop optimization
- Pattern-based trading signals
- Adaptive learning systems

Mathematical Foundation:
- Glyph pattern matching algorithms
- Conditional probability models
- Feedback loop dynamics
- Pattern evolution tracking
- Adaptive threshold adjustment
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from core.unified_math_system import unified_math
from enum import Enum

logger = logging.getLogger(__name__)


class GlyphType(Enum):
    """Types of glyph patterns."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BREAKOUT = "breakout"


class FeedbackType(Enum):
    """Types of feedback mechanisms."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    ADAPTIVE = "adaptive"


@dataclass
class GlyphPattern:
    """Represents a glyph pattern."""
    pattern_id: str
    glyph_type: GlyphType
    confidence: float  # 0.0 to 1.0
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalRule:
    """Represents a conditional rule."""
    rule_id: str
    condition: str
    action: str
    threshold: float
    confidence: float
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackLoop:
    """Represents a feedback loop."""
    loop_id: str
    input_pattern: GlyphPattern
    output_signal: str
    feedback_type: FeedbackType
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlyphAnalysis:
    """Result of glyph pattern analysis."""
    detected_patterns: List[GlyphPattern]
    active_rules: List[ConditionalRule]
    feedback_loops: List[FeedbackLoop]
    confidence_score: float
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConditionalGlyphFeedbackLoop:
    """
    Advanced glyph pattern recognition and conditional feedback system.
    
    Provides mathematical models for:
    - Pattern recognition and classification
    - Conditional logic processing
    - Feedback loop optimization
    - Adaptive learning
    """
    
    def __init__(self):
        """Initialize conditional glyph feedback loop system."""
        self.patterns: List[GlyphPattern] = []
        self.rules: List[ConditionalRule] = []
        self.feedback_loops: List[FeedbackLoop] = []
        self.max_history = 1000
        
        # Pattern recognition thresholds
        self.pattern_thresholds = {
            GlyphType.BULLISH: 0.6,
            GlyphType.BEARISH: 0.6,
            GlyphType.NEUTRAL: 0.5,
            GlyphType.REVERSAL: 0.7,
            GlyphType.CONTINUATION: 0.6,
            GlyphType.BREAKOUT: 0.8
        }
        
        # Feedback loop parameters
        self.feedback_decay = 0.95
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.1
        
        logger.info("ConditionalGlyphFeedbackLoop initialized")
    
    def detect_glyph_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """
        Detect glyph patterns in market data.
        
        Parameters:
        -----------
        price_data : List[float]
            Historical price data
        volume_data : List[float]
            Historical volume data
        technical_indicators : Dict[str, List[float]]
            Technical indicator data
            
        Returns:
        --------
        List[GlyphPattern]
            Detected glyph patterns
        """
        try:
            patterns = []
            
            if len(price_data) < 10:
                return patterns
            
            # Detect different types of patterns
            patterns.extend(self._detect_bullish_patterns(price_data, volume_data, technical_indicators))
            patterns.extend(self._detect_bearish_patterns(price_data, volume_data, technical_indicators))
            patterns.extend(self._detect_reversal_patterns(price_data, volume_data, technical_indicators))
            patterns.extend(self._detect_continuation_patterns(price_data, volume_data, technical_indicators))
            patterns.extend(self._detect_breakout_patterns(price_data, volume_data, technical_indicators))
            
            # Store patterns
            self.patterns.extend(patterns)
            if len(self.patterns) > self.max_history:
                self.patterns = self.patterns[-self.max_history:]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting glyph patterns: {e}")
            return []
    
    def _detect_bullish_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """Detect bullish glyph patterns."""
        patterns = []
        
        try:
            if len(price_data) < 5:
                return patterns
            
            # Calculate bullish indicators
            price_trend = self._calculate_trend(price_data[-5:])
            volume_trend = self._calculate_trend(volume_data[-5:])
            
            # Check for bullish conditions
            bullish_score = 0.0
            features = {}
            
            if price_trend > 0.02:  # 2% upward trend
                bullish_score += 0.4
                features['price_trend'] = price_trend
            
            if volume_trend > 0.1:  # 10% volume increase
                bullish_score += 0.3
                features['volume_trend'] = volume_trend
            
            # Check technical indicators
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi'][-1]
                if 30 < rsi < 70:  # Not overbought/oversold
                    bullish_score += 0.2
                    features['rsi'] = rsi
            
            if 'macd' in technical_indicators:
                macd = technical_indicators['macd'][-1]
                if macd > 0:  # Positive MACD
                    bullish_score += 0.1
                    features['macd'] = macd
            
            if bullish_score > self.pattern_thresholds[GlyphType.BULLISH]:
                pattern = GlyphPattern(
                    pattern_id=f"bullish_{int(time.time())}",
                    glyph_type=GlyphType.BULLISH,
                    confidence=bullish_score,
                    strength=bullish_score,
                    timestamp=datetime.now(),
                    features=features
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting bullish patterns: {e}")
            return []
    
    def _detect_bearish_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """Detect bearish glyph patterns."""
        patterns = []
        
        try:
            if len(price_data) < 5:
                return patterns
            
            # Calculate bearish indicators
            price_trend = self._calculate_trend(price_data[-5:])
            volume_trend = self._calculate_trend(volume_data[-5:])
            
            # Check for bearish conditions
            bearish_score = 0.0
            features = {}
            
            if price_trend < -0.02:  # 2% downward trend
                bearish_score += 0.4
                features['price_trend'] = price_trend
            
            if volume_trend > 0.1:  # High volume on decline
                bearish_score += 0.3
                features['volume_trend'] = volume_trend
            
            # Check technical indicators
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi'][-1]
                if rsi > 70:  # Overbought
                    bearish_score += 0.2
                    features['rsi'] = rsi
            
            if 'macd' in technical_indicators:
                macd = technical_indicators['macd'][-1]
                if macd < 0:  # Negative MACD
                    bearish_score += 0.1
                    features['macd'] = macd
            
            if bearish_score > self.pattern_thresholds[GlyphType.BEARISH]:
                pattern = GlyphPattern(
                    pattern_id=f"bearish_{int(time.time())}",
                    glyph_type=GlyphType.BEARISH,
                    confidence=bearish_score,
                    strength=bearish_score,
                    timestamp=datetime.now(),
                    features=features
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting bearish patterns: {e}")
            return []
    
    def _detect_reversal_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """Detect reversal glyph patterns."""
        patterns = []
        
        try:
            if len(price_data) < 10:
                return patterns
            
            # Look for reversal patterns
            reversal_score = 0.0
            features = {}
            
            # Check for double top/bottom
            if self._is_double_top(price_data[-10:]):
                reversal_score += 0.4
                features['pattern'] = 'double_top'
            
            if self._is_double_bottom(price_data[-10:]):
                reversal_score += 0.4
                features['pattern'] = 'double_bottom'
            
            # Check for divergence
            if 'rsi' in technical_indicators:
                if self._has_divergence(price_data[-5:], technical_indicators['rsi'][-5:]):
                    reversal_score += 0.3
                    features['divergence'] = True
            
            if reversal_score > self.pattern_thresholds[GlyphType.REVERSAL]:
                pattern = GlyphPattern(
                    pattern_id=f"reversal_{int(time.time())}",
                    glyph_type=GlyphType.REVERSAL,
                    confidence=reversal_score,
                    strength=reversal_score,
                    timestamp=datetime.now(),
                    features=features
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {e}")
            return []
    
    def _detect_continuation_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """Detect continuation glyph patterns."""
        patterns = []
        
        try:
            if len(price_data) < 8:
                return patterns
            
            # Look for continuation patterns
            continuation_score = 0.0
            features = {}
            
            # Check for flag pattern
            if self._is_flag_pattern(price_data[-8:]):
                continuation_score += 0.5
                features['pattern'] = 'flag'
            
            # Check for triangle pattern
            if self._is_triangle_pattern(price_data[-8:]):
                continuation_score += 0.5
                features['pattern'] = 'triangle'
            
            if continuation_score > self.pattern_thresholds[GlyphType.CONTINUATION]:
                pattern = GlyphPattern(
                    pattern_id=f"continuation_{int(time.time())}",
                    glyph_type=GlyphType.CONTINUATION,
                    confidence=continuation_score,
                    strength=continuation_score,
                    timestamp=datetime.now(),
                    features=features
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting continuation patterns: {e}")
            return []
    
    def _detect_breakout_patterns(
        self,
        price_data: List[float],
        volume_data: List[float],
        technical_indicators: Dict[str, List[float]]
    ) -> List[GlyphPattern]:
        """Detect breakout glyph patterns."""
        patterns = []
        
        try:
            if len(price_data) < 10:
                return patterns
            
            # Look for breakout patterns
            breakout_score = 0.0
            features = {}
            
            # Check for resistance/support break
            if self._is_resistance_break(price_data[-10:]):
                breakout_score += 0.6
                features['breakout_type'] = 'resistance'
            
            if self._is_support_break(price_data[-10:]):
                breakout_score += 0.6
                features['breakout_type'] = 'support'
            
            # Check for volume confirmation
            if volume_data and volume_data[-1] > unified_math.unified_math.mean(volume_data[-5:]) * 1.5:
                breakout_score += 0.2
                features['volume_confirmation'] = True
            
            if breakout_score > self.pattern_thresholds[GlyphType.BREAKOUT]:
                pattern = GlyphPattern(
                    pattern_id=f"breakout_{int(time.time())}",
                    glyph_type=GlyphType.BREAKOUT,
                    confidence=breakout_score,
                    strength=breakout_score,
                    timestamp=datetime.now(),
                    features=features
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting breakout patterns: {e}")
            return []
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend of data series."""
        try:
            if len(data) < 2:
                return 0.0
            
            # Simple linear trend
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            
            # Normalize by first value
            if data[0] != 0:
                return slope / data[0]
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    def _is_double_top(self, price_data: List[float]) -> bool:
        """Check for double top pattern."""
        try:
            if len(price_data) < 5:
                return False
            
            # Find peaks
            peaks = []
            for i in range(1, len(price_data) - 1):
                if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                    peaks.append((i, price_data[i]))
            
            if len(peaks) >= 2:
                # Check if peaks are similar in height and separated
                peak1, peak2 = peaks[-2], peaks[-1]
                height_diff = unified_math.abs(peak1[1] - peak2[1]) / peak1[1]
                separation = peak2[0] - peak1[0]
                
                return height_diff < 0.05 and separation >= 2
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking double top: {e}")
            return False
    
    def _is_double_bottom(self, price_data: List[float]) -> bool:
        """Check for double bottom pattern."""
        try:
            if len(price_data) < 5:
                return False
            
            # Find troughs
            troughs = []
            for i in range(1, len(price_data) - 1):
                if price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                    troughs.append((i, price_data[i]))
            
            if len(troughs) >= 2:
                # Check if troughs are similar in depth and separated
                trough1, trough2 = troughs[-2], troughs[-1]
                depth_diff = unified_math.abs(trough1[1] - trough2[1]) / trough1[1]
                separation = trough2[0] - trough1[0]
                
                return depth_diff < 0.05 and separation >= 2
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking double bottom: {e}")
            return False
    
    def _has_divergence(self, price_data: List[float], indicator_data: List[float]) -> bool:
        """Check for price-indicator divergence."""
        try:
            if len(price_data) < 3 or len(indicator_data) < 3:
                return False
            
            price_trend = self._calculate_trend(price_data)
            indicator_trend = self._calculate_trend(indicator_data)
            
            # Check for opposite trends
            return (price_trend > 0 and indicator_trend < 0) or (price_trend < 0 and indicator_trend > 0)
            
        except Exception as e:
            logger.error(f"Error checking divergence: {e}")
            return False
    
    def _is_flag_pattern(self, price_data: List[float]) -> bool:
        """Check for flag pattern."""
        try:
            if len(price_data) < 6:
                return False
            
            # Simple flag detection (consolidation after strong move)
            first_half = price_data[:len(price_data)//2]
            second_half = price_data[len(price_data)//2:]
            
            first_trend = self._calculate_trend(first_half)
            second_trend = self._calculate_trend(second_half)
            
            # Strong move followed by consolidation
            return unified_math.abs(first_trend) > 0.03 and unified_math.abs(second_trend) < 0.01
            
        except Exception as e:
            logger.error(f"Error checking flag pattern: {e}")
            return False
    
    def _is_triangle_pattern(self, price_data: List[float]) -> bool:
        """Check for triangle pattern."""
        try:
            if len(price_data) < 6:
                return False
            
            # Simple triangle detection (converging trendlines)
            first_half = price_data[:len(price_data)//2]
            second_half = price_data[len(price_data)//2:]
            
            first_volatility = unified_math.unified_math.std(first_half)
            second_volatility = unified_math.unified_math.std(second_half)
            
            # Decreasing volatility indicates triangle
            return second_volatility < first_volatility * 0.8
            
        except Exception as e:
            logger.error(f"Error checking triangle pattern: {e}")
            return False
    
    def _is_resistance_break(self, price_data: List[float]) -> bool:
        """Check for resistance break."""
        try:
            if len(price_data) < 5:
                return False
            
            # Find resistance level
            resistance = unified_math.max(price_data[:-1])
            current_price = price_data[-1]
            
            return current_price > resistance * 1.01  # 1% break
            
        except Exception as e:
            logger.error(f"Error checking resistance break: {e}")
            return False
    
    def _is_support_break(self, price_data: List[float]) -> bool:
        """Check for support break."""
        try:
            if len(price_data) < 5:
                return False
            
            # Find support level
            support = unified_math.min(price_data[:-1])
            current_price = price_data[-1]
            
            return current_price < support * 0.99  # 1% break
            
        except Exception as e:
            logger.error(f"Error checking support break: {e}")
            return False
    
    def add_conditional_rule(
        self,
        condition: str,
        action: str,
        threshold: float,
        confidence: float = 0.8
    ) -> ConditionalRule:
        """
        Add a conditional rule to the system.
        
        Parameters:
        -----------
        condition : str
            Condition description
        action : str
            Action to take when condition is met
        threshold : float
            Threshold for condition
        confidence : float
            Confidence in the rule
            
        Returns:
        --------
        ConditionalRule
            Created conditional rule
        """
        try:
            rule = ConditionalRule(
                rule_id=f"rule_{int(time.time())}",
                condition=condition,
                action=action,
                threshold=threshold,
                confidence=confidence
            )
            
            self.rules.append(rule)
            return rule
            
        except Exception as e:
            logger.error(f"Error adding conditional rule: {e}")
            raise
    
    def process_conditional_logic(self, patterns: List[GlyphPattern]) -> List[str]:
        """
        Process conditional logic based on detected patterns.
        
        Parameters:
        -----------
        patterns : List[GlyphPattern]
            Detected glyph patterns
            
        Returns:
        --------
        List[str]
            Actions to take
        """
        try:
            actions = []
            
            for rule in self.rules:
                if not rule.active:
                    continue
                
                # Check if rule condition is met
                if self._evaluate_condition(rule, patterns):
                    actions.append(rule.action)
                    
                    # Create feedback loop
                    if patterns:
                        feedback = FeedbackLoop(
                            loop_id=f"feedback_{int(time.time())}",
                            input_pattern=patterns[0],
                            output_signal=rule.action,
                            feedback_type=FeedbackType.POSITIVE,
                            strength=rule.confidence,
                            timestamp=datetime.now()
                        )
                        self.feedback_loops.append(feedback)
            
            return actions
            
        except Exception as e:
            logger.error(f"Error processing conditional logic: {e}")
            return []
    
    def _evaluate_condition(self, rule: ConditionalRule, patterns: List[GlyphPattern]) -> bool:
        """Evaluate if a condition is met."""
        try:
            # Simple condition evaluation
            # In a real system, this would be more sophisticated
            
            if not patterns:
                return False
            
            # Check if any pattern meets the threshold
            for pattern in patterns:
                if pattern.confidence >= rule.threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def analyze_glyph_patterns(self) -> GlyphAnalysis:
        """
        Perform comprehensive glyph pattern analysis.
        
        Returns:
        --------
        GlyphAnalysis
            Complete glyph analysis result
        """
        try:
            # Get recent patterns
            recent_patterns = self.patterns[-10:] if self.patterns else []
            
            # Get active rules
            active_rules = [rule for rule in self.rules if rule.active]
            
            # Get recent feedback loops
            recent_feedback = self.feedback_loops[-10:] if self.feedback_loops else []
            
            # Calculate overall confidence
            confidence_score = 0.0
            if recent_patterns:
                confidence_score = unified_math.mean([p.confidence for p in recent_patterns])
            
            # Generate recommendation
            recommendation = self._generate_recommendation(recent_patterns, confidence_score)
            
            return GlyphAnalysis(
                detected_patterns=recent_patterns,
                active_rules=active_rules,
                feedback_loops=recent_feedback,
                confidence_score=confidence_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error in glyph pattern analysis: {e}")
            return GlyphAnalysis(
                detected_patterns=[],
                active_rules=[],
                feedback_loops=[],
                confidence_score=0.0,
                recommendation="Error in analysis"
            )
    
    def _generate_recommendation(
        self,
        patterns: List[GlyphPattern],
        confidence_score: float
    ) -> str:
        """Generate recommendation based on patterns."""
        try:
            if not patterns:
                return "No patterns detected"
            
            # Count pattern types
            pattern_counts = {}
            for pattern in patterns:
                pattern_type = pattern.glyph_type.value
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Generate recommendation
            if pattern_counts.get('bullish', 0) > pattern_counts.get('bearish', 0):
                return "Bullish patterns detected - consider long positions"
            elif pattern_counts.get('bearish', 0) > pattern_counts.get('bullish', 0):
                return "Bearish patterns detected - consider short positions"
            elif confidence_score > 0.7:
                return "Strong patterns detected - monitor for confirmation"
            else:
                return "Weak patterns detected - wait for stronger signals"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Error generating recommendation"
    
    def get_glyph_statistics(self) -> Dict[str, Any]:
        """Get glyph pattern statistics."""
        try:
            total_patterns = len(self.patterns)
            total_rules = len(self.rules)
            total_feedback = len(self.feedback_loops)
            
            # Pattern type distribution
            pattern_types = {}
            for pattern in self.patterns:
                pattern_type = pattern.glyph_type.value
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            # Rule effectiveness
            active_rules = sum(1 for rule in self.rules if rule.active)
            
            # Average confidence
            avg_confidence = unified_math.mean([p.confidence for p in self.patterns]) if self.patterns else 0.0
            
            return {
                "total_patterns": total_patterns,
                "total_rules": total_rules,
                "total_feedback": total_feedback,
                "pattern_type_distribution": pattern_types,
                "active_rules": active_rules,
                "average_confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting glyph statistics: {e}")
            return {"error": str(e)}


def main() -> None:
    """Test function for ConditionalGlyphFeedbackLoop."""
    safe_print("🔮 Testing Conditional Glyph Feedback Loop...")
    
    system = ConditionalGlyphFeedbackLoop()
    
    # Simulate market data
    price_data = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    volume_data = [1000000, 1100000, 950000, 1200000, 1300000, 1150000, 1400000, 1500000, 1350000, 1600000]
    technical_indicators = {
        'rsi': [45, 50, 48, 55, 60, 58, 65, 70, 68, 75],
        'macd': [0.1, 0.2, 0.15, 0.3, 0.4, 0.35, 0.5, 0.6, 0.55, 0.7]
    }
    
    # Detect patterns
    patterns = system.detect_glyph_patterns(price_data, volume_data, technical_indicators)
    safe_print(f"✅ Detected {len(patterns)} glyph patterns")
    
    # Add conditional rules
    rule1 = system.add_conditional_rule(
        condition="bullish_pattern_detected",
        action="open_long_position",
        threshold=0.6
    )
    
    rule2 = system.add_conditional_rule(
        condition="bearish_pattern_detected",
        action="open_short_position",
        threshold=0.6
    )
    
    # Process conditional logic
    actions = system.process_conditional_logic(patterns)
    safe_print(f"✅ Generated {len(actions)} actions: {actions}")
    
    # Perform analysis
    analysis = system.analyze_glyph_patterns()
    safe_print(f"📊 Analysis results:")
    safe_print(f"   Confidence score: {analysis.confidence_score:.3f}")
    safe_print(f"   Recommendation: {analysis.recommendation}")
    safe_print(f"   Active rules: {len(analysis.active_rules)}")
    safe_print(f"   Feedback loops: {len(analysis.feedback_loops)}")
    
    # Get statistics
    stats = system.get_glyph_statistics()
    safe_print(f"📈 Glyph statistics: {stats}")
    
    return 0

if __name__ == "__main__":
    exit(main())
