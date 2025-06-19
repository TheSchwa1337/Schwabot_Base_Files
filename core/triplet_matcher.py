#!/usr/bin/env python3
"""
Triplet Matcher - Schwabot Mathematical Framework.

Advanced pattern matching system for identifying mathematical triplets and
patterns within trading data, thermal signatures, and profit vectors.

Key Features:
- Vector triplet pattern matching
- Thermal signature pattern detection  
- Profit correlation analysis
- Mathematical sequence identification
- Windows CLI compatibility
- Integration with unified mathematical systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


@dataclass
class TripletPattern:
    """Container for a mathematical triplet pattern."""
    
    pattern_id: str
    values: Tuple[Decimal, Decimal, Decimal]
    pattern_type: str
    confidence: float
    thermal_signature: Optional[Decimal] = None
    correlation_strength: Optional[float] = None


@dataclass  
class MatchResult:
    """Container for pattern matching results."""
    
    found_match: bool
    pattern: Optional[TripletPattern]
    similarity_score: float
    match_indices: Optional[Tuple[int, int, int]]
    deviation_metrics: Dict[str, float]


class MathematicalSequenceDetector:
    """Detects mathematical sequences and patterns."""
    
    def __init__(self) -> None:
        """Initialize sequence detector."""
        self.epsilon = Decimal("1e-10")
    
    def detect_arithmetic_progression(self, values: List[Decimal]) -> Dict[str, Any]:
        """Detect arithmetic progression in sequence."""
        if len(values) < 3:
            return {'is_ap': False, 'common_difference': None}
        
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        # Check if all differences are approximately equal
        first_diff = differences[0]
        is_ap = all(abs(d - first_diff) < self.epsilon for d in differences)
        
        return {
            'is_ap': is_ap,
            'common_difference': float(first_diff) if is_ap else None,
            'deviation': float(max(abs(d - first_diff) for d in differences)) if differences else 0.0
        }
    
    def detect_geometric_progression(self, values: List[Decimal]) -> Dict[str, Any]:
        """Detect geometric progression in sequence."""
        if len(values) < 3 or any(v == 0 for v in values[:-1]):
            return {'is_gp': False, 'common_ratio': None}
        
        ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
        
        # Check if all ratios are approximately equal
        first_ratio = ratios[0]
        is_gp = all(abs(r - first_ratio) < self.epsilon for r in ratios)
        
        return {
            'is_gp': is_gp,
            'common_ratio': float(first_ratio) if is_gp else None,
            'deviation': float(max(abs(r - first_ratio) for r in ratios)) if ratios else 0.0
        }
    
    def detect_fibonacci_like(self, values: List[Decimal]) -> Dict[str, Any]:
        """Detect Fibonacci-like sequences."""
        if len(values) < 3:
            return {'is_fibonacci_like': False, 'ratio_to_golden': None}
        
        # Check if each term is sum of previous two
        fibonacci_like = True
        deviations = []
        
        for i in range(2, len(values)):
            expected = values[i-2] + values[i-1]
            deviation = abs(values[i] - expected)
            deviations.append(float(deviation))
            
            if deviation > self.epsilon * abs(expected):
                fibonacci_like = False
        
        # Calculate ratio approximation to golden ratio
        golden_ratio = Decimal("1.618033988749895")
        if len(values) >= 4:
            recent_ratio = values[-1] / values[-2] if values[-2] != 0 else Decimal("0")
            ratio_to_golden = float(abs(recent_ratio - golden_ratio))
        else:
            ratio_to_golden = None
        
        return {
            'is_fibonacci_like': fibonacci_like,
            'ratio_to_golden': ratio_to_golden,
            'max_deviation': max(deviations) if deviations else 0.0,
            'avg_deviation': sum(deviations) / len(deviations) if deviations else 0.0
        }


class ThermalSignatureAnalyzer:
    """Analyzes thermal signatures for pattern matching."""
    
    def __init__(self) -> None:
        """Initialize thermal signature analyzer."""
        self.signature_cache = {}
        self.pattern_threshold = 0.8
    
    def compute_thermal_signature(self, thermal_values: List[float]) -> Decimal:
        """Compute thermal signature from values."""
        if not thermal_values:
            return Decimal("0.0")
        
        # Weighted sum with exponential decay
        weights = [Decimal(str(np.exp(-0.1 * i))) for i in range(len(thermal_values))]
        weighted_sum = sum(Decimal(str(v)) * w for v, w in zip(thermal_values, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else Decimal("0.0")
    
    def compare_thermal_signatures(self, sig1: Decimal, sig2: Decimal) -> float:
        """Compare two thermal signatures and return similarity score."""
        if sig1 == 0 and sig2 == 0:
            return 1.0
        
        max_sig = max(abs(sig1), abs(sig2))
        if max_sig == 0:
            return 1.0
        
        difference = abs(sig1 - sig2)
        similarity = 1.0 - float(difference / max_sig)
        
        return max(0.0, similarity)
    
    def analyze_thermal_triplet(self, t1: float, t2: float, t3: float) -> Dict[str, Any]:
        """Analyze thermal triplet for patterns."""
        values = [Decimal(str(v)) for v in [t1, t2, t3]]
        
        # Calculate thermal momentum
        momentum1 = values[1] - values[0]
        momentum2 = values[2] - values[1]
        momentum_change = momentum2 - momentum1
        
        # Calculate thermal stability
        mean_thermal = sum(values) / Decimal("3")
        variance = sum((v - mean_thermal) ** 2 for v in values) / Decimal("3")
        stability = float(Decimal("1") / (Decimal("1") + variance))
        
        # Pattern classification
        if abs(momentum_change) < Decimal("0.01"):
            pattern_type = "linear"
        elif momentum_change > Decimal("0.05"):
            pattern_type = "accelerating"
        elif momentum_change < Decimal("-0.05"):
            pattern_type = "decelerating"
        else:
            pattern_type = "irregular"
        
        return {
            'thermal_signature': self.compute_thermal_signature([t1, t2, t3]),
            'momentum_change': float(momentum_change),
            'stability_score': stability,
            'pattern_type': pattern_type,
            'mean_thermal': float(mean_thermal)
        }


class VectorTripletMatcher:
    """Matches vector triplets for trading patterns."""
    
    def __init__(self) -> None:
        """Initialize vector triplet matcher."""
        self.known_patterns = []
        self.match_threshold = 0.75
        self.sequence_detector = MathematicalSequenceDetector()
        self.thermal_analyzer = ThermalSignatureAnalyzer()
    
    def register_pattern(self, pattern: TripletPattern) -> None:
        """Register a known pattern for matching."""
        self.known_patterns.append(pattern)
        logger.info(f"Registered pattern {pattern.pattern_id} of type {pattern.pattern_type}")
    
    def create_triplet_pattern(self, values: Tuple[float, float, float], 
                             pattern_type: str, confidence: float = 1.0) -> TripletPattern:
        """Create a new triplet pattern."""
        decimal_values = tuple(Decimal(str(v)) for v in values)
        
        # Generate pattern ID
        import hashlib
        pattern_data = f"{values}{pattern_type}{confidence}"
        pattern_id = hashlib.md5(pattern_data.encode()).hexdigest()[:8]
        
        # Compute thermal signature
        thermal_sig = self.thermal_analyzer.compute_thermal_signature(list(values))
        
        return TripletPattern(
            pattern_id=pattern_id,
            values=decimal_values,
            pattern_type=pattern_type,
            confidence=confidence,
            thermal_signature=thermal_sig
        )
    
    def match_vector_triplet(self, test_values: Tuple[float, float, float]) -> MatchResult:
        """Match a vector triplet against known patterns."""
        test_decimals = [Decimal(str(v)) for v in test_values]
        best_match = None
        best_score = 0.0
        best_indices = None
        
        # Check against all known patterns
        for pattern in self.known_patterns:
            score = self._calculate_similarity(test_decimals, list(pattern.values))
            
            if score > best_score and score >= self.match_threshold:
                best_score = score
                best_match = pattern
                best_indices = (0, 1, 2)  # Direct triplet match
        
        # Calculate deviation metrics
        deviation_metrics = {}
        if best_match:
            deviations = [abs(test_decimals[i] - best_match.values[i]) 
                         for i in range(3)]
            deviation_metrics = {
                'max_deviation': float(max(deviations)),
                'avg_deviation': float(sum(deviations) / 3),
                'relative_deviation': float(max(deviations) / max(abs(v) for v in best_match.values))
            }
        
        return MatchResult(
            found_match=best_match is not None,
            pattern=best_match,
            similarity_score=best_score,
            match_indices=best_indices,
            deviation_metrics=deviation_metrics
        )
    
    def _calculate_similarity(self, values1: List[Decimal], values2: List[Decimal]) -> float:
        """Calculate similarity between two triplets."""
        if len(values1) != len(values2):
            return 0.0
        
        # Normalize values for comparison
        max_val1 = max(abs(v) for v in values1)
        max_val2 = max(abs(v) for v in values2)
        
        if max_val1 == 0 or max_val2 == 0:
            return 1.0 if max_val1 == max_val2 else 0.0
        
        norm_vals1 = [v / max_val1 for v in values1]
        norm_vals2 = [v / max_val2 for v in values2]
        
        # Calculate Euclidean distance
        distance = sum((v1 - v2) ** 2 for v1, v2 in zip(norm_vals1, norm_vals2))
        distance = float(distance ** Decimal("0.5"))
        
        # Convert to similarity score
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def analyze_triplet_patterns(self, values: List[float]) -> Dict[str, Any]:
        """Analyze all possible triplet patterns in a sequence."""
        if len(values) < 3:
            return {'error': 'Insufficient values for triplet analysis'}
        
        triplet_analyses = []
        
        # Analyze all consecutive triplets
        for i in range(len(values) - 2):
            triplet = values[i:i+3]
            
            # Mathematical sequence analysis
            decimal_triplet = [Decimal(str(v)) for v in triplet]
            ap_analysis = self.sequence_detector.detect_arithmetic_progression(decimal_triplet)
            gp_analysis = self.sequence_detector.detect_geometric_progression(decimal_triplet)
            fib_analysis = self.sequence_detector.detect_fibonacci_like(decimal_triplet)
            
            # Thermal analysis
            thermal_analysis = self.thermal_analyzer.analyze_thermal_triplet(*triplet)
            
            # Pattern matching
            match_result = self.match_vector_triplet(tuple(triplet))
            
            triplet_analyses.append({
                'triplet_index': i,
                'values': triplet,
                'arithmetic_progression': ap_analysis,
                'geometric_progression': gp_analysis,
                'fibonacci_like': fib_analysis,
                'thermal_analysis': thermal_analysis,
                'pattern_match': {
                    'found_match': match_result.found_match,
                    'similarity_score': match_result.similarity_score,
                    'pattern_type': match_result.pattern.pattern_type if match_result.pattern else None
                }
            })
        
        return {
            'total_triplets': len(triplet_analyses),
            'triplet_analyses': triplet_analyses,
            'summary': self._summarize_triplet_analysis(triplet_analyses)
        }
    
    def _summarize_triplet_analysis(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize triplet analysis results."""
        if not analyses:
            return {}
        
        # Count pattern types
        ap_count = sum(1 for a in analyses if a['arithmetic_progression']['is_ap'])
        gp_count = sum(1 for a in analyses if a['geometric_progression']['is_gp'])
        fib_count = sum(1 for a in analyses if a['fibonacci_like']['is_fibonacci_like'])
        match_count = sum(1 for a in analyses if a['pattern_match']['found_match'])
        
        # Average metrics
        avg_similarity = sum(a['pattern_match']['similarity_score'] for a in analyses) / len(analyses)
        avg_stability = sum(a['thermal_analysis']['stability_score'] for a in analyses) / len(analyses)
        
        return {
            'arithmetic_progressions': ap_count,
            'geometric_progressions': gp_count,
            'fibonacci_like_sequences': fib_count,
            'pattern_matches': match_count,
            'average_similarity_score': avg_similarity,
            'average_thermal_stability': avg_stability,
            'dominant_pattern_type': self._find_dominant_pattern_type(analyses)
        }
    
    def _find_dominant_pattern_type(self, analyses: List[Dict[str, Any]]) -> str:
        """Find the most common pattern type in analyses."""
        pattern_counts = {}
        
        for analysis in analyses:
            thermal_type = analysis['thermal_analysis']['pattern_type']
            pattern_counts[thermal_type] = pattern_counts.get(thermal_type, 0) + 1
        
        return max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "unknown"


class TripletMatcher:
    """Main triplet matching system."""
    
    def __init__(self) -> None:
        """Initialize triplet matcher."""
        self.version = "1.0.0"
        self.vector_matcher = VectorTripletMatcher()
        self.sequence_detector = MathematicalSequenceDetector()
        self.thermal_analyzer = ThermalSignatureAnalyzer()
        
        # Register some default patterns
        self._register_default_patterns()
        
        logger.info(f"TripletMatcher v{self.version} initialized")
    
    def _register_default_patterns(self) -> None:
        """Register default mathematical patterns."""
        # Common trading patterns
        default_patterns = [
            ((1.0, 1.1, 1.21), "geometric_growth", 0.9),  # 10% growth pattern
            ((100.0, 105.0, 110.0), "linear_growth", 0.85),  # Linear increase
            ((1.0, 0.9, 0.81), "geometric_decay", 0.9),  # 10% decay pattern
            ((50.0, 55.0, 50.0), "oscillation", 0.8),  # Simple oscillation
            ((1.0, 1.0, 1.0), "stable", 0.95),  # Stability pattern
        ]
        
        for values, pattern_type, confidence in default_patterns:
            pattern = self.vector_matcher.create_triplet_pattern(values, pattern_type, confidence)
            self.vector_matcher.register_pattern(pattern)
    
    def match_triplet(self, values: Tuple[float, float, float]) -> Dict[str, Any]:
        """Main triplet matching interface."""
        try:
            # Vector pattern matching
            match_result = self.vector_matcher.match_vector_triplet(values)
            
            # Mathematical sequence analysis
            decimal_values = [Decimal(str(v)) for v in values]
            ap_analysis = self.sequence_detector.detect_arithmetic_progression(decimal_values)
            gp_analysis = self.sequence_detector.detect_geometric_progression(decimal_values)
            fib_analysis = self.sequence_detector.detect_fibonacci_like(decimal_values)
            
            # Thermal analysis
            thermal_analysis = self.thermal_analyzer.analyze_thermal_triplet(*values)
            
            return {
                'status': 'success',
                'triplet_values': values,
                'pattern_match': {
                    'found_match': match_result.found_match,
                    'pattern_id': match_result.pattern.pattern_id if match_result.pattern else None,
                    'pattern_type': match_result.pattern.pattern_type if match_result.pattern else None,
                    'similarity_score': match_result.similarity_score,
                    'deviation_metrics': match_result.deviation_metrics
                },
                'mathematical_analysis': {
                    'arithmetic_progression': ap_analysis,
                    'geometric_progression': gp_analysis,
                    'fibonacci_like': fib_analysis
                },
                'thermal_analysis': thermal_analysis,
                'version': self.version
            }
            
        except Exception as e:
            logger.error(f"Error in triplet matching: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'triplet_values': values
            }
    
    def analyze_sequence(self, values: List[float]) -> Dict[str, Any]:
        """Analyze a full sequence for triplet patterns."""
        return self.vector_matcher.analyze_triplet_patterns(values)
    
    def get_registered_patterns(self) -> List[Dict[str, Any]]:
        """Get list of all registered patterns."""
        return [
            {
                'pattern_id': p.pattern_id,
                'values': [float(v) for v in p.values],
                'pattern_type': p.pattern_type,
                'confidence': p.confidence,
                'thermal_signature': float(p.thermal_signature) if p.thermal_signature else None
            }
            for p in self.vector_matcher.known_patterns
        ]


def main() -> None:
    """Demo of triplet matcher system."""
    try:
        matcher = TripletMatcher()
        print(f"✅ TripletMatcher v{matcher.version} initialized")
        
        # Test triplets
        test_triplets = [
            (100.0, 110.0, 121.0),  # Should match geometric growth
            (10.0, 15.0, 20.0),     # Should match linear growth
            (50.0, 50.0, 50.0),     # Should match stable pattern
            (1.0, 1.0, 2.0),        # Fibonacci-like start
            (42.5, 39.8, 37.1),     # Custom pattern
        ]
        
        print(f"🔍 Testing {len(test_triplets)} triplet patterns:")
        
        for i, triplet in enumerate(test_triplets):
            result = matcher.match_triplet(triplet)
            
            if result['status'] == 'success':
                match_info = result['pattern_match']
                thermal_info = result['thermal_analysis']
                
                print(f"   Triplet {i+1}: {triplet}")
                print(f"      Match: {'✅' if match_info['found_match'] else '❌'}")
                if match_info['found_match']:
                    print(f"      Pattern: {match_info['pattern_type']}")
                    print(f"      Similarity: {match_info['similarity_score']:.3f}")
                print(f"      Thermal: {thermal_info['pattern_type']} "
                      f"(stability: {thermal_info['stability_score']:.3f})")
        
        # Test sequence analysis
        print(f"\n📊 Sequence Analysis:")
        test_sequence = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]
        sequence_result = matcher.analyze_sequence(test_sequence)
        
        if 'summary' in sequence_result:
            summary = sequence_result['summary']
            print(f"   Total triplets analyzed: {sequence_result['total_triplets']}")
            print(f"   Pattern matches: {summary['pattern_matches']}")
            print(f"   Dominant pattern: {summary['dominant_pattern_type']}")
            print(f"   Avg similarity: {summary['average_similarity_score']:.3f}")
        
        print("🎉 Triplet matcher demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main() 