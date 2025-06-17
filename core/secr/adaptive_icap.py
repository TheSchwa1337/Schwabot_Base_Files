"""
SECR Adaptive ICAP Tuner
========================

Dynamically adjusts ICAP thresholds based on SECR feedback and SchwaFit
learning to optimize profit corridor navigation and entry probability.
"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .failure_logger import FailureKey, FailureGroup, FailureSubGroup
from .watchdog import OutcomeMetrics

logger = logging.getLogger(__name__)

@dataclass
class ICAPAdjustment:
    """Record of ICAP threshold adjustment"""
    timestamp: float
    old_threshold: float
    new_threshold: float
    reason: str
    failure_context: Optional[str] = None
    expected_improvement: float = 0.0
    actual_improvement: Optional[float] = None
    
    def effectiveness(self) -> Optional[float]:
        """Calculate adjustment effectiveness"""
        if self.actual_improvement is None:
            return None
        return self.actual_improvement / max(abs(self.expected_improvement), 0.01)

class AdaptiveICAPTuner:
    """
    Adaptive ICAP threshold tuning system
    
    Adjusts ICAP thresholds based on:
    - Failure patterns (ENTROPY/ICAP_COLLAPSE events)
    - Profit performance feedback  
    - Market condition changes
    - Resource utilization patterns
    """
    
    def __init__(self, 
                 initial_threshold: float = 0.4,
                 adjustment_alpha: float = 0.05,
                 bounds: Tuple[float, float] = (0.1, 0.85),
                 learning_window: int = 100):
        
        self.current_threshold = initial_threshold
        self.adjustment_alpha = adjustment_alpha  # Learning rate
        self.min_threshold, self.max_threshold = bounds
        self.learning_window = learning_window
        
        # Adjustment history
        self.adjustments: deque[ICAPAdjustment] = deque(maxlen=learning_window)
        self.performance_history: deque[Tuple[float, float, float]] = deque(maxlen=learning_window)  # (timestamp, threshold, profit)
        
        # Pattern tracking
        self.failure_patterns: Dict[str, List[float]] = {
            'icap_collapse_count': [],
            'entropy_spike_severity': [],
            'profit_correlation': []
        }
        
        # Tuning parameters
        self.volatility_sensitivity = 0.3
        self.profit_weight = 0.6
        self.stability_weight = 0.4
        
        # State tracking
        self.last_market_volatility = 0.0
        self.consecutive_adjustments = 0
        self.adjustment_cooldown = 0
        
        logger.info(f"Initialized ICAP tuner with threshold={initial_threshold:.3f}")
    
    def process_failure(self, failure_key: FailureKey) -> Optional[float]:
        """
        Process a failure event and potentially adjust ICAP threshold
        
        Args:
            failure_key: The failure event to process
            
        Returns:
            New ICAP threshold if adjusted, None otherwise
        """
        if failure_key.group != FailureGroup.ENTROPY:
            return None
        
        # Reduce cooldown
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
        
        if failure_key.subgroup == FailureSubGroup.ICAP_COLLAPSE:
            return self._handle_icap_collapse(failure_key)
        elif failure_key.subgroup == FailureSubGroup.ENTROPY_SPIKE:
            return self._handle_entropy_spike(failure_key)
        
        return None
    
    def _handle_icap_collapse(self, failure_key: FailureKey) -> Optional[float]:
        """Handle ICAP collapse events"""
        icap_value = failure_key.ctx.get('icap_value', 0.0)
        severity = failure_key.severity
        
        # Calculate suggested adjustment
        # Higher severity or lower ICAP value suggests threshold too low
        adjustment_magnitude = self.adjustment_alpha * (severity + (0.5 - icap_value))
        
        # Increase threshold to make entries more selective
        suggested_increase = adjustment_magnitude
        
        new_threshold = self._apply_adjustment(
            suggested_increase, 
            f"ICAP collapse (value={icap_value:.3f}, severity={severity:.3f})",
            failure_key.hash
        )
        
        # Track pattern
        self.failure_patterns['icap_collapse_count'].append(time.time())
        if len(self.failure_patterns['icap_collapse_count']) > 20:
            self.failure_patterns['icap_collapse_count'] = self.failure_patterns['icap_collapse_count'][-20:]
        
        return new_threshold
    
    def _handle_entropy_spike(self, failure_key: FailureKey) -> Optional[float]:
        """Handle entropy spike events"""
        delta_psi = failure_key.ctx.get('delta_psi', 0.0)
        severity = failure_key.severity
        
        # Entropy spikes suggest market volatility
        # May need to widen corridors (lower threshold) for more opportunities
        # Or narrow corridors (higher threshold) for stability
        
        # Decision based on recent performance
        recent_profit_trend = self._get_recent_profit_trend()
        
        if recent_profit_trend < 0:
            # Losing money - be more selective (higher threshold)
            adjustment = self.adjustment_alpha * severity * 0.5
        else:
            # Making money - potentially lower threshold for more opportunities
            adjustment = -self.adjustment_alpha * severity * 0.3
        
        new_threshold = self._apply_adjustment(
            adjustment,
            f"Entropy spike (delta_psi={delta_psi:.3f}, profit_trend={recent_profit_trend:.3f})",
            failure_key.hash
        )
        
        # Track pattern
        self.failure_patterns['entropy_spike_severity'].append(severity)
        if len(self.failure_patterns['entropy_spike_severity']) > 50:
            self.failure_patterns['entropy_spike_severity'] = self.failure_patterns['entropy_spike_severity'][-50:]
        
        return new_threshold
    
    def _apply_adjustment(self, 
                         adjustment: float, 
                         reason: str, 
                         failure_context: Optional[str] = None) -> Optional[float]:
        """Apply threshold adjustment with bounds checking"""
        
        # Check cooldown
        if self.adjustment_cooldown > 0:
            logger.debug(f"ICAP adjustment on cooldown ({self.adjustment_cooldown} ticks remaining)")
            return None
        
        # Calculate new threshold
        old_threshold = self.current_threshold
        new_threshold = np.clip(
            old_threshold + adjustment,
            self.min_threshold,
            self.max_threshold
        )
        
        # Check if adjustment is significant enough
        if abs(new_threshold - old_threshold) < 0.005:
            logger.debug(f"ICAP adjustment too small ({adjustment:.4f}), skipping")
            return None
        
        # Apply adjustment
        self.current_threshold = new_threshold
        
        # Record adjustment
        adj_record = ICAPAdjustment(
            timestamp=time.time(),
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            reason=reason,
            failure_context=failure_context,
            expected_improvement=adjustment * 10  # Rough expected profit improvement %
        )
        self.adjustments.append(adj_record)
        
        # Set cooldown to prevent rapid oscillation
        self.adjustment_cooldown = max(8, int(abs(adjustment) * 100))
        
        # Track consecutive adjustments
        if len(self.adjustments) >= 2:
            if self.adjustments[-1].timestamp - self.adjustments[-2].timestamp < 60:
                self.consecutive_adjustments += 1
            else:
                self.consecutive_adjustments = 0
        
        logger.info(f"ICAP threshold adjusted: {old_threshold:.3f} -> {new_threshold:.3f} ({reason})")
        
        return new_threshold
    
    def update_performance(self, profit_delta: float, stability_score: float = 1.0) -> None:
        """Update performance tracking with recent results"""
        timestamp = time.time()
        
        # Record performance
        self.performance_history.append((timestamp, self.current_threshold, profit_delta))
        
        # Update recent adjustment effectiveness
        if self.adjustments:
            latest_adj = self.adjustments[-1]
            if latest_adj.actual_improvement is None:
                # This is the first performance update since adjustment
                latest_adj.actual_improvement = profit_delta
        
        # Update profit correlation tracking
        if len(self.performance_history) >= 10:
            self._update_profit_correlation()
    
    def _update_profit_correlation(self) -> None:
        """Update correlation between threshold and profit"""
        if len(self.performance_history) < 10:
            return
        
        # Get recent data
        recent_data = list(self.performance_history)[-20:]
        thresholds = [data[1] for data in recent_data]
        profits = [data[2] for data in recent_data]
        
        # Calculate correlation
        if len(set(thresholds)) > 1:  # Need variation in thresholds
            correlation = np.corrcoef(thresholds, profits)[0, 1]
            if not np.isnan(correlation):
                self.failure_patterns['profit_correlation'].append(correlation)
                if len(self.failure_patterns['profit_correlation']) > 20:
                    self.failure_patterns['profit_correlation'] = self.failure_patterns['profit_correlation'][-20:]
    
    def _get_recent_profit_trend(self) -> float:
        """Get recent profit trend"""
        if len(self.performance_history) < 5:
            return 0.0
        
        recent_profits = [data[2] for data in list(self.performance_history)[-10:]]
        if len(recent_profits) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_profits))
        slope, _ = np.polyfit(x, recent_profits, 1)
        return float(slope)
    
    def suggest_market_based_adjustment(self, market_metrics: Dict[str, float]) -> Optional[float]:
        """Suggest threshold adjustment based on market conditions"""
        
        volatility = market_metrics.get('volatility', 0.0)
        volume = market_metrics.get('volume', 1.0)
        price_momentum = market_metrics.get('price_momentum', 0.0)
        
        # Track volatility changes
        volatility_change = volatility - self.last_market_volatility
        self.last_market_volatility = volatility
        
        # High volatility might require more selective entries (higher threshold)
        # Low volatility might allow more opportunities (lower threshold)
        
        volatility_adjustment = 0.0
        reason_parts = []
        
        # Volatility-based adjustment
        if volatility > 0.15:  # High volatility
            if self.current_threshold < 0.6:
                volatility_adjustment += self.adjustment_alpha * self.volatility_sensitivity
                reason_parts.append(f"high volatility ({volatility:.3f})")
        elif volatility < 0.05:  # Low volatility
            if self.current_threshold > 0.3:
                volatility_adjustment -= self.adjustment_alpha * self.volatility_sensitivity * 0.5
                reason_parts.append(f"low volatility ({volatility:.3f})")
        
        # Volume-based adjustment
        if volume < 0.5:  # Low volume
            if self.current_threshold > 0.25:
                volatility_adjustment -= self.adjustment_alpha * 0.2
                reason_parts.append(f"low volume ({volume:.3f})")
        
        # Apply adjustment if significant
        if abs(volatility_adjustment) > 0.01 and reason_parts:
            reason = f"Market conditions: {', '.join(reason_parts)}"
            return self._apply_adjustment(volatility_adjustment, reason)
        
        return None
    
    def analyze_adjustment_effectiveness(self) -> Dict[str, float]:
        """Analyze the effectiveness of recent adjustments"""
        if not self.adjustments:
            return {}
        
        # Get adjustments with measured outcomes
        effective_adjustments = [adj for adj in self.adjustments if adj.actual_improvement is not None]
        
        if not effective_adjustments:
            return {'insufficient_data': True}
        
        # Calculate metrics
        effectiveness_scores = [adj.effectiveness() for adj in effective_adjustments if adj.effectiveness() is not None]
        
        if not effectiveness_scores:
            return {'no_measurable_outcomes': True}
        
        return {
            'avg_effectiveness': np.mean(effectiveness_scores),
            'effectiveness_std': np.std(effectiveness_scores),
            'positive_adjustments': sum(1 for score in effectiveness_scores if score > 0),
            'total_adjustments': len(effectiveness_scores),
            'success_rate': sum(1 for score in effectiveness_scores if score > 0) / len(effectiveness_scores),
            'best_effectiveness': max(effectiveness_scores),
            'worst_effectiveness': min(effectiveness_scores)
        }
    
    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for optimizing ICAP tuning parameters"""
        analysis = self.analyze_adjustment_effectiveness()
        
        if 'success_rate' not in analysis:
            return {'status': 'insufficient_data'}
        
        suggestions = {}
        
        # Adjust learning rate based on success rate
        success_rate = analysis['success_rate']
        if success_rate < 0.3:
            suggestions['reduce_alpha'] = {
                'current': self.adjustment_alpha,
                'suggested': self.adjustment_alpha * 0.8,
                'reason': 'Low success rate suggests too aggressive adjustments'
            }
        elif success_rate > 0.8:
            suggestions['increase_alpha'] = {
                'current': self.adjustment_alpha,
                'suggested': min(0.1, self.adjustment_alpha * 1.2),
                'reason': 'High success rate suggests adjustments could be more aggressive'
            }
        
        # Analyze patterns
        if self.failure_patterns['icap_collapse_count']:
            recent_collapses = len([t for t in self.failure_patterns['icap_collapse_count'] 
                                 if time.time() - t < 3600])  # Last hour
            if recent_collapses > 5:
                suggestions['increase_base_threshold'] = {
                    'current': self.current_threshold,
                    'suggested': min(0.7, self.current_threshold + 0.1),
                    'reason': f'Frequent ICAP collapses ({recent_collapses} in last hour)'
                }
        
        # Correlation analysis
        if self.failure_patterns['profit_correlation']:
            avg_correlation = np.mean(self.failure_patterns['profit_correlation'])
            if avg_correlation < -0.3:
                suggestions['review_strategy'] = {
                    'correlation': avg_correlation,
                    'reason': 'Strong negative correlation between threshold and profit'
                }
        
        return suggestions
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current ICAP tuner status"""
        return {
            'current_threshold': self.current_threshold,
            'adjustment_alpha': self.adjustment_alpha,
            'bounds': [self.min_threshold, self.max_threshold],
            'adjustment_cooldown': self.adjustment_cooldown,
            'consecutive_adjustments': self.consecutive_adjustments,
            'total_adjustments': len(self.adjustments),
            'recent_adjustments': [
                {
                    'timestamp': adj.timestamp,
                    'old_threshold': adj.old_threshold,
                    'new_threshold': adj.new_threshold,
                    'reason': adj.reason,
                    'effectiveness': adj.effectiveness()
                }
                for adj in list(self.adjustments)[-5:]  # Last 5 adjustments
            ],
            'performance_samples': len(self.performance_history),
            'pattern_tracking': {
                'icap_collapses_tracked': len(self.failure_patterns['icap_collapse_count']),
                'entropy_spikes_tracked': len(self.failure_patterns['entropy_spike_severity']),
                'profit_correlations': len(self.failure_patterns['profit_correlation'])
            }
        }
    
    def manual_override(self, new_threshold: float, reason: str = "Manual override") -> bool:
        """Manually set ICAP threshold"""
        if not (self.min_threshold <= new_threshold <= self.max_threshold):
            logger.warning(f"Manual threshold {new_threshold} outside bounds [{self.min_threshold}, {self.max_threshold}]")
            return False
        
        old_threshold = self.current_threshold
        self.current_threshold = new_threshold
        
        # Record as adjustment
        adj_record = ICAPAdjustment(
            timestamp=time.time(),
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            reason=reason,
            expected_improvement=0.0  # Unknown for manual override
        )
        self.adjustments.append(adj_record)
        
        # Reset consecutive counter since this is manual
        self.consecutive_adjustments = 0
        self.adjustment_cooldown = 5  # Brief cooldown after manual override
        
        logger.info(f"Manual ICAP override: {old_threshold:.3f} -> {new_threshold:.3f}")
        return True 