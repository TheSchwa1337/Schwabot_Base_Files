"""
Hash Affinity Vault v2.0
========================

Advanced cognitive substrate for recursive intelligence with integrated
profit tier navigation, SHA256 hash correlation, and dynamic backend optimization.

Features:
- Multi-dimensional signal correlation tracking
- Profit tier transition analysis  
- Hash-based future value validation
- Backend performance optimization
- Real-time anomaly detection
"""

import hashlib
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import asyncio

@dataclass
class TickSignature:
    """Complete signature for a single tick with correlation metadata"""
    tick_id: str
    timestamp: datetime
    signal_strength: float
    backend: str
    matrix_id: str
    profit_tier: Optional[str]
    sha256_hash: str
    btc_price: float
    volume: float
    error: Optional[Dict] = None
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    correlation_score: float = 0.0

@dataclass
class ProfitTierTransition:
    """Tracks transitions between profit tiers"""
    from_tier: str
    to_tier: str
    transition_time: datetime
    signal_delta: float
    backend_at_transition: str
    success_rate: float

class HashAffinityVault:
    """
    Advanced vault for tracking signal/error correlations with profit optimization
    """
    
    def __init__(self, max_history: int = 10000, correlation_window: int = 100):
        """
        Initialize the vault with advanced correlation tracking
        
        Args:
            max_history: Maximum number of ticks to store
            correlation_window: Window size for correlation analysis
        """
        self.vault: Dict[str, TickSignature] = {}
        self.max_history = max_history
        self.correlation_window = correlation_window
        
        # Specialized tracking structures
        self.error_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.profit_transitions: List[ProfitTierTransition] = []
        self.backend_performance: Dict[str, Dict] = defaultdict(lambda: {
            'total_ticks': 0,
            'error_count': 0,
            'avg_signal_strength': 0.0,
            'avg_processing_time': 0.0,
            'profit_correlation': 0.0
        })
        
        # Hash correlation tracking
        self.hash_correlations: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.price_prediction_accuracy: deque = deque(maxlen=1000)
        
        # Real-time analysis
        self.recent_ticks: deque = deque(maxlen=correlation_window)
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        
        # Profit tier hierarchy
        self.tier_hierarchy = {
            'PLATINUM': 4,
            'GOLD': 3, 
            'SILVER': 2,
            'BRONZE': 1,
            'NEUTRAL': 0
        }
    
    def generate_tick_hash(self, price: float, volume: float, timestamp: datetime) -> str:
        """Generate SHA256 hash for tick correlation"""
        hash_input = f"{price:.8f}_{volume:.2f}_{timestamp.timestamp()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def log_tick(self, tick_id: str, signal_strength: float, backend: str, 
                matrix_id: str, btc_price: float, volume: float,
                profit_tier: Optional[str] = None, error: Optional[Dict] = None,
                gpu_util: float = 0.0, cpu_util: float = 0.0) -> TickSignature:
        """
        Enhanced tick logging with complete signature tracking
        """
        timestamp = datetime.utcnow()
        sha_hash = self.generate_tick_hash(btc_price, volume, timestamp)
        
        # Calculate correlation score with recent history
        correlation_score = self._calculate_correlation_score(signal_strength, profit_tier)
        
        signature = TickSignature(
            tick_id=tick_id,
            timestamp=timestamp,
            signal_strength=signal_strength,
            backend=backend,
            matrix_id=matrix_id,
            profit_tier=profit_tier,
            sha256_hash=sha_hash,
            btc_price=btc_price,
            volume=volume,
            error=error,
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util,
            correlation_score=correlation_score
        )
        
        # Store in vault
        self.vault[tick_id] = signature
        self.recent_ticks.append(signature)
        
        # Update tracking structures
        self._update_backend_performance(signature)
        self._track_profit_transitions(signature)
        self._update_hash_correlations(signature)
        
        if error:
            self._log_error_pattern(signature)
        
        # Maintain history limit
        if len(self.vault) > self.max_history:
            oldest_tick = min(self.vault.keys())
            del self.vault[oldest_tick]
        
        return signature
    
    def _calculate_correlation_score(self, signal_strength: float, 
                                   profit_tier: Optional[str]) -> float:
        """Calculate correlation score with recent signals"""
        if len(self.recent_ticks) < 10:
            return 0.5  # Neutral score for insufficient data
        
        recent_signals = [t.signal_strength for t in self.recent_ticks]
        recent_avg = np.mean(recent_signals)
        recent_std = np.std(recent_signals)
        
        if recent_std == 0:
            return 0.5
        
        # Z-score based correlation
        z_score = (signal_strength - recent_avg) / recent_std
        
        # Adjust for profit tier
        tier_bonus = 0.0
        if profit_tier:
            tier_bonus = self.tier_hierarchy.get(profit_tier, 0) * 0.1
        
        # Normalize to 0-1 range
        correlation = 0.5 + (z_score * 0.2) + tier_bonus
        return max(0.0, min(1.0, correlation))
    
    def _update_backend_performance(self, signature: TickSignature):
        """Update backend performance metrics"""
        backend = signature.backend
        perf = self.backend_performance[backend]
        
        # Update rolling averages
        total = perf['total_ticks']
        perf['avg_signal_strength'] = (
            (perf['avg_signal_strength'] * total + signature.signal_strength) / (total + 1)
        )
        
        if signature.error:
            perf['error_count'] += 1
        
        perf['total_ticks'] += 1
        
        # Update profit correlation
        if signature.profit_tier:
            tier_value = self.tier_hierarchy.get(signature.profit_tier, 0)
            perf['profit_correlation'] = (
                (perf['profit_correlation'] * total + tier_value) / (total + 1)
            )
    
    def _track_profit_transitions(self, signature: TickSignature):
        """Track profit tier transitions"""
        if not signature.profit_tier or len(self.recent_ticks) < 2:
            return
        
        prev_signature = self.recent_ticks[-2]
        if prev_signature.profit_tier and prev_signature.profit_tier != signature.profit_tier:
            transition = ProfitTierTransition(
                from_tier=prev_signature.profit_tier,
                to_tier=signature.profit_tier,
                transition_time=signature.timestamp,
                signal_delta=signature.signal_strength - prev_signature.signal_strength,
                backend_at_transition=signature.backend,
                success_rate=self._calculate_transition_success_rate(
                    prev_signature.profit_tier, signature.profit_tier
                )
            )
            self.profit_transitions.append(transition)
    
    def _calculate_transition_success_rate(self, from_tier: str, to_tier: str) -> float:
        """Calculate historical success rate for this transition type"""
        similar_transitions = [
            t for t in self.profit_transitions 
            if t.from_tier == from_tier and t.to_tier == to_tier
        ]
        
        if not similar_transitions:
            return 0.5  # Neutral for unknown transitions
        
        # Simple success metric: higher tier = success
        from_value = self.tier_hierarchy.get(from_tier, 0)
        to_value = self.tier_hierarchy.get(to_tier, 0)
        
        if to_value > from_value:
            return min(1.0, 0.7 + (len(similar_transitions) * 0.05))
        else:
            return max(0.0, 0.3 - (len(similar_transitions) * 0.02))
    
    def _update_hash_correlations(self, signature: TickSignature):
        """Update hash-based correlations for future value prediction"""
        hash_prefix = signature.sha256_hash[:8]  # Use first 8 chars as key
        
        # Store correlation with signal strength
        self.hash_correlations[hash_prefix].append(
            (signature.tick_id, signature.signal_strength)
        )
        
        # Limit correlation history per hash prefix
        if len(self.hash_correlations[hash_prefix]) > 50:
            self.hash_correlations[hash_prefix] = self.hash_correlations[hash_prefix][-50:]
    
    def _log_error_pattern(self, signature: TickSignature):
        """Log error patterns for analysis"""
        error_pattern = {
            'timestamp': signature.timestamp.isoformat(),
            'backend': signature.backend,
            'signal_strength': signature.signal_strength,
            'profit_tier': signature.profit_tier,
            'error_type': signature.error.get('type', 'unknown'),
            'error_details': signature.error.get('details', ''),
            'gpu_utilization': signature.gpu_utilization,
            'cpu_utilization': signature.cpu_utilization
        }
        
        self.error_patterns[signature.backend].append(error_pattern)
        
        # Limit error history per backend
        if len(self.error_patterns[signature.backend]) > 100:
            self.error_patterns[signature.backend] = self.error_patterns[signature.backend][-100:]
    
    def get_recent_errors(self, backend: Optional[str] = None, 
                         time_window: Optional[timedelta] = None) -> Dict[str, TickSignature]:
        """Enhanced error retrieval with time filtering"""
        cutoff_time = None
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
        
        filtered_errors = {}
        for tick_id, signature in self.vault.items():
            if signature.error is None:
                continue
            
            if backend and signature.backend != backend:
                continue
            
            if cutoff_time and signature.timestamp < cutoff_time:
                continue
            
            filtered_errors[tick_id] = signature
        
        return filtered_errors
    
    def suggest_backend_swap(self, threshold: int = 3, 
                           consider_profit_correlation: bool = True) -> Optional[str]:
        """Advanced backend swap suggestion with profit correlation"""
        recent_errors = self.get_recent_errors(time_window=timedelta(minutes=10))
        
        if len(recent_errors) < threshold:
            return None
        
        # Count errors by backend
        error_counts = defaultdict(int)
        for signature in recent_errors.values():
            error_counts[signature.backend] += 1
        
        if not error_counts:
            return None
        
        # Find most problematic backend
        failing_backend = max(error_counts.items(), key=lambda x: x[1])[0]
        
        # Consider profit correlation in decision
        if consider_profit_correlation:
            failing_perf = self.backend_performance[failing_backend]
            if failing_perf['profit_correlation'] > 0.7:  # High profit correlation
                return None  # Don't swap profitable backend
        
        return failing_backend
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect signal anomalies using statistical analysis"""
        if len(self.recent_ticks) < 20:
            return []
        
        signals = np.array([t.signal_strength for t in self.recent_ticks])
        mean_signal = np.mean(signals)
        std_signal = np.std(signals)
        
        anomalies = []
        # Get last 10 ticks from deque safely
        last_10_ticks = []
        for i, signature in enumerate(self.recent_ticks):
            if i >= len(self.recent_ticks) - 10:
                last_10_ticks.append(signature)
        
        for signature in last_10_ticks:
            z_score = abs(signature.signal_strength - mean_signal) / max(std_signal, 0.001)
            
            if z_score > self.anomaly_threshold:
                anomalies.append({
                    'tick_id': signature.tick_id,
                    'signal_strength': signature.signal_strength,
                    'z_score': z_score,
                    'backend': signature.backend,
                    'profit_tier': signature.profit_tier,
                    'anomaly_type': 'signal_deviation'
                })
        
        return anomalies
    
    def predict_hash_correlation(self, hash_prefix: str) -> Optional[float]:
        """Predict signal strength based on hash correlation"""
        if hash_prefix not in self.hash_correlations:
            return None
        
        correlations = self.hash_correlations[hash_prefix]
        if len(correlations) < 3:
            return None
        
        # Return weighted average of recent correlations
        recent_correlations = correlations[-10:]
        weights = np.linspace(0.1, 1.0, len(recent_correlations))
        weighted_signals = [c[1] * w for c, w in zip(recent_correlations, weights)]
        
        return sum(weighted_signals) / sum(weights)
    
    def get_profit_tier_analysis(self) -> Dict[str, Any]:
        """Comprehensive profit tier analysis"""
        if not self.profit_transitions:
            return {'error': 'No profit transitions recorded'}
        
        # Transition frequency analysis
        transition_counts = defaultdict(int)
        successful_transitions = 0
        
        for transition in self.profit_transitions:
            transition_key = f"{transition.from_tier}→{transition.to_tier}"
            transition_counts[transition_key] += 1
            
            if transition.success_rate > 0.5:
                successful_transitions += 1
        
        # Backend correlation with profit tiers
        backend_profit_correlation = {}
        for backend, perf in self.backend_performance.items():
            backend_profit_correlation[backend] = perf['profit_correlation']
        
        return {
            'total_transitions': len(self.profit_transitions),
            'successful_transitions': successful_transitions,
            'success_rate': successful_transitions / len(self.profit_transitions),
            'transition_patterns': dict(transition_counts),
            'backend_profit_correlation': backend_profit_correlation,
            'most_profitable_backend': max(backend_profit_correlation.items(), 
                                         key=lambda x: x[1])[0] if backend_profit_correlation else None
        }
    
    def export_comprehensive_state(self) -> Dict[str, Any]:
        """Export complete vault state for analysis"""
        return {
            'total_ticks': len(self.vault),
            'backend_performance': dict(self.backend_performance),
            'profit_analysis': self.get_profit_tier_analysis(),
            'recent_anomalies': self.detect_anomalies(),
            'hash_correlation_count': len(self.hash_correlations),
            'error_patterns_summary': {
                backend: len(patterns) for backend, patterns in self.error_patterns.items()
            },
            'vault_utilization': len(self.vault) / self.max_history
        } 