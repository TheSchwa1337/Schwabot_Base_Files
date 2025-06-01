"""
Strategic Dormancy Management
============================

Implements dormancy state management and trigger logic for the Forever Fractal system.
Handles pattern scoring and periodic strategy audits.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from .spectral_state import SpectralState
from .behavior_pattern_tracker import BehaviorPatternTracker

class StrategicDormancy:
    """Manages strategic dormancy states and triggers"""
    
    def __init__(self,
                 entropy_threshold: float = 0.7,
                 coherence_threshold: float = 0.8,
                 min_pattern_frequency: float = 0.1,
                 audit_interval: int = 3600):
        """
        Initialize dormancy manager
        
        Args:
            entropy_threshold: Threshold for entropy-based dormancy
            coherence_threshold: Threshold for coherence-based dormancy
            min_pattern_frequency: Minimum pattern frequency for activation
            audit_interval: Interval between strategy audits (seconds)
        """
        self.entropy_threshold = entropy_threshold
        self.coherence_threshold = coherence_threshold
        self.min_pattern_frequency = min_pattern_frequency
        self.audit_interval = audit_interval
        
        self.last_audit = datetime.now().timestamp()
        self.dormant_states = {}
        self.activation_history = []
        self.pattern_scores = {}
        
    def evaluate_dormancy(self,
                         state: SpectralState,
                         pattern_tracker: BehaviorPatternTracker) -> Tuple[bool, float]:
        """
        Evaluate if system should enter dormancy
        
        Args:
            state: Current spectral state
            pattern_tracker: Behavior pattern tracker
            
        Returns:
            Tuple of (should_dormant, confidence)
        """
        # Check entropy condition
        entropy_condition = state.entropy_gradient > self.entropy_threshold
        
        # Check coherence condition
        coherence_condition = state.spectral_coherence > self.coherence_threshold
        
        # Check pattern frequency
        pattern_frequency = pattern_tracker.get_pattern_frequency(state.pattern_hash)
        pattern_condition = pattern_frequency < self.min_pattern_frequency
        
        # Calculate dormancy confidence
        confidence = np.mean([
            float(entropy_condition),
            float(coherence_condition),
            float(pattern_condition)
        ])
        
        should_dormant = confidence > 0.5
        
        if should_dormant:
            self.dormant_states[state.pattern_hash] = {
                'timestamp': datetime.now().timestamp(),
                'confidence': confidence,
                'entropy': state.entropy_gradient,
                'coherence': state.spectral_coherence
            }
            
        return should_dormant, confidence
    
    def check_activation_trigger(self,
                               state: SpectralState,
                               pattern_tracker: BehaviorPatternTracker) -> bool:
        """
        Check if system should activate from dormancy
        
        Args:
            state: Current spectral state
            pattern_tracker: Behavior pattern tracker
            
        Returns:
            Whether to activate
        """
        if state.pattern_hash not in self.dormant_states:
            return True
            
        dormant_state = self.dormant_states[state.pattern_hash]
        
        # Check if enough time has passed
        time_in_dormancy = datetime.now().timestamp() - dormant_state['timestamp']
        if time_in_dormancy < self.audit_interval:
            return False
            
        # Check entropy recovery
        entropy_recovered = state.entropy_gradient < self.entropy_threshold
        
        # Check coherence recovery
        coherence_recovered = state.spectral_coherence < self.coherence_threshold
        
        # Check pattern frequency recovery
        pattern_frequency = pattern_tracker.get_pattern_frequency(state.pattern_hash)
        pattern_recovered = pattern_frequency >= self.min_pattern_frequency
        
        # Calculate activation confidence
        activation_confidence = np.mean([
            float(entropy_recovered),
            float(coherence_recovered),
            float(pattern_recovered)
        ])
        
        should_activate = activation_confidence > 0.5
        
        if should_activate:
            self.activation_history.append({
                'pattern_hash': state.pattern_hash,
                'timestamp': datetime.now().timestamp(),
                'confidence': activation_confidence,
                'dormancy_duration': time_in_dormancy
            })
            del self.dormant_states[state.pattern_hash]
            
        return should_activate
    
    def update_pattern_score(self,
                           pattern_hash: str,
                           success: bool,
                           profit: float) -> None:
        """
        Update pattern success score
        
        Args:
            pattern_hash: Pattern hash
            success: Whether the pattern was successful
            profit: Profit achieved
        """
        if pattern_hash not in self.pattern_scores:
            self.pattern_scores[pattern_hash] = {
                'success_count': 0,
                'total_count': 0,
                'total_profit': 0.0
            }
            
        score = self.pattern_scores[pattern_hash]
        score['total_count'] += 1
        if success:
            score['success_count'] += 1
        score['total_profit'] += profit
        
    def get_pattern_score(self, pattern_hash: str) -> Dict:
        """
        Get pattern success score
        
        Args:
            pattern_hash: Pattern hash to look up
            
        Returns:
            Pattern score dictionary
        """
        return self.pattern_scores.get(pattern_hash, {
            'success_count': 0,
            'total_count': 0,
            'total_profit': 0.0
        })
    
    def run_strategy_audit(self) -> Dict:
        """
        Run periodic strategy audit
        
        Returns:
            Audit results
        """
        current_time = datetime.now().timestamp()
        if current_time - self.last_audit < self.audit_interval:
            return {}
            
        # Calculate dormancy metrics
        dormant_count = len(self.dormant_states)
        total_patterns = len(self.pattern_scores)
        dormancy_rate = dormant_count / total_patterns if total_patterns > 0 else 0
        
        # Calculate success metrics
        success_rates = []
        for pattern_hash, score in self.pattern_scores.items():
            if score['total_count'] > 0:
                success_rate = score['success_count'] / score['total_count']
                success_rates.append(success_rate)
                
        avg_success_rate = np.mean(success_rates) if success_rates else 0
        
        # Calculate profit metrics
        total_profit = sum(score['total_profit'] for score in self.pattern_scores.values())
        avg_profit = total_profit / total_patterns if total_patterns > 0 else 0
        
        # Update audit timestamp
        self.last_audit = current_time
        
        return {
            'dormant_count': dormant_count,
            'total_patterns': total_patterns,
            'dormancy_rate': dormancy_rate,
            'avg_success_rate': avg_success_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'timestamp': current_time
        }
    
    def save_state(self, filepath: str) -> None:
        """
        Save dormancy state to file
        
        Args:
            filepath: Path to save state
        """
        state = {
            'entropy_threshold': self.entropy_threshold,
            'coherence_threshold': self.coherence_threshold,
            'min_pattern_frequency': self.min_pattern_frequency,
            'audit_interval': self.audit_interval,
            'last_audit': self.last_audit,
            'dormant_states': self.dormant_states,
            'activation_history': self.activation_history,
            'pattern_scores': self.pattern_scores
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'StrategicDormancy':
        """
        Load dormancy state from file
        
        Args:
            filepath: Path to load state from
            
        Returns:
            Loaded StrategicDormancy instance
        """
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        dormancy = cls(
            entropy_threshold=state['entropy_threshold'],
            coherence_threshold=state['coherence_threshold'],
            min_pattern_frequency=state['min_pattern_frequency'],
            audit_interval=state['audit_interval']
        )
        
        dormancy.last_audit = state['last_audit']
        dormancy.dormant_states = state['dormant_states']
        dormancy.activation_history = state['activation_history']
        dormancy.pattern_scores = state['pattern_scores']
        
        return dormancy 