"""
Pattern Metrics Module
===================

Handles quantum-cellular pattern calculations and state management.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

@dataclass
class PatternState:
    """Current state of a quantum-cellular pattern"""
    timestamp: str
    magnitude: float
    stability: float
    coherence: float
    entropy: float
    state_distribution: Dict[str, float]
    centroid_distance: float

class PatternMetrics:
    """Calculates and manages quantum-cellular pattern metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.log_dir / "pattern_metrics.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PatternMetrics')
        
        # Pattern state thresholds
        self.state_thresholds = {
            'stable': 0.7,      # 70% stability
            'chaotic': 0.3,     # 30% stability
            'transition': 0.5,  # 50% stability
            'quantum': 0.8,     # 80% coherence
            'cellular': 0.6     # 60% coherence
        }
    
    def calculate_magnitude(self, pattern: np.ndarray) -> float:
        """Calculate pattern magnitude"""
        return np.linalg.norm(pattern)
    
    def calculate_stability(self, 
                          current_pattern: np.ndarray,
                          previous_pattern: Optional[np.ndarray] = None) -> float:
        """Calculate pattern stability"""
        if previous_pattern is None:
            return 1.0
        
        # Calculate pattern difference
        diff = np.linalg.norm(current_pattern - previous_pattern)
        max_diff = np.linalg.norm(current_pattern) + np.linalg.norm(previous_pattern)
        
        # Normalize stability to [0, 1]
        return 1.0 - (diff / max_diff if max_diff > 0 else 0)
    
    def calculate_coherence(self, pattern: np.ndarray) -> float:
        """Calculate pattern coherence"""
        # Calculate correlation matrix
        corr = np.corrcoef(pattern)
        
        # Remove diagonal elements
        np.fill_diagonal(corr, 0)
        
        # Calculate average correlation
        return np.mean(np.abs(corr))
    
    def calculate_entropy(self, pattern: np.ndarray) -> float:
        """Calculate pattern entropy"""
        # Normalize pattern
        norm_pattern = pattern / np.sum(np.abs(pattern))
        
        # Calculate Shannon entropy
        return -np.sum(norm_pattern * np.log2(norm_pattern + 1e-10))
    
    def determine_state_distribution(self,
                                   stability: float,
                                   coherence: float) -> Dict[str, float]:
        """Determine pattern state distribution"""
        states = {
            'stable': 0.0,
            'chaotic': 0.0,
            'transition': 0.0,
            'quantum': 0.0,
            'cellular': 0.0
        }
        
        # Calculate state probabilities
        if stability >= self.state_thresholds['stable']:
            states['stable'] = 1.0
        elif stability <= self.state_thresholds['chaotic']:
            states['chaotic'] = 1.0
        else:
            states['transition'] = 1.0
        
        if coherence >= self.state_thresholds['quantum']:
            states['quantum'] = 1.0
        elif coherence >= self.state_thresholds['cellular']:
            states['cellular'] = 1.0
        
        # Normalize probabilities
        total = sum(states.values())
        if total > 0:
            states = {k: v/total for k, v in states.items()}
        
        return states
    
    def calculate_centroid_distance(self, 
                                  pattern: np.ndarray,
                                  ideal_center: float = 7.5) -> float:
        """Calculate distance from pattern centroid to ideal center"""
        centroid = np.mean(pattern)
        return abs(centroid - ideal_center)
    
    def update_pattern_state(self,
                           pattern: np.ndarray,
                           previous_pattern: Optional[np.ndarray] = None) -> PatternState:
        """Update pattern state with current metrics"""
        # Calculate basic metrics
        magnitude = self.calculate_magnitude(pattern)
        stability = self.calculate_stability(pattern, previous_pattern)
        coherence = self.calculate_coherence(pattern)
        entropy = self.calculate_entropy(pattern)
        
        # Determine state distribution
        state_distribution = self.determine_state_distribution(stability, coherence)
        
        # Calculate centroid distance
        centroid_distance = self.calculate_centroid_distance(pattern)
        
        # Create pattern state
        state = PatternState(
            timestamp=datetime.now().isoformat(),
            magnitude=magnitude,
            stability=stability,
            coherence=coherence,
            entropy=entropy,
            state_distribution=state_distribution,
            centroid_distance=centroid_distance
        )
        
        # Log state update
        self.logger.info(
            f"Pattern state updated - Magnitude: {magnitude:.2f}, "
            f"Stability: {stability:.2f}, Coherence: {coherence:.2f}"
        )
        
        return state
    
    def get_pattern_metrics(self, state: PatternState) -> Dict:
        """Get pattern metrics in dictionary format"""
        return {
            'magnitude': state.magnitude,
            'stability': state.stability,
            'coherence': state.coherence,
            'entropy': state.entropy,
            'state_distribution': state.state_distribution,
            'centroid_distance': state.centroid_distance
        }

# Example usage:
"""
from ncco_core.pattern_metrics import PatternMetrics
import numpy as np

# Initialize
metrics = PatternMetrics()

# Create sample pattern
pattern = np.random.rand(4, 4)
previous_pattern = np.random.rand(4, 4)

# Update pattern state
state = metrics.update_pattern_state(pattern, previous_pattern)

# Get metrics
pattern_metrics = metrics.get_pattern_metrics(state)
""" 