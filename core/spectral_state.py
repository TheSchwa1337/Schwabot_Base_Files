"""
Spectral State Management
========================

Implements the core spectral state schema and management for the Forever Fractal system.
Provides unified state representation across all reflection and action pipelines.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
import json

@dataclass
class SpectralState:
    """Core spectral state representation"""
    confidence: float
    entropy_gradient: float
    fft_correlation: float
    dormant_energy: float
    fractal_depth: int
    timestamp: float
    pattern_hash: str
    context_vector: List[float]
    recursive_awareness: float
    profit_bias: float
    success_weight: float
    spectral_coherence: float
    memory_weight: float
    triplet_sequence: List[str]
    historical_outcomes: Dict[str, float]
    
    @classmethod
    def create_initial_state(cls) -> 'SpectralState':
        """Create initial spectral state"""
        return cls(
            confidence=0.0,
            entropy_gradient=0.0,
            fft_correlation=0.0,
            dormant_energy=0.0,
            fractal_depth=0,
            timestamp=datetime.now().timestamp(),
            pattern_hash="",
            context_vector=[],
            recursive_awareness=0.0,
            profit_bias=0.0,
            success_weight=0.0,
            spectral_coherence=0.0,
            memory_weight=1.0,
            triplet_sequence=[],
            historical_outcomes={}
        )
    
    def calculate_confidence_decay(self, current_time: float, decay_rate: float = 1/86400) -> float:
        """
        Calculate confidence decay using exponential decay function:
        C(t) = C₀ * e^(-λt)
        
        Args:
            current_time: Current timestamp
            decay_rate: Decay rate (λ)
            
        Returns:
            Decayed confidence value
        """
        dt = current_time - self.timestamp
        return self.confidence * np.exp(-decay_rate * dt)
    
    def update_weighted_confidence(self, 
                                 base_confidence: float,
                                 historical_confidence: float,
                                 profit_expectation: float,
                                 awareness_bonus: float) -> None:
        """
        Update confidence using weighted fusion:
        Confidence_weighted = 0.4B + 0.3H + 0.2P + 0.1A
        
        Args:
            base_confidence: Base spectral confidence (B)
            historical_confidence: Historical lattice confidence (H)
            profit_expectation: Clamped profit expectation (P)
            awareness_bonus: Recursive awareness bonus (A)
        """
        # Clamp profit expectation to [-0.5, 0.5]
        clamped_profit = np.clip(profit_expectation, -0.5, 0.5)
        
        # Calculate weighted confidence
        self.confidence = (
            0.4 * base_confidence +
            0.3 * historical_confidence +
            0.2 * clamped_profit +
            0.1 * awareness_bonus
        )
    
    def update_profit_bias(self, profit: float, success: bool, alpha: float = 0.1) -> None:
        """
        Update profit bias using EMA smoothing:
        Profit_bias = α * (profit * success_weight) + (1-α) * prior_bias
        
        Args:
            profit: Current profit
            success: Whether the trade was successful
            alpha: EMA smoothing factor
        """
        success_weight = 1.0 if success else -0.5
        self.profit_bias = alpha * (profit * success_weight) + (1 - alpha) * self.profit_bias
    
    def calculate_pattern_hash(self, triplet: str, context: str, action: str) -> str:
        """
        Calculate behavior pattern hash:
        H(b) = MD5(T,C,A,D)
        
        Args:
            triplet: Triplet sequence
            context: Context information
            action: Action taken
            
        Returns:
            MD5 hash of the pattern
        """
        import hashlib
        pattern_data = f"{triplet}|{context}|{action}|{self.fractal_depth}"
        return hashlib.md5(pattern_data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "confidence": self.confidence,
            "entropy_gradient": self.entropy_gradient,
            "fft_correlation": self.fft_correlation,
            "dormant_energy": self.dormant_energy,
            "fractal_depth": self.fractal_depth,
            "timestamp": self.timestamp,
            "pattern_hash": self.pattern_hash,
            "context_vector": self.context_vector,
            "recursive_awareness": self.recursive_awareness,
            "profit_bias": self.profit_bias,
            "success_weight": self.success_weight,
            "spectral_coherence": self.spectral_coherence,
            "memory_weight": self.memory_weight,
            "triplet_sequence": self.triplet_sequence,
            "historical_outcomes": self.historical_outcomes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpectralState':
        """Create state from dictionary"""
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SpectralState':
        """Load state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data) 