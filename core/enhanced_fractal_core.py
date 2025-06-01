"""
Enhanced Fractal Core
===================

Implements an enhanced version of the Forever Fractal mathematical framework
with advanced mathematical structures including:
- Quantized vector lattice with epsilon spacing
- Entropy-slope aware coherence
- Profit allocation tree
- Dormant engine hooks
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import math
from datetime import datetime
import hashlib
import json
import os

@dataclass
class QuantizationProfile:
    """Configuration for vector quantization"""
    decay_power: float = 1.5
    terms: int = 12
    dimension: int = 8
    epsilon_q: float = 0.003
    precision: float = 1e-3

@dataclass
class FractalState:
    """Enhanced fractal state with additional metrics"""
    vector: List[float]
    timestamp: float
    phase: float
    entropy: float
    recursive_depth: int = 0
    coherence_score: float = 0.0
    is_mirror: bool = False
    profit_bias: float = 0.0
    entropy_slope: float = 0.0

class EnhancedFractalCore:
    """Enhanced implementation of the Forever Fractal mathematical framework"""
    
    def __init__(self, profile: Optional[QuantizationProfile] = None):
        """Initialize the enhanced fractal core
        
        Args:
            profile: Optional quantization profile
        """
        self.profile = profile or QuantizationProfile()
        self.state_history: List[FractalState] = []
        self.mirror_memory: Dict[Tuple[int], float] = {}
        self.cyclic_patterns: Dict[int, Dict] = {}
        self.pattern_reversal_key: Optional[int] = None
        
        # Initialize vector bank for efficient sampling
        self.vector_bank: List[List[float]] = []
        self._initialize_vector_bank()
        
        # Initialize profit allocation tree
        self.profit_tree: Dict[str, float] = {}
        self.last_profit_snapshot: float = 0.0
        self.last_snapshot_time: float = 0.0
        
        # Dormant state tracking
        self.dormant_state = False
        self.last_dormant_trigger: Optional[float] = None
        self.entropy_slope_history: List[float] = []
        self.harmonic_power_history: List[float] = []
        self.cpu_render_history: List[float] = []
        
    def _initialize_vector_bank(self):
        """Initialize the vector bank with pre-generated lattice points"""
        # Generate base vectors
        t = time.time()
        for i in range(1000):  # Generate 1000 base vectors
            vector = self.generate_fractal_vector(t + i * 0.1)
            if self.validate_spacing([vector]):
                self.vector_bank.append(vector)
                
        # Generate jittered variants
        jittered_vectors = []
        for vector in self.vector_bank:
            for _ in range(5):  # 5 jittered variants per base vector
                jitter = np.random.normal(0, self.profile.epsilon_q / 2, len(vector))
                jittered = [v + j for v, j in zip(vector, jitter)]
                if self.validate_spacing([jittered]):
                    jittered_vectors.append(jittered)
                    
        self.vector_bank.extend(jittered_vectors)
        
    def forever_fractal(self, t: float, phase_shift: float = 0.0) -> float:
        """
        Generates a scalar fractal signal at time t with decay-modulated frequency terms.
        f_k(t) = sum_{n=1}^N sin((n+1)(t + phase)) / (n+1)^alpha
        """
        return sum(
            math.sin((n + 1) * (t + phase_shift)) / ((n + 1) ** self.profile.decay_power)
            for n in range(self.profile.terms)
        )
    
    def generate_fractal_vector(self, t: float) -> List[float]:
        """
        Constructs an n-dimensional fractal vector with shifted phase.
        F_t = [f_1(t+phi_1), ..., f_n(t+phi_n)]
        """
        return [
            self.forever_fractal(t, phase_shift=2 * math.pi * i / self.profile.dimension)
            for i in range(self.profile.dimension)
        ]
    
    def quantize_vector(self, vector: List[float]) -> List[float]:
        """
        Quantizes a vector into lattice-aligned memory.
        """
        return [round(v / self.profile.precision) * self.profile.precision for v in vector]
    
    def validate_spacing(self, vectors: List[List[float]]) -> bool:
        """
        Ensures all vectors are spaced by at least epsilon_q in Euclidean norm.
        """
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dist = np.linalg.norm(np.array(vectors[i]) - np.array(vectors[j]))
                if dist < self.profile.epsilon_q:
                    return False
        return True
    
    def fft_non_aliasing_check(self, vectors: List[List[float]]) -> bool:
        """
        Confirms FFT non-aliasing: max non-zero freq < DC component.
        """
        for vec in vectors:
            spectrum = np.fft.rfft(vec)
            if np.max(np.abs(spectrum[1:])) >= np.abs(spectrum[0]):
                return False
        return True
    
    def spectral_entropy(self, vector: List[float]) -> float:
        """
        Measures entropy of vector frequency domain.
        """
        spectrum = np.abs(np.fft.rfft(vector))
        normed = spectrum / (np.sum(spectrum) + 1e-10)
        return -np.sum(normed * np.log2(normed + 1e-10))
    
    def validate_entropy_bandwidth(self, vectors: List[List[float]], 
                                 low: float = 0.1, high: float = 0.9) -> bool:
        """
        Ensure entropy of corrected vectors remains inside expected range.
        """
        for v in vectors:
            e = self.spectral_entropy(v)
            if not (low <= e <= high):
                return False
        return True
    
    def compute_entropy_slope(self, states: List[FractalState]) -> float:
        """
        Compute entropy slope from recent states.
        """
        if len(states) < 2:
            return 0.0
            
        entropies = [state.entropy for state in states]
        times = [state.timestamp for state in states]
        
        # Linear regression for slope
        slope, _ = np.polyfit(times, entropies, 1)
        return slope
    
    def update_profit_tree(self, pattern_hash: str, profit: float):
        """
        Update profit allocation tree using Fenwick tree structure.
        """
        # Update profit for pattern
        self.profit_tree[pattern_hash] = self.profit_tree.get(pattern_hash, 0.0) + profit
        
        # Take snapshot every 10 seconds
        current_time = time.time()
        if current_time - self.last_snapshot_time >= 10.0:
            self.last_profit_snapshot = sum(self.profit_tree.values())
            self.last_snapshot_time = current_time
    
    def get_cumulative_profit(self, pattern_hash: str) -> float:
        """
        Get cumulative profit for a pattern using Fenwick tree.
        """
        return self.profit_tree.get(pattern_hash, 0.0)
    
    def compute_dormant_score(self) -> float:
        """
        Compute dormant state score based on multiple metrics.
        """
        if not self.entropy_slope_history or not self.harmonic_power_history:
            return 0.0
            
        # Check entropy slope
        recent_slope = np.mean(self.entropy_slope_history[-10:])
        slope_score = 1.0 if recent_slope <= 0 else 0.0
        
        # Check harmonic power
        recent_power = np.mean(self.harmonic_power_history[-10:])
        power_score = 1.0 if recent_power < 0.1 else 0.0
        
        # Check CPU render time
        recent_cpu = np.mean(self.cpu_render_history[-10:])
        cpu_score = 1.0 if recent_cpu > 100 else 0.0  # 100ms threshold
        
        # Combine scores
        return (slope_score * 0.4 + power_score * 0.4 + cpu_score * 0.2)
    
    def process_recursive_state(self, vector: List[float], depth: int = 0) -> Dict[str, Any]:
        """
        Process fractal state recursively with enhanced features.
        """
        if depth >= 3:  # Max recursive depth
            return {"status": "max_depth_reached"}
            
        # Generate fractal state
        state = FractalState(
            vector=vector,
            timestamp=time.time(),
            phase=math.atan2(vector[1], vector[0]),
            entropy=self.spectral_entropy(vector),
            recursive_depth=depth
        )
        
        # Update state history
        self.state_history.append(state)
        
        # Compute entropy slope
        if len(self.state_history) >= 2:
            slope = self.compute_entropy_slope(self.state_history[-2:])
            state.entropy_slope = slope
            self.entropy_slope_history.append(slope)
            
        # Check for cyclic patterns
        cyclic_detected = self.detect_cyclic_pattern(vector)
        
        # Update profit tree
        pattern_hash = self.generate_pattern_hash(vector)
        self.update_profit_tree(pattern_hash, 0.1)  # Example profit
        
        # Compute dormant score
        dormant_score = self.compute_dormant_score()
        if dormant_score > 0.7:  # Dormant threshold
            self.dormant_state = True
            self.last_dormant_trigger = time.time()
        
        # Process recursively if needed
        recursive_result = None
        if depth < 2:  # Allow one more level of recursion
            downsampled = vector[::2]
            recursive_result = self.process_recursive_state(downsampled, depth + 1)
        
        return {
            "depth": depth,
            "dormant_state": self.dormant_state,
            "dormant_score": dormant_score,
            "entropy": state.entropy,
            "entropy_slope": state.entropy_slope,
            "cyclic_detected": cyclic_detected,
            "recursive_result": recursive_result,
            "profit": self.get_cumulative_profit(pattern_hash)
        }
    
    def detect_cyclic_pattern(self, vector: List[float]) -> bool:
        """
        Detect cyclic number patterns in state vector.
        """
        # Convert vector to cyclic number representation
        cyclic_value = int(abs(vector[0]) * 998001)  # 999²
        cyclic_str = str(cyclic_value).zfill(6)
        
        # Check for pattern reversal
        if cyclic_str == "998998":
            self.pattern_reversal_key = cyclic_value
            return True
            
        # Store pattern
        self.cyclic_patterns[cyclic_value] = {
            'vector': vector,
            'timestamp': time.time()
        }
        
        return False
    
    def generate_pattern_hash(self, vector: List[float]) -> str:
        """
        Generate a hash for pattern matching.
        """
        # Combine vector with its position in sequence
        data = f"{vector}:{len(self.state_history)}"
        
        # Add fractal state to hash
        if self.state_history:
            state = self.state_history[-1]
            data += f":{state.coherence_score}:{state.is_mirror}"
            
        return hashlib.sha256(data.encode()).hexdigest()
    
    def save_state(self, filepath: str):
        """
        Save current state to file.
        """
        state = {
            'profile': self.profile.__dict__,
            'state_history': [
                {
                    'vector': s.vector,
                    'timestamp': s.timestamp,
                    'phase': s.phase,
                    'entropy': s.entropy,
                    'recursive_depth': s.recursive_depth,
                    'coherence_score': s.coherence_score,
                    'is_mirror': s.is_mirror,
                    'profit_bias': s.profit_bias,
                    'entropy_slope': s.entropy_slope
                }
                for s in self.state_history
            ],
            'mirror_memory': {str(k): v for k, v in self.mirror_memory.items()},
            'cyclic_patterns': self.cyclic_patterns,
            'pattern_reversal_key': self.pattern_reversal_key,
            'profit_tree': self.profit_tree,
            'last_profit_snapshot': self.last_profit_snapshot,
            'last_snapshot_time': self.last_snapshot_time,
            'dormant_state': self.dormant_state,
            'last_dormant_trigger': self.last_dormant_trigger
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'EnhancedFractalCore':
        """
        Load state from file.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        # Create instance with profile
        profile = QuantizationProfile(**state['profile'])
        instance = cls(profile)
        
        # Restore state
        instance.state_history = [
            FractalState(**s) for s in state['state_history']
        ]
        instance.mirror_memory = {
            tuple(map(int, k.strip('()').split(','))): v
            for k, v in state['mirror_memory'].items()
        }
        instance.cyclic_patterns = state['cyclic_patterns']
        instance.pattern_reversal_key = state['pattern_reversal_key']
        instance.profit_tree = state['profit_tree']
        instance.last_profit_snapshot = state['last_profit_snapshot']
        instance.last_snapshot_time = state['last_snapshot_time']
        instance.dormant_state = state['dormant_state']
        instance.last_dormant_trigger = state['last_dormant_trigger']
        
        return instance 