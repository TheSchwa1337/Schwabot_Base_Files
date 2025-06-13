"""
Cursor Math Integration
======================

Core mathematical functions for Schwabot's Euler-coded triggers, binding energy
calculations, and phase drift analysis. This module serves as the mathematical
foundation for the entire system's stability and pattern recognition.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging
import json
import yaml
from pathlib import Path
from .braid_pattern_engine import BraidPatternEngine, BraidPattern

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class BindingEnergyParams:
    """Parameters for binding energy calculations"""
    alpha_v: float = 15.8  # Volume term
    alpha_s: float = 18.3  # Surface term
    alpha_c: float = 0.714  # Coulomb term
    alpha_a: float = 23.2  # Asymmetry term
    alpha_p: float = 12.0  # Pairing term

@dataclass
class PhaseShell:
    """Represents a phase shell state"""
    theta: float  # Phase angle
    drift: float  # Drift from π
    stability: float  # Shell stability (0-1)
    shell_type: str  # 'symmetry', 'drift_positive', 'drift_negative'

@dataclass
class CursorState:
    """Represents the current state of the cursor"""
    triplet: Tuple[float, float, float]
    timestamp: float
    velocity: float = 0.0
    entropy: float = 0.0

def load_config():
    config_path = Path(__file__).resolve().parent / 'config/matrix_response_paths.yaml'
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return {"defaults": True}

class CursorMath:
    """Core mathematical functions for Schwabot's stability analysis"""
    
    def __init__(self, binding_params: Optional[BindingEnergyParams] = None):
        self.binding_params = binding_params or BindingEnergyParams()
        
    def compute_binding_energy(self, Z: int, N: int) -> float:
        """
        Compute binding energy per nucleon (B/A) using the semi-empirical mass formula.
        
        Args:
            Z: Proton number (volume trend)
            N: Neutron number (volatility buffer)
            
        Returns:
            Binding energy per nucleon
        """
        A = Z + N
        if A <= 0:
            return -np.inf
            
        B = (
            self.binding_params.alpha_v * A
            - self.binding_params.alpha_s * A ** (2/3)
            - self.binding_params.alpha_c * Z ** 2 / A ** (1/3)
            - self.binding_params.alpha_a * (N - Z) ** 2 / A
            + self.binding_params.alpha_p / A ** 0.5
        )
        
        return B / A
    
    def compute_stability_index(self, Z: int, N: int) -> float:
        """
        Compute stability index based on binding energy.
        
        Args:
            Z: Proton number (volume trend)
            N: Neutron number (volatility buffer)
            
        Returns:
            Stability index (0-1)
        """
        B_A = self.compute_binding_energy(Z, N)
        # Normalize to 0-1 range based on known stable nuclei
        return max(0.0, min(1.0, (B_A - 5.0) / 4.0))
    
    def compute_phase_drift(self, theta: float) -> Tuple[float, float]:
        """
        Compute phase drift from Euler collapse point (π).
        
        Args:
            theta: Current phase angle
            
        Returns:
            (loss_shell, profit_shell) tuple
        """
        # Loss shell function
        loss_shell = abs(math.sin(theta)) * math.exp(-10 * (theta - math.pi) ** 2)
        # Profit shell is complement
        profit_shell = 1.0 - loss_shell
        return loss_shell, profit_shell
    
    def classify_phase_shell(self, theta: float) -> PhaseShell:
        """
        Classify current phase into shell type.
        
        Args:
            theta: Current phase angle
            
        Returns:
            PhaseShell object with classification
        """
        drift = abs(theta - math.pi)
        loss_shell, profit_shell = self.compute_phase_drift(theta)
        
        if drift < 0.1:  # Near π
            shell_type = 'symmetry'
        elif theta > math.pi:
            shell_type = 'drift_positive'
        else:
            shell_type = 'drift_negative'
            
        return PhaseShell(
            theta=theta,
            drift=drift,
            stability=profit_shell,
            shell_type=shell_type
        )
    
    def compute_decay_index(self, index: int, base_value: float = 100.0,
                          decay_rate: float = 0.015) -> float:
        """
        Compute memory decay index.
        
        Args:
            index: Time index
            base_value: Initial value
            decay_rate: Decay rate constant
            
        Returns:
            Decayed value
        """
        return base_value * math.exp(-decay_rate * index)
    
    def classify_entropy_shell(self, z_score: float) -> str:
        """
        Classify entropy shell based on z-score.
        
        Args:
            z_score: Z-score of entropy
            
        Returns:
            Shell classification
        """
        if z_score > 2.5:
            return 'critical_bloom'
        elif z_score > 1.0:
            return 'unstable'
        else:
            return 'stable'
    
    def compute_totient_phase(self, n: int) -> List[int]:
        """
        Compute valid phase indices using Euler's totient function.
        
        Args:
            n: Modulus for phase calculation
            
        Returns:
            List of valid phase indices
        """
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
            
        return [i for i in range(1, n) if gcd(i, n) == 1]
    
    def compute_collision_score(self, hash1: int, hash2: int) -> float:
        """
        Compute collision score between two hashes.
        
        Args:
            hash1: First hash value
            hash2: Second hash value
            
        Returns:
            Collision score (0-1)
        """
        # XOR the hashes and count set bits
        xor_result = hash1 ^ hash2
        set_bits = bin(xor_result).count('1')
        return set_bits / 256.0  # Normalize by hash size
    
    def compute_matrix_stability(self, binding_energy: float,
                               phase_stability: float,
                               entropy_class: str) -> str:
        """
        Compute overall matrix stability state.
        
        Args:
            binding_energy: Binding energy value
            phase_stability: Phase shell stability
            entropy_class: Entropy shell classification
            
        Returns:
            Matrix state classification
        """
        if entropy_class == 'critical_bloom':
            return 'cooldown_abort'
        elif entropy_class == 'unstable' or binding_energy < 5.0:
            return 'entropy_realign'
        elif phase_stability < 0.5:
            return 'phase_correction'
        else:
            return 'matrix_safe'

    def calculate_angular_similarity(self, triplet1: Tuple[float, float, float], triplet2: Tuple[float, float, float]) -> float:
        """Calculate angular similarity between two triplets"""
        vec1 = np.array(triplet1)
        vec2 = np.array(triplet2)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        return angle

class CursorEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_config()
        self.pattern_engine = BraidPatternEngine()
        self.state: Optional[CursorState] = None
        self.history: List[CursorState] = []
        
    def tick(self, triplet: Tuple[float, float, float], timestamp: float, recursive: bool = False) -> Optional[BraidPattern]:
        """Process a new tick with triplet data"""
        # Basic type check for triplet
        if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
            raise ValueError("Triplet must be a 3-element tuple or list of floats.")
        
        # Calculate entropy using normalized vector magnitude
        velocity = np.linalg.norm(np.array(triplet))
        entropy = self._calculate_entropy(velocity)
        
        self.state = CursorState(triplet=triplet, timestamp=timestamp, velocity=velocity, entropy=entropy)
        self.history.append(self.state)
        
        # Update pattern engine
        self.pattern_engine.add_strand(velocity, entropy)
        
        # Check for patterns
        pattern = self.pattern_engine.classify()
        
        if recursive:
            # Add future integration: fractal sync, mirrored cache, or GAN callback
            pass
        
        return pattern
    
    def tick_vector(self, triplet_series: List[Tuple[float, float, float]], timestamps: List[float], recursive: bool = False) -> List[BraidPattern]:
        """Process multiple ticks with triplet data"""
        patterns = []
        for t, ts in zip(triplet_series, timestamps):
            pattern = self.tick(t, ts, recursive=recursive)
            if pattern:
                patterns.append(pattern)
        return patterns
        
    def get_current_pattern(self) -> Optional[BraidPattern]:
        """Get the current braid pattern"""
        return self.pattern_engine.classify()
        
    def get_pattern_frequency(self, pattern_name: str) -> float:
        """Calculate frequency of a specific pattern in recent history"""
        count = 0
        for s in self.history[-3:]:
            sim = self.pattern_engine.simulate_pattern_from_states([s.triplet])
            if sim and sim[0].name == pattern_name:
                count += 1
        return count / len(self.history) if self.history else 0.0
        
    def get_history(self, limit: Optional[int] = None) -> List[CursorState]:
        """Get cursor history, optionally limited to last N states"""
        if limit is None:
            return self.history
        return self.history[-limit:]

    def clear_history(self):
        """Clear cursor history"""
        self.history.clear()
        self.pattern_engine.clear_history()

    def add_custom_pattern(self, name: str, pattern: List[Tuple[int, float]]):
        """Add a custom pattern to the pattern library"""
        self.pattern_engine.add_custom_pattern(name, pattern)

    def reverse_tick(self, steps: int = 1) -> None:
        """Reverse the cursor by a specified number of steps"""
        for _ in range(min(steps, len(self.history))):
            self.history.pop()
        self.state = self.history[-1] if self.history else None

    def save_to_file(self, path: Path) -> None:
        """Save cursor state to a JSON file"""
        with open(path, 'w') as f:
            f.write(self.to_json())

    def _calculate_entropy(self, velocity: float) -> float:
        """Calculate pseudo-entropy with safe log protection"""
        safe_velocity = max(velocity, 1e-8)
        return -np.log(safe_velocity)

# Example usage
if __name__ == "__main__":
    math_core = CursorMath()
    
    # Test binding energy
    B_A = math_core.compute_binding_energy(26, 30)  # Iron-56
    print(f"Binding energy (Fe-56): {B_A:.3f}")
    
    # Test phase drift
    loss, profit = math_core.compute_phase_drift(math.pi + 0.1)
    print(f"Phase drift (π + 0.1): loss={loss:.3f}, profit={profit:.3f}")
    
    # Test entropy classification
    shell = math_core.classify_entropy_shell(2.7)
    print(f"Entropy shell (z=2.7): {shell}")
    
    # Test matrix stability
    state = math_core.compute_matrix_stability(
        binding_energy=7.5,
        phase_stability=0.8,
        entropy_class='stable'
    )
    print(f"Matrix state: {state}")

    engine = CursorEngine()
    engine.add_custom_pattern("example", [(1, 0.5), (2, 0.3)])
    triplet1 = (1.0, 2.0, 3.0)
    triplet2 = (4.0, 5.0, 6.0)
    similarity = engine.calculate_angular_similarity(triplet1, triplet2)
    print(f"Angular Similarity: {similarity} degrees") 