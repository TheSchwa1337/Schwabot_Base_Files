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