"""
Security Vector Allocator.

Implements χ(t) = ∂Φ/∂t + ∇·Ψ for Schwabot's security system.
Computes security drift and applies trust rotation matrices.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SecurityAllocation:
    """Represents a security allocation with drift and rotation."""
    chi_value: float
    phi_drift: float
    psi_divergence: float
    rotation_angle: float
    metadata: Dict[str, float] = None


class SecurityVectorAllocator:
    """
    Implements χ(t) = ∂Φ/∂t + ∇·Ψ for security vector allocation.
    
    This class computes security drift from strategy confidence changes
    and applies trust rotation matrices to strategy vectors.
    """

    def __init__(self: 'SecurityVectorAllocator', phi: float, psi_field: np.ndarray) -> None:
        """
        Initialize the security vector allocator.

        Args:
            phi: Strategy confidence change rate
            psi_field: Vector field representing market entropy
        """
        self.phi = phi
        self.psi_field = psi_field
        self.allocation_history = []

    def compute_chi(self: 'SecurityVectorAllocator') -> float:
        """
        Compute χ(t) = ∂Φ/∂t + ∇·Ψ.
        
        χ(t) represents the security drift from strategy confidence
        changes and entropy field divergence.

        Returns:
            float: Computed χ(t) value
        """
        try:
            # ∂Φ/∂t component (strategy confidence drift)
            phi_drift = self.phi
            
            # ∇·Ψ component (divergence of entropy field)
            if self.psi_field.ndim > 1:
                psi_divergence = np.sum(np.gradient(self.psi_field))
            else:
                psi_divergence = np.sum(np.diff(self.psi_field))
            
            # χ(t) = ∂Φ/∂t + ∇·Ψ
            chi = phi_drift + psi_divergence
            
            return float(chi)
            
        except Exception as e:
            print(f"Error computing χ(t): {e}")
            return 0.0

    def compute_trust_rotation_matrix(self: 'SecurityVectorAllocator', theta: float) -> np.ndarray:
        """
        Compute trust rotation matrix Λ_secure(θ).
        
        This matrix applies security-based rotation to strategy vectors.

        Args:
            theta: Rotation angle in radians
        Returns:
            np.ndarray: 2x2 rotation matrix
        """
        # Standard 2D rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        return rotation_matrix

    def apply_security_allocation(
        self: 'SecurityVectorAllocator', 
        strategy_vector: np.ndarray, 
        theta: float
    ) -> np.ndarray:
        """
        Apply security allocation to strategy vector.

        Args:
            strategy_vector: Input strategy vector
            theta: Security rotation angle
        Returns:
            np.ndarray: Security-allocated strategy vector
        """
        try:
            # Compute χ(t)
            chi = self.compute_chi()
            
            # Get rotation matrix
            rotation_matrix = self.compute_trust_rotation_matrix(theta)
            
            # Apply rotation: secured_vector = Λ_secure(θ) @ strategy_vector
            if strategy_vector.shape[0] == 2:
                secured_vector = rotation_matrix @ strategy_vector
            else:
                # For higher dimensions, apply rotation to first 2 components
                secured_vector = strategy_vector.copy()
                secured_vector[:2] = rotation_matrix @ strategy_vector[:2]
            
            # Store allocation
            allocation = SecurityAllocation(
                chi_value=chi,
                phi_drift=self.phi,
                psi_divergence=chi - self.phi,
                rotation_angle=theta,
                metadata={'vector_norm': float(np.linalg.norm(secured_vector))}
            )
            self.allocation_history.append(allocation)
            
            return secured_vector
            
        except Exception as e:
            print(f"Error applying security allocation: {e}")
            return strategy_vector


if __name__ == "__main__":
    # Demo the security vector allocator
    print("🛡️ Security Vector Allocator Demo")
    print("=" * 40)
    
    # Initialize allocator
    phi = 0.015
    psi_field = np.array([0.012, -0.004, 0.006])
    allocator = SecurityVectorAllocator(phi, psi_field)
    
    # Compute χ(t)
    chi = allocator.compute_chi()
    print(f"χ(t) = {chi:.6f}")
    
    # Test security allocation
    strategy_vector = np.array([0.8, 0.6])
    theta = np.pi / 4
    secured_vector = allocator.apply_security_allocation(strategy_vector, theta)
    print(f"Original: {strategy_vector}")
    print(f"Secured: {secured_vector}") 