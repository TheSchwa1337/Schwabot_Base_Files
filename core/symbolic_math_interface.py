#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symbolic Math Interface for Schwabot
====================================

Provides Cursor-friendly wrappers for complex mathematical operations:
• Symbolic operator aliases (∇, Ω, ψ, λ)
• Hardware-optimized computation (CPU/GPU)
• Type-safe mathematical operations
• Recursive logic preservation

This interface allows Cursor to understand complex math while maintaining
the recursive depth and symbolic richness of Schwabot's mathematical framework.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type aliases for Cursor clarity
SignalField = np.ndarray
TimeIndex = int
PhaseValue = float
DriftCoefficient = float
EntropyWeight = float
HashValue = str

@dataclass
class SymbolicContext:
    """Context for symbolic mathematical operations."""
    cycle_id: int
    vault_state: str
    entropy_index: int
    phantom_layer: bool = False

class EntropicGradient:
    """
    Symbolic gradient operations (∇).
    Handles gradient computation with hardware optimization.
    """
    
    @staticmethod
    def derive(field: SignalField, time_idx: TimeIndex, context: Optional[SymbolicContext] = None) -> SignalField:
        """
        Compute gradient of signal field at specific time index.
        
        Symbolic: ∇ψ(t)
        Cursor-friendly: EntropicGradient.derive(ψ, t)
        """
        try:
            # Hardware-optimized gradient computation
            if hasattr(np, 'gradient'):
                gradient = np.gradient(field)
                return gradient[time_idx] if time_idx < len(gradient) else gradient[-1]
            else:
                # Fallback gradient computation
                if len(field) > 1:
                    return np.diff(field)[min(time_idx, len(field)-2)]
                return np.array([0.0])
        except Exception as e:
            logger.error(f"Gradient computation error: {e}")
            return np.array([0.0])
    
    @staticmethod
    def derive_with_context(field: SignalField, time_idx: TimeIndex, context: SymbolicContext) -> SignalField:
        """
        Compute gradient with contextual awareness.
        
        Symbolic: ∇ψ(t) | context
        """
        base_gradient = EntropicGradient.derive(field, time_idx)
        
        # Apply context-specific modifications
        if context.phantom_layer:
            # Phantom layer entropy boost
            base_gradient *= 1.2
        
        if context.vault_state == 'phantom':
            # Vault state modification
            base_gradient *= 0.8
            
        return base_gradient

class PhaseOmega:
    """
    Phase Omega operations (Ω).
    Handles phase computation and momentum signals.
    """
    
    @staticmethod
    def compute(gradient: SignalField, drift: DriftCoefficient, context: Optional[SymbolicContext] = None) -> PhaseValue:
        """
        Compute phase omega from gradient and drift.
        
        Symbolic: Ω = ∇ψ(t) * D
        Cursor-friendly: PhaseOmega.compute(gradient, drift)
        """
        try:
            if isinstance(gradient, np.ndarray):
                gradient_value = gradient[0] if gradient.size > 0 else 0.0
            else:
                gradient_value = float(gradient)
            
            omega = gradient_value * drift
            
            # Apply context modifications
            if context and context.phantom_layer:
                omega *= 1.1  # Phantom layer boost
                
            return float(omega)
        except Exception as e:
            logger.error(f"Phase Omega computation error: {e}")
            return 0.0
    
    @staticmethod
    def compute_stable(gradient: SignalField, drift: DriftCoefficient, noise_factor: float = 1.0) -> PhaseValue:
        """
        Compute stable phase omega with noise consideration.
        
        Symbolic: Ω = (∇ψ(t) * D) / Σnoise
        """
        try:
            base_omega = PhaseOmega.compute(gradient, drift)
            return base_omega / max(noise_factor, 1e-6)  # Prevent division by zero
        except Exception as e:
            logger.error(f"Stable Phase Omega computation error: {e}")
            return 0.0

class SignalPsi:
    """
    Signal Psi operations (ψ).
    Handles signal field operations and state management.
    """
    
    @staticmethod
    def extract_field(signal_data: Union[List, np.ndarray], entropy_index: int) -> SignalField:
        """
        Extract signal field at specific entropy index.
        
        Symbolic: ψ = signal_field[entropy_index]
        Cursor-friendly: SignalPsi.extract_field(signal_data, entropy_index)
        """
        try:
            if isinstance(signal_data, list):
                signal_array = np.array(signal_data)
            else:
                signal_array = signal_data
                
            if entropy_index < len(signal_array):
                return signal_array[entropy_index:entropy_index+1]
            else:
                return signal_array[-1:] if len(signal_array) > 0 else np.array([0.0])
        except Exception as e:
            logger.error(f"Signal field extraction error: {e}")
            return np.array([0.0])
    
    @staticmethod
    def compute_entropy_weight(signal_field: SignalField, time_idx: TimeIndex) -> EntropyWeight:
        """
        Compute entropy weight for signal field.
        
        Symbolic: λ = entropy_weight(ψ, t)
        """
        try:
            if len(signal_field) == 0:
                return 0.0
                
            # Shannon entropy approximation
            unique_values, counts = np.unique(signal_field, return_counts=True)
            probabilities = counts / len(signal_field)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return float(entropy)
        except Exception as e:
            logger.error(f"Entropy weight computation error: {e}")
            return 0.0

class DriftField:
    """
    Drift field operations (D).
    Handles drift coefficient computation and management.
    """
    
    @staticmethod
    def compute_drift(signal_history: List[float], context: Optional[SymbolicContext] = None) -> DriftCoefficient:
        """
        Compute drift coefficient from signal history.
        
        Symbolic: D = drift(signal_history)
        Cursor-friendly: DriftField.compute_drift(signal_history)
        """
        try:
            if len(signal_history) < 2:
                return 0.1  # Default drift
            
            # Compute drift as rate of change
            signal_array = np.array(signal_history)
            drift = np.mean(np.diff(signal_array)) / max(np.std(signal_array), 1e-6)
            
            # Apply context modifications
            if context and context.vault_state == 'phantom':
                drift *= 1.5  # Phantom state drift boost
                
            return float(np.clip(drift, 0.001, 0.25))  # Clamp to reasonable range
        except Exception as e:
            logger.error(f"Drift computation error: {e}")
            return 0.1

class NoiseField:
    """
    Noise field operations (Σ).
    Handles noise computation and filtering.
    """
    
    @staticmethod
    def sum_noise(signal_field: SignalField) -> float:
        """
        Compute noise sum for signal field.
        
        Symbolic: Σnoise = sum_noise(ψ)
        Cursor-friendly: NoiseField.sum_noise(signal_field)
        """
        try:
            if len(signal_field) == 0:
                return 1.0
                
            # Compute noise as standard deviation
            noise = np.std(signal_field)
            return float(max(noise, 1e-6))  # Prevent zero noise
        except Exception as e:
            logger.error(f"Noise computation error: {e}")
            return 1.0

class SymbolicMathEngine:
    """
    Main symbolic math engine that coordinates all operations.
    Provides unified interface for complex mathematical computations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'hardware_preference': 'auto',  # 'cpu', 'gpu', 'auto'
            'enable_phantom_boost': True,
            'enable_context_awareness': True,
            'max_iterations': 100,
            'convergence_threshold': 1e-6,
        }
    
    def _initialize_system(self) -> None:
        """Initialize the symbolic math engine."""
        try:
            self.logger.info("Initializing SymbolicMathEngine")
            self.initialized = True
            self.active = True
            self.logger.info("✅ SymbolicMathEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing SymbolicMathEngine: {e}")
            self.initialized = False
    
    def compute_phase_omega(self, signal_data: Union[List, np.ndarray], time_idx: TimeIndex, 
                          context: Optional[SymbolicContext] = None) -> PhaseValue:
        """
        Complete phase omega computation pipeline.
        
        Symbolic: Ω = ∇ψ(t) * D
        Cursor-friendly: engine.compute_phase_omega(signal_data, time_idx, context)
        """
        try:
            # Extract signal field
            signal_field = SignalPsi.extract_field(signal_data, 
                                                 context.entropy_index if context else 0)
            
            # Compute gradient
            gradient = EntropicGradient.derive_with_context(signal_field, time_idx, context)
            
            # Compute drift
            if isinstance(signal_data, list):
                drift = DriftField.compute_drift(signal_data, context)
            else:
                drift = 0.1  # Default drift for numpy arrays
            
            # Compute noise factor
            noise_factor = NoiseField.sum_noise(signal_field)
            
            # Compute final phase omega
            omega = PhaseOmega.compute_stable(gradient, drift, noise_factor)
            
            return omega
            
        except Exception as e:
            self.logger.error(f"Phase omega computation error: {e}")
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
        }

# Factory function for easy instantiation
def create_symbolic_math_engine(config: Optional[Dict[str, Any]] = None) -> SymbolicMathEngine:
    """Create a symbolic math engine instance."""
    return SymbolicMathEngine(config)

# Global instance for easy access
symbolic_math_engine = SymbolicMathEngine() 