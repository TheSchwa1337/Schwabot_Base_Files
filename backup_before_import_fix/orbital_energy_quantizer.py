#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbital Energy Quantizer for Schwabot
=====================================
Converts orbital states (s,p,d,f) into quantized energy values:
• orbital_energy(Ω,Ξ,Φ) = (Ω² + Φ) * log(Ξ + 1e-6)
• Energy thresholds for orbital classification
• Phase depth calculations
• Quantum-inspired orbital transitions
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class OrbitalState(Enum):
    """Orbital states with energy thresholds."""
    S = "s"      # Sharp/stable: < 0.3
    P = "p"      # Phase/periodic: 0.3-0.7
    D = "d"      # Drift/diffuse: 0.7-1.1
    F = "f"      # Fast/volatile: > 1.1
    NULL = "null" # No orbital: > 2.0

@dataclass
class OrbitalEnergyResult:
    """Result of orbital energy calculation."""
    energy_value: float
    orbital_state: OrbitalState
    phase_depth: float
    quantum_coherence: float
    transition_probability: float
    confidence: float

class OrbitalEnergyQuantizer:
    """Quantizes orbital states into energy values."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize orbital energy quantizer."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Energy thresholds for orbital states
        self.energy_thresholds = self._build_energy_thresholds()
        
        # Transition probabilities between orbitals
        self.transition_matrix = self._build_transition_matrix()
        
        # Historical energy tracking
        self.energy_history: List[Tuple[float, OrbitalState, float]] = []  # (energy, state, time)
        
        self.logger.info("✅ Orbital Energy Quantizer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'energy_thresholds': {
                's_max': 0.3,
                'p_min': 0.3,
                'p_max': 0.7,
                'd_min': 0.7,
                'd_max': 1.1,
                'f_min': 1.1,
                'f_max': 2.0,
                'null_min': 2.0
            },
            'phase_depth_weight': 0.4,
            'quantum_coherence_weight': 0.3,
            'transition_smoothing': 0.1,
            'enable_quantum_effects': True,
            'coherence_decay_rate': 0.05,
            'max_phase_depth': 1.0
        }
    
    def _build_energy_thresholds(self) -> Dict[OrbitalState, Tuple[float, float]]:
        """Build energy thresholds for each orbital state."""
        thresholds = self.config['energy_thresholds']
        
        return {
            OrbitalState.S: (0.0, thresholds['s_max']),
            OrbitalState.P: (thresholds['p_min'], thresholds['p_max']),
            OrbitalState.D: (thresholds['d_min'], thresholds['d_max']),
            OrbitalState.F: (thresholds['f_min'], thresholds['f_max']),
            OrbitalState.NULL: (thresholds['null_min'], float('inf'))
        }
    
    def _build_transition_matrix(self) -> Dict[OrbitalState, Dict[OrbitalState, float]]:
        """Build transition probability matrix between orbitals."""
        return {
            OrbitalState.S: {
                OrbitalState.S: 0.8,    # Stay in S
                OrbitalState.P: 0.15,   # S → P
                OrbitalState.D: 0.04,   # S → D
                OrbitalState.F: 0.01,   # S → F
                OrbitalState.NULL: 0.0  # S → NULL
            },
            OrbitalState.P: {
                OrbitalState.S: 0.1,    # P → S
                OrbitalState.P: 0.7,    # Stay in P
                OrbitalState.D: 0.15,   # P → D
                OrbitalState.F: 0.04,   # P → F
                OrbitalState.NULL: 0.01 # P → NULL
            },
            OrbitalState.D: {
                OrbitalState.S: 0.05,   # D → S
                OrbitalState.P: 0.1,    # D → P
                OrbitalState.D: 0.6,    # Stay in D
                OrbitalState.F: 0.2,    # D → F
                OrbitalState.NULL: 0.05 # D → NULL
            },
            OrbitalState.F: {
                OrbitalState.S: 0.02,   # F → S
                OrbitalState.P: 0.05,   # F → P
                OrbitalState.D: 0.15,   # F → D
                OrbitalState.F: 0.5,    # Stay in F
                OrbitalState.NULL: 0.28 # F → NULL
            },
            OrbitalState.NULL: {
                OrbitalState.S: 0.3,    # NULL → S
                OrbitalState.P: 0.3,    # NULL → P
                OrbitalState.D: 0.2,    # NULL → D
                OrbitalState.F: 0.15,   # NULL → F
                OrbitalState.NULL: 0.05 # Stay in NULL
            }
        }
    
    def calculate_orbital_energy(self, omega_values: np.ndarray, 
                               phi_values: np.ndarray,
                               xi_values: np.ndarray,
                               current_time: float = None) -> OrbitalEnergyResult:
        """
        Calculate orbital energy: orbital_energy(Ω,Ξ,Φ) = (Ω² + Φ) * log(Ξ + 1e-6)
        """
        try:
            if len(omega_values) == 0 or len(phi_values) == 0 or len(xi_values) == 0:
                return self._fallback_energy_result()
            
            # Calculate means for energy calculation
            omega_mean = np.mean(omega_values)
            phi_mean = np.mean(phi_values)
            xi_mean = np.mean(xi_values)
            
            # Core orbital energy calculation
            # orbital_energy(Ω,Ξ,Φ) = (Ω² + Φ) * log(Ξ + 1e-6)
            energy_value = (omega_mean**2 + phi_mean) * np.log(xi_mean + 1e-6)
            
            # Determine orbital state based on energy
            orbital_state = self._classify_orbital_state(energy_value)
            
            # Calculate phase depth
            phase_depth = self._calculate_phase_depth(omega_values, phi_values, xi_values)
            
            # Calculate quantum coherence
            quantum_coherence = self._calculate_quantum_coherence(energy_value, phase_depth)
            
            # Calculate transition probability
            transition_probability = self._calculate_transition_probability(orbital_state)
            
            # Calculate confidence
            confidence = self._calculate_confidence(energy_value, phase_depth, quantum_coherence)
            
            # Store in history
            current_time = current_time or time.time()
            self.energy_history.append((energy_value, orbital_state, current_time))
            
            return OrbitalEnergyResult(
                energy_value=float(energy_value),
                orbital_state=orbital_state,
                phase_depth=phase_depth,
                quantum_coherence=quantum_coherence,
                transition_probability=transition_probability,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating orbital energy: {e}")
            return self._fallback_energy_result()
    
    def _classify_orbital_state(self, energy_value: float) -> OrbitalState:
        """Classify energy value into orbital state."""
        for orbital_state, (min_energy, max_energy) in self.energy_thresholds.items():
            if min_energy <= energy_value < max_energy:
                return orbital_state
        
        # Fallback to NULL if no match
        return OrbitalState.NULL
    
    def _calculate_phase_depth(self, omega_values: np.ndarray, 
                             phi_values: np.ndarray, 
                             xi_values: np.ndarray) -> float:
        """Calculate phase depth based on signal complexity."""
        try:
            # Calculate phase complexity using multiple metrics
            
            # 1. Omega-Phi correlation
            if len(omega_values) == len(phi_values) and len(omega_values) > 1:
                omega_phi_corr = abs(np.corrcoef(omega_values, phi_values)[0, 1])
                if np.isnan(omega_phi_corr):
                    omega_phi_corr = 0.0
            else:
                omega_phi_corr = 0.0
            
            # 2. Xi variability
            xi_std = np.std(xi_values) if len(xi_values) > 1 else 0.0
            
            # 3. Omega-Phi phase difference
            if len(omega_values) == len(phi_values):
                phase_diff = np.mean(np.abs(omega_values - phi_values))
            else:
                phase_diff = 0.5
            
            # Combine metrics into phase depth
            phase_depth = (omega_phi_corr * 0.4 + 
                          xi_std * 0.3 + 
                          (1 - phase_diff) * 0.3)
            
            return min(self.config['max_phase_depth'], max(0.0, phase_depth))
            
        except Exception as e:
            self.logger.error(f"Error calculating phase depth: {e}")
            return 0.5
    
    def _calculate_quantum_coherence(self, energy_value: float, phase_depth: float) -> float:
        """Calculate quantum coherence based on energy and phase depth."""
        try:
            if not self.config['enable_quantum_effects']:
                return 0.5
            
            # Quantum coherence decreases with energy and increases with phase depth
            base_coherence = 1.0 - (energy_value / 5.0)  # Normalize energy
            
            # Phase depth enhances coherence
            coherence_enhancement = phase_depth * 0.3
            
            # Apply coherence decay
            decay_rate = self.config['coherence_decay_rate']
            if self.energy_history:
                time_factor = len(self.energy_history) * decay_rate
                coherence_decay = np.exp(-time_factor)
            else:
                coherence_decay = 1.0
            
            quantum_coherence = (base_coherence + coherence_enhancement) * coherence_decay
            
            return min(1.0, max(0.0, quantum_coherence))
            
        except Exception as e:
            self.logger.error(f"Error calculating quantum coherence: {e}")
            return 0.5
    
    def _calculate_transition_probability(self, current_state: OrbitalState) -> float:
        """Calculate probability of transitioning to a different orbital state."""
        try:
            if not self.energy_history:
                return 0.5
            
            # Get transition probabilities for current state
            transitions = self.transition_matrix.get(current_state, {})
            
            # Calculate probability of staying in current state
            stay_probability = transitions.get(current_state, 0.5)
            
            # Transition probability is inverse of stay probability
            transition_probability = 1.0 - stay_probability
            
            # Apply smoothing
            smoothing = self.config['transition_smoothing']
            transition_probability = (transition_probability * (1 - smoothing) + 
                                   0.5 * smoothing)
            
            return transition_probability
            
        except Exception as e:
            self.logger.error(f"Error calculating transition probability: {e}")
            return 0.5
    
    def _calculate_confidence(self, energy_value: float, phase_depth: float,
                            quantum_coherence: float) -> float:
        """Calculate confidence in orbital energy calculation."""
        try:
            # Energy stability confidence
            energy_confidence = 1.0 - abs(energy_value - 1.0) / 5.0  # Center around 1.0
            
            # Phase depth confidence
            phase_confidence = phase_depth
            
            # Quantum coherence confidence
            coherence_confidence = quantum_coherence
            
            # Weighted average
            confidence = (energy_confidence * 0.4 + 
                         phase_confidence * 0.3 + 
                         coherence_confidence * 0.3)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def predict_orbital_transition(self, current_state: OrbitalState, 
                                 energy_trend: float) -> Tuple[OrbitalState, float]:
        """Predict next orbital state based on energy trend."""
        try:
            transitions = self.transition_matrix.get(current_state, {})
            
            # Adjust transition probabilities based on energy trend
            adjusted_transitions = {}
            for target_state, base_prob in transitions.items():
                if target_state != current_state:
                    # Higher energy trend favors higher energy orbitals
                    if energy_trend > 0:
                        if target_state.value in ['p', 'd', 'f']:
                            adjusted_prob = base_prob * (1 + abs(energy_trend) * 0.2)
                        else:
                            adjusted_prob = base_prob * (1 - abs(energy_trend) * 0.1)
                    else:
                        if target_state.value in ['s', 'p']:
                            adjusted_prob = base_prob * (1 + abs(energy_trend) * 0.2)
                        else:
                            adjusted_prob = base_prob * (1 - abs(energy_trend) * 0.1)
                    
                    adjusted_transitions[target_state] = min(1.0, max(0.0, adjusted_prob))
            
            # Find most likely transition
            if adjusted_transitions:
                most_likely_state = max(adjusted_transitions.items(), key=lambda x: x[1])
                return most_likely_state[0], most_likely_state[1]
            else:
                return current_state, 1.0
            
        except Exception as e:
            self.logger.error(f"Error predicting orbital transition: {e}")
            return current_state, 0.5
    
    def get_orbital_statistics(self) -> Dict[str, Any]:
        """Get orbital energy statistics."""
        try:
            if not self.energy_history:
                return {
                    'total_calculations': 0,
                    'orbital_distribution': {},
                    'average_energy': 0.0,
                    'energy_std': 0.0,
                    'most_common_orbital': None
                }
            
            energies = [energy for energy, _, _ in self.energy_history]
            states = [state.value for _, state, _ in self.energy_history]
            
            # Calculate orbital distribution
            orbital_distribution = {}
            for state in states:
                orbital_distribution[state] = orbital_distribution.get(state, 0) + 1
            
            # Find most common orbital
            most_common_orbital = max(orbital_distribution.items(), key=lambda x: x[1])[0] if orbital_distribution else None
            
            return {
                'total_calculations': len(self.energy_history),
                'orbital_distribution': orbital_distribution,
                'average_energy': np.mean(energies) if energies else 0.0,
                'energy_std': np.std(energies) if energies else 0.0,
                'most_common_orbital': most_common_orbital,
                'recent_energies': energies[-10:] if len(energies) >= 10 else energies
            }
            
        except Exception as e:
            self.logger.error(f"Error getting orbital statistics: {e}")
            return {}
    
    def _fallback_energy_result(self) -> OrbitalEnergyResult:
        """Return fallback energy result on error."""
        return OrbitalEnergyResult(
            energy_value=0.5,
            orbital_state=OrbitalState.S,
            phase_depth=0.5,
            quantum_coherence=0.5,
            transition_probability=0.5,
            confidence=0.0
        )

# Factory function
def create_orbital_energy_quantizer(config: Optional[Dict[str, Any]] = None) -> OrbitalEnergyQuantizer:
    """Create an orbital energy quantizer instance."""
    return OrbitalEnergyQuantizer(config) 