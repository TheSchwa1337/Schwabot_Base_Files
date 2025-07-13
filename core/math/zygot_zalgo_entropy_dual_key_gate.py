#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zygot-Zalgo Entropy Dual Key Gate - Zygotic Hash Duality & Zalgo Field Access

Implements Nexus mathematics for recursive security lock + prediction logic:
- Dual Key Collapse Gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
- Hash Echo Mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
- Forms recursive security lock + prediction logic layer using 2-key entropy mirroring
- Essential for Ferris-Wheel profit-layer hashing
- Zygot/Zalgo keys introduced in memory points 34–35, referencing access control and AI echo field triggers
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, optimize, stats

logger = logging.getLogger(__name__)


class GateState(Enum):
    """Zygot-Zalgo gate states."""
    CLOSED = "closed"           # Gate is closed
    OPEN = "open"              # Gate is open
    PARTIAL = "partial"        # Partial access
    CONFLICTED = "conflicted"  # Key conflict detected
    RESET = "reset"            # Gate reset required


class KeyType(Enum):
    """Key types for dual-key system."""
    ZYGOT = "zygot"    # Recursive root entropy stabilizer
    ZALGO = "zalgo"    # Phase inversion entropy spike detector


@dataclass
class DualKeyResult:
    """Result of dual-key gate evaluation."""
    timestamp: float
    gate_state: GateState
    zygot_entropy: float
    zalgo_entropy: float
    combined_entropy: float
    access_granted: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashEchoResult:
    """Result of hash echo mirroring."""
    timestamp: float
    volume_hash: str
    momentum_hash: str
    echo_hash: str
    echo_strength: float
    phase_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZygotZalgoEntropyDualKeyGate:
    """
    Zygot-Zalgo Entropy Dual Key Gate - Zygotic Hash Duality & Zalgo Field Access
    
    Implements the Nexus mathematics for recursive security lock + prediction logic:
    - Dual Key Collapse Gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
    - Hash Echo Mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
    - Forms recursive security lock + prediction logic layer using 2-key entropy mirroring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Zygot-Zalgo Entropy Dual Key Gate."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        
        # Gate parameters
        self.zygot_entropy_threshold = self.config.get('zygot_entropy_threshold', 0.3)
        self.zalgo_entropy_threshold = self.config.get('zalgo_entropy_threshold', 0.3)
        self.combined_threshold = self.config.get('combined_threshold', 0.5)
        self.adaptive_thresholding = self.config.get('adaptive_thresholding', True)
        
        # Key parameters
        self.alpha_coeff = self.config.get('alpha_coeff', 1.0)
        self.beta_coeff = self.config.get('beta_coeff', 1.0)
        self.r2_radius = self.config.get('r2_radius', 2.0)
        
        # Key storage
        self.zygot_key = self._generate_key()
        self.zalgo_key = self._generate_key()
        
        # State tracking
        self.gate_state = GateState.CLOSED
        self.evaluation_history: List[DualKeyResult] = []
        self.echo_history: List[HashEchoResult] = []
        
        self._initialize_gate()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Zygot-Zalgo Entropy Dual Key Gate."""
        return {
            'zygot_entropy_threshold': 0.3,  # Zygot entropy threshold
            'zalgo_entropy_threshold': 0.3,  # Zalgo entropy threshold
            'combined_threshold': 0.5,       # Combined entropy threshold
            'adaptive_thresholding': True,   # Adaptive thresholding
            'alpha_coeff': 1.0,             # Alpha coefficient for dual key
            'beta_coeff': 1.0,              # Beta coefficient for dual key
            'r2_radius': 2.0,               # R² radius for entropy calculation
            'echo_threshold': 0.7,          # Echo strength threshold
            'phase_threshold': 0.8,         # Phase alignment threshold
        }
    
    def _generate_key(self) -> str:
        """Generate a random key for the dual-key system."""
        return hashlib.sha256(f"{time.time()}{np.random.random()}".encode()).hexdigest()
    
    def _initialize_gate(self):
        """Initialize the dual-key gate."""
        try:
            self.logger.info("Initializing Zygot-Zalgo Entropy Dual Key Gate...")
            
            # Validate thresholds
            if not (0.0 <= self.zygot_entropy_threshold <= 1.0):
                raise ValueError("zygot_entropy_threshold must be between 0.0 and 1.0")
            if not (0.0 <= self.zalgo_entropy_threshold <= 1.0):
                raise ValueError("zalgo_entropy_threshold must be between 0.0 and 1.0")
            
            # Initialize SHA256 contexts
            self.zygot_sha256 = hashlib.sha256()
            self.zalgo_sha256 = hashlib.sha256()
            
            self.initialized = True
            self.logger.info("[SUCCESS] Zygot-Zalgo Entropy Dual Key Gate initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error initializing Zygot-Zalgo Entropy Dual Key Gate: {e}")
            self.initialized = False
    
    def compute_dual_key_collapse_gate(self, data_vector: np.ndarray) -> float:
        """
        Compute dual key collapse gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
        
        Args:
            data_vector: Input data vector
            
        Returns:
            Dual key collapse gate entropy value
        """
        try:
            # Define R² region: R² = {ψ | κ(ψ) = 2}
            # For simplicity, we'll use a circular region with radius r2_radius
            r2_radius = self.r2_radius
            
            # Generate ψ values in R² region
            theta_values = np.linspace(0, 2 * np.pi, 100)
            r_values = np.linspace(0, r2_radius, 50)
            
            # Initialize entropy sum
            zk_entropy = 0.0
            
            # Compute Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
            for r in r_values:
                for theta in theta_values:
                    # ψ = r * e^(i*theta) in complex form
                    psi_real = r * np.cos(theta)
                    psi_imag = r * np.imag(theta)
                    
                    # Compute α⋅cos(ψ) + β⋅sin(ψ)
                    cos_component = self.alpha_coeff * np.cos(psi_real)
                    sin_component = self.beta_coeff * np.sin(psi_imag)
                    
                    # Add to entropy sum
                    zk_entropy += cos_component + sin_component
            
            # Normalize by number of points
            total_points = len(r_values) * len(theta_values)
            zk_entropy /= total_points
            
            return zk_entropy
            
        except Exception as e:
            self.logger.error(f"Error computing dual key collapse gate: {e}")
            return 0.0
    
    def compute_hash_echo_mirror(self, volume_data: np.ndarray, 
                               momentum_data: np.ndarray) -> HashEchoResult:
        """
        Compute hash echo mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
        
        Args:
            volume_data: Volume data array
            momentum_data: Momentum data array
            
        Returns:
            Hash echo mirror result
        """
        try:
            # Compute volume hash: SHA256(H_volume)
            volume_str = f"{np.mean(volume_data):.6f}{np.std(volume_data):.6f}{len(volume_data)}"
            self.zygot_sha256.update(volume_str.encode())
            volume_hash = self.zygot_sha256.hexdigest()
            
            # Compute momentum hash: SHA256(H_momentum)
            momentum_str = f"{np.mean(momentum_data):.6f}{np.std(momentum_data):.6f}{len(momentum_data)}"
            self.zalgo_sha256.update(momentum_str.encode())
            momentum_hash = self.zalgo_sha256.hexdigest()
            
            # Compute XOR of hashes: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
            volume_int = int(volume_hash[:16], 16)
            momentum_int = int(momentum_hash[:16], 16)
            echo_int = volume_int ^ momentum_int
            echo_hash = f"{echo_int:016x}"
            
            # Compute echo strength (normalized)
            echo_strength = (echo_int / (16 ** 16)) % 1.0
            
            # Compute phase alignment between volume and momentum
            volume_phase = np.angle(np.mean(volume_data) + 1j * np.std(volume_data))
            momentum_phase = np.angle(np.mean(momentum_data) + 1j * np.std(momentum_data))
            phase_alignment = np.cos(volume_phase - momentum_phase)
            
            return HashEchoResult(
                timestamp=time.time(),
                volume_hash=volume_hash,
                momentum_hash=momentum_hash,
                echo_hash=echo_hash,
                echo_strength=echo_strength,
                phase_alignment=phase_alignment
            )
            
        except Exception as e:
            self.logger.error(f"Error computing hash echo mirror: {e}")
            return HashEchoResult(
                timestamp=time.time(),
                volume_hash="",
                momentum_hash="",
                echo_hash="",
                echo_strength=0.0,
                phase_alignment=0.0
            )
    
    def evaluate_zygot_entropy(self, data: np.ndarray) -> float:
        """
        Evaluate Zygot entropy (recursive root entropy stabilizer).
        
        Args:
            data: Input data array
            
        Returns:
            Zygot entropy value
        """
        try:
            # Compute Shannon entropy for Zygot
            if len(data) == 0:
                return 0.0
            
            # Normalize data to [0, 1]
            data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Compute histogram
            hist, _ = np.histogram(data_normalized, bins=min(20, len(data)), range=(0, 1))
            hist = hist / np.sum(hist)
            
            # Compute Shannon entropy: S = -Σ pᵢ log(pᵢ)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            
            # Normalize to [0, 1]
            max_entropy = np.log2(len(hist))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception as e:
            self.logger.error(f"Error evaluating Zygot entropy: {e}")
            return 0.0
    
    def evaluate_zalgo_entropy(self, data: np.ndarray) -> float:
        """
        Evaluate Zalgo entropy (phase inversion entropy spike detector).
        
        Args:
            data: Input data array
            
        Returns:
            Zalgo entropy value
        """
        try:
            if len(data) < 2:
                return 0.0
            
            # Compute phase differences
            phase_diffs = np.diff(data)
            
            # Detect spikes (sudden changes)
            spike_threshold = np.std(phase_diffs) * 2
            spikes = np.abs(phase_diffs) > spike_threshold
            
            # Compute spike entropy
            spike_ratio = np.sum(spikes) / len(spikes)
            
            # Compute phase inversion entropy
            # Higher entropy when there are many phase inversions
            inversions = np.sum(np.diff(np.sign(phase_diffs)) != 0)
            inversion_entropy = inversions / len(phase_diffs)
            
            # Combine spike and inversion entropy
            zalgo_entropy = (spike_ratio + inversion_entropy) / 2.0
            
            return zalgo_entropy
            
        except Exception as e:
            self.logger.error(f"Error evaluating Zalgo entropy: {e}")
            return 0.0
    
    def evaluate_dual_key_gate(self, volume_data: np.ndarray, 
                             momentum_data: np.ndarray) -> DualKeyResult:
        """
        Evaluate the dual-key gate with volume and momentum data.
        
        Args:
            volume_data: Volume data array
            momentum_data: Momentum data array
            
        Returns:
            Dual key evaluation result
        """
        try:
            # Evaluate Zygot entropy
            zygot_entropy = self.evaluate_zygot_entropy(volume_data)
            
            # Evaluate Zalgo entropy
            zalgo_entropy = self.evaluate_zalgo_entropy(momentum_data)
            
            # Compute dual key collapse gate
            combined_data = np.concatenate([volume_data, momentum_data])
            dual_key_entropy = self.compute_dual_key_collapse_gate(combined_data)
            
            # Compute combined entropy
            combined_entropy = (zygot_entropy + zalgo_entropy + dual_key_entropy) / 3.0
            
            # Determine gate state
            if (zygot_entropy >= self.zygot_entropy_threshold and 
                zalgo_entropy >= self.zalgo_entropy_threshold and
                combined_entropy >= self.combined_threshold):
                gate_state = GateState.OPEN
                access_granted = True
            elif (zygot_entropy >= self.zygot_entropy_threshold or 
                  zalgo_entropy >= self.zalgo_entropy_threshold):
                gate_state = GateState.PARTIAL
                access_granted = False
            elif abs(zygot_entropy - zalgo_entropy) > 0.5:
                gate_state = GateState.CONFLICTED
                access_granted = False
            else:
                gate_state = GateState.CLOSED
                access_granted = False
            
            # Compute confidence
            confidence = min(1.0, combined_entropy)
            
            # Create result
            result = DualKeyResult(
                timestamp=time.time(),
                gate_state=gate_state,
                zygot_entropy=zygot_entropy,
                zalgo_entropy=zalgo_entropy,
                combined_entropy=combined_entropy,
                access_granted=access_granted,
                confidence=confidence
            )
            
            # Store result
            self.evaluation_history.append(result)
            self.gate_state = gate_state
            
            # Keep history manageable
            max_history = 1000
            if len(self.evaluation_history) > max_history:
                self.evaluation_history = self.evaluation_history[-max_history:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating dual key gate: {e}")
            return DualKeyResult(
                timestamp=time.time(),
                gate_state=GateState.CLOSED,
                zygot_entropy=0.0,
                zalgo_entropy=0.0,
                combined_entropy=0.0,
                access_granted=False,
                confidence=0.0
            )
    
    def get_gate_summary(self) -> Dict[str, Any]:
        """Get comprehensive gate summary."""
        if not self.evaluation_history:
            return {'status': 'no_evaluations'}
        
        # Compute gate statistics
        zygot_entropies = [r.zygot_entropy for r in self.evaluation_history]
        zalgo_entropies = [r.zalgo_entropy for r in self.evaluation_history]
        combined_entropies = [r.combined_entropy for r in self.evaluation_history]
        confidences = [r.confidence for r in self.evaluation_history]
        
        # Count gate states
        state_counts = {}
        for state in GateState:
            state_counts[state.value] = sum(1 for r in self.evaluation_history if r.gate_state == state)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'current_gate_state': self.gate_state.value,
            'mean_zygot_entropy': np.mean(zygot_entropies),
            'mean_zalgo_entropy': np.mean(zalgo_entropies),
            'mean_combined_entropy': np.mean(combined_entropies),
            'mean_confidence': np.mean(confidences),
            'state_distribution': state_counts,
            'access_granted_ratio': sum(1 for r in self.evaluation_history if r.access_granted) / len(self.evaluation_history),
            'initialized': self.initialized
        }
    
    def reset_gate(self):
        """Reset the dual-key gate."""
        self.gate_state = GateState.RESET
        self.evaluation_history.clear()
        self.echo_history.clear()
        self.zygot_key = self._generate_key()
        self.zalgo_key = self._generate_key()
        self.logger.info("Zygot-Zalgo Entropy Dual Key Gate reset")


# Factory function
def create_zygot_zalgo_entropy_dual_key_gate(config: Optional[Dict[str, Any]] = None) -> ZygotZalgoEntropyDualKeyGate:
    """Create a Zygot-Zalgo Entropy Dual Key Gate instance."""
    return ZygotZalgoEntropyDualKeyGate(config)