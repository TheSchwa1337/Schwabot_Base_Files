"""
Anti-Pole Vector Implementation
===============================

Core mathematical engine for Anti-Pole Theory calculations.
Implements inverse drift-shell gradients and ICAP probability calculations.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class AntiPoleConfig:
    """Configuration for Anti-Pole Vector calculations"""
    mu_c: float = 0.015              # Cool-state mean threshold
    sigma_c: float = 0.007           # Cool-state standard deviation
    tau_icap: float = 0.65           # ICAP activation threshold
    epsilon: float = 1e-9            # Division by zero protection
    hash_window: int = 256           # SHA-256 entropy window
    thermal_decay: float = 0.95      # Thermal decay coefficient
    profit_amplification: float = 1.2  # Profit signal amplification
    recursion_depth: int = 8         # Maximum recursion depth for stability

@dataclass
class AntiPoleState:
    """Current state of anti-pole calculations"""
    timestamp: datetime
    delta_psi_bar: float
    icap_probability: float
    hash_entropy: float
    thermal_coefficient: float
    is_ready: bool
    profit_tier: Optional[str] = None
    phase_lock: bool = False
    recursion_stability: float = 1.0

class AntiPoleVector:
    """
    Core Anti-Pole Vector Calculator
    
    Implements the mathematical foundation:
    Δ̄Ψᵢ = ∇ₜ[1/(Hₙ+ε)] ⊗ (1-Λᵢ(t))
    P̄(χ) = e^(-Δ̄Ψᵢ) · (1-Fₖ(t))
    """
    
    def __init__(self, config: Optional[AntiPoleConfig] = None):
        self.config = config or AntiPoleConfig()
        self.entropy_buffer = np.zeros(self.config.hash_window)
        self.lambda_buffer = np.zeros(64)  # Strategy activation history
        self.f_k_buffer = np.zeros(32)     # UFS family echo scores
        self.buffer_index = 0
        self.last_hash_entropy = 0.0
        self.recursion_count = 0
        
        # Statistical tracking for adaptive thresholds
        self.delta_psi_history = []
        self.icap_history = []
        
        logger.info(f"AntiPoleVector initialized with config: {self.config}")

    def update_hash_entropy(self, btc_price: float, volume: float, timestamp: float) -> float:
        """
        Calculate normalized hash entropy from BTC market data
        Uses SHA-256 based entropy calculation for price/volume correlation
        """
        # Create hash input from market data
        hash_input = f"{btc_price:.8f}:{volume:.2f}:{timestamp:.6f}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        
        # Convert to normalized entropy (0-1 scale)
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        normalized_entropy = (hash_int % 1000000) / 1000000.0
        
        # Update circular buffer
        self.entropy_buffer[self.buffer_index % self.config.hash_window] = normalized_entropy
        self.buffer_index += 1
        
        # Calculate smoothed entropy
        if self.buffer_index >= 2:
            current_entropy = np.mean(self.entropy_buffer[-20:])  # 20-period moving average
            self.last_hash_entropy = current_entropy
            return current_entropy
        
        return normalized_entropy

    def calculate_inverse_drift_gradient(self, hash_entropy: float, lambda_i: float) -> float:
        """
        Calculate Inverse Drift-Shell Gradient: Δ̄Ψᵢ = ∇ₜ[1/(Hₙ+ε)] ⊗ (1-Λᵢ(t))
        
        Args:
            hash_entropy: Normalized hash entropy Hₙ
            lambda_i: Strategy activation mask Λᵢ(t) ∈ [0,1]
        
        Returns:
            Inverse drift-shell gradient value
        """
        # Calculate inverse entropy with protection
        inverse_entropy = 1.0 / (hash_entropy + self.config.epsilon)
        
        # Calculate temporal gradient if we have history
        if len(self.delta_psi_history) > 0:
            # Approximate gradient using last calculation
            dt = 1.0  # Assuming 1-second intervals
            last_inverse = 1.0 / (self.delta_psi_history[-1] + self.config.epsilon)
            gradient = (inverse_entropy - last_inverse) / dt
        else:
            gradient = inverse_entropy
        
        # Apply strategy activation mask
        activation_factor = 1.0 - lambda_i
        
        # Calculate final drift gradient
        delta_psi_bar = gradient * activation_factor
        
        return delta_psi_bar

    def calculate_icap(self, delta_psi_bar: float, f_k: float) -> float:
        """
        Calculate Inverse Cluster Activation Probability: P̄(χ) = e^(-Δ̄Ψᵢ) · (1-Fₖ(t))
        
        Args:
            delta_psi_bar: Inverse drift-shell gradient
            f_k: UFS family echo score ∈ [0,1]
        
        Returns:
            ICAP probability value
        """
        # Exponential decay term
        exp_term = np.exp(-abs(delta_psi_bar))
        
        # Family echo complement
        echo_complement = 1.0 - f_k
        
        # ICAP calculation
        icap = exp_term * echo_complement
        
        return icap

    def check_trigger_gate(self, delta_psi_bar: float, icap: float) -> bool:
        """
        Check trigger gate condition: φₛ(t) = 1 ⟺ [Δ̄Ψᵢ > μc + σc] ∧ [P̄(χ) ≥ τᵢcₐₚ]
        """
        # Adaptive threshold calculation
        if len(self.delta_psi_history) > 10:
            # Use rolling statistics
            recent_history = self.delta_psi_history[-100:]
            mu_c = np.mean(recent_history)
            sigma_c = np.std(recent_history)
        else:
            # Use config defaults
            mu_c = self.config.mu_c
            sigma_c = self.config.sigma_c
        
        # Check both conditions
        drift_condition = delta_psi_bar > (mu_c + sigma_c)
        icap_condition = icap >= self.config.tau_icap
        
        return drift_condition and icap_condition

    def detect_profit_tier(self, delta_psi_bar: float, icap: float, 
                          btc_price: float) -> Optional[str]:
        """
        Detect profit tier based on anti-pole alignment
        """
        # Calculate profit potential score
        profit_score = delta_psi_bar * icap * self.config.profit_amplification
        
        if profit_score > 0.8:
            return "PLATINUM"
        elif profit_score > 0.6:
            return "GOLD"
        elif profit_score > 0.4:
            return "SILVER"
        elif profit_score > 0.2:
            return "BRONZE"
        else:
            return None

    def calculate_phase_lock(self, current_state: AntiPoleState) -> bool:
        """
        Calculate recursive phase lock for stability detection
        """
        if self.recursion_count >= self.config.recursion_depth:
            return False
        
        # Phase lock occurs when ICAP and drift are both stable
        if len(self.icap_history) >= 5:
            icap_stability = np.std(self.icap_history[-5:])
            drift_stability = np.std(self.delta_psi_history[-5:])
            
            return icap_stability < 0.05 and drift_stability < 0.02
        
        return False

    def process_tick(self, btc_price: float, volume: float, 
                    lambda_i: float = 0.0, f_k: float = 0.0) -> AntiPoleState:
        """
        Process a single tick of market data through Anti-Pole calculations
        
        Args:
            btc_price: Current BTC price
            volume: Current trading volume
            lambda_i: Strategy activation mask (0=cold, 1=hot)
            f_k: UFS family echo score
        
        Returns:
            Complete Anti-Pole state for this tick
        """
        timestamp = datetime.now()
        
        # Step 1: Update hash entropy
        hash_entropy = self.update_hash_entropy(btc_price, volume, timestamp.timestamp())
        
        # Step 2: Calculate inverse drift gradient
        delta_psi_bar = self.calculate_inverse_drift_gradient(hash_entropy, lambda_i)
        
        # Step 3: Calculate ICAP
        icap_probability = self.calculate_icap(delta_psi_bar, f_k)
        
        # Step 4: Check trigger gate
        is_ready = self.check_trigger_gate(delta_psi_bar, icap_probability)
        
        # Step 5: Detect profit tier
        profit_tier = self.detect_profit_tier(delta_psi_bar, icap_probability, btc_price)
        
        # Step 6: Create state object
        state = AntiPoleState(
            timestamp=timestamp,
            delta_psi_bar=delta_psi_bar,
            icap_probability=icap_probability,
            hash_entropy=hash_entropy,
            thermal_coefficient=self.config.thermal_decay,
            is_ready=is_ready,
            profit_tier=profit_tier
        )
        
        # Step 7: Calculate phase lock
        state.phase_lock = self.calculate_phase_lock(state)
        
        # Step 8: Update history buffers
        self.delta_psi_history.append(delta_psi_bar)
        self.icap_history.append(icap_probability)
        
        # Keep buffers manageable
        if len(self.delta_psi_history) > 1000:
            self.delta_psi_history = self.delta_psi_history[-500:]
        if len(self.icap_history) > 1000:
            self.icap_history = self.icap_history[-500:]
        
        # Step 9: Log significant events
        if is_ready:
            logger.info(f"🔥 Anti-Pole READY: Tier={profit_tier}, ICAP={icap_probability:.3f}, "
                       f"Drift={delta_psi_bar:.6f}, Phase={state.phase_lock}")
        
        return state

    def get_statistics(self) -> Dict:
        """Get current anti-pole statistics for monitoring"""
        return {
            'buffer_fill': min(self.buffer_index, self.config.hash_window),
            'entropy_mean': np.mean(self.entropy_buffer) if self.buffer_index > 0 else 0,
            'entropy_std': np.std(self.entropy_buffer) if self.buffer_index > 1 else 0,
            'delta_psi_mean': np.mean(self.delta_psi_history) if self.delta_psi_history else 0,
            'icap_mean': np.mean(self.icap_history) if self.icap_history else 0,
            'ready_rate': sum(1 for h in self.delta_psi_history[-100:] 
                             if h > self.config.mu_c + self.config.sigma_c) / 
                         min(100, len(self.delta_psi_history)) if self.delta_psi_history else 0,
            'recursion_count': self.recursion_count,
            'last_hash_entropy': self.last_hash_entropy
        } 