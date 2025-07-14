#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Logic Engine for Schwabot
=============================
Implements all core mathematical behaviors for recursive, immune-inspired, memory-driven trading.
"""

import hashlib
import math
from typing import List, Tuple

import numpy as np


# 1. Unified Entropy Drift Function (𝓓(t))
def entropy_drift(psi: np.ndarray, phi: np.ndarray, xi: np.ndarray, n: int = 8) -> float:
    """Compute unified entropy drift function.
    
    Args:
        psi: Main signal array (e.g., price, pressure)
        phi: Phi signal array
        xi: Xi signal array
        n: Window size for std/gradient
        
    Returns:
        Drift value (float)
        
    Raises:
        ValueError: If inputs are invalid or arrays are empty
    """
    # Input validation
    if not isinstance(psi, np.ndarray) or not isinstance(phi, np.ndarray) or not isinstance(xi, np.ndarray):
        raise ValueError("All inputs must be numpy arrays")
    
    if len(psi) == 0 or len(phi) == 0 or len(xi) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if n <= 0:
        raise ValueError("Window size must be positive")
    
    if len(psi) < n:
        return 0.0
    
    try:
        std_val = np.std(psi[-n:])
        grad_val = np.gradient(psi)[-1]
        phi_mean = np.mean(phi[-n:])
        xi_mean = np.mean(xi[-n:])
        
        # Numerical stability check
        if xi_mean <= 0:
            xi_mean = 1e-8
        
        drift = std_val * grad_val - np.log((phi_mean + 1e-8) / xi_mean)
        return float(drift)
    except Exception as e:
        raise ValueError(f"Error in entropy drift calculation: {e}")


# 2. Cross-Asset Drift Chain Weight Matrix (W_{A→B})
def drift_chain_weight(omega_a: np.ndarray, omega_b: np.ndarray, delta_t: int,
                      roi_weight: float, xi_score: float) -> float:
    """Compute cross-asset drift chain weight.
    
    Args:
        omega_a: Source asset omega array
        omega_b: Target asset omega array
        delta_t: Tick offset
        roi_weight: ROI weight (float)
        xi_score: Xi score (float)
        
    Returns:
        Weight (float)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(omega_a, np.ndarray) or not isinstance(omega_b, np.ndarray):
        raise ValueError("Omega arrays must be numpy arrays")
    
    if len(omega_b) <= abs(delta_t):
        return 0.0
    
    try:
        if delta_t >= 0:
            b_shifted = omega_b[delta_t:]
            a_trunc = omega_a[:len(b_shifted)]
        else:
            a_trunc = omega_a[-delta_t:]
            b_shifted = omega_b[:len(a_trunc)]
        
        if len(a_trunc) == 0 or len(b_shifted) == 0:
            return 0.0
        
        corr = np.corrcoef(a_trunc, b_shifted)[0, 1]
        if np.isnan(corr):
            return 0.0
        
        return float(corr * roi_weight * xi_score)
    except Exception as e:
        raise ValueError(f"Error in drift chain weight calculation: {e}")


# 3. Recursive Vault Re-entry Delay Function (T_delay)
def vault_reentry_delay(xi_exit: float, phi_entry: float, vault_mass: float,
                       tick_entropy: float) -> int:
    """Compute vault re-entry delay (in ticks)."""
    if phi_entry == 0 or tick_entropy <= 0:
        return 1
    
    try:
        delay = math.ceil((xi_exit / phi_entry) * vault_mass * math.log(tick_entropy + 1e-8))
        return max(1, delay)
    except Exception:
        return 1


# 4. Phase Rotation Engine (θ(t))
def phase_rotation(xi: float, phi: float, omega: float, period: int = 16) -> float:
    """Compute rotational phase offset."""
    try:
        return float((xi * phi * omega) % period)
    except Exception:
        return 0.0


# 5. Vault Mass Function (𝓥(t))
def vault_mass(xi: List[float], phi: List[float], roi: List[float],
               holding_weight: List[float]) -> float:
    """Compute vault mass (pressure)."""
    if not all(isinstance(x, list) for x in [xi, phi, roi, holding_weight]):
        raise ValueError("All inputs must be lists")
    
    if len(set(len(x) for x in [xi, phi, roi, holding_weight])) != 1:
        raise ValueError("All input lists must have the same length")
    
    try:
        arr = np.array(xi) * np.array(phi) * np.array(roi) * np.array(holding_weight)
        return float(np.sum(arr))
    except Exception as e:
        raise ValueError(f"Error in vault mass calculation: {e}")


# 6. Bitmap Folding Logic
def bitmap_fold(bitmap: List[int], k: int = 3) -> int:
    """Fold bitmap signal for memory/hash compression."""
    if not isinstance(bitmap, list) or len(bitmap) < 3:
        return 0
    
    try:
        folded = bitmap[-1] ^ bitmap[-2]
        rotated = ((bitmap[-3] << k) | (bitmap[-3] >> (8 - k))) & 0xFF
        return (folded + rotated) & 0xFF
    except Exception:
        return 0


# 7. ΦΞ Orbital Energy Quantization
def orbital_energy(omega: float, phi: float, xi: float) -> Tuple[float, str]:
    """Compute orbital energy and classify orbital state.
    
    Returns:
        (energy, state) where state is 's', 'p', 'd', or 'f'.
    """
    try:
        energy = (omega ** 2 + phi) * math.log(xi + 1e-8)
        
        if energy < 0.3:
            state = 's'
        elif energy < 0.7:
            state = 'p'
        elif energy < 1.1:
            state = 'd'
        else:
            state = 'f'
        
        return float(energy), state
    except Exception:
        return 0.0, 's'


# 8. Strategy DNA Recombination Hash Tree
def strategy_hash_evolution(prev_hash: str, delta_roi: float,
                          entropy_deviation: float) -> str:
    """Evolve strategy hash using SHA256."""
    try:
        seed = f"{prev_hash}_{delta_roi:.6f}_{entropy_deviation:.6f}".encode()
        return hashlib.sha256(seed).hexdigest()
    except Exception:
        return prev_hash


# 9. Clonal Expansion Coefficient (𝓒)
def clonal_expansion_coefficient(tcell_activation: float, roi: float,
                               xi_weight: float) -> float:
    """Compute clonal expansion coefficient."""
    try:
        return float(tcell_activation * roi * xi_weight)
    except Exception:
        return 0.0


# 10. Strategy Hash Mutation Rate Function (μ)
def mutation_rate(roi: float, phi: float, volatility: float) -> float:
    """Compute mutation rate for strategy hashes."""
    try:
        return float((1 - roi) * (1 - phi) * volatility)
    except Exception:
        return 0.0


# 11. Rebuy Probability Function (𝓟_rebuy)
def rebuy_probability(omega: float, xi: float, phi: float,
                     vault_pressure: float) -> float:
    """Compute rebuy probability (sigmoid)."""
    try:
        x = omega + xi + phi - vault_pressure
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0


# 12. Hash Priority Score (HPS)
def hash_priority_score(roi: float, clonal_coeff: float, xi_weight: float,
                       asset_drift_alignment: float) -> float:
    """Compute hash priority score for strategy selection."""
    try:
        return float(roi * clonal_coeff * xi_weight * asset_drift_alignment)
    except Exception:
        return 0.0


# 13. Echo Trigger Zone (𝓔𝓣𝓩)
def echo_trigger_zone(xi_score: float, phi_score: float, omega: float,
                     omega_mean: float) -> bool:
    """Detect echo/warp trigger zone."""
    try:
        return xi_score > 0.9 and phi_score > 0.85 and omega > omega_mean
    except Exception:
        return False


# --- ENTRY/EXIT LOGIC WRAPPERS ---
def should_enter(tcell_activation: float, clonal_coeff: float, rebuy_prob: float,
                echo_zone: bool) -> bool:
    """Decide if Schwabot should enter a position."""
    try:
        return (tcell_activation > 0.76 and clonal_coeff > 0.65 and
                rebuy_prob > 0.84 and echo_zone)
    except Exception:
        return False


def should_exit(tcell_activation: float, clonal_coeff: float, rebuy_prob: float,
               echo_zone: bool) -> bool:
    """Decide if Schwabot should exit a position."""
    try:
        return (tcell_activation < 0.3 or clonal_coeff < 0.3 or
                rebuy_prob < 0.2 or not echo_zone)
    except Exception:
        return True


# --- UTILITY: Sigmoid ---
def sigmoid(x: float) -> float:
    """Compute sigmoid function."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0 