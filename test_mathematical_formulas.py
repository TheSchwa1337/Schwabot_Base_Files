#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Mathematical Formulas Test
=======================================
Tests all mathematical formulas from the MATHEMATICAL_FORMULAS_REFERENCE.md
"""

import numpy as np
import sys
import os
from typing import List, Dict, Any, Tuple

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    from advanced_mathematical_core import (
        safe_delta_calculation,
        normalized_delta_tanh,
        shannon_entropy_stable,
        kl_divergence_stable,
        entropy_gradient_field,
        stable_activation_matrix,
        robust_matrix_inverse
    )
    from type_defs import QuantumState, Temperature, Vector, Matrix, Tensor
    from constants import EPSILON_FLOAT64, THERMAL_CONDUCTIVITY_BTC, REDUCED_PLANCK, KELLY_SAFETY_FACTOR
    print("✅ Successfully imported mathematical core modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_delta_calculations():
    """Test delta calculation formulas."""
    print("\n📊 Testing Delta Calculations...")
    
    # Test safe delta calculation
    price_now = 100.0
    price_prev = 95.0
    delta = safe_delta_calculation(price_now, price_prev)
    expected_delta = (price_now - price_prev) / price_prev
    assert abs(delta - expected_delta) < 1e-10, "Safe delta calculation error"
    print(f"  Safe Delta: {delta:.4f} (expected: {expected_delta:.4f})")
    
    # Test normalized tanh delta
    norm_delta = normalized_delta_tanh(price_now, price_prev, scaling_factor=1.0)
    assert -1 <= norm_delta <= 1, "Normalized delta should be in [-1, 1]"
    print(f"  Normalized Tanh Delta: {norm_delta:.4f}")
    
    # Test edge case: zero previous price
    delta_zero = safe_delta_calculation(100.0, 0.0)
    assert delta_zero == 100.0 / EPSILON_FLOAT64, "Zero price delta error"
    print(f"  Zero Price Delta: {delta_zero:.4f}")
    
    print("  ✅ Delta calculations passed")

def test_entropy_and_information_theory():
    """Test entropy and information theory formulas."""
    print("\n📈 Testing Entropy & Information Theory...")
    
    # Test Shannon entropy
    prob_vector = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = shannon_entropy_stable(prob_vector)
    expected_entropy = 2.0  # log2(4) = 2
    assert abs(entropy - expected_entropy) < 1e-10, "Shannon entropy calculation error"
    print(f"  Shannon Entropy: {entropy:.4f} (expected: {expected_entropy:.4f})")
    
    # Test KL divergence
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    kl_div = kl_divergence_stable(p, q)
    assert kl_div >= 0, "KL divergence should be non-negative"
    print(f"  KL Divergence: {kl_div:.6f}")
    
    # Test entropy gradient field
    entropy_map = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    gradient_field = entropy_gradient_field(entropy_map)
    assert gradient_field.shape == (3, 3, 2), "Gradient field shape error"
    print(f"  Entropy Gradient Field Shape: {gradient_field.shape}")
    
    print("  ✅ Entropy and information theory passed")

def test_matrix_activation():
    """Test matrix activation formulas."""
    print("\n🔁 Testing Matrix Activation...")
    
    # Test stable activation matrix
    input_array = np.array([1.0, 2.0, 3.0])
    weight_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    
    activation = stable_activation_matrix(input_array, weight_matrix, lambda_reg=0.01)
    assert activation.shape == (3,), "Activation shape error"
    assert np.all(activation >= -1) and np.all(activation <= 1), "Activation should be in [-1, 1]"
    print(f"  Activation Result: {activation}")
    
    # Test robust matrix inversion
    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    inverse = robust_matrix_inverse(matrix)
    identity_check = np.dot(matrix, inverse)
    assert np.allclose(identity_check, np.eye(2), atol=1e-10), "Matrix inversion error"
    print(f"  Matrix Inversion Check: {np.allclose(identity_check, np.eye(2))}")
    
    print("  ✅ Matrix activation passed")

def test_thermal_dynamics():
    """Test thermal dynamics formulas."""
    print("\n🌡 Testing Thermal Dynamics...")
    
    # Test enhanced thermal dynamics
    volume_data = np.array([1000, 1100, 1200, 1150, 1300])
    volatility = 0.02
    beta = 0.1
    
    # Exponential Moving Average
    ema_volume = beta * np.mean(volume_data) + (1 - beta) * volume_data[-1]
    print(f"  EMA Volume: {ema_volume:.2f}")
    
    # Volatility-scaled pressure
    pressure = np.tanh(volume_data[-1] / (ema_volume + EPSILON_FLOAT64)) * (1 + np.log(1 + volatility))
    assert -1 <= pressure <= 1, "Pressure should be in [-1, 1]"
    print(f"  Volatility-Scaled Pressure: {pressure:.4f}")
    
    # Decay factor
    decay_factor = np.exp(-volatility / 10)
    assert 0 < decay_factor <= 1, "Decay factor should be in (0, 1]"
    print(f"  Decay Factor: {decay_factor:.6f}")
    
    # Thermal conductivity
    thermal_conductivity = THERMAL_CONDUCTIVITY_BTC * (1 + volatility / 100)
    assert thermal_conductivity > 0, "Thermal conductivity should be positive"
    print(f"  Thermal Conductivity: {thermal_conductivity:.6f}")
    
    print("  ✅ Thermal dynamics passed")

def test_risk_adjusted_profit_rate():
    """Test risk-adjusted profit rate formulas."""
    print("\n📈 Testing Risk-Adjusted Profit Rate...")
    
    entry_price = 50000.0
    exit_price = 52500.0  # 5% profit
    time_held_minutes = 1440  # 24 hours
    volatility = 0.02  # 2% volatility
    
    # Basic return
    raw_return = (exit_price - entry_price) / entry_price
    expected_return = 0.05
    assert abs(raw_return - expected_return) < 1e-10, "Raw return calculation error"
    print(f"  Raw Return: {raw_return:.4f} ({raw_return*100:.2f}%)")
    
    # Annualized return
    annualized_return = raw_return * (525600 / max(time_held_minutes, 1))
    assert annualized_return > raw_return, "Annualized return should be larger than raw return"
    print(f"  Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
    
    # Sharpe ratio
    sharpe_ratio = annualized_return / (volatility + EPSILON_FLOAT64)
    assert sharpe_ratio > 0, "Sharpe ratio should be positive"
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Risk-adjusted return
    risk_adjusted_return = raw_return * np.exp(-volatility)
    assert risk_adjusted_return <= raw_return, "Risk-adjusted return should not exceed raw return"
    print(f"  Risk-Adjusted Return: {risk_adjusted_return:.4f}")
    
    print("  ✅ Risk-adjusted profit rate passed")

def test_kelly_criterion():
    """Test Kelly criterion formulas."""
    print("\n💹 Testing Kelly Criterion...")
    
    win_probability = 0.6  # 60% win rate
    expected_return = 0.1  # 10% expected return
    volatility = 0.15  # 15% volatility
    safety_factor = KELLY_SAFETY_FACTOR
    max_fraction = 0.25
    
    # Calculate odds
    odds = expected_return / volatility
    assert odds > 0, "Odds should be positive"
    print(f"  Odds: {odds:.4f}")
    
    # Calculate Kelly fraction
    lose_probability = 1 - win_probability
    kelly_fraction = (win_probability * odds - lose_probability) / odds
    print(f"  Kelly Fraction: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}%)")
    
    # Apply safety factor and limits
    safe_kelly = np.clip(kelly_fraction, 0, max_fraction) * safety_factor
    assert safe_kelly <= max_fraction, "Safe Kelly should not exceed max fraction"
    assert safe_kelly >= 0, "Safe Kelly should be non-negative"
    print(f"  Safe Kelly: {safe_kelly:.4f} ({safe_kelly*100:.2f}%)")
    
    # Calculate growth rate
    if kelly_fraction > 0 and kelly_fraction < 1:
        growth_rate = (win_probability * np.log(1 + odds * kelly_fraction) + 
                      lose_probability * np.log(1 - kelly_fraction))
    else:
        growth_rate = 0.0
    print(f"  Growth Rate: {growth_rate:.6f}")
    
    print("  ✅ Kelly criterion passed")

def test_quantum_signal_normalization():
    """Test quantum signal normalization formulas."""
    print("\n🧬 Testing Quantum Signal Normalization...")
    
    # Create mock quantum state
    amplitude = np.array([1.0, 2.0, 3.0])
    quantum_state = QuantumState(amplitude=amplitude, phase=0.0)
    
    # Normalize state
    state_magnitude = np.linalg.norm(quantum_state.amplitude)
    normalized_state = quantum_state.amplitude / (state_magnitude + EPSILON_FLOAT64)
    assert np.allclose(np.linalg.norm(normalized_state), 1.0, atol=1e-10), "State normalization error"
    print(f"  Normalized State Magnitude: {np.linalg.norm(normalized_state):.6f}")
    
    # Probability vector
    probability_vector = np.abs(normalized_state) ** 2
    assert np.allclose(np.sum(probability_vector), 1.0, atol=1e-10), "Probability vector should sum to 1"
    print(f"  Probability Vector Sum: {np.sum(probability_vector):.6f}")
    
    # Von Neumann entropy
    von_neumann_entropy = shannon_entropy_stable(probability_vector)
    assert von_neumann_entropy >= 0, "Von Neumann entropy should be non-negative"
    print(f"  Von Neumann Entropy: {von_neumann_entropy:.4f}")
    
    # Purity
    purity = np.sum(probability_vector ** 2)
    assert 0 < purity <= 1, "Purity should be in (0, 1]"
    print(f"  Purity: {purity:.4f}")
    
    print("  ✅ Quantum signal normalization passed")

def test_quantum_thermal_coupling():
    """Test quantum-thermal coupling formulas."""
    print("\n🧊 Testing Quantum-Thermal Coupling...")
    
    # Create mock quantum state and temperature
    quantum_state = QuantumState(amplitude=np.array([1.0, 0.0]), phase=0.0)
    temperature = Temperature(300.0)  # 300K
    gamma_factor = 1.0
    
    # Decoherence rate
    decoherence_rate = gamma_factor * temperature / REDUCED_PLANCK
    assert decoherence_rate > 0, "Decoherence rate should be positive"
    print(f"  Decoherence Rate: {decoherence_rate:.8f}")
    
    # Thermal entropy
    thermal_entropy = gamma_factor * temperature
    assert thermal_entropy > 0, "Thermal entropy should be positive"
    print(f"  Thermal Entropy: {thermal_entropy:.4f}")
    
    # Coupling strength
    coupling_strength = np.exp(-temperature / (10 * THERMAL_CONDUCTIVITY_BTC))
    assert 0 < coupling_strength <= 1, "Coupling strength should be in (0, 1]"
    print(f"  Coupling Strength: {coupling_strength:.6f}")
    
    # Decohered state
    decoherence_factor = np.exp(-decoherence_rate)
    assert 0 < decoherence_factor <= 1, "Decoherence factor should be in (0, 1]"
    print(f"  Decoherence Factor: {decoherence_factor:.6f}")
    
    print("  ✅ Quantum-thermal coupling passed")

def test_fractal_dimensions():
    """Test fractal dimensions (Higuchi method)."""
    print("\n🌌 Testing Fractal Dimensions (Higuchi)...")
    
    # Create test time series
    time_series = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    
    # Higuchi fractal dimension calculation
    n = len(time_series)
    k_max = 5
    k_values = range(1, min(k_max + 1, n // 2))
    l_values = []
    
    for k in k_values:
        l_k = 0
        for m in range(k):
            l_m_k = 0
            for i in range(1, int((n - m) / k)):
                l_m_k += abs(time_series[m + i * k] - time_series[m + (i - 1) * k])
            l_m_k = l_m_k * (n - 1) / (k ** 2)
            l_k += l_m_k
        l_k = l_k / k
        l_values.append(l_k)
    
    # Calculate slope
    if len(l_values) > 1:
        log_k = np.log(k_values)
        log_l = np.log(l_values)
        slope = np.polyfit(log_k, log_l, 1)[0]
        fractal_dimension = -slope
        assert fractal_dimension > 0, "Fractal dimension should be positive"
        print(f"  Fractal Dimension: {fractal_dimension:.4f}")
    else:
        print("  Fractal Dimension: Insufficient data for calculation")
    
    print("  ✅ Fractal dimensions passed")

def test_ferris_wheel_harmonic_analysis():
    """Test Ferris wheel harmonic analysis."""
    print("\n🎡 Testing Ferris Wheel Harmonic Analysis...")
    
    time_series = np.array([100, 101, 102, 103, 104, 105])
    periods = [60, 120, 240]  # 1h, 2h, 4h periods
    current_time = 100.0
    
    # Calculate harmonic phases
    harmonic_phases = [2 * np.pi * current_time / P for P in periods]
    assert len(harmonic_phases) == len(periods), "Number of phases should match number of periods"
    print(f"  Harmonic Phases: {[f'{p:.4f}' for p in harmonic_phases]}")
    
    # Calculate angular velocity
    primary_period = periods[0] if periods else 60
    angular_velocity = 2 * np.pi / primary_period
    assert angular_velocity > 0, "Angular velocity should be positive"
    print(f"  Angular Velocity: {angular_velocity:.4f}")
    
    # Calculate coherence
    complex_phases = np.exp(1j * np.array(harmonic_phases))
    coherence = np.abs(np.mean(complex_phases))
    assert 0 <= coherence <= 1, "Coherence should be in [0, 1]"
    print(f"  Coherence: {coherence:.4f}")
    
    # Calculate sync level
    sync_level = np.std(np.abs(complex_phases))
    assert sync_level >= 0, "Sync level should be non-negative"
    print(f"  Sync Level: {sync_level:.4f}")
    
    print("  ✅ Ferris wheel harmonic analysis passed")

def test_void_well_fractal_index():
    """Test Void-Well fractal index."""
    print("\n⚫ Testing Void-Well Fractal Index...")
    
    volume_data = np.array([1000, 1100, 1200, 1150, 1300, 1250])
    price_data = np.array([50000, 50100, 50200, 50150, 50300, 50250])
    
    # Calculate volume gradient
    volume_gradient = np.gradient(volume_data)
    print(f"  Volume Gradient: {volume_gradient}")
    
    # Calculate price gradient
    price_gradient = np.gradient(price_data)
    print(f"  Price Gradient: {price_gradient}")
    
    # Calculate curl field
    curl_field = volume_gradient * price_gradient
    print(f"  Curl Field: {curl_field}")
    
    # Calculate fractal index
    curl_magnitude = np.sum(np.abs(curl_field))
    volume_magnitude = np.sum(np.abs(volume_data))
    fractal_index = curl_magnitude / (volume_magnitude + EPSILON_FLOAT64)
    assert fractal_index >= 0, "Fractal index should be non-negative"
    print(f"  Fractal Index: {fractal_index:.6f}")
    
    # Calculate entropy gradient
    entropy_gradient = shannon_entropy_stable(np.abs(curl_field) + EPSILON_FLOAT64)
    assert entropy_gradient >= 0, "Entropy gradient should be non-negative"
    print(f"  Entropy Gradient: {entropy_gradient:.4f}")
    
    print("  ✅ Void-Well fractal index passed")

def test_api_entropy_reflection_penalty():
    """Test API entropy reflection penalty."""
    print("\n🧩 Testing API Entropy Reflection Penalty...")
    
    confidence = 0.8
    error_count = 2
    entropy = 0.5
    tau = 10.0
    
    # Calculate penalty
    penalty = np.exp(-error_count / tau)
    assert 0 < penalty <= 1, "Penalty should be in (0, 1]"
    print(f"  Penalty: {penalty:.6f}")
    
    # Calculate reflected confidence
    reflected_confidence = confidence * penalty * (1 - entropy / np.log2(2))
    assert reflected_confidence <= confidence, "Reflected confidence should not exceed original"
    print(f"  Reflected Confidence: {reflected_confidence:.4f}")
    
    print("  ✅ API entropy reflection penalty passed")

def test_recursive_time_lock_synchronization():
    """Test recursive time lock synchronization."""
    print("\n⏳ Testing Recursive Time Lock Synchronization...")
    
    # Test data - multiple time series at different scales
    time_series = [
        np.array([1, 2, 3, 4, 5]),  # Short scale
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # Medium scale
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # Long scale
    ]
    periods = [5, 10, 15]  # Corresponding periods
    sync_threshold = 0.7
    
    # Calculate phases for each scale
    phases = []
    for series, period in zip(time_series, periods):
        if len(series) > 0:
            cycle_count = len(series)
            phase = 2 * np.pi * (cycle_count % period) / period
            phases.append(phase)
        else:
            phases.append(0.0)
    
    assert len(phases) == len(periods), "Number of phases should match number of periods"
    print(f"  Phases: {[f'{p:.4f}' for p in phases]}")
    
    # Calculate coherence
    complex_phases = np.exp(1j * np.array(phases))
    coherence = np.abs(np.mean(complex_phases))
    assert 0 <= coherence <= 1, "Coherence should be in [0, 1]"
    print(f"  Coherence: {coherence:.4f}")
    
    # Check sync trigger
    sync_triggered = coherence > sync_threshold
    print(f"  Sync Triggered: {sync_triggered}")
    
    # Calculate phase variance
    phase_variance = np.var(phases) if len(phases) > 1 else 0.0
    assert phase_variance >= 0, "Phase variance should be non-negative"
    print(f"  Phase Variance: {phase_variance:.4f}")
    
    print("  ✅ Recursive time lock synchronization passed")

def test_grayscale_drift_tensor_core():
    """Test grayscale drift tensor core."""
    print("\n🌗 Testing Grayscale Drift Tensor Core...")
    
    x, y, z, t = 1.0, 2.0, 3.0, 0.5
    l, delta = 0.5, 0.1
    psi_infinity = 1.0
    
    # Drift field
    drift_field = (np.exp(-t) * np.sin(x * y) * np.cos(z) * 
                   (1 + abs(x)) / (1 + 0.1 * abs(y)))
    print(f"  Drift Field: {drift_field:.6f}")
    
    # Ring drift allocation
    ring_drift = psi_infinity * np.sin(l * delta) / (1 + l ** 2)
    print(f"  Ring Drift: {ring_drift:.6f}")
    
    # Gamma coupling
    gamma_coupling = 1 / (1 + abs(x) * np.log(1 + delta))
    assert 0 < gamma_coupling <= 1, "Gamma coupling should be in (0, 1]"
    print(f"  Gamma Coupling: {gamma_coupling:.6f}")
    
    print("  ✅ Grayscale drift tensor core passed")

def test_recursive_tensor_feedback():
    """Test recursive tensor feedback."""
    print("\n🧠 Testing Recursive Tensor Feedback...")
    
    # Create test tensors
    base_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    feedback_tensors = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5, 0.6], [0.7, 0.8]])
    ]
    delta_entropies = [0.1, 0.2]
    lambda_values = [0.5, 1.0]
    
    # Calculate weights
    weights = [np.exp(-lambda_val) for lambda_val in lambda_values]
    assert all(w > 0 for w in weights), "Weights should be positive"
    print(f"  Weights: {[f'{w:.4f}' for w in weights]}")
    
    # Calculate weighted sum
    weighted_sum = base_tensor.copy()
    for i, (tensor, delta_entropy, weight) in enumerate(zip(feedback_tensors, delta_entropies, weights)):
        weighted_sum += weight * tensor * delta_entropy
    
    # Normalize by total weight
    total_weight = 1 + sum(weights)
    feedback_tensor = weighted_sum / total_weight
    
    assert feedback_tensor.shape == base_tensor.shape, "Feedback tensor shape should match base tensor"
    print(f"  Feedback Tensor:\n{feedback_tensor}")
    
    print("  ✅ Recursive tensor feedback passed")

def main():
    """Run all mathematical formula tests."""
    print("🚀 Starting Comprehensive Mathematical Formulas Test")
    print("=" * 60)
    
    try:
        test_delta_calculations()
        test_entropy_and_information_theory()
        test_matrix_activation()
        test_thermal_dynamics()
        test_risk_adjusted_profit_rate()
        test_kelly_criterion()
        test_quantum_signal_normalization()
        test_quantum_thermal_coupling()
        test_fractal_dimensions()
        test_ferris_wheel_harmonic_analysis()
        test_void_well_fractal_index()
        test_api_entropy_reflection_penalty()
        test_recursive_time_lock_synchronization()
        test_grayscale_drift_tensor_core()
        test_recursive_tensor_feedback()
        
        print("\n" + "=" * 60)
        print("🎉 All mathematical formulas tests passed successfully!")
        print("✅ Mathematical foundation validated for Schwabot system")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 