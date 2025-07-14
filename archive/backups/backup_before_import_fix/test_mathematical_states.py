import os
import sys
import traceback

import numpy as np
from constants import EPSILON_FLOAT64, FERRIS_PRIMARY_CYCLE, KELLY_SAFETY_FACTOR
from type_defs import QuantumState, Temperature

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Mathematical State Structures
=================================
Comprehensive test suite for all mathematical state structures in advanced_mathematical_core.py
"""


# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

try:
        FerrisWheelState,
        QuantumThermalState,
        VoidWellMetrics,
        ProfitState,
        RecursiveTimeLockSync,
        KellyMetrics,
        calculate_ferris_wheel_state,
        calculate_quantum_thermal_state,
        calculate_void_well_metrics,
        calculate_profit_state,
        calculate_recursive_time_lock_sync,
        calculate_kelly_metrics,
        shannon_entropy_stable,
        safe_delta_calculation,
        normalized_delta_tanh,
    )

    print("✅ Successfully imported all mathematical state modules")
    except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_ferris_wheel_state():
    """Test FerrisWheelState calculations."""
    print("\n🧠 Testing FerrisWheelState...")

    # Test data
    time_series = np.array([100, 101, 102, 103, 104, 105])
    periods = [FERRIS_PRIMARY_CYCLE, FERRIS_PRIMARY_CYCLE * 2, FERRIS_PRIMARY_CYCLE * 4]
    current_time = 100.0

    # Calculate state
    state = calculate_ferris_wheel_state(time_series, periods, current_time)

    print(f"  Cycle Position: {state.cycle_position:.4f}")
    print(f"  Angular Velocity: {state.angular_velocity:.4f}")
    print(f"  Phase Coherence: {state.phase_coherence:.4f}")
    print(f"  Synchronization Level: {state.synchronization_level:.4f}")
    print(f"  Harmonic Phases: {[f'{p:.4f}' for p in state.harmonic_phases]}")

    # Validate mathematical properties
    assert 0 <= state.cycle_position <= 2 * np.pi, "Cycle position should be in [0, 2π]"
    assert state.angular_velocity > 0, "Angular velocity should be positive"
    assert 0 <= state.phase_coherence <= 1, "Phase coherence should be in [0, 1]"
    assert state.synchronization_level >= 0, ()
        "Synchronization level should be non-negative"
    )

    print("  ✅ FerrisWheelState tests passed")


def test_quantum_thermal_state():
    """Test QuantumThermalState calculations."""
    print("\n🔥 Testing QuantumThermalState...")

    # Mock quantum state and temperature
    quantum_state = QuantumState(amplitude=1.0, phase=0.0)
    temperature = Temperature(300.0)  # 300K

    # Calculate state
    state = calculate_quantum_thermal_state()
        quantum_state, temperature, gamma_factor=1.0
    )

    print(f"  Temperature: {state.temperature:.2f}K")
    print(f"  Thermal Entropy: {state.thermal_entropy:.4f}")
    print(f"  Coupling Strength: {state.coupling_strength:.6f}")
    print(f"  Decoherence Rate: {state.decoherence_rate:.8f}")

    # Validate mathematical properties
    assert state.temperature > 0, "Temperature should be positive"
    assert state.thermal_entropy > 0, "Thermal entropy should be positive"
    assert 0 < state.coupling_strength <= 1, "Coupling strength should be in (0, 1]")
    assert state.decoherence_rate >= 0, "Decoherence rate should be non-negative"

    print("  ✅ QuantumThermalState tests passed")


def test_void_well_metrics():
    """Test VoidWellMetrics calculations."""
    print("\n🌀 Testing VoidWellMetrics...")

    # Test data
    volume_data = np.array([1000, 1100, 1200, 1150, 1300, 1250])
    price_data = np.array([50000, 50100, 50200, 50150, 50300, 50250])

    # Calculate metrics
    metrics = calculate_void_well_metrics(volume_data, price_data)

    print(f"  Fractal Index (VFI): {metrics.fractal_index:.6f}")
    print(f"  Volume Divergence: {metrics.volume_divergence:.4f}")
    print(f"  Curl Magnitude: {metrics.curl_magnitude:.4f}")
    print(f"  Entropy Gradient: {metrics.entropy_gradient:.4f}")
    print(f"  Price Variance Field: {metrics.price_variance_field}")

    # Validate mathematical properties
    assert metrics.fractal_index >= 0, "Fractal index should be non-negative"
    assert metrics.volume_divergence >= 0, "Volume divergence should be non-negative"
    assert metrics.curl_magnitude >= 0, "Curl magnitude should be non-negative"
    assert metrics.entropy_gradient >= 0, "Entropy gradient should be non-negative"

    print("  ✅ VoidWellMetrics tests passed")


def test_profit_state():
    """Test ProfitState calculations."""
    print("\n🧮 Testing ProfitState...")

    # Test data
    entry_price = 50000.0
    exit_price = 52500.0  # 5% profit
    time_held_minutes = 1440  # 24 hours
    volatility = 0.2  # 2% volatility

    # Calculate profit state
    state = calculate_profit_state()
        entry_price, exit_price, time_held_minutes, volatility
    )

    print(f"  Raw Return: {state.raw_return:.4f} ({state.raw_return * 100:.2f}%)")
    print()
        f"  Annualized Return: {state.annualized_return:.4f} ({state.annualized_return * 100:.2f}%)"
    )
    print(f"  Sharpe Ratio: {state.sharpe_ratio:.4f}")
    print(f"  Risk-Adjusted Return: {state.risk_adjusted_return:.4f}")
    print(f"  Risk Penalty: {state.risk_penalty:.4f}")

    # Validate mathematical properties
    expected_raw_return = (exit_price - entry_price) / entry_price
    assert abs(state.raw_return - expected_raw_return) < 1e-10, ()
        "Raw return calculation error"
    )
    assert state.risk_penalty > 0 and state.risk_penalty <= 1, ()
        "Risk penalty should be in (0, 1]")
    )
    assert state.risk_adjusted_return <= state.raw_return, ()
        "Risk-adjusted return should not exceed raw return"
    )

    print("  ✅ ProfitState tests passed")


def test_recursive_time_lock_sync():
    """Test RecursiveTimeLockSync calculations."""
    print("\n🧿 Testing RecursiveTimeLockSync...")

    # Test data - multiple time series at different scales
    time_series = []
        np.array([1, 2, 3, 4, 5]),  # Short scale
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # Medium scale
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),  # Long scale
    ]
    periods = [5, 10, 15]  # Corresponding periods
    sync_threshold = 0.7

    # Calculate sync state
    sync_state = calculate_recursive_time_lock_sync()
        time_series, periods, sync_threshold
    )

    print(f"  Coherence: {sync_state.coherence:.4f}")
    print(f"  Sync Triggered: {sync_state.sync_triggered}")
    print(f"  Phase Variance: {sync_state.phase_variance:.4f}")
    print(f"  Ratios: {sync_state.ratios}")

    # Validate mathematical properties
    assert 0 <= sync_state.coherence <= 1, "Coherence should be in [0, 1]"
    assert sync_state.phase_variance >= 0, "Phase variance should be non-negative"
    assert len(sync_state.ratios) == 2, "Should have exactly 2 ratios"

    print("  ✅ RecursiveTimeLockSync tests passed")


def test_kelly_metrics():
    """Test KellyMetrics calculations."""
    print("\n📐 Testing KellyMetrics...")

    # Test data
    win_probability = 0.6  # 60% win rate
    expected_return = 0.1  # 10% expected return
    volatility = 0.15  # 15% volatility
    safety_factor = KELLY_SAFETY_FACTOR
    max_fraction = 0.25

    # Calculate Kelly metrics
    metrics = calculate_kelly_metrics()
        win_probability, expected_return, volatility, safety_factor, max_fraction
    )

    print()
        f"  Kelly Fraction: {metrics.kelly_fraction:.4f} ({metrics.kelly_fraction * 100:.2f}%)"
    )
    print(f"  Safe Kelly: {metrics.safe_kelly:.4f} ({metrics.safe_kelly * 100:.2f}%)")
    print(f"  Odds: {metrics.odds:.4f}")
    print(f"  Growth Rate: {metrics.growth_rate:.6f}")
    print(f"  ROI Volatility: {metrics.roi_volatility:.4f}")

    # Validate mathematical properties
    assert metrics.odds > 0, "Odds should be positive"
    assert metrics.safe_kelly <= max_fraction, ()
        "Safe Kelly should not exceed max fraction"
    )
    assert metrics.safe_kelly >= 0, "Safe Kelly should be non-negative"
    assert metrics.roi_volatility == volatility, "ROI volatility should match input"

    print("  ✅ KellyMetrics tests passed")


def test_utility_functions():
    """Test utility mathematical functions."""
    print("\n🔧 Testing Utility Functions...")

    # Test safe delta calculation
    delta = safe_delta_calculation(100, 95)
    expected_delta = (100 - 95) / 95
    assert abs(delta - expected_delta) < 1e-10, "Delta calculation error"
    print(f"  Safe Delta: {delta:.4f}")

    # Test normalized delta with tanh
    norm_delta = normalized_delta_tanh(100, 95, scaling_factor=1.0)
    assert -1 <= norm_delta <= 1, "Normalized delta should be in [-1, 1]"
    print(f"  Normalized Delta: {norm_delta:.4f}")

    # Test Shannon entropy
    prob_vector = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = shannon_entropy_stable(prob_vector)
    expected_entropy = 2.0  # log2(4) = 2
    assert abs(entropy - expected_entropy) < 1e-10, "Shannon entropy calculation error"
    print(f"  Shannon Entropy: {entropy:.4f}")

    print("  ✅ Utility function tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n⚠️ Testing Edge Cases...")

    # Test empty data
    empty_state = calculate_ferris_wheel_state(np.array([]), [], 0.0)
    assert empty_state.cycle_position == 0.0, ()
        "Empty data should return zero cycle position"
    )

    # Test zero price
    zero_profit = calculate_profit_state(0.0, 100.0, 60.0, 0.1)
    assert zero_profit.raw_return == 0.0, "Zero entry price should return zero profit"

    # Test invalid Kelly parameters
    invalid_kelly = calculate_kelly_metrics(0.0, 0.1, 0.15)  # Zero win probability
    assert invalid_kelly.kelly_fraction == 0.0, ()
        "Invalid Kelly should return zero fraction"
    )

    print("  ✅ Edge case tests passed")


def main():
    """Run all mathematical state tests."""
    print("🚀 Starting Mathematical State Structure Tests")
    print("=" * 50)

    try:
        test_ferris_wheel_state()
        test_quantum_thermal_state()
        test_void_well_metrics()
        test_profit_state()
        test_recursive_time_lock_sync()
        test_kelly_metrics()
        test_utility_functions()
        test_edge_cases()

        print("\n" + "=" * 50)
        print("🎉 All mathematical state tests passed successfully!")
        print("✅ Mathematical viability confirmed for all state structures")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
))
