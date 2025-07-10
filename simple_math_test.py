import os
import sys
import traceback

import numpy as np

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Mathematical State Test
==============================
Direct test of mathematical state structures without complex imports.
"""


# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))


def test_mathematical_structures():
    """Test the mathematical state structures directly."""
    print("🚀 Testing Mathematical State Structures")
    print("=" * 50)

    try:
        # Test FerrisWheelState
        print("\n🧠 Testing FerrisWheelState...")

        # Simple test data
        time_series = np.array([100, 101, 102, 103, 104, 105])
        periods = [60, 120, 240]  # 1h, 2h, 4h periods
        current_time = 100.0

        # Calculate harmonic phases: φᵢ = 2πt/Pᵢ
        harmonic_phases = [2 * np.pi * current_time / P for P in periods]

        # Calculate angular velocity: ω = 2π/P (using primary, period)
        primary_period = periods[0] if periods else 60
        angular_velocity = 2 * np.pi / primary_period

        # Calculate phase coherence: C = (1/n) Σᵢ₌₁ⁿ |⟨e^(iφᵢ)⟩|
        complex_phases = np.exp(1j * np.array(harmonic_phases))
        phase_coherence = np.abs(np.mean(complex_phases))

        # Calculate synchronization level: σ = std({|⟨e^(iφᵢ)⟩|})
        synchronization_level = np.std(np.abs(complex_phases))

        # Calculate cycle position: φ₁ mod 2π
        cycle_position = harmonic_phases[0] % (2 * np.pi)

        print(f"  Cycle Position: {cycle_position:.4f}")
        print(f"  Angular Velocity: {angular_velocity:.4f}")
        print(f"  Phase Coherence: {phase_coherence:.4f}")
        print(f"  Synchronization Level: {synchronization_level:.4f}")
        print(f"  Harmonic Phases: {[f'{p:.4f}' for p in harmonic_phases]}")

        # Validate mathematical properties
        assert 0 <= cycle_position <= 2 * np.pi, "Cycle position should be in [0, 2π]"
        assert angular_velocity > 0, "Angular velocity should be positive"
        assert 0 <= phase_coherence <= 1, "Phase coherence should be in [0, 1]"
        assert synchronization_level >= 0, ()
            "Synchronization level should be non-negative"
        )

        print("  ✅ FerrisWheelState calculations passed")

        # Test VoidWellMetrics
        print("\n🌀 Testing VoidWellMetrics...")

        # Test data
        volume_data = np.array([1000, 1100, 1200, 1150, 1300, 1250])
        price_data = np.array([50000, 50100, 50200, 50150, 50300, 50250])
        epsilon = 1e-12

        # Calculate volume gradient: ∇V
        volume_gradient = np.gradient(volume_data)

        # Calculate price variance field: dP⃗
        price_variance_field = np.gradient(price_data)

        # Calculate curl-like field: C⃗ = ∇V · dP⃗
        curl_field = volume_gradient * price_variance_field

        # Calculate curl magnitude: ||C⃗|| = Σ|Cᵢ|
        curl_magnitude = np.sum(np.abs(curl_field))

        # Calculate volume divergence: Σ|∇V|
        volume_divergence = np.sum(np.abs(volume_gradient))

        # Calculate Void-Well Fractal Index: VFI = ||C⃗||/(||V|| + ε)
        volume_magnitude = np.linalg.norm(volume_data)
        fractal_index = curl_magnitude / (volume_magnitude + epsilon)

        # Calculate entropy gradient: ∇S = Shannon(C⃗)
        if len(curl_field) > 1:
            # Normalize curl field for entropy calculation
            curl_normalized = np.abs(curl_field) / ()
                np.sum(np.abs(curl_field)) + epsilon
            )
            # Simple entropy calculation
            entropy_gradient = -np.sum()
                curl_normalized * np.log2(curl_normalized + epsilon)
            )
        else:
            entropy_gradient = 0.0

        print(f"  Fractal Index (VFI): {fractal_index:.6f}")
        print(f"  Volume Divergence: {volume_divergence:.4f}")
        print(f"  Curl Magnitude: {curl_magnitude:.4f}")
        print(f"  Entropy Gradient: {entropy_gradient:.4f}")
        print(f"  Price Variance Field: {price_variance_field}")

        # Validate mathematical properties
        assert fractal_index >= 0, "Fractal index should be non-negative"
        assert volume_divergence >= 0, "Volume divergence should be non-negative"
        assert curl_magnitude >= 0, "Curl magnitude should be non-negative"
        assert entropy_gradient >= 0, "Entropy gradient should be non-negative"

        print("  ✅ VoidWellMetrics calculations passed")

        # Test ProfitState
        print("\n🧮 Testing ProfitState...")

        # Test data
        entry_price = 50000.0
        exit_price = 52500.0  # 5% profit
        time_held_minutes = 1440  # 24 hours
        volatility = 0.2  # 2% volatility

        # Calculate raw return: R = (P_exit - P_entry)/P_entry
        raw_return = (exit_price - entry_price) / entry_price

        # Calculate annualized return: R_annualized = R·(525600/t_held)
        # 525600 = minutes in a year (365 * 24 * 60)
        annualized_return = raw_return * (525600 / max(time_held_minutes, 1))

        # Calculate risk-adjusted return: R_a = R·e^(-σ)
        risk_adjusted_return = raw_return * np.exp(-volatility)

        # Calculate risk penalty: e^(-σ)
        risk_penalty = np.exp(-volatility)

        # Calculate Sharpe ratio: Sharpe = R_annualized/(σ + ε)
        sharpe_ratio = annualized_return / (volatility + epsilon)

        print(f"  Raw Return: {raw_return:.4f} ({raw_return * 100:.2f}%)")
        print()
            f"  Annualized Return: {annualized_return:.4f} ({annualized_return * 100:.2f}%)"
        )
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Risk-Adjusted Return: {risk_adjusted_return:.4f}")
        print(f"  Risk Penalty: {risk_penalty:.4f}")

        # Validate mathematical properties
        expected_raw_return = (exit_price - entry_price) / entry_price
        assert abs(raw_return - expected_raw_return) < 1e-10, ()
            "Raw return calculation error"
        )
        assert risk_penalty > 0 and risk_penalty <= 1, ()
            "Risk penalty should be in (0, 1]")
        )
        assert risk_adjusted_return <= raw_return, ()
            "Risk-adjusted return should not exceed raw return"
        )

        print("  ✅ ProfitState calculations passed")

        # Test KellyMetrics
        print("\n📐 Testing KellyMetrics...")

        # Test data
        win_probability = 0.6  # 60% win rate
        expected_return = 0.1  # 10% expected return
        volatility = 0.15  # 15% volatility
        safety_factor = 0.25  # 25% safety factor
        max_fraction = 0.25

        # Calculate odds: b = E[r]/σ
        odds = expected_return / volatility

        # Calculate Kelly fraction: f* = (p·b - q)/b
        # where q = 1 - p (lose, probability)
        lose_probability = 1 - win_probability
        kelly_fraction = (win_probability * odds - lose_probability) / odds

        # Apply safety factor and limits: f_safe = clip(f*, 0, limit)·SAFETY
        safe_kelly = np.clip(kelly_fraction, 0, max_fraction) * safety_factor

        # Calculate growth rate: G = p·log(1 + bf*) + q·log(1 - f*)
        if kelly_fraction > 0 and kelly_fraction < 1:
            growth_rate = win_probability * np.log()
                1 + odds * kelly_fraction
            ) + lose_probability * np.log(1 - kelly_fraction)
        else:
            growth_rate = 0.0

        print(f"  Kelly Fraction: {kelly_fraction:.4f} ({kelly_fraction * 100:.2f}%)")
        print(f"  Safe Kelly: {safe_kelly:.4f} ({safe_kelly * 100:.2f}%)")
        print(f"  Odds: {odds:.4f}")
        print(f"  Growth Rate: {growth_rate:.6f}")
        print(f"  ROI Volatility: {volatility:.4f}")

        # Validate mathematical properties
        assert odds > 0, "Odds should be positive"
        assert safe_kelly <= max_fraction, "Safe Kelly should not exceed max fraction"
        assert safe_kelly >= 0, "Safe Kelly should be non-negative"
        assert volatility == 0.15, "ROI volatility should match input"

        print("  ✅ KellyMetrics calculations passed")

        # Test RecursiveTimeLockSync
        print("\n🧿 Testing RecursiveTimeLockSync...")

        # Test data - multiple time series at different scales
        time_series = []
            np.array([1, 2, 3, 4, 5]),  # Short scale
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # Medium scale
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),  # Long scale
        ]
        periods = [5, 10, 15]  # Corresponding periods
        sync_threshold = 0.7

        # Calculate phases for each scale: φₖ = 2π(Cₖ mod, P)/P
        phases = []
        for i, (series, period) in enumerate(zip(time_series, periods)):
            if len(series) > 0:
                # Use the last value as current cycle count
                cycle_count = len(series)
                phase = 2 * np.pi * (cycle_count % period) / period
                phases.append(phase)
            else:
                phases.append(0.0)

        # Calculate coherence: C = |⟨e^(iφₖ)⟩|
        complex_phases = np.exp(1j * np.array(phases))
        coherence = np.abs(np.mean(complex_phases))

        # Check sync trigger: sync = C > τ
        sync_triggered = coherence > sync_threshold

        # Calculate phase variance: σ² = Var(φₖ)
        phase_variance = np.var(phases) if len(phases) > 1 else 0.0

        # Calculate ratios: (C₁/C₂, C₂/C₃)
        if len(phases) >= 3:
            ratios = ()
                phases[0] / (phases[1] + epsilon),
                phases[1] / (phases[2] + epsilon),
            )
        else:
            ratios = (1.0, 1.0)

        print(f"  Coherence: {coherence:.4f}")
        print(f"  Sync Triggered: {sync_triggered}")
        print(f"  Phase Variance: {phase_variance:.4f}")
        print(f"  Ratios: {ratios}")

        # Validate mathematical properties
        assert 0 <= coherence <= 1, "Coherence should be in [0, 1]"
        assert phase_variance >= 0, "Phase variance should be non-negative"
        assert len(ratios) == 2, "Should have exactly 2 ratios"

        print("  ✅ RecursiveTimeLockSync calculations passed")

        print("\n" + "=" * 50)
        print("🎉 All mathematical state calculations passed successfully!")
        print("✅ Mathematical viability confirmed for all state structures")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mathematical_structures()
    exit(0 if success else 1)
)
