#!/usr/bin/env python3
"""
Test ZPE Mathematical Framework
==============================

Simple test to verify the ZPE core mathematical functions work correctly.
"""

import math
import sys

def calculate_zpe_work(trend_strength: float, entry_exit_range: float) -> float:
    """ZPE Work Core: W = F · d = ΔP"""
    market_force = math.tanh(trend_strength)  # Bounded between -1 and 1
    work = market_force * entry_exit_range
    return work

def calculate_rotational_torque(liquidity_depth: float, trend_change_rate: float) -> float:
    """Rotational Vectorization: τ = I · α"""
    inertia = 1.0 / (1.0 + liquidity_depth)  # Higher liquidity = lower inertia
    angular_acceleration = math.atan(trend_change_rate)  # Bounded acceleration
    torque = inertia * angular_acceleration
    return torque

def calculate_thermal_efficiency(profit_generated: float, capital_exposure: float) -> float:
    """Thermal Integrity Differential: η = W_out / Q_in"""
    if capital_exposure <= 0:
        return 0.0
    efficiency = profit_generated / capital_exposure
    return efficiency

def map_news_lantern_signals(news_density: float, sentiment_delta: float) -> float:
    """News / Lantern API Signal Mapping: Lₜ = g(nₜ, ΔSₜ)"""
    normalized_density = max(0.0, min(1.0, news_density))
    normalized_sentiment = max(-1.0, min(1.0, sentiment_delta))
    lantern_signal = normalized_density * (1.0 + normalized_sentiment)
    return lantern_signal

def spin_profit_wheel(market_data: dict) -> dict:
    """Main ZPE Profit Wheel function - where Schwabot becomes the wheel."""
    print("🔄 Spinning ZPE Profit Wheel...")
    
    # Extract market data
    trend_strength = market_data.get('trend_strength', 0.0)
    entry_exit_range = market_data.get('entry_exit_range', 0.0)
    liquidity_depth = market_data.get('liquidity_depth', 1.0)
    trend_change_rate = market_data.get('trend_change_rate', 0.0)
    news_density = market_data.get('news_density', 0.0)
    sentiment_delta = market_data.get('sentiment_delta', 0.0)
    
    # Execute ZPE mathematical framework
    zpe_work = calculate_zpe_work(trend_strength, entry_exit_range)
    rotational_torque = calculate_rotational_torque(liquidity_depth, trend_change_rate)
    lantern_signal = map_news_lantern_signals(news_density, sentiment_delta)
    
    # Calculate spin decision
    spin_threshold = 0.5
    spin_score = (zpe_work + lantern_signal) / 2.0
    should_spin = spin_score > spin_threshold
    
    result = {
        'zpe_work': zpe_work,
        'rotational_torque': rotational_torque,
        'lantern_signal': lantern_signal,
        'spin_score': spin_score,
        'should_spin': should_spin
    }
    
    print(f"🎯 ZPE Wheel Decision: {'SPIN' if should_spin else 'HOLD'} (score: {spin_score:.6f})")
    return result

def main():
    """Test the ZPE Mathematical Framework."""
    print("🧠 Testing Schwabot ZPE Mathematical Framework")
    print("=" * 50)
    
    # Test market data
    market_data = {
        'trend_strength': 0.8,
        'entry_exit_range': 0.05,
        'liquidity_depth': 0.7,
        'trend_change_rate': 0.3,
        'news_density': 0.6,
        'sentiment_delta': 0.2
    }
    
    # Spin the profit wheel
    result = spin_profit_wheel(market_data)
    
    print(f"\n📊 Results:")
    print(f"ZPE Work: {result['zpe_work']:.6f}")
    print(f"Rotational Torque: {result['rotational_torque']:.6f}")
    print(f"Lantern Signal: {result['lantern_signal']:.6f}")
    print(f"Spin Score: {result['spin_score']:.6f}")
    print(f"Should Spin: {result['should_spin']}")
    
    # Test thermal efficiency
    efficiency = calculate_thermal_efficiency(100.0, 1000.0)
    print(f"Thermal Efficiency: {efficiency:.6f}")
    
    print("\n🎉 ZPE Mathematical Framework test complete!")
    print("\n🔥 Schwabot is now the wheel - spinning into profit, not pinging against it!")

if __name__ == "__main__":
    main() 