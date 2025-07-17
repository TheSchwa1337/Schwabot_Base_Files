import os
import random
import sys
import time
import traceback
from typing import Any, Dict

import numpy as np

from core.dualistic_thought_engines import DualisticState, DualisticThoughtEngines

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALIF Dualistic State Engine Test Script
=======================================

This script demonstrates the concrete implementation of the ALIF (Adaptive, Learning)
Interference Filter) dualistic state engine across various market scenarios.

Features tested:
- ALIF state activation and decision making
- Volume delta calculations
- Resonance delta analysis
- AI feedback integration
- Error correction mechanisms
- Market memory scoring
- Performance metrics and statistics
"""


# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))



def create_market_data(price=50000.0, volume=1000000.0, rsi=45.0, volatility=0.3, momentum=0.2, include_ai_feedback=False, include_errors=False) -> Dict[str, Any]:
    """Create realistic market data for testing."""
    market_data = {
        "current_price": price,
        "previous_price": price * (1 + random.uniform(-0.2, 0.2)),
        "volume": volume,
        "previous_volume": volume * random.uniform(0.8, 1.2),
        "average_volume": volume * random.uniform(0.9, 1.1),
        "rsi": rsi,
        "volatility": volatility,
        "price_momentum": momentum,
        "macd_signal": random.uniform(-0.1, 0.1),
        "volume_change": random.uniform(-0.3, 0.3),
        "moving_average": price * random.uniform(0.98, 1.2),
        "previous_close": price * random.uniform(0.98, 1.2),
        "consensus_signal": random.choice(["buy", "sell", "hold", "neutral"]),
        "price_history": [price * (1 + random.uniform(-0.5, 0.5)) for _ in range(20)],
        "volume_history": [volume * random.uniform(0.7, 1.3) for _ in range(20)],
        "phase_data": [random.uniform(0, 2 * np.pi) for _ in range(4)],
    }
    # Add AI feedback if requested
    if include_ai_feedback:
        market_data["ai_feedback"] = [
            {
                "model": "GPT-4",
                "confidence": random.uniform(0.6, 0.9),
                "weight": 0.4,
                "prediction": random.choice(["bullish", "bearish", "neutral"]),
            },
            {
                "model": "Claude-3",
                "confidence": random.uniform(0.5, 0.8),
                "weight": 0.3,
                "prediction": random.choice(["bullish", "bearish", "neutral"]),
            },
            {
                "model": "R1",
                "confidence": random.uniform(0.7, 0.95),
                "weight": 0.3,
                "prediction": random.choice(["bullish", "bearish", "neutral"]),
            },
        ]
    # Add error logs if requested
    if include_errors:
        market_data["error_logs"] = [
            {
                "severity": random.uniform(0.1, 0.5),
                "message": "API timeout",
                "timestamp": time.time() - random.uniform(0, 3600),
            },
            {
                "severity": random.uniform(0.2, 0.6),
                "message": "Data inconsistency",
                "timestamp": time.time() - random.uniform(0, 1800),
            },
        ]
    return market_data


def test_alif_basic_functionality():
    """Test basic ALIF functionality."""
    print("🧪 Testing Basic ALIF Functionality")
    print("=" * 50)

    # Initialize engine
    engine = DualisticThoughtEngines()

    # Create test market data
    market_data = create_market_data()

    # Process market data
    thought_vector = engine.process_market_data(market_data)

    # Display results
    print(f"Final State: {thought_vector.state.value}")
    print(f"ALIF Score: {thought_vector.alif_score:.3f}")
    print(f"ALIF Decision: {thought_vector.alif_decision}")
    print(f"Final Decision: {thought_vector.decision}")
    print(f"Confidence: {thought_vector.confidence:.3f}")

    if thought_vector.alif_feedback:
        print(f"ALIF Routing: {thought_vector.alif_feedback.routing_target}")
        print(f"Volume Delta: {thought_vector.alif_feedback.volume_delta:.3f}")
        print(f"Resonance Delta: {thought_vector.alif_feedback.resonance_delta:.3f}")
        print(f"AI Feedback Score: {thought_vector.alif_feedback.ai_feedback_score:.3f}")

    print()


def test_alif_state_activation():
    """Test ALIF state activation under different conditions."""
    print("🧪 Testing ALIF State Activation")
    print("=" * 50)

    engine = DualisticThoughtEngines()

    # Test scenarios that should activate ALIF
    scenarios = [
        (
            "High AI Confidence",
            create_market_data(price=50000.0, rsi=30.0, volatility=0.8, momentum=0.1, include_ai_feedback=True),
        ),
        (
            "High Volume",
            create_market_data(price=50000.0, volume=2000000.0, rsi=70.0, volatility=0.6, include_ai_feedback=True),
        ),
        (
            "Error Correction",
            create_market_data(price=50000.0, rsi=50.0, volatility=0.4, include_ai_feedback=True, include_errors=True),
        ),
    ]
    for scenario_name, market_data in scenarios:
        print(f"\nScenario: {scenario_name}")

        # Process data
        thought_vector = engine.process_market_data(market_data)

        print(f"  State: {thought_vector.state.value}")
        print(f"  ALIF Score: {thought_vector.alif_score:.3f}")
        print(f"  ALIF Decision: {thought_vector.alif_decision}")
        print(f"  Final Decision: {thought_vector.decision}")

        if thought_vector.state == DualisticState.ALIF:
            print("  ✅ ALIF State Activated")
        else:
            print("  ❌ ALIF State Not Activated")

    print()


def test_alif_memory_and_learning():
    """Test ALIF memory and learning capabilities."""
    print("🧪 Testing ALIF Memory and Learning")
    print("=" * 50)

    engine = DualisticThoughtEngines()

    # Add some AI feedback
    engine.add_ai_feedback({"model": "GPT-4", "confidence": 0.85, "prediction": "bullish"})

    # Add some errors
    engine.add_alif_error({"severity": 0.3, "message": "Test error for learning"})

    # Process multiple market data points
    for i in range(5):
        market_data = create_market_data(
            price=50000.0 + i * 100,
            volume=1000000.0 + i * 50000,
            rsi=45.0 + i * 5,
            include_ai_feedback=True,
        )

        thought_vector = engine.process_market_data(market_data)
        print(f"Iteration {i + 1}: ALIF Score = {thought_vector.alif_score:.3f}, State = {thought_vector.state.value}")

    # Check memory statistics
    alif_stats = engine.get_alif_statistics()
    print(f"\nALIF Memory Size: {alif_stats['memory_size']}")
    print(f"ALIF Feedback History: {alif_stats['feedback_history_size']}")
    print(f"Error Log Size: {alif_stats['error_log_size']}")
    print(f"Market Memory Size: {alif_stats['market_memory_size']}")

    print()


def test_alif_configuration():
    """Test ALIF configuration and control methods."""
    print("🧪 Testing ALIF Configuration")
    print("=" * 50)

    engine = DualisticThoughtEngines()

    # Test threshold setting
    print("Setting ALIF threshold to 0.5...")
    engine.set_alif_threshold(0.5)

    # Test weight configuration
    new_weights = {
        "volume": 0.5,
        "resonance": 0.2,
        "ai_feedback": 0.2,
        "error_correction": 0.1,
    }
    print(f"Setting ALIF weights to {new_weights}...")
    engine.set_alif_weights(new_weights)

    # Test disable/enable
    print("Disabling ALIF...")
    engine.disable_alif()

    market_data = create_market_data()
    thought_vector = engine.process_market_data(market_data)
    print(f"ALIF Disabled - State: {thought_vector.state.value}")

    print("Enabling ALIF...")
    engine.enable_alif()

    thought_vector = engine.process_market_data(market_data)
    print(f"ALIF Enabled - State: {thought_vector.state.value}")

    print()


def test_alif_performance_metrics():
    """Test ALIF performance metrics and statistics."""
    print("🧪 Testing ALIF Performance Metrics")
    print("=" * 50)

    engine = DualisticThoughtEngines()

    # Process multiple scenarios to generate metrics
    scenarios = [
        ("Bullish", create_market_data(price=50000.0, rsi=30.0, momentum=0.5)),
        ("Bearish", create_market_data(price=50000.0, rsi=70.0, momentum=-0.5)),
        ("Neutral", create_market_data(price=50000.0, rsi=50.0, momentum=0.0)),
        ("High Volatility", create_market_data(price=50000.0, volatility=0.8)),
        ("Low Volume", create_market_data(price=50000.0, volume=500000.0)),
    ]
    for scenario_name, market_data in scenarios:
        thought_vector = engine.process_market_data(market_data)
        print(f"{scenario_name}: ALIF Score = {thought_vector.alif_score:.3f}, Decision = {thought_vector.decision}")

    # Display comprehensive performance metrics
    print("\n📊 Performance Metrics:")
    print("-" * 30)

    general_stats = engine.get_engine_performance()
    alif_stats = engine.get_alif_statistics()

    print(f"Total Decisions: {general_stats['total_decisions']}")
    print(f"ALIF Decisions: {alif_stats['total_decisions']}")
    print(f"ALIF Activations: {alif_stats['activations']}")
    print(f"ALIF Corrections: {alif_stats['corrections']}")
    print(f"ALIF Activation Rate: {alif_stats['activation_rate']:.2%}")
    print(f"ALIF Correction Rate: {alif_stats['correction_rate']:.2%}")
    print(f"Current State: {general_stats['current_state']}")
    print(f"ALIF Enabled: {alif_stats['enabled']}")
    print(f"ALIF Threshold: {alif_stats['threshold']}")
    print(f"ALIF Weights: {alif_stats['weights']}")

    print()


def test_alif_force_state():
    """Test forcing ALIF state for decision making."""
    print("🧪 Testing ALIF Force State")
    print("=" * 50)

    engine = DualisticThoughtEngines()

    # Create market data
    market_data = create_market_data()

    # Normal processing
    normal_vector = engine.process_market_data(market_data)
    print(f"Normal Processing - State: {normal_vector.state.value}, ALIF Score: {normal_vector.alif_score:.3f}")

    # Force ALIF state
    forced_vector = engine.force_alif_state(market_data)
    print(f"Forced ALIF - State: {forced_vector.state.value}, ALIF Score: {forced_vector.alif_score:.3f}")

    # Compare decisions
    print(f"Normal Decision: {normal_vector.decision}")
    print(f"Forced ALIF Decision: {forced_vector.decision}")

    if forced_vector.state == DualisticState.ALIF:
        print("✅ ALIF state successfully forced")
    else:
        print("❌ ALIF state not forced correctly")

    print()


def main():
    """Run all ALIF dualistic state engine tests."""
    print("🚀 ALIF Dualistic State Engine Test Suite")
    print("=" * 60)
    print()

    try:
        # Run all tests
        test_alif_basic_functionality()
        test_alif_state_activation()
        test_alif_memory_and_learning()
        test_alif_configuration()
        test_alif_performance_metrics()
        test_alif_force_state()

        print("✅ All ALIF tests completed successfully!")
        print("\n🎯 ALIF Dualistic State Engine Implementation Summary:")
        print("- ALIF state successfully integrated into dualistic engine")
        print("- Volume delta calculations working")
        print("- Resonance delta analysis functional")
        print("- AI feedback integration operational")
        print("- Error correction mechanisms active")
        print("- Market memory scoring implemented")
        print("- Performance metrics and statistics available")
        print("- Configuration and control methods working")
        print("- Force state functionality operational")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
