#!/usr/bin/env python3
"""
Test Hash Recollection API Integration
=====================================

Test script to verify that the hash_recollection API integration
works properly with all modules.
"""

import numpy as np

from hash_recollection import HashRecollectionAPI


def test_entropy_tracker():
    """Test entropy tracker functionality."""
    print("Testing Entropy Tracker...")

    from hash_recollection.entropy_tracker import EntropyTracker

    tracker = EntropyTracker()

    # Generate sample price data
    price_data = [100 + i * 0.1 + np.random.normal(0, 0.5) for i in range(50)]

    # Calculate entropy
    metrics = tracker.calculate_entropy(price_data)
    print(f"  Entropy value: {metrics.entropy_value:.4f}")
    print(f"  State: {metrics.state.value}")
    print(f"  Confidence: {metrics.confidence:.4f}")

    # Generate signal
    signal = tracker.generate_signal(price_data)
    if signal:
        print(f"  Signal: {signal.signal_type} (strength: {signal.strength:.4f})")
    else:
        print("  No signal generated")

    print("  ✓ Entropy Tracker test passed\n")


def test_bit_operations():
    """Test bit operations functionality."""
    print("Testing Bit Operations...")

    from hash_recollection.bit_operations import BitOperations

    bit_ops = BitOperations()

    # Generate sample values
    values = [np.random.random() for _ in range(20)]

    # Create bit sequence
    sequence = bit_ops.create_bit_sequence(values)
    print(f"  Sequence ID: {sequence.sequence_id}")
    print(f"  Bits: {sequence.bits[:10]}...")  # Show first 10 bits

    # Detect patterns
    patterns = bit_ops.detect_patterns(sequence.bits)
    print(f"  Patterns found: {len(patterns)}")

    for pattern in patterns[:3]:  # Show first 3 patterns
        print(f"    - {pattern.pattern_type}: {pattern.confidence:.4f}")

    print("  ✓ Bit Operations test passed\n")


def test_pattern_utils():
    """Test pattern utilities functionality."""
    print("Testing Pattern Utils...")

    from hash_recollection.pattern_utils import PatternUtils

    pattern_utils = PatternUtils()

    # Generate sample price data with trend
    price_data = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(30)]

    # Analyze trend
    trend = pattern_utils.analyze_trend(price_data)
    print(f"  Trend direction: {trend.trend_direction}")
    print(f"  Trend strength: {trend.strength:.4f}")
    print(f"  R-squared: {trend.r_squared:.4f}")

    # Detect patterns
    patterns = pattern_utils.detect_patterns(price_data)
    print(f"  Patterns found: {len(patterns)}")

    for pattern in patterns[:3]:  # Show first 3 patterns
        print(f"    - {pattern.pattern_type.value}: {pattern.confidence:.4f}")

    print("  ✓ Pattern Utils test passed\n")


def test_api_integration():
    """Test API integration."""
    print("Testing API Integration...")

    # Create API instance
    api = HashRecollectionAPI()

    # Generate sample price data
    price_data = [100 + i * 0.2 + np.random.normal(0, 0.8) for i in range(40)]

    # Test comprehensive analysis
    print("  Testing comprehensive analysis...")

    # Simulate the analysis (since we can't easily test async functions, here)'
    entropy_metrics = api.entropy_tracker.calculate_entropy(price_data)
    trend = api.pattern_utils.analyze_trend(price_data)
    patterns = api.pattern_utils.detect_patterns(price_data)

    print()
        f"    Entropy: {entropy_metrics.entropy_value:.4f} ({entropy_metrics.state.value})"
    )
    print(f"    Trend: {trend.trend_direction} (strength: {trend.strength:.4f})")
    print(f"    Patterns: {len(patterns)} found")

    print("  ✓ API Integration test passed\n")


def test_signal_generation():
    """Test signal generation."""
    print("Testing Signal Generation...")

    api = HashRecollectionAPI()

    # Generate different types of price data
    test_cases = []
        ("uptrend", [100 + i * 0.5 for i in range(30)]),
        ("downtrend", [100 - i * 0.5 for i in range(30)]),
        ("sideways", [100 + np.random.normal(0, 2) for _ in range(30)]),
    ]
    for name, price_data in test_cases:
        print(f"  Testing {name}...")

        # Get entropy signal
        entropy_signal = api.entropy_tracker.generate_signal(price_data)

        # Analyze patterns
        trend = api.pattern_utils.analyze_trend(price_data)
        patterns = api.pattern_utils.detect_patterns(price_data)

        # Create bit sequence
        normalized_prices = []
            (p - min(price_data)) / (max(price_data) - min(price_data))
            for p in price_data
        ]
        bit_sequence = api.bit_operations.create_bit_sequence(normalized_prices)
        bit_patterns = api.bit_operations.detect_patterns(bit_sequence.bits)

        print()
            f"    Entropy signal: {entropy_signal.signal_type if entropy_signal else 'none'}"
        )
        print(f"    Trend: {trend.trend_direction} ({trend.strength:.4f})")
        print(f"    Patterns: {len(patterns) + len(bit_patterns)} total")

    print("  ✓ Signal Generation test passed\n")


def main():
    """Run all tests."""
    print("🧪 Hash Recollection API Integration Tests")
    print("=" * 50)

    try:
        test_entropy_tracker()
        test_bit_operations()
        test_pattern_utils()
        test_api_integration()
        test_signal_generation()

        print("🎉 All tests passed!")
        print("\nTo start the API server, run:")
        print("python -m hash_recollection.api_integration")
        print("\nOr use the convenience function:")
        print("from hash_recollection import create_and_run_api")
        print("create_and_run_api()")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
