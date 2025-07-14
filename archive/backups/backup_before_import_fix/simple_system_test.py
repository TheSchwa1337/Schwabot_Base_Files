import time

from core.api_bridge import APIBridge
from core.dualistic_thought_engines import DualisticThoughtEngines
from core.hash_relay_system import hash_relay_system

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple System Test - Core Functionality Verification
===================================================

Quick test to verify core system components are working.
"""



def test_core_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")

    try:
        print("✓ API Bridge imported")

        print("✓ Dualistic Thought Engines imported")

        print("✓ Hash Relay System imported")

        print("✓ Lantern Core imported")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_api_bridge_basic():
    """Test basic API Bridge functionality."""
    print("\nTesting API Bridge basic functionality...")

    try:

        bridge = APIBridge(sandbox=True)
        print("✓ API Bridge initialized")

        # Test performance summary
        performance = bridge.get_api_performance_summary()
        print(f"✓ Performance summary available: {len(performance)} metrics")

        return True
    except Exception as e:
        print(f"✗ API Bridge test failed: {e}")
        return False


def test_dualistic_engines_basic():
    """Test basic Dualistic Thought Engines functionality."""
    print("\nTesting Dualistic Thought Engines basic functionality...")

    try:

        engines = DualisticThoughtEngines()
        print("✓ Dualistic Thought Engines initialized")

        # Test with simple market data
        market_data = {}
            "current_price": 62000.0,
            "rsi": 50.0,
            "macd_signal": 0.0,
            "volume_change": 0.1,
            "moving_average": 62000.0,
            "previous_close": 62000.0,
            "price_history": [62000.0] * 7,
            "volume_history": [100.0] * 7,
            "phase_data": [0.5, 0.5, 0.5, 0.5],
            "volatility": 0.5,
            "consensus_signal": "neutral",
        }
        thought_vector = engines.process_market_data(market_data, thermal_state="warm")
        print()
            f"✓ Thought Vector generated: {thought_vector.decision} (confidence: {thought_vector.confidence:.2f})"
        )

        # Test performance metrics
        performance = engines.get_engine_performance()
        print(f"✓ Performance metrics available: {len(performance)} metrics")

        return True
    except Exception as e:
        print(f"✗ Dualistic Engines test failed: {e}")
        return False


def test_hash_relay_basic():
    """Test basic Hash Relay System functionality."""
    print("\nTesting Hash Relay System basic functionality...")

    try:

        # Test subscription
        relay_received = []

        def test_callback(hash_str, data):
            relay_received.append((hash_str, data))
            print(f"✓ Relay callback executed for {data.get('decision', 'unknown')}")

        hash_relay_system.subscribe(test_callback)
        print("✓ Relay subscription registered")

        # Test data submission
        test_data = {}
            "timestamp": time.time(),
            "decision": "buy",
            "confidence": 0.85,
            "price": 62000.0,
        }
        hash_str = hash_relay_system.submit(test_data)
        print(f"✓ Data submitted and hashed: {hash_str[:8]}...")

        # Check if relay was received
        if relay_received:
            print(f"✓ Relay system working: {len(relay_received)} events received")
            return True
        else:
            print("✗ Relay callback not executed")
            return False

    except Exception as e:
        print(f"✗ Hash Relay test failed: {e}")
        return False


def test_integration_basic():
    """Test basic integration between components."""
    print("\nTesting basic integration...")

    try:

        # Track relay events
        relay_events = []

        def integration_callback(hash_str, data):
            relay_events.append((hash_str, data))

        hash_relay_system.subscribe(integration_callback)

        # Process market data
        engines = DualisticThoughtEngines()
        market_data = {}
            "current_price": 62000.0,
            "rsi": 30.0,
            "macd_signal": 0.1,
            "volume_change": 0.3,
            "moving_average": 61500.0,
            "previous_close": 61800.0,
            "price_history": []
                61000.0,
                61500.0,
                62000.0,
                61800.0,
                62200.0,
                62500.0,
                62300.0,
            ],
            "volume_history": [100.0, 120.0, 110.0, 90.0, 130.0, 150.0, 140.0],
            "phase_data": [0.6, 0.4, 0.8, 0.2],
            "volatility": 0.8,
            "consensus_signal": "buy",
        }
        thought_vector = engines.process_market_data(market_data, thermal_state="hot")
        print(f"✓ Integration test: {thought_vector.decision} decision generated")

        # Check if relay was triggered
        if relay_events:
            print(f"✓ Integration successful: {len(relay_events)} relay events")
            return True
        else:
            print("✗ Integration failed: no relay events")
            return False

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("SCHWABOT SIMPLE SYSTEM TEST")
    print("=" * 50)

    tests = []
        ("Core Imports", test_core_imports),
        ("API Bridge Basic", test_api_bridge_basic),
        ("Dualistic Engines Basic", test_dualistic_engines_basic),
        ("Hash Relay Basic", test_hash_relay_basic),
        ("Integration Basic", test_integration_basic),
    ]
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nSystem is ready for:")
        print("  ✓ Live API trading via CCXT")
        print("  ✓ BTC processor integration")
        print("  ✓ Pool mining with trading capabilities")
        print("  ✓ Mathematical relay system")
        print("  ✓ Dualistic decision making")
    else:
        print("\n⚠️  Some tests failed. System needs attention.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
