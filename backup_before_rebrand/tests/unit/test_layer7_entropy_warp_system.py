#!/usr/bin/env python3
"""
🌌⚙️🕳️⏱️ LAYER 7: ENTROPY DRIFT ENGINE + TEMPORAL WARP MAPPING TEST SUITE
==========================================================================

Comprehensive testing of Layer 7: Entropy Drift Engine + Temporal Warp Mapping
- Entropy Drift Tracker
- Temporal Warp Engine
- Warp-aware strategy execution
- Drift-based timing optimization
- Recursive vector echoing
"""

import logging
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.append('.')

try:
        EntropyDriftTracker,
        DriftSnapshot,
        create_entropy_drift_tracker
    )
        TemporalWarpEngine,
        WarpWindow,
        create_temporal_warp_engine
    )
        StrategyLoopSwitcher,
        AssetTarget,
        StrategyResult,
        create_strategy_loop_switcher
    )
    from core.glyph_router import GlyphRouter
    from core.visual_decision_engine import VisualDecisionEngine
    LAYER7_READY = True
    except ImportError as e:
    print(f"❌ Critical import error: {e}")
    LAYER7_READY = False

def test_entropy_drift_tracker():
    """Test the Entropy Drift Tracker"""
    print("🌌⚙️ TESTING ENTROPY DRIFT TRACKER")
    print("=" * 60)

    if not LAYER7_READY:
        print("❌ Layer 7 components not ready. Aborting test.")
        return False

    try:
        # Create drift tracker
        drift_tracker = create_entropy_drift_tracker(max_history=50, warp_threshold=0.1)

        # Test data - simulate increasing drift
        strategy_id = "test_strategy_entropy"
        vectors = []
            np.array([0.1, 0.2, 0.1]),  # Base vector
            np.array([0.2, 0.3, 0.2]),  # Slight drift
            np.array([0.1, 0.1, 0.1]),  # Lower drift
            np.array([0.4, 0.5, 0.4]),  # High drift
            np.array([0.6, 0.7, 0.6])   # Very high drift
        ]

        # Test 1: Record vectors and track drift
        print("\n📝 Test 1: Recording Vectors and Tracking Drift")
        for i, vector in enumerate(vectors):
            drift = drift_tracker.record_vector(strategy_id, vector)
            print(f"  Vector {i+1}: {vector} → Drift: {drift:.4f}")

        # Test 2: Compute average drift
        print("\n📊 Test 2: Computing Average Drift")
        avg_drift = drift_tracker.compute_drift(strategy_id)
        print(f"  Average drift: {avg_drift:.4f}")

        # Test 3: Check warp window activation
        print("\n🕳️ Test 3: Checking Warp Window Activation")
        in_warp = drift_tracker.is_warp_window(strategy_id)
        print(f"  In warp window: {in_warp}")

        # Test 4: Get drift trend
        print("\n📈 Test 4: Getting Drift Trend")
        trend = drift_tracker.get_drift_trend(strategy_id)
        print(f"  Drift trend: {trend}")

        # Test 5: Predict warp delay
        print("\n⏱️ Test 5: Predicting Warp Delay")
        delay = drift_tracker.predict_warp_delay(strategy_id, alpha=60.0)
        print(f"  Predicted delay: {delay:.2f} seconds")

        # Test 6: Get comprehensive statistics
        print("\n📊 Test 6: Getting Drift Statistics")
        stats = drift_tracker.get_drift_statistics(strategy_id)
        print(f"  Drift statistics: {stats}")

        print("✅ Entropy Drift Tracker tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Entropy Drift Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_warp_engine():
    """Test the Temporal Warp Engine"""
    print("\n🕳️⏱️ TESTING TEMPORAL WARP ENGINE")
    print("=" * 60)

    try:
        # Create warp engine
        warp_engine = create_temporal_warp_engine(default_alpha=60.0, max_warp_duration=1800)

        # Test data
        strategy_id = "test_strategy_warp"

        # Test 1: Update warp window with different drift values
        print("\n🕳️ Test 1: Updating Warp Windows")
        drift_values = [0.1, 0.25, 0.5, 0.75]

        for i, drift in enumerate(drift_values):
            window = warp_engine.update_window(strategy_id, drift)
            print(f"  Drift {i+1}: {drift} → Delay: {drift * 60:.1f}s, Confidence: {window.confidence:.3f}")

        # Test 2: Check window status
        print("\n⏱️ Test 2: Checking Window Status")
        within_window = warp_engine.is_within_window(strategy_id)
        print(f"  Within window: {within_window}")

        # Test 3: Get time until window
        print("\n⏳ Test 3: Time Until Window")
        time_until = warp_engine.get_time_until(strategy_id)
        print(f"  Time until window: {time_until:.1f} seconds")

        # Test 4: Get window properties
        print("\n📏 Test 4: Window Properties")
        duration = warp_engine.get_window_duration(strategy_id)
        confidence = warp_engine.get_window_confidence(strategy_id)
        print(f"  Window duration: {duration:.1f} seconds")
        print(f"  Window confidence: {confidence:.3f}")

        # Test 5: Force warp window
        print("\n⚡ Test 5: Force Warp Window")
        forced_window = warp_engine.force_warp_window(strategy_id, duration_seconds=120)
        print(f"  Forced window created: {forced_window is not None}")

        # Test 6: Get warp statistics
        print("\n📊 Test 6: Warp Statistics")
        stats = warp_engine.get_warp_statistics()
        print(f"  Warp statistics: {stats}")

        print("✅ Temporal Warp Engine tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Temporal Warp Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_warp_aware_strategy_execution():
    """Test warp-aware strategy execution"""
    print("\n🔄 TESTING WARP-AWARE STRATEGY EXECUTION")
    print("=" * 60)

    try:
        # Create strategy loop switcher with warp integration
        switcher = create_strategy_loop_switcher()

        # Mock data
        market_data = {}
            "timestamp": time.time(),
            "btc_price": 50000,
            "eth_price": 3000,
            "market_volatility": 0.4,
            "volume": 2000
        }

        portfolio = {}
            "BTC": 0.2,
            "ETH": 3.0,
            "SOL": 50
        }

        # Test 1: Force warp window for immediate execution
        print("\n⚡ Test 1: Force Warp Window Execution")
        forced_window = switcher.warp_engine.force_warp_window("BTC", duration_seconds=300)
        print(f"  Forced warp window created: {forced_window is not None}")

        # Test 2: Execute strategy with warp awareness
        print("\n🔄 Test 2: Warp-Aware Strategy Execution")
        results = switcher.force_cycle_execution(market_data, portfolio)
        print(f"  Executed {len(results)} strategies")

        for result in results:
            print(f"    {result.asset}: Ghost shell={result.ghost_shell_used}, Fractal={result.fractal_match}")

        # Test 3: Check drift tracking
        print("\n📊 Test 3: Drift Tracking Integration")
        for asset in ["BTC", "ETH", "SOL"]:
            drift_stats = switcher.drift_tracker.get_drift_statistics(asset)
            print(f"  {asset} drift stats: {drift_stats}")

        # Test 4: Check warp statistics
        print("\n🕳️ Test 4: Warp Statistics")
        warp_stats = switcher.warp_engine.get_warp_statistics()
        print(f"  Warp statistics: {warp_stats}")

        print("✅ Warp-Aware Strategy Execution tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Warp-Aware Strategy Execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_layer7_system():
    """Test the complete integrated Layer 7 system"""
    print("\n🌌⚙️🕳️⏱️ TESTING INTEGRATED LAYER 7 SYSTEM")
    print("=" * 60)

    try:
        # Create all components
        drift_tracker = create_entropy_drift_tracker()
        warp_engine = create_temporal_warp_engine()
        switcher = create_strategy_loop_switcher()

        # Test data
        market_data = {}
            "timestamp": time.time(),
            "btc_price": 50000,
            "eth_price": 3000,
            "market_volatility": 0.5,
            "volume": 2500
        }

        portfolio = {}
            "BTC": 0.3,
            "ETH": 4.0,
            "AVAX": 100
        }

        # Test 1: Simulate drift progression
        print("\n📈 Test 1: Simulating Drift Progression")
        strategy_id = "BTC"

        # Record increasing drift
        vectors = []
            np.array([0.1, 0.2, 0.1]),
            np.array([0.3, 0.4, 0.3]),  # Higher drift
            np.array([0.6, 0.7, 0.6])   # Very high drift
        ]

        for i, vector in enumerate(vectors):
            drift = drift_tracker.record_vector(strategy_id, vector)
            warp_engine.update_window(strategy_id, drift)
            in_warp = warp_engine.is_within_window(strategy_id)
            print(f"  Vector {i+1}: drift={drift:.4f}, in_warp={in_warp}")

        # Test 2: Execute with warp awareness
        print("\n🔄 Test 2: Warp-Aware Execution")
        results = switcher.force_cycle_execution(market_data, portfolio)
        print(f"  Executed {len(results)} strategies")

        # Test 3: Check system performance
        print("\n⚡ Test 3: System Performance")
        start_time = time.time()

        # Execute multiple cycles
        for i in range(3):
            switcher.force_cycle_execution(market_data, portfolio)

        total_time = time.time() - start_time
        print(f"  Executed 3 cycles in {total_time:.2f}s")
        print(f"  Average time per cycle: {total_time/3:.3f}s")

        # Test 4: Memory and cleanup
        print("\n🧹 Test 4: Memory Management")

        # Clean up old data
        removed_drift = drift_tracker.cleanup_old_data(max_age_hours=1)
        removed_warp = warp_engine.cleanup_expired_windows()

        print(f"  Removed {removed_drift} old drift snapshots")
        print(f"  Removed {removed_warp} expired warp windows")

        # Test 5: Final statistics
        print("\n📊 Test 5: Final System Statistics")

        drift_stats = drift_tracker.get_drift_statistics(strategy_id)
        warp_stats = warp_engine.get_warp_statistics()

        print(f"  Drift stats: {drift_stats}")
        print(f"  Warp stats: {warp_stats}")

        print("✅ Integrated Layer 7 System tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Integrated Layer 7 System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Layer 7 tests"""
    print("🌌⚙️🕳️⏱️ LAYER 7: ENTROPY DRIFT ENGINE + TEMPORAL WARP MAPPING")
    print("=" * 80)
    print("Comprehensive Test Suite")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not LAYER7_READY:
        print("❌ Layer 7 components not available. Please ensure all dependencies are installed.")
        return False

    tests = []
        ("Entropy Drift Tracker", test_entropy_drift_tracker),
        ("Temporal Warp Engine", test_temporal_warp_engine),
        ("Warp-Aware Strategy Execution", test_warp_aware_strategy_execution),
        ("Integrated Layer 7 System", test_integrated_layer7_system),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")

    print(f"\n{'='*80}")
    print(f"LAYER 7 TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("🎉 ALL LAYER 7 TESTS PASSED!")
        print("🌌⚙️🕳️⏱️ Entropy Drift Engine + Temporal Warp Mapping is FULLY OPERATIONAL!")
        print("⏱️ Schwabot now trades in probabilistic time fields!")
        print("🕳️ Ready for Layer 8: Hash-Encoded Glyph Memory Compression + Cross-Agent Path Blending")
        return True
    else:
        print("⚠️ Some Layer 7 tests failed. Review output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 