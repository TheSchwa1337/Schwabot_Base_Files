#!/usr/bin/env python3
"""
🧠⚛️ LAYER 4: GHOST SHELL MEMORY + FRACTAL RECURSION ENGINE TEST SUITE
======================================================================

Comprehensive testing of Layer 4: Ghost Shell Memory + Fractal Recursion Engine
- Ghost Shell Memory Engine
- Fractal Memory Tracker  
- Strategy Loop Switcher
- Memory persistence and retrieval
- Fractal pattern recognition
- Recursive strategy replay
"""

import json
import logging
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.append('.')

try:
        ShellMemoryEngine, 
        GhostShellMemory,
        MemoryState,
        create_shell_memory_engine
    )
        FractalMemoryTracker,
        FractalSnapshot,
        FractalMatch,
        FractalMatchType,
        create_fractal_memory_tracker
    )
        StrategyLoopSwitcher,
        AssetTarget,
        StrategyResult,
        create_strategy_loop_switcher
    )
    from core.qutrit_signal_matrix import QutritSignalMatrix, QutritState
    LAYER4_READY = True
    except ImportError as e:
    print(f"❌ Critical import error: {e}")
    LAYER4_READY = False

def test_ghost_shell_memory_engine():
    """Test the Ghost Shell Memory Engine"""
    print("🧠⚛️ TESTING GHOST SHELL MEMORY ENGINE")
    print("=" * 60)

    if not LAYER4_READY:
        print("❌ Layer 4 components not ready. Aborting test.")
        return False

    try:
        # Create memory engine
        memory_engine = create_shell_memory_engine(max_size=100, ttl=3600)

        # Test data
        strategy_id = "test_strategy_789"
        q_matrix = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
        profit_vector = np.array([0.1, 0.2, 0.1])

        # Test 1: Save ghost shell
        print("\n📝 Test 1: Saving Ghost Shell")
        hash_key = memory_engine.save_shell(strategy_id, q_matrix, profit_vector, confidence=0.8)
        print(f"✅ Saved ghost shell with hash: {hash_key}")
        assert len(hash_key) > 0, "Hash key should not be empty"

        # Test 2: Load ghost shell
        print("\n📖 Test 2: Loading Ghost Shell")
        loaded_memory = memory_engine.load_shell(strategy_id, q_matrix)
        if loaded_memory:
            print(f"✅ Loaded ghost shell:")
            print(f"   - Profit vector: {loaded_memory['profit_vector']}")
            print(f"   - Confidence: {loaded_memory['confidence']:.3f}")
            print(f"   - Access count: {loaded_memory['access_count']}")
            print(f"   - State: {loaded_memory['state']}")
        else:
            print("❌ Failed to load ghost shell")
            return False

        # Test 3: Similar shell search
        print("\n🔍 Test 3: Similar Shell Search")
        similar_q_matrix = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 1]])  # Slightly different
        similar_shells = memory_engine.find_similar_shells(similar_q_matrix, similarity_threshold=0.7)
        print(f"✅ Found {len(similar_shells)} similar shells")

        # Test 4: Memory statistics
        print("\n📊 Test 4: Memory Statistics")
        stats = memory_engine.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        assert stats['total_memories'] > 0, "Should have at least one memory"

        # Test 5: Memory export/import
        print("\n💾 Test 5: Memory Export/Import")
        export_file = "test_ghost_memories.json"
        if memory_engine.export_memories(export_file):
            print(f"✅ Exported memories to {export_file}")

            # Create new engine and import
            new_engine = create_shell_memory_engine()
            if new_engine.import_memories(export_file):
                print("✅ Successfully imported memories")
                new_stats = new_engine.get_memory_stats()
                assert new_stats['total_memories'] == stats['total_memories'], "Import should preserve memory count"
            else:
                print("❌ Failed to import memories")
                return False
        else:
            print("❌ Failed to export memories")
            return False

        # Cleanup
        if os.path.exists(export_file):
            os.remove(export_file)

        print("✅ Ghost Shell Memory Engine tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Ghost Shell Memory Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fractal_memory_tracker():
    """Test the Fractal Memory Tracker"""
    print("\n🔄 TESTING FRACTAL MEMORY TRACKER")
    print("=" * 60)

    try:
        # Create fractal tracker
        fractal_tracker = create_fractal_memory_tracker(max_snapshots=100, similarity_threshold=0.8)

        # Test data
        strategy_id = "fractal_test_strategy"
        q_matrix_1 = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
        q_matrix_2 = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 1]])  # Slightly different
        q_matrix_3 = np.array([[1, 0, 2], [2, 1, 0], [0, 2, 1]])  # Very different

        # Test 1: Save snapshots
        print("\n📝 Test 1: Saving Fractal Snapshots")
        snapshot_id_1 = fractal_tracker.save_snapshot(q_matrix_1, strategy_id, profit_result=0.5)
        snapshot_id_2 = fractal_tracker.save_snapshot(q_matrix_2, strategy_id, profit_result=-0.2)
        print(f"✅ Saved snapshots: {snapshot_id_1}, {snapshot_id_2}")

        # Test 2: Fractal matching
        print("\n🔍 Test 2: Fractal Pattern Matching")
        match_1 = fractal_tracker.match_fractal(q_matrix_1, strategy_id)
        if match_1:
            print(f"✅ Exact match found: {match_1.match_type.value}")
            print(f"   - Similarity: {match_1.similarity_score:.3f}")
            print(f"   - Confidence: {match_1.confidence:.3f}")
            print(f"   - Replay recommended: {match_1.replay_recommended}")
        else:
            print("❌ No exact match found")
            return False

        match_2 = fractal_tracker.match_fractal(q_matrix_2, strategy_id)
        if match_2:
            print(f"✅ Similar match found: {match_2.match_type.value}")
            print(f"   - Similarity: {match_2.similarity_score:.3f}")
        else:
            print("❌ No similar match found")

        match_3 = fractal_tracker.match_fractal(q_matrix_3, strategy_id)
        if match_3:
            print(f"✅ Different match found: {match_3.match_type.value}")
        else:
            print("✅ No match found for very different matrix (expected)")

        # Test 3: Pattern statistics
        print("\n📊 Test 3: Pattern Statistics")
        stats = fractal_tracker.get_pattern_statistics()
        print(f"✅ Pattern stats: {stats}")
        assert stats['total_patterns'] > 0, "Should have at least one pattern"

        # Test 4: Recent patterns
        print("\n⏰ Test 4: Recent Pattern Search")
        recent_patterns = fractal_tracker.find_recent_patterns(hours_back=24)
        print(f"✅ Found {len(recent_patterns)} recent patterns")

        # Test 5: Pattern export/import
        print("\n💾 Test 5: Pattern Export/Import")
        export_file = "test_fractal_patterns.json"
        if fractal_tracker.export_patterns(export_file):
            print(f"✅ Exported patterns to {export_file}")

            # Create new tracker and import
            new_tracker = create_fractal_memory_tracker()
            if new_tracker.import_patterns(export_file):
                print("✅ Successfully imported patterns")
                new_stats = new_tracker.get_pattern_statistics()
                assert new_stats['total_patterns'] == stats['total_patterns'], "Import should preserve pattern count"
            else:
                print("❌ Failed to import patterns")
                return False
        else:
            print("❌ Failed to export patterns")
            return False

        # Cleanup
        if os.path.exists(export_file):
            os.remove(export_file)

        print("✅ Fractal Memory Tracker tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Fractal Memory Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_loop_switcher():
    """Test the Strategy Loop Switcher"""
    print("\n🔄 TESTING STRATEGY LOOP SWITCHER")
    print("=" * 60)

    try:
        # Create strategy loop switcher
        switcher = create_strategy_loop_switcher()

        # Mock data
        market_data = {}
            "timestamp": time.time(),
            "btc_price": 50000,
            "eth_price": 3000,
            "market_volatility": 0.3,
            "volume": 1500
        }

        portfolio = {}
            "BTC": 0.1,
            "ETH": 2.5,
            "XRP": 1000
        }

        # Test 1: Force cycle execution
        print("\n🔄 Test 1: Strategy Cycle Execution")
        results = switcher.force_cycle_execution(market_data, portfolio)
        print(f"✅ Executed {len(results)} strategies")

        for i, result in enumerate(results):
            print(f"  Strategy {i+1}: {result.asset}")
            print(f"    - Strategy ID: {result.strategy_id}")
            print(f"    - Ghost shell used: {result.ghost_shell_used}")
            print(f"    - Fractal match: {result.fractal_match}")
            print(f"    - Confidence: {result.confidence:.3f}")
            print(f"    - Profit vector: {result.profit_vector}")
            print(f"    - Execution time: {result.execution_time:.4f}s")

        # Test 2: Loop statistics
        print("\n📊 Test 2: Loop Statistics")
        stats = switcher.get_loop_statistics()
        print(f"✅ Loop stats:")
        print(f"   - Cycle count: {stats['cycle_count']}")
        print(f"   - Memory stats: {stats['memory_stats']}")
        print(f"   - Fractal stats: {stats['fractal_stats']}")

        # Test 3: Asset weight updates
        print("\n⚖️ Test 3: Asset Weight Updates")
        if switcher.update_asset_weights("BTC", 1.5):
            print("✅ Updated BTC weight to 1.5")
        else:
            print("❌ Failed to update BTC weight")
            return False

        # Test 4: Multiple cycle execution
        print("\n🔄 Test 4: Multiple Cycle Execution")
        for i in range(3):
            print(f"  Executing cycle {i+1}...")
            cycle_results = switcher.force_cycle_execution(market_data, portfolio)
            print(f"    - Results: {len(cycle_results)} strategies")

            # Check for ghost shell usage
            ghost_shells = sum(1 for r in cycle_results if r.ghost_shell_used)
            fractal_matches = sum(1 for r in cycle_results if r.fractal_match)
            print(f"    - Ghost shells: {ghost_shells}, Fractal matches: {fractal_matches}")

        # Test 5: Final statistics
        print("\n📊 Test 5: Final Statistics")
        final_stats = switcher.get_loop_statistics()
        print(f"✅ Final stats:")
        print(f"   - Total cycles: {final_stats['cycle_count']}")
        print(f"   - Total memories: {final_stats['memory_stats']['total_memories']}")
        print(f"   - Total patterns: {final_stats['fractal_stats']['total_patterns']}")

        print("✅ Strategy Loop Switcher tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Strategy Loop Switcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_layer4_system():
    """Test the complete integrated Layer 4 system"""
    print("\n🧠⚛️ TESTING INTEGRATED LAYER 4 SYSTEM")
    print("=" * 60)

    try:
        # Create all components
        memory_engine = create_shell_memory_engine()
        fractal_tracker = create_fractal_memory_tracker()
        loop_switcher = create_strategy_loop_switcher()

        # Test data
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

        # Test 1: Complete cycle with memory persistence
        print("\n🔄 Test 1: Complete Memory Cycle")

        # First cycle - should create new strategies
        print("  Executing first cycle...")
        results_1 = loop_switcher.force_cycle_execution(market_data, portfolio)
        print(f"    - First cycle results: {len(results_1)} strategies")

        # Second cycle - should reuse some ghost shells
        print("  Executing second cycle...")
        results_2 = loop_switcher.force_cycle_execution(market_data, portfolio)
        print(f"    - Second cycle results: {len(results_2)} strategies")

        # Check ghost shell usage
        ghost_shells_1 = sum(1 for r in results_1 if r.ghost_shell_used)
        ghost_shells_2 = sum(1 for r in results_2 if r.ghost_shell_used)
        print(f"    - Ghost shells used: Cycle 1: {ghost_shells_1}, Cycle 2: {ghost_shells_2}")

        # Test 2: Memory persistence across components
        print("\n💾 Test 2: Memory Persistence")
        memory_stats = memory_engine.get_memory_stats()
        fractal_stats = fractal_tracker.get_pattern_statistics()

        print(f"  Memory engine: {memory_stats['total_memories']} memories")
        print(f"  Fractal tracker: {fractal_stats['total_patterns']} patterns")

        # Test 3: Fractal pattern recognition
        print("\n🔄 Test 3: Fractal Pattern Recognition")

        # Create similar market conditions
        similar_market_data = {}
            "timestamp": time.time() + 100,
            "btc_price": 50100,  # Similar price
            "eth_price": 3010,   # Similar price
            "market_volatility": 0.41,  # Similar volatility
            "volume": 2100       # Similar volume
        }

        # Execute with similar conditions
        similar_results = loop_switcher.force_cycle_execution(similar_market_data, portfolio)
        fractal_matches = sum(1 for r in similar_results if r.fractal_match)
        print(f"  Similar conditions: {fractal_matches} fractal matches")

        # Test 4: System performance
        print("\n⚡ Test 4: System Performance")
        start_time = time.time()

        # Execute multiple cycles
        for i in range(5):
            loop_switcher.force_cycle_execution(market_data, portfolio)

        total_time = time.time() - start_time
        print(f"  Executed 5 cycles in {total_time:.2f}s")
        print(f"  Average time per cycle: {total_time/5:.3f}s")

        # Test 5: Memory cleanup and management
        print("\n🧹 Test 5: Memory Management")

        # Check memory utilization
        final_memory_stats = memory_engine.get_memory_stats()
        final_fractal_stats = fractal_tracker.get_pattern_statistics()

        print(f"  Memory utilization: {final_memory_stats['memory_utilization']:.2%}")
        print(f"  Pattern utilization: {final_fractal_stats['memory_utilization']:.2%}")

        # Test cleanup
        removed_patterns = fractal_tracker.cleanup_old_patterns(max_age_hours=1)
        print(f"  Cleaned up {removed_patterns} old patterns")

        print("✅ Integrated Layer 4 System tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Integrated Layer 4 System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Layer 4 tests"""
    print("🧠⚛️ LAYER 4: GHOST SHELL MEMORY + FRACTAL RECURSION ENGINE")
    print("=" * 80)
    print("Comprehensive Test Suite")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not LAYER4_READY:
        print("❌ Layer 4 components not available. Please ensure all dependencies are installed.")
        return False

    tests = []
        ("Ghost Shell Memory Engine", test_ghost_shell_memory_engine),
        ("Fractal Memory Tracker", test_fractal_memory_tracker),
        ("Strategy Loop Switcher", test_strategy_loop_switcher),
        ("Integrated Layer 4 System", test_integrated_layer4_system),
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
    print(f"LAYER 4 TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("🎉 ALL LAYER 4 TESTS PASSED!")
        print("🧠⚛️ Ghost Shell Memory + Fractal Recursion Engine is FULLY OPERATIONAL!")
        print("👻 Schwabot now has recursive memory intelligence!")
        print("🔄 Ready for Layer 5: Advanced Neural Integration + Live Trading")
        return True
    else:
        print("⚠️ Some Layer 4 tests failed. Review output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 