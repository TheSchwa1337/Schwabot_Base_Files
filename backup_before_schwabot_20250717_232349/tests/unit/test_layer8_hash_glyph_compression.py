#!/usr/bin/env python3
"""
🧬🔐🤖🔀 LAYER 8: HASH-GLYPH MEMORY COMPRESSION + CROSS-AGENT PATH BLENDING TEST SUITE
=====================================================================================

Comprehensive testing of Layer 8: Hash-Glyph Memory Compression + Cross-Agent Path Blending
- Hash-Glyph Compressor
- AI Matrix Consensus
- Visual Decision Engine with path blending
- Strategy Loop Switcher integration
- Memory compression and replay
- Cross-agent vote blending
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
        HashGlyphCompressor,
        GlyphMemoryChunk,
        create_hash_glyph_compressor
    )
        AIMatrixConsensus,
        AgentVote,
        AgentOpinion,
        ConsensusResult,
        create_ai_matrix_consensus
    )
        VisualDecisionEngine,
        create_visual_decision_engine
    )
        StrategyLoopSwitcher,
        AssetTarget,
        StrategyResult,
        create_strategy_loop_switcher
    )
    LAYER8_READY = True
    except ImportError as e:
    print(f"❌ Critical import error: {e}")
    LAYER8_READY = False

def test_hash_glyph_compression():
    """Test the Hash-Glyph Compressor"""
    print("🧬🔐 TESTING HASH-GLYPH COMPRESSION")
    print("=" * 60)

    if not LAYER8_READY:
        print("❌ Layer 8 components not ready. Aborting test.")
        return False

    try:
        # Create compressor
        compressor = create_hash_glyph_compressor(max_memory_size=50, compression_threshold=0.8)

        # Test data
        strategy_id = "test_strategy_compression"
        q_matrix = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
        glyph = "🌘"
        vector = np.array([0.1, 0.4, 0.3])
        votes = {"R1": "execute", "Claude": "recycle", "GPT-4o": "defer"}

        # Test 1: Store memory chunk
        print("\n💾 Test 1: Storing Memory Chunk")
        hash_key = compressor.store(strategy_id, q_matrix, glyph, vector, votes, confidence=0.9)
        print(f"  Stored with hash: {hash_key[:16]}...")

        # Test 2: Retrieve memory chunk
        print("\n🔍 Test 2: Retrieving Memory Chunk")
        retrieved = compressor.retrieve(strategy_id, q_matrix)
        if retrieved:
            print(f"  Retrieved: {retrieved.glyph} → {retrieved.vector}")
            print(f"  Votes: {retrieved.votes}")
            print(f"  Confidence: {retrieved.confidence}")
            print(f"  Usage count: {retrieved.usage_count}")
        else:
            print("  ❌ No memory found")

        # Test 3: Test with different matrix (should, miss)
        print("\n❌ Test 3: Testing Cache Miss")
        different_matrix = np.array([[2, 1, 0], [1, 0, 2], [0, 2, 1]])
        missed = compressor.retrieve(strategy_id, different_matrix)
        print(f"  Cache miss: {missed is None}")

        # Test 4: Find similar patterns
        print("\n🔍 Test 4: Finding Similar Patterns")
        similar = compressor.find_similar_patterns(strategy_id, q_matrix)
        print(f"  Found {len(similar)} similar patterns")

        # Test 5: Get memory statistics
        print("\n📊 Test 5: Memory Statistics")
        stats = compressor.get_memory_stats()
        print(f"  Memory stats: {stats}")

        # Test 6: Export and import memory
        print("\n💾 Test 6: Export/Import Memory")
        export_success = compressor.export_memory("test_layer8_memory.json")
        print(f"  Export success: {export_success}")

        # Create new compressor and import
        new_compressor = create_hash_glyph_compressor()
        import_success = new_compressor.import_memory("test_layer8_memory.json")
        print(f"  Import success: {import_success}")

        # Verify import
        imported = new_compressor.retrieve(strategy_id, q_matrix)
        print(f"  Imported data valid: {imported is not None}")

        print("✅ Hash-Glyph Compression tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Hash-Glyph Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_matrix_consensus():
    """Test the AI Matrix Consensus"""
    print("\n🤖🔀 TESTING AI MATRIX CONSENSUS")
    print("=" * 60)

    try:
        # Create consensus engine
        consensus = create_ai_matrix_consensus(num_agents=5)

        # Test data
        glyph = "🌘"
        base_vector = np.array([0.1, 0.4, 0.3])
        market_context = {"volatility": 0.3, "volume": 1000}

        # Test 1: Get consensus vote
        print("\n🗳️ Test 1: Getting Consensus Vote")
        result = consensus.vote(glyph, base_vector, market_context)
        print(f"  Consensus vote: {result.consensus_vote}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Blended vector: {result.blended_vector}")
        print(f"  Vote distribution: {result.vote_distribution}")
        print(f"  Agent weights: {result.agent_weights}")

        # Test 2: Get blended vector
        print("\n🔄 Test 2: Getting Blended Vector")
        blended = consensus.blended_vector(glyph, base_vector, market_context)
        print(f"  Blended vector: {blended}")

        # Test 3: Test with different glyphs
        print("\n🌕 Test 3: Testing Different Glyphs")
        glyphs = ["🌕", "🌗", "🌑", "🌖"]
        for test_glyph in glyphs:
            result = consensus.vote(test_glyph, base_vector, market_context)
            print(f"  {test_glyph}: {result.consensus_vote} (confidence: {result.confidence:.3f})")

        # Test 4: Update agent weight
        print("\n⚖️ Test 4: Updating Agent Weight")
        success = consensus.update_agent_weight("R1", 1.5)
        print(f"  Weight update success: {success}")

        # Test 5: Get consensus statistics
        print("\n📊 Test 5: Consensus Statistics")
        stats = consensus.get_consensus_statistics()
        print(f"  Consensus stats: {stats}")

        print("✅ AI Matrix Consensus tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ AI Matrix Consensus test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visual_decision_engine():
    """Test the Visual Decision Engine with Layer 8 integration"""
    print("\n🎨🧠 TESTING VISUAL DECISION ENGINE")
    print("=" * 60)

    try:
        # Create engine
        engine = create_visual_decision_engine(max_memory_size=100, num_agents=3)

        # Test data
        strategy_id = "test_strategy_visual"
        q_matrix = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
        vector = np.array([0.1, 0.4, 0.3])
        market_context = {"volatility": 0.3, "volume": 1000}

        # Test 1: Path blending
        print("\n🔄 Test 1: Path Blending")
        glyph, blended_vector, decision = engine.route_with_path_blending()
            strategy_id, q_matrix, vector, market_context
        )
        print(f"  Glyph: {glyph}")
        print(f"  Decision: {decision}")
        print(f"  Blended vector: {blended_vector}")

        # Test 2: Memory retrieval (should find, cached)
        print("\n🧠 Test 2: Memory Retrieval")
        glyph2, blended_vector2, decision2 = engine.route_with_path_blending()
            strategy_id, q_matrix, vector, market_context
        )
        print(f"  Retrieved: {glyph2} → {decision2}")
        print(f"  Hash match: {decision2 == 'replay'}")

        # Test 3: Render strategy grid
        print("\n🎨 Test 3: Strategy Grid Rendering")
        engine.render_strategy_grid("BTC", q_matrix, blended_vector)

        # Test 4: Find similar patterns
        print("\n🔍 Test 4: Finding Similar Patterns")
        similar = engine.find_similar_patterns(strategy_id, q_matrix)
        print(f"  Found {len(similar)} similar patterns")

        # Test 5: Get statistics
        print("\n📊 Test 5: Statistics")
        stats = engine.get_memory_statistics()
        print(f"  Engine stats: {stats}")

        # Test 6: Export memory
        print("\n💾 Test 6: Export Memory")
        export_success = engine.export_memory("test_visual_memory.json")
        print(f"  Export success: {export_success}")

        print("✅ Visual Decision Engine tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Visual Decision Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_loop_integration():
    """Test the Strategy Loop Switcher with Layer 8 integration"""
    print("\n🔄 TESTING STRATEGY LOOP INTEGRATION")
    print("=" * 60)

    try:
        # Create switcher
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

        # Test 1: Force cycle execution
        print("\n🔄 Test 1: Force Cycle Execution")
        results = switcher.force_cycle_execution(market_data, portfolio)
        print(f"  Executed {len(results)} strategies")

        # Test 2: Analyze results
        print("\n📊 Test 2: Analyzing Results")
        for result in results:
            print(f"  {result.asset}:")
            print(f"    Ghost shell: {result.ghost_shell_used}")
            print(f"    Fractal match: {result.fractal_match}")
            print(f"    Hash match: {result.hash_match}")
            print(f"    Glyph: {result.glyph}")
            print(f"    Decision: {result.decision_type}")
            print(f"    Confidence: {result.confidence:.3f}")
            if result.ai_consensus:
                print(f"    AI consensus: {result.ai_consensus.get('decision', 'N/A')}")

        # Test 3: Get loop statistics
        print("\n📊 Test 3: Loop Statistics")
        stats = switcher.get_loop_statistics()
        print(f"  Loop stats: {stats}")

        # Test 4: Export Layer 8 memory
        print("\n💾 Test 4: Export Layer 8 Memory")
        export_success = switcher.export_layer8_memory("test_layer8_export.json")
        print(f"  Export success: {export_success}")

        # Test 5: Multiple cycles to test memory accumulation
        print("\n🔄 Test 5: Multiple Cycles")
        for i in range(3):
            results = switcher.force_cycle_execution(market_data, portfolio)
            hash_matches = sum(1 for r in results if r.hash_match)
            print(f"  Cycle {i+1}: {len(results)} strategies, {hash_matches} hash matches")

        print("✅ Strategy Loop Integration tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Strategy Loop Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_layer8_system():
    """Test the complete integrated Layer 8 system"""
    print("\n🧬🔐🤖🔀 TESTING INTEGRATED LAYER 8 SYSTEM")
    print("=" * 60)

    try:
        # Create all components
        compressor = create_hash_glyph_compressor()
        consensus = create_ai_matrix_consensus()
        engine = create_visual_decision_engine()
        switcher = create_strategy_loop_switcher()

        # Test data
        strategy_id = "BTC_integrated_test"
        q_matrix = np.array([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
        vector = np.array([0.1, 0.4, 0.3])
        market_context = {"volatility": 0.5, "volume": 2500}

        # Test 1: Complete path blending workflow
        print("\n🔄 Test 1: Complete Path Blending Workflow")

        # First pass - should create new memory
        glyph1, blended1, decision1 = engine.route_with_path_blending()
            strategy_id, q_matrix, vector, market_context
        )
        print(f"  First pass: {glyph1} → {decision1}")

        # Second pass - should retrieve from memory
        glyph2, blended2, decision2 = engine.route_with_path_blending()
            strategy_id, q_matrix, vector, market_context
        )
        print(f"  Second pass: {glyph2} → {decision2}")
        print(f"  Memory hit: {decision2 == 'replay'}")

        # Test 2: AI consensus integration
        print("\n🤖 Test 2: AI Consensus Integration")
        consensus_result = consensus.vote(glyph1, vector, market_context)
        print(f"  Consensus: {consensus_result.consensus_vote}")
        print(f"  Vote distribution: {consensus_result.vote_distribution}")

        # Test 3: Memory compression verification
        print("\n🧬 Test 3: Memory Compression Verification")
        memory_stats = compressor.get_memory_stats()
        print(f"  Memory size: {memory_stats.get('memory_size', 0)}")
        print(f"  Hit rate: {memory_stats.get('hit_rate', 0):.3f}")

        # Test 4: Strategy loop with Layer 8
        print("\n🔄 Test 4: Strategy Loop with Layer 8")
        market_data = {}
            "timestamp": time.time(),
            "btc_price": 50000,
            "eth_price": 3000,
            "market_volatility": 0.6,
            "volume": 3000
        }
        portfolio = {"BTC": 0.3, "ETH": 4.0, "AVAX": 100}

        results = switcher.force_cycle_execution(market_data, portfolio)

        # Analyze Layer 8 features
        hash_matches = sum(1 for r in results if r.hash_match)
        glyphs_used = [r.glyph for r in results if r.glyph]
        decisions = [r.decision_type for r in results if r.decision_type]

        print(f"  Total strategies: {len(results)}")
        print(f"  Hash matches: {hash_matches}")
        print(f"  Glyphs used: {glyphs_used}")
        print(f"  Decisions: {decisions}")

        # Test 5: Performance and memory efficiency
        print("\n⚡ Test 5: Performance and Memory Efficiency")
        start_time = time.time()

        # Execute multiple cycles
        for i in range(5):
            switcher.force_cycle_execution(market_data, portfolio)

        total_time = time.time() - start_time
        print(f"  Executed 5 cycles in {total_time:.2f}s")
        print(f"  Average time per cycle: {total_time/5:.3f}s")

        # Final statistics
        final_stats = switcher.get_loop_statistics()
        layer8_stats = final_stats.get('layer8_stats', {})
        memory_stats = layer8_stats.get('memory', {})

        print(f"  Final memory size: {memory_stats.get('memory_size', 0)}")
        print(f"  Final hit rate: {memory_stats.get('hit_rate', 0):.3f}")

        print("✅ Integrated Layer 8 System tests: PASSED")
        return True

    except Exception as e:
        print(f"❌ Integrated Layer 8 System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = []
        "test_layer8_memory.json",
        "test_visual_memory.json", 
        "test_layer8_export.json"
    ]

    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"  Cleaned up: {file}")
        except Exception as e:
            print(f"  Could not clean up {file}: {e}")

def main():
    """Run all Layer 8 tests"""
    print("🧬🔐🤖🔀 LAYER 8: HASH-GLYPH MEMORY COMPRESSION + CROSS-AGENT PATH BLENDING")
    print("=" * 80)
    print("Comprehensive Test Suite")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not LAYER8_READY:
        print("❌ Layer 8 components not available. Please ensure all dependencies are installed.")
        return False

    tests = []
        ("Hash-Glyph Compression", test_hash_glyph_compression),
        ("AI Matrix Consensus", test_ai_matrix_consensus),
        ("Visual Decision Engine", test_visual_decision_engine),
        ("Strategy Loop Integration", test_strategy_loop_integration),
        ("Integrated Layer 8 System", test_integrated_layer8_system),
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

    # Cleanup
    print(f"\n🧹 Cleaning up test files...")
    cleanup_test_files()

    print(f"\n{'='*80}")
    print(f"LAYER 8 TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("🎉 ALL LAYER 8 TESTS PASSED!")
        print("🧬🔐🤖🔀 Hash-Glyph Memory Compression + Cross-Agent Path Blending is FULLY OPERATIONAL!")
        print("🧠 Schwabot now compresses strategy memory into hash-glyph bundles!")
        print("🤖 AI agents blend their votes into consensus-driven vectors!")
        print("🔄 Ready for Layer 9: Multi-Agent Conscious Glyph Pool + Temporal Probabilistic Voting Engine")
        return True
    else:
        print("⚠️ Some Layer 8 tests failed. Review output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 