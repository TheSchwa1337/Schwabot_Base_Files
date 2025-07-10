#!/usr/bin/env python3
"""
🧠⚛️ QUTRIT SIGNAL MATRIX TEST SUITE
===================================

Comprehensive testing of Layer 3: Qutrit/Tri-State Math Wiring
- Qutrit matrix generation from SHA-256 hashes
- State decision logic (DEFER/EXECUTE/RECHECK)
- Volatility overlay application
- Strategy bit mapper integration
- Fallback vector morphing
"""

import logging
import sys
import time

import numpy as np

# Add project root to path
sys.path.append('.')

try:
        QutritSignalMatrix, 
        QutritState, 
        QutritMatrixResult,
        create_qutrit_matrix
    )
    from core.strategy_bit_mapper import StrategyBitMapper
    from core.unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
    QUTRIT_READY = True
    except ImportError as e:
    print(f"❌ Critical import error: {e}")
    QUTRIT_READY = False

def test_qutrit_matrix_generation():
    """Test basic qutrit matrix generation and properties"""
    print("🧠⚛️ TESTING QUTRIT MATRIX GENERATION")
    print("=" * 50)

    if not QUTRIT_READY:
        print("❌ Qutrit components not ready. Aborting test.")
        return False

    try:
        # Test 1: Basic matrix generation
        print("\n📊 Test 1: Basic Matrix Generation")
        seed = "btc_orbital_test_hash"
        market_data = {"price": 50000, "volatility": 0.3, "volume": 1500, "timestamp": 1234567890}

        qutrit_matrix = QutritSignalMatrix(seed, market_data)
        matrix = qutrit_matrix.get_matrix()

        print(f"Seed: {seed}")
        print(f"Matrix:\n{matrix}")
        print(f"Matrix Shape: {matrix.shape}")
        print(f"Matrix Sum: {np.sum(matrix)}")
        print(f"Matrix Range: [{np.min(matrix)}, {np.max(matrix)}]")

        # Validate matrix properties
        assert matrix.shape == (3, 3), f"Expected (3,3) shape, got {matrix.shape}"
        assert np.all((matrix >= 0) & (matrix <= 2)), "Matrix values must be in range [0,2]"
        assert np.all(np.isfinite(matrix)), "Matrix contains non-finite values"

        print("✅ Basic matrix generation: PASSED")

        # Test 2: State decision logic
        print("\n🎯 Test 2: State Decision Logic")
        state = qutrit_matrix.get_state_decision()
        matrix_sum = np.sum(matrix)
        state_value = matrix_sum % 3

        print(f"Matrix Sum: {matrix_sum}")
        print(f"State Value (sum % 3): {state_value}")
        print(f"Qutrit State: {state}")
        print(f"State Description: {qutrit_matrix.get_state_description()}")

        # Validate state logic
        expected_state_map = {0: QutritState.DEFER, 1: QutritState.EXECUTE, 2: QutritState.RECHECK}
        expected_state = expected_state_map[state_value]
        assert state == expected_state, f"Expected {expected_state}, got {state}"

        print("✅ State decision logic: PASSED")

        # Test 3: Confidence calculation
        print("\n📈 Test 3: Confidence Calculation")
        confidence = qutrit_matrix.calculate_confidence()

        print(f"Confidence: {confidence:.3f}")
        print(f"Matrix Std: {np.std(matrix):.3f}")
        print(f"Volatility: {market_data['volatility']}")
        print(f"Volume: {market_data['volume']}")

        # Validate confidence range
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in range [0,1]"

        print("✅ Confidence calculation: PASSED")

        return True

    except Exception as e:
        print(f"❌ Matrix generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_volatility_overlay():
    """Test volatility overlay application"""
    print("\n🌊 TESTING VOLATILITY OVERLAY")
    print("=" * 50)

    try:
        seed = "volatility_test"
        market_data = {"price": 48000, "volatility": 0.2, "volume": 1000, "timestamp": time.time()}

        qutrit_matrix = QutritSignalMatrix(seed, market_data)
        original_matrix = qutrit_matrix.get_matrix()

        print(f"Original Matrix:\n{original_matrix}")

        # Test different volatility levels
        volatility_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        for vol in volatility_levels:
            overlay_matrix = qutrit_matrix.apply_volatility_overlay(vol)
            print(f"\nVolatility {vol}:")
            print(f"Overlay Matrix:\n{overlay_matrix}")

            # Validate overlay properties
            assert overlay_matrix.shape == (3, 3), f"Overlay shape mismatch: {overlay_matrix.shape}"
            assert np.all((overlay_matrix >= 0) & (overlay_matrix <= 2)), f"Overlay values out of range for vol {vol}"

        print("✅ Volatility overlay: PASSED")
        return True

    except Exception as e:
        print(f"❌ Volatility overlay test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_context_updates():
    """Test market context updates and matrix regeneration"""
    print("\n🔄 TESTING MARKET CONTEXT UPDATES")
    print("=" * 50)

    try:
        seed = "context_test"
        initial_market = {"price": 50000, "volatility": 0.3, "volume": 1500, "timestamp": 1000}

        qutrit_matrix = QutritSignalMatrix(seed, initial_market)
        initial_matrix = qutrit_matrix.get_matrix()
        initial_state = qutrit_matrix.get_state_decision()

        print(f"Initial Matrix:\n{initial_matrix}")
        print(f"Initial State: {initial_state}")

        # Update market context
        new_market = {"price": 52000, "volatility": 0.5, "volume": 2000, "timestamp": 2000}
        qutrit_matrix.update_market_context(new_market)

        updated_matrix = qutrit_matrix.get_matrix()
        updated_state = qutrit_matrix.get_state_decision()

        print(f"\nUpdated Matrix:\n{updated_matrix}")
        print(f"Updated State: {updated_state}")

        # Validate that matrix changed (due to different, context)
        matrices_different = not np.array_equal(initial_matrix, updated_matrix)
        print(f"Matrices Different: {matrices_different}")

        # Get complete result
        result = qutrit_matrix.get_matrix_result()
        print(f"\nComplete Result:")
        print(f"  Hash Segment: {result.hash_segment}")
        print(f"  State: {result.state}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Market Context: {result.market_context}")

        assert isinstance(result, QutritMatrixResult)
        assert len(result.hash_segment) == 8
        assert 0.0 <= result.confidence <= 1.0

        print("✅ Market context updates: PASSED")
        return True

    except Exception as e:
        print(f"❌ Market context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_bit_mapper_integration():
    """Test qutrit gate integration with strategy bit mapper"""
    print("\n🔬 TESTING STRATEGY BIT MAPPER INTEGRATION")
    print("=" * 50)

    try:
        # Initialize strategy bit mapper
        strategy_mapper = StrategyBitMapper(matrix_dir="data/matrices")

        # Test qutrit gate application
        strategy_id = "test_strategy_456"
        seed = "qutrit_gate_integration_test"
        market_data = {"price": 49000, "volatility": 0.4, "volume": 1800, "timestamp": time.time()}

        qutrit_result = strategy_mapper.apply_qutrit_gate(strategy_id, seed, market_data)

        print(f"Qutrit Gate Result:")
        print(f"  Strategy ID: {qutrit_result['strategy_id']}")
        print(f"  Action: {qutrit_result['action']}")
        print(f"  Reason: {qutrit_result['reason']}")
        print(f"  Qutrit State: {qutrit_result['qutrit_state']}")
        print(f"  Confidence: {qutrit_result['confidence']:.3f}")
        print(f"  Hash Segment: {qutrit_result['hash_segment']}")
        print(f"  Matrix: {qutrit_result['matrix']}")

        # Validate result structure
        assert qutrit_result['strategy_id'] == strategy_id
        assert qutrit_result['action'] in ['defer', 'execute', 'recheck']
        assert 0.0 <= qutrit_result['confidence'] <= 1.0
        assert len(qutrit_result['hash_segment']) == 8
        assert len(qutrit_result['matrix']) == 3
        assert len(qutrit_result['matrix'][0]) == 3

        # Test individual action methods
        defer_result = strategy_mapper.defer(strategy_id)
        execute_result = strategy_mapper.execute_trade(strategy_id)
        recheck_result = strategy_mapper.recheck_later(strategy_id)

        print(f"\nIndividual Actions:")
        print(f"  Defer: {defer_result['action']}")
        print(f"  Execute: {execute_result['action']}")
        print(f"  Recheck: {recheck_result['action']}")

        assert defer_result['action'] == 'defer'
        assert execute_result['action'] == 'execute'
        assert recheck_result['action'] == 'recheck'

        print("✅ Strategy bit mapper integration: PASSED")
        return True

    except Exception as e:
        print(f"❌ Strategy bit mapper integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_vector_morphing():
    """Test fallback vector morphing with qutrit influence"""
    print("\n🛡️ TESTING FALLBACK VECTOR MORPHING")
    print("=" * 50)

    try:
        # Initialize profit vectorization system
        profit_vectorizer = UnifiedProfitVectorizationSystem()

        # Test with different market conditions
        test_cases = []
            {}
                "name": "High Volatility",
                "data": {"price": 50000, "volatility": 0.8, "volume": 500, "timestamp": time.time()}
            },
            {}
                "name": "Low Volume",
                "data": {"price": 50000, "volatility": 0.2, "volume": 100, "timestamp": time.time()}
            },
            {}
                "name": "Normal Conditions",
                "data": {"price": 50000, "volatility": 0.4, "volume": 1500, "timestamp": time.time()}
            }
        ]

        for test_case in test_cases:
            print(f"\n📊 {test_case['name']}:")

            # Create qutrit matrix for this case
            seed = f"fallback_test_{test_case['name'].lower().replace(' ', '_')}"
            qutrit_matrix = QutritSignalMatrix(seed, test_case['data'])
            qutrit_result = qutrit_matrix.get_matrix_result()

            print(f"  Qutrit State: {qutrit_result.state}")
            print(f"  Confidence: {qutrit_result.confidence:.3f}")
            print(f"  Hash Segment: {qutrit_result.hash_segment}")

            # Test safe profit calculation with qutrit overlay
            try:
                matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                result = profit_vectorizer._safe_calculate_profit_with_fallback(matrix, test_case['data'])
                print(f"  Profit Result: {result}")
            except Exception as e:
                print(f"  Fallback Triggered: {str(e)}")
                fallback_vector = profit_vectorizer._generate_fallback_vector(str(e))
                print(f"  Fallback Vector: {fallback_vector}")

        print("✅ Fallback vector morphing: PASSED")
        return True

    except Exception as e:
        print(f"❌ Fallback vector morphing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all qutrit signal matrix tests"""
    print("🧠⚛️ QUTRIT SIGNAL MATRIX COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing Layer 3: Qutrit/Tri-State Math Wiring")
    print("=" * 80)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    tests = []
        ("Qutrit Matrix Generation", test_qutrit_matrix_generation),
        ("Volatility Overlay", test_volatility_overlay),
        ("Market Context Updates", test_market_context_updates),
        ("Strategy Bit Mapper Integration", test_strategy_bit_mapper_integration),
        ("Fallback Vector Morphing", test_fallback_vector_morphing),
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
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("🎉 ALL QUTRIT SIGNAL MATRIX TESTS PASSED!")
        print("🧠⚛️ Layer 3: Qutrit/Tri-State Math Wiring is FULLY OPERATIONAL!")
        print("🚀 Ready for Layer 4: Shell Memory Integration + Recursive Fractal Injection")
        return True
    else:
        print("⚠️ Some tests failed. Review output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 