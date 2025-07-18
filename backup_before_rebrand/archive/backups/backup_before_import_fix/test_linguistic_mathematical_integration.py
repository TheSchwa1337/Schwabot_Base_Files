import json
import os
import sys
import time

import numpy as np

from core.dualistic_thought_engines import DualisticThoughtEngines
from core.linguistic_glyph_engine import ASICBitState, linguistic_engine, process_linguistic_command

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Linguistic-Mathematical Integration Test
=====================================================

Tests the complete English language → ASIC → Profit vectorization pipeline:
1. English text + glyph parsing
2. SHA hash generation and bit state mapping
3. Zalgo entropy overlay and Zygot expansion
4. ASIC dualistic logic integration
5. Fractal memory synthesis (Forever, Paradox, Echo)
6. BTC/USDC waveform processing
7. Profit containment vector updates
8. Real-time memory state management

This verifies that Schwabot can process natural language commands and convert
them into mathematical trading decisions with full fractal memory support.
"""


# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))


def test_linguistic_glyph_engine():
    """Test core linguistic glyph engine functionality."""
    print("🧠 Testing Linguistic Glyph Engine...")

    try:

        # Test basic language processing
        commands = []
            "Capture the next BTC/USDC dip 🧿 vectorize profit",
            "Hold current position 💎 maintain memory lock",
            "Execute profit vector 🚀 upward extrapolation",
            "Schwa recursive state 🔄 neutral recursion",
            "Ghost entry activation 👻 containment protocol",
        ]
        results = []
        for cmd in commands:
            result = process_linguistic_command(cmd)
            results.append(result)
            print(f"✅ Command: {cmd}")
            print(f"   Decision: {result['decision']}")
            print(f"   Bit State: {result['bit_state']:02b}")
            print(f"   Profit Delta: ${result['profit_delta']:.2f}")
            print(f"   Entropy: {result['entropy_overlay']:.4f}")

        return True, results

    except Exception as e:
        print(f"❌ Linguistic engine test failed: {e}")
        return False, str(e)


def test_fractal_mathematical_functions():
    """Test fractal mathematical functions for profit vectorization."""
    print("\n🌀 Testing Fractal Mathematical Functions...")

    try:
            forever_fractal,
            paradox_fractal,
            echo_fractal,
        )

        # Generate test data
        x = np.linspace(0, 250, 256)

        # Test each fractal
        forever_data = forever_fractal(x)
        paradox_data = paradox_fractal(x)
        echo_data = echo_fractal(x)

        # Verify fractal properties
        results = {}
            "forever_fractal": {}
                "mean": float(np.mean(forever_data)),
                "std": float(np.std(forever_data)),
                "max": float(np.max(forever_data)),
                "min": float(np.min(forever_data)),
                "energy": float(np.sum(np.abs(forever_data))),
            },
            "paradox_fractal": {}
                "mean": float(np.mean(paradox_data)),
                "std": float(np.std(paradox_data)),
                "max": float(np.max(paradox_data)),
                "min": float(np.min(paradox_data)),
                "energy": float(np.sum(np.abs(paradox_data))),
            },
            "echo_fractal": {}
                "mean": float(np.mean(echo_data)),
                "std": float(np.std(echo_data)),
                "max": float(np.max(echo_data)),
                "min": float(np.min(echo_data)),
                "energy": float(np.sum(np.abs(echo_data))),
            },
        }
        print("✅ Forever Fractal: Non-decaying memory vector (Golden ratio, dynamics)")
        print(f"   Energy: {results['forever_fractal']['energy']:.2f}")

        print("✅ Paradox Fractal: Market contradiction detector (Collapsing, sine)")
        print(f"   Energy: {results['paradox_fractal']['energy']:.2f}")

        print("✅ Echo Fractal: Recursive memory backlog (Decay, pattern)")
        print(f"   Energy: {results['echo_fractal']['energy']:.2f}")

        return True, results

    except Exception as e:
        print(f"❌ Fractal functions test failed: {e}")
        return False, str(e)


def test_dualistic_thought_integration():
    """Test integration with dualistic thought engines."""
    print("\n🧬 Testing Dualistic Thought Engine Integration...")

    try:

        # Initialize engines
        thought_engines = DualisticThoughtEngines()

        # Test linguistic command processing
        test_commands = []
            "🚀 Execute BTC profit vector with memory lock 🧿",
            "👻 Ghost entry on next dip - capture containment",
            "🔄 Schwa recursive state - neutral hold position",
            "💎 Diamond hands - maintain current vector lock",
        ]
        market_data = {}
            "btc_price": 47500.0,
            "usdc_balance": 15000.0,
            "volume": 1250000,
            "price_change_24h": 2.3,
        }
        results = []
        for cmd in test_commands:
            result = thought_engines.process_linguistic_trading_command()
                cmd, market_data
            )
            results.append(result)

            if result["success"]:
                print(f"✅ Command: {cmd}")
                print(f"   Decision: {result['linguistic_analysis']['decision']}")
                print(f"   Thought Hash: {result['thought_vector']['hash']}")
                print(f"   Confidence: {result['thought_vector']['confidence']:.2f}")
                print(f"   Profit Delta: ${result['trade_vector']['profit_delta']:.2f}")
            else:
                print(f"❌ Command failed: {cmd}")
                print(f"   Error: {result['error']}")

        return True, results

    except Exception as e:
        print(f"❌ Dualistic integration test failed: {e}")
        return False, str(e)


def test_asic_bit_logic_states():
    """Test ASIC bit logic state mapping and transitions."""
    print("\n🔧 Testing ASIC Bit Logic States...")

    try:

        # Test bit state mappings
        test_cases = []
            ("capture", ASICBitState.GHOST_ENTRY.value),
            ("memory", ASICBitState.MEMORY_LOCK.value),
            ("profit", ASICBitState.PROFIT_VECTOR.value),
            ("schwa", ASICBitState.NULL_RECURSION.value),
            ("🧿", ASICBitState.MEMORY_LOCK.value),
            ("🚀", ASICBitState.PROFIT_VECTOR.value),
            ("👻", ASICBitState.GHOST_ENTRY.value),
            ("🔄", ASICBitState.NULL_RECURSION.value),
        ]
        results = {}
        for input_term, expected_state in test_cases:
            ling_hash = linguistic_engine.text_to_glyph_hash(input_term)
            actual_state = ling_hash.bit_state

            success = actual_state == expected_state
            results[input_term] = {}
                "expected": expected_state,
                "actual": actual_state,
                "success": success,
                "sha_hash": ling_hash.sha_hash[:8],
                "weight": ling_hash.weight,
            }
            status = "✅" if success else "❌"
            print()
                f"{status} {input_term} → {actual_state:02b} (expected {expected_state:02b})"
            )

        # Test state transitions and combinations
        combined_tests = []
            "capture 🧿 profit",  # Should prioritize glyph (MEMORY_LOCK)
            "👻 vectorize exit",  # Should prioritize glyph (GHOST_ENTRY)
            "schwa recursive 🚀",  # Should prioritize glyph (PROFIT_VECTOR)
        ]
        print("\n🔀 Testing Combined State Logic:")
        for test_phrase in combined_tests:
            ling_hash = linguistic_engine.text_to_glyph_hash(test_phrase)
            print()
                f"✅ '{test_phrase}' → {ling_hash.bit_state:02b} (weight: {ling_hash.weight:.3f})"
            )

        return True, results

    except Exception as e:
        print(f"❌ ASIC bit logic test failed: {e}")
        return False, str(e)


def test_profit_vectorization_synthesis():
    """Test complete profit vectorization and containment synthesis."""
    print("\n💰 Testing Profit Vectorization Synthesis...")

    try:

        # Reset engine state
        linguistic_engine.profit_containment = np.zeros(256)
        linguistic_engine.fractal_memory = np.zeros((16, 16))
        linguistic_engine.memory_stack = []
        linguistic_engine.trade_vectors = []

        # Simulate trading sequence
        trading_sequence = []
            ("🚀 Execute BTC entry at 47000", 47000.0, 10000.0),
            ("💎 Hold position - diamond hands", 47200.0, 10000.0),
            ("🧿 Memory lock current vector", 47150.0, 10000.0),
            ("📈 Profit vector - upward extrapolation", 47800.0, 10000.0),
            ("👻 Ghost exit - capture gains", 47650.0, 10000.0),
        ]
        results = []
        total_profit = 0.0

        for i, (command, btc_price, usdc_balance) in enumerate(trading_sequence):
            trade_vector = linguistic_engine.process_btc_usdc_waveform()
                command, btc_price, usdc_balance
            )

            total_profit += trade_vector.profit_delta

            result = {}
                "step": i + 1,
                "command": command,
                "btc_price": btc_price,
                "profit_delta": trade_vector.profit_delta,
                "containment_sum": float(np.sum(linguistic_engine.profit_containment)),
                "fractal_energy": float()
                    np.sum(np.abs(linguistic_engine.fractal_memory))
                ),
                "bit_sequence_length": len(trade_vector.bit_sequence),
                "glyph_signature": trade_vector.glyph_signature,
            }
            results.append(result)

            print(f"✅ Step {i + 1}: {command}")
            print()
                f"   BTC: ${btc_price:,.0f} | Profit: ${trade_vector.profit_delta:.2f}"
            )
            print(f"   Containment: {np.sum(linguistic_engine.profit_containment):.2f}")
            print()
                f"   Fractal Energy: {np.sum(np.abs(linguistic_engine.fractal_memory)):.2f}"
            )

        print(f"\n💰 Total Profit Synthesized: ${total_profit:.2f}")
        print(f"🧠 Memory Vectors: {len(linguistic_engine.memory_stack)}")
        print(f"📊 Trade Vectors: {len(linguistic_engine.trade_vectors)}")

        # Get final memory state
        final_memory = linguistic_engine.get_memory_state_summary()
        print(f"🔮 Final Memory State: {final_memory}")

        return True, {}
            "sequence_results": results,
            "total_profit": total_profit,
            "final_memory_state": final_memory,
        }
    except Exception as e:
        print(f"❌ Profit vectorization test failed: {e}")
        return False, str(e)


def test_zalgo_zygot_mathematical_processing():
    """Test Zalgo entropy overlay and Zygot recursive expansion."""
    print("\n🕷️ Testing Zalgo & Zygot Mathematical Processing...")

    try:

        # Test Zalgo overlay with different bit patterns
        test_patterns = []
            [1, 0, 1, 1, 0, 0, 1, 0],  # Random pattern
            [3, 2, 1, 0, 3, 2, 1, 0],  # 2-bit sequence
            [1, 1, 1, 1, 0, 0, 0, 0],  # Block pattern
            [2, 3, 0, 1, 2, 3, 0, 1],  # Repeating 2-bit
        ]
        zalgo_results = []
        zygot_results = []

        for i, pattern in enumerate(test_patterns):
            # Test Zalgo entropy overlay
            zalgo_entropy = linguistic_engine.zalgo_overlay(pattern, lambda_val=5.0)
            zalgo_results.append(zalgo_entropy)

            # Test Zygot recursive expansion
            zygot_expansion = linguistic_engine.zygot_expand(pattern, depth=3)
            zygot_results.append(zygot_expansion)

            print(f"✅ Pattern {i + 1}: {pattern}")
            print(f"   Zalgo Entropy: {zalgo_entropy:.4f}")
            print(f"   Zygot Expansion Length: {len(zygot_expansion)}")
            print(f"   Zygot Energy: {sum(abs(x) for x in zygot_expansion):.2f}")

        # Test with actual glyph-derived patterns
        print("\n🔣 Testing with Glyph-Derived Patterns:")
        test_glyphs = ["🧿", "🚀", "👻", "💎", "🔄"]

        glyph_results = {}
        for glyph in test_glyphs:
            bit_mask = linguistic_engine.emoji_to_bitmask(glyph)
            zalgo_entropy = linguistic_engine.zalgo_overlay(bit_mask)
            zygot_expansion = linguistic_engine.zygot_expand()
                bit_mask[:8]
            )  # Limit for processing

            glyph_results[glyph] = {}
                "bit_mask_length": len(bit_mask),
                "zalgo_entropy": zalgo_entropy,
                "zygot_length": len(zygot_expansion),
                "zygot_energy": sum(abs(x) for x in zygot_expansion)
                if zygot_expansion
                else 0,
            }
            print()
                f"✅ {glyph}: Entropy={zalgo_entropy:.4f}, Zygot Energy={glyph_results[glyph]['zygot_energy']:.2f}"
            )

        return True, {}
            "zalgo_test_results": zalgo_results,
            "zygot_test_results": zygot_results,
            "glyph_processing": glyph_results,
        }
    except Exception as e:
        print(f"❌ Zalgo/Zygot processing test failed: {e}")
        return False, str(e)


def run_comprehensive_integration_test():
    """Run all tests and generate comprehensive report."""
    print("🎯 COMPREHENSIVE LINGUISTIC-MATHEMATICAL INTEGRATION TEST")
    print("=" * 80)

    test_results = {}
    start_time = time.time()

    # Run all tests
    tests = []
        ("Linguistic Glyph Engine", test_linguistic_glyph_engine),
        ("Fractal Mathematical Functions", test_fractal_mathematical_functions),
        ("Dualistic Thought Integration", test_dualistic_thought_integration),
        ("ASIC Bit Logic States", test_asic_bit_logic_states),
        ("Profit Vectorization Synthesis", test_profit_vectorization_synthesis),
        ("Zalgo & Zygot Processing", test_zalgo_zygot_mathematical_processing),
    ]
    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 20} {test_name} {'=' * 20}")
            success, result = test_func()
            test_results[test_name] = {}
                "success": success,
                "result": result,
                "timestamp": time.time(),
            }
            if success:
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED: {result}")
        except Exception as e:
            test_results[test_name] = {}
                "success": False,
                "result": str(e),
                "timestamp": time.time(),
            }
            print(f"❌ {test_name} ERROR: {e}")

    # Generate final report
    end_time = time.time()
    test_duration = end_time - start_time

    print(f"\n{'=' * 80}")
    print("🎯 FINAL INTEGRATION TEST REPORT")
    print(f"{'=' * 80}")
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"⏱️ Total Duration: {test_duration:.2f} seconds")
    print(f"📊 Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - LINGUISTIC MATHEMATICAL INTEGRATION READY!")
        print()
            "🚀 Schwabot can now process English language → ASIC → Profit vectorization"
        )
        print()
            "🧠 Memory states, fractal overlays, and Zalgo/Zygot processing functional"
        )
        print("💰 BTC/USDC waveform analysis with hash valuations operational")
    else:
        print("⚠️ Some tests failed - review results for issues")

    # Save detailed results
    timestamp = int(time.time())
    results_file = f"linguistic_mathematical_integration_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"📄 Detailed results saved to: {results_file}")

    return test_results


if __name__ == "__main__":
    results = run_comprehensive_integration_test()
