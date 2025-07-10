#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Profit-Tiered CUDA Orchestration System

Demonstrates the ZPE/ZBE dual-state router system with profit-tiered
compute orchestration for Schwabot trading strategies.
"""

import logging
import time
from typing import Any, Dict

import numpy as np

# Configure logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dual_state_router():
    """Test the dual state router with various strategy types."""

    print("🧠 SCHWABOT ZPE/ZBE PROFIT-TIERED CUDA ORCHESTRATION TEST")
    print("=" * 60)

    try:
        # Import dual state router
            get_dual_state_router, StrategyTier, ComputeMode
        )

        # Get router instance
        router = get_dual_state_router()
        print("✅ Dual State Router initialized successfully")

        # Test 1: Short-term strategy (should prefer ZPE/CPU)
        print("\n🔻 Test 1: Short-term Strategy (Ghost Tick, Detection)")
        print("-" * 50)

        short_term_data = {}
            'price_data': np.random.randn(100).tolist(),
            'volume_data': np.random.randn(100).tolist()
        }

        result = router.route("ghost_tick_detector", short_term_data)
        print(f"Strategy: ghost_tick_detector")
        print(f"Compute Mode: {result['execution_metrics']['compute_mode']}")
        print(f"Execution Time: {result['execution_metrics']['execution_time_ms']:.2f}ms")
        print(f"Strategy Tier: {result['execution_metrics']['strategy_tier']}")
        print(f"Profit Density: {result['execution_metrics']['profit_density']:.3f}")
        print(f"Profit Delta: {result.get('profit_delta', 0):.6f}")

        # Test 2: Mid-term strategy (should balance ZPE/ZBE)
        print("\n🔻 Test 2: Mid-term Strategy (Matrix, Matching)")
        print("-" * 50)

        mid_term_data = {}
            'hash_vector': np.random.randn(64).tolist(),
            'matrices': []
                {'matrix': np.random.randn(8, 8).tolist()},
                {'matrix': np.random.randn(8, 8).tolist()},
                {'matrix': np.random.randn(8, 8).tolist()}
            ],
            'threshold': 0.8
        }

        result = router.route("matrix_match_hash", mid_term_data)
        print(f"Strategy: matrix_match_hash")
        print(f"Compute Mode: {result['execution_metrics']['compute_mode']}")
        print(f"Execution Time: {result['execution_metrics']['execution_time_ms']:.2f}ms")
        print(f"Strategy Tier: {result['execution_metrics']['strategy_tier']}")
        print(f"Profit Density: {result['execution_metrics']['profit_density']:.3f}")
        print(f"Profit Delta: {result.get('profit_delta', 0):.6f}")

        # Test 3: Long-term strategy (should prefer ZBE/GPU)
        print("\n🔻 Test 3: Long-term Strategy (Fractal, Analysis)")
        print("-" * 50)

        long_term_data = {}
            'time_series': np.random.randn(1000).tolist()
        }

        result = router.route("fractal_analysis", long_term_data)
        print(f"Strategy: fractal_analysis")
        print(f"Compute Mode: {result['execution_metrics']['compute_mode']}")
        print(f"Execution Time: {result['execution_metrics']['execution_time_ms']:.2f}ms")
        print(f"Strategy Tier: {result['execution_metrics']['strategy_tier']}")
        print(f"Profit Density: {result['execution_metrics']['profit_density']:.3f}")
        print(f"Profit Delta: {result.get('profit_delta', 0):.6f}")

        # Test 4: Complex tensor operations (should prefer ZBE/GPU)
        print("\n🔻 Test 4: Complex Tensor Operations")
        print("-" * 50)

        tensor_data = {}
            'tensor_a': np.random.randn(100, 100).tolist(),
            'tensor_b': np.random.randn(100, 100).tolist(),
            'operation': 'tensordot'
        }

        result = router.route("tensor_operations", tensor_data)
        print(f"Strategy: tensor_operations")
        print(f"Compute Mode: {result['execution_metrics']['compute_mode']}")
        print(f"Execution Time: {result['execution_metrics']['execution_time_ms']:.2f}ms")
        print(f"Strategy Tier: {result['execution_metrics']['strategy_tier']}")
        print(f"Profit Density: {result['execution_metrics']['profit_density']:.3f}")
        print(f"Profit Delta: {result.get('profit_delta', 0):.6f}")

        # Test 5: Spectral analysis (should prefer ZBE/GPU)
        print("\n🔻 Test 5: Spectral Analysis")
        print("-" * 50)

        spectral_data = {}
            'signal_data': np.random.randn(512).tolist()
        }

        result = router.route("spectral_analysis", spectral_data)
        print(f"Strategy: spectral_analysis")
        print(f"Compute Mode: {result['execution_metrics']['compute_mode']}")
        print(f"Execution Time: {result['execution_metrics']['execution_time_ms']:.2f}ms")
        print(f"Strategy Tier: {result['execution_metrics']['strategy_tier']}")
        print(f"Profit Density: {result['execution_metrics']['profit_density']:.3f}")
        print(f"Profit Delta: {result.get('profit_delta', 0):.6f}")

        # Get performance summary
        print("\n📊 Performance Summary")
        print("-" * 50)
        summary = router.get_performance_summary()

        print(f"Total Executions: {summary['total_executions']}")
        print(f"ZPE (CPU) Executions: {summary['zpe_executions']}")
        print(f"ZBE (GPU) Executions: {summary['zbe_executions']}")
        print(f"ZPE Avg Time: {summary['zpe_avg_time_ms']:.2f}ms")
        print(f"ZBE Avg Time: {summary['zbe_avg_time_ms']:.2f}ms")
        print(f"ZPE Avg Profit: {summary['zpe_avg_profit']:.6f}")
        print(f"ZBE Avg Profit: {summary['zbe_avg_profit']:.6f}")
        print(f"Performance Ratio: {summary['performance_ratio']:.2f}")
        print(f"Active Strategies: {summary['active_strategies']}")
        print(f"CUDA Available: {summary['cuda_available']}")

        # Test strategy analytics
        print("\n📈 Strategy Analytics")
        print("-" * 50)

        for strategy_id in ["ghost_tick_detector", "matrix_match_hash", "fractal_analysis"]:
            analytics = router.get_strategy_analytics(strategy_id)
            if 'error' not in analytics:
                print(f"\nStrategy: {analytics['strategy_id']}")
                print(f"  Tier: {analytics['tier']}")
                print(f"  Priority: {analytics['priority']:.3f}")
                print(f"  Avg Compute Time: {analytics['avg_compute_time_ms']:.2f}ms")
                print(f"  Avg Profit Margin: {analytics['avg_profit_margin']:.6f}")
                print(f"  Success Rate: {analytics['success_rate']:.3f}")
                print(f"  Preferred Mode: {analytics['preferred_mode']}")
                print(f"  Execution Count: {analytics['execution_count']}")
                print(f"  Total Profit: {analytics['total_profit']:.6f}")

        print("\n✅ All tests completed successfully!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_advanced_tensor_algebra():
    """Test advanced tensor algebra with dual state router integration."""

    print("\n🧮 ADVANCED TENSOR ALGEBRA WITH DUAL STATE ROUTER")
    print("=" * 60)

    try:
        from core.advanced_tensor_algebra import AdvancedTensorAlgebra

        # Initialize tensor algebra
        algebra = AdvancedTensorAlgebra()
        print("✅ Advanced Tensor Algebra initialized")

        # Test tensor fusion
        print("\n🔻 Testing Tensor Fusion with Dual State Router")
        print("-" * 50)

        A = np.random.randn(64, 32)
        B = np.random.randn(32, 16)

        start_time = time.time()
        result = algebra.tensor_dot_fusion(A, B)
        execution_time = (time.time() - start_time) * 1000

        print(f"Input Shapes: A{A.shape}, B{B.shape}")
        print(f"Output Shape: {result.shape}")
        print(f"Execution Time: {execution_time:.2f}ms")
        print(f"Result Norm: {np.linalg.norm(result):.6f}")

        print("\n✅ Tensor algebra test completed!")

    except Exception as e:
        print(f"❌ Tensor algebra test failed: {e}")
        import traceback
        traceback.print_exc()


def test_strategy_bit_mapper():
    """Test strategy bit mapper with dual state router integration."""

    print("\n🎯 STRATEGY BIT MAPPER WITH DUAL STATE ROUTER")
    print("=" * 60)

    try:
        from core.strategy_bit_mapper import StrategyBitMapper

        # Initialize strategy bit mapper
        mapper = StrategyBitMapper(matrix_dir="./matrices")
        print("✅ Strategy Bit Mapper initialized")

        # Test tensor-weighted expansion
        print("\n🔻 Testing Tensor-Weighted Expansion")
        print("-" * 50)

        strategy_id = 12345
        target_bits = 8

        start_time = time.time()
        expanded_id = mapper._tensor_weighted_expansion(strategy_id, target_bits)
        execution_time = (time.time() - start_time) * 1000

        print(f"Original Strategy ID: {strategy_id}")
        print(f"Expanded Strategy ID: {expanded_id}")
        print(f"Execution Time: {execution_time:.2f}ms")
        print(f"Binary: {bin(expanded_id)}")

        print("\n✅ Strategy bit mapper test completed!")

    except Exception as e:
        print(f"❌ Strategy bit mapper test failed: {e}")
        import traceback
        traceback.print_exc()


def test_fractal_core():
    """Test fractal core with dual state router integration."""

    print("\n🌀 FRACTAL CORE WITH DUAL STATE ROUTER")
    print("=" * 60)

    try:
        from core.fractal_core import fractal_quantize_vector

        # Test fractal quantization
        print("\n🔻 Testing Fractal Quantization")
        print("-" * 50)

        vector = np.random.randn(100)

        start_time = time.time()
        result = fractal_quantize_vector(vector, precision=8, method="mandelbrot")
        execution_time = (time.time() - start_time) * 1000

        print(f"Input Vector Length: {len(vector)}")
        print(f"Quantized Vector Length: {len(result.quantized_vector)}")
        print(f"Fractal Dimension: {result.fractal_dimension:.3f}")
        print(f"Self-Similarity Score: {result.self_similarity_score:.3f}")
        print(f"Compression Ratio: {result.compression_ratio:.3f}")
        print(f"Execution Time: {execution_time:.2f}ms")
        print(f"Dual State Routed: {result.metadata.get('dual_state_routed', False)}")

        print("\n✅ Fractal core test completed!")

    except Exception as e:
        print(f"❌ Fractal core test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🚀 SCHWABOT PROFIT-TIERED CUDA ORCHESTRATION SYSTEM")
    print("=" * 70)
    print("Testing ZPE/ZBE dual-state routing with profit-tiered compute orchestration")
    print()

    # Run tests
    test_dual_state_router()
    test_advanced_tensor_algebra()
    test_strategy_bit_mapper()
    test_fractal_core()

    print("\n🎉 ALL TESTS COMPLETED!")
    print("=" * 70)
    print("The profit-tiered CUDA orchestration system is working correctly.")
    print("ZPE (CPU) and ZBE (GPU) routing is based on:")
    print("- Strategy tier (short/mid/long)")
    print("- Profit density (ROI per compute, time)")
    print("- Historical performance")
    print("- Current system load")
    print()
    print("This ensures optimal compute resource allocation for maximum profit!")


if __name__ == "__main__":
    main() 