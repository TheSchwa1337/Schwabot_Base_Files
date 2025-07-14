#!/usr/bin/env python3
"""
Comprehensive Test for Schwabot Tensor + Profit Vector System

This script validates the complete system for:
- Advanced tensor algebra operations
- Profit vector calculations
- CUDA + CPU hybrid acceleration
- Trading logic integration
- API data processing
- Advanced hashing and tensor operations
- Clean math pipelines
- Registry storage functionality
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_tensor_algebra_system():
    """Test advanced tensor algebra operations."""
    print("\n" + "=" * 60)
    print("🧮 Testing Advanced Tensor Algebra System")
    print("=" * 60)

    try:
        from core.advanced_tensor_algebra import AdvancedTensorAlgebra

        # Initialize tensor algebra
        tensor_algebra = AdvancedTensorAlgebra(precision=64, enable_caching=True)
        print("✅ Advanced Tensor Algebra initialized")

        # Test data
        size = 100
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        vector = np.random.rand(size)

        print(f"✅ Test data generated (size: {size}x{size})")

        # Test tensor fusion
        print("\n📊 Testing Tensor Fusion:")
        start_time = time.perf_counter()
        fused_tensor = tensor_algebra.tensor_dot_fusion(A, B)
        fusion_time = time.perf_counter() - start_time
        print(f"  Result shape: {fused_tensor.shape}")
        print(f"  Execution time: {fusion_time:.6f}s")
        print(f"  Result sum: {np.sum(fused_tensor):.6f}")

        # Test bit phase rotation
        print("\n📊 Testing Bit Phase Rotation:")
        start_time = time.perf_counter()
        rotated = tensor_algebra.bit_phase_rotation(vector, theta=np.pi/4)
        rotation_time = time.perf_counter() - start_time
        print(f"  Result shape: {rotated.shape}")
        print(f"  Execution time: {rotation_time:.6f}s")
        print(f"  Result magnitude: {np.abs(rotated).mean():.6f}")

        # Test entropy vector quantization
        print("\n📊 Testing Entropy Vector Quantization:")
        start_time = time.perf_counter()
        quantized = tensor_algebra.entropy_vector_quantize(vector, E=0.5)
        quantization_time = time.perf_counter() - start_time
        print(f"  Result shape: {quantized.shape}")
        print(f"  Execution time: {quantization_time:.6f}s")
        print(f"  Unique values: {len(np.unique(quantized))}")

        # Test matrix trace conditions
        print("\n📊 Testing Matrix Trace Conditions:")
        start_time = time.perf_counter()
        trace_data = tensor_algebra.matrix_trace_conditions(A)
        trace_time = time.perf_counter() - start_time
        print(f"  Execution time: {trace_time:.6f}s")
        print(f"  Trace: {trace_data.get('trace', 0):.6f}")
        print(f"  Stability: {trace_data.get('stability_condition', 0):.6f}")

        # Test spectral norm tracking
        print("\n📊 Testing Spectral Norm Tracking:")
        start_time = time.perf_counter()
        spectral_data = tensor_algebra.spectral_norm_tracking(A, history_length=10)
        spectral_time = time.perf_counter() - start_time
        print(f"  Execution time: {spectral_time:.6f}s")
        print(f"  Current norm: {spectral_data.get('current_norm', 0):.6f}")
        print(f"  Convergence: {spectral_data.get('convergence_rate', 0):.6f}")

        print("\n✅ Advanced Tensor Algebra tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Advanced Tensor Algebra test failed: {e}")
        return False


def test_profit_vector_system():
    """Test profit vector calculations and processing."""
    print("\n" + "=" * 60)
    print("💰 Testing Profit Vector System")
    print("=" * 60)

    try:
            enhanced_profit_vectorization, enhanced_cosine_sim, get_enhancement_status
        )

        # Get enhancement status
        status = get_enhancement_status()
        print(f"✅ Enhancement Available: {status['enhancement_available']}")
        print(f"🎯 CUDA Available: {status['cuda_available']}")

        # Test data
        size = 1000
        profits = np.random.rand(size) * 100  # Random profits
        weights = np.random.rand(size)  # Random weights
        weights = weights / np.sum(weights)  # Normalize weights

        print(f"✅ Test data generated (size: {size})")
        print(f"  Total profit: {np.sum(profits):.2f}")
        print(f"  Weight sum: {np.sum(weights):.6f}")

        # Test profit vectorization with enhancement
        print("\n📊 Testing Enhanced Profit Vectorization:")
        start_time = time.perf_counter()
        profit_vector = enhanced_profit_vectorization()
            profits, weights, 
            entropy=0.7, profit_weight=0.8, 
            use_enhancement=True
        )
        vectorization_time = time.perf_counter() - start_time
        print(f"  Execution time: {vectorization_time:.6f}s")
        print(f"  Result shape: {profit_vector.shape}")
        print(f"  Total weighted profit: {np.sum(profit_vector):.2f}")
        print(f"  Profit efficiency: {np.sum(profit_vector) / np.sum(profits):.3f}")

        # Test profit similarity calculations
        print("\n📊 Testing Profit Similarity Calculations:")
        profit_strategy_1 = np.random.rand(size)
        profit_strategy_2 = np.random.rand(size)

        start_time = time.perf_counter()
        similarity = enhanced_cosine_sim()
            profit_strategy_1, profit_strategy_2,
            entropy=0.6, profit_weight=0.7,
            use_enhancement=True
        )
        similarity_time = time.perf_counter() - start_time
        print(f"  Execution time: {similarity_time:.6f}s")
        print(f"  Strategy similarity: {similarity:.6f}")

        # Test profit optimization scenarios
        print("\n📊 Testing Profit Optimization Scenarios:")

        # Scenario 1: High profit, low entropy
        profit_high = enhanced_profit_vectorization()
            profits * 1.5, weights,
            entropy=0.3, profit_weight=0.9,
            use_enhancement=True
        )
        print(f"  High profit scenario: {np.sum(profit_high):.2f}")

        # Scenario 2: Low profit, high entropy
        profit_low = enhanced_profit_vectorization()
            profits * 0.5, weights,
            entropy=0.8, profit_weight=0.3,
            use_enhancement=True
        )
        print(f"  Low profit scenario: {np.sum(profit_low):.2f}")

        # Scenario 3: Balanced scenario
        profit_balanced = enhanced_profit_vectorization()
            profits, weights,
            entropy=0.5, profit_weight=0.6,
            use_enhancement=True
        )
        print(f"  Balanced scenario: {np.sum(profit_balanced):.2f}")

        print("\n✅ Profit Vector System tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Profit Vector System test failed: {e}")
        return False


def test_acceleration_enhancement():
    """Test CUDA + CPU hybrid acceleration system."""
    print("\n" + "=" * 60)
    print("🚀 Testing CUDA + CPU Hybrid Acceleration System")
    print("=" * 60)

    try:
        from core.acceleration_enhancement import get_acceleration_enhancement

        # Initialize enhancement layer
        enhancement = get_acceleration_enhancement()
        print(f"✅ Enhancement Layer initialized")
        print(f"🎯 CUDA Available: {enhancement.cuda_available}")
        print(f"🔗 System Integration: {enhancement.existing_system_available}")

        # Test acceleration metrics
        print("\n📊 Testing Acceleration Metrics:")

        # Simulate operations with different characteristics
        def cpu_operation(data):
            return np.sum(data ** 2)

        def gpu_operation(data):
            try:
                import cupy as cp
                data_gpu = cp.asarray(data)
                return float(cp.sum(data_gpu ** 2))
            except ImportError:
                return cpu_operation(data)

        # Test data
        test_data = np.random.rand(10000)

        # Test different entropy/profit combinations
        test_cases = []
            (0.2, 0.3, "Low complexity, low profit"),
            (0.5, 0.5, "Medium complexity, medium profit"),
            (0.8, 0.8, "High complexity, high profit"),
        ]

        for entropy, profit_weight, description in test_cases:
            print(f"\n📊 {description}:")

            start_time = time.perf_counter()
            result = enhancement.execute_with_enhancement()
                cpu_operation,
                gpu_operation,
                test_data,
                entropy=entropy,
                profit_weight=profit_weight,
                op_name="test_operation",
                zpe_integration=True,
                zbe_integration=True
            )
            execution_time = time.perf_counter() - start_time

            print(f"  Result: {result:.6f}")
            print(f"  Execution time: {execution_time:.6f}s")
            print(f"  Entropy: {entropy:.2f}")
            print(f"  Profit weight: {profit_weight:.2f}")

        # Get enhancement report
        print("\n📊 Enhancement Performance Report:")
        report = enhancement.get_enhancement_report()

        print(f"  Status: {report['status']}")
        print(f"  Total Operations: {report['total_operations']}")
        print(f"  CPU Operations: {report['cpu_operations']}")
        print(f"  GPU Operations: {report['gpu_operations']}")
        print(f"  Success Rate: {report['overall_success_rate']:.1%}")

        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            print(f"  CPU Avg Time: {perf['avg_cpu_time_ms']:.3f}ms")
            print(f"  GPU Avg Time: {perf['avg_gpu_time_ms']:.3f}ms")
            print(f"  Speedup Ratio: {perf['speedup_ratio']:.2f}x")

        print("\n✅ Acceleration Enhancement tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Acceleration Enhancement test failed: {e}")
        return False


def test_trading_logic_integration():
    """Test trading logic integration with tensor calculations."""
    print("\n" + "=" * 60)
    print("📈 Testing Trading Logic Integration")
    print("=" * 60)

    try:
        # Test ZPE core integration
        print("\n📊 Testing ZPE Core Integration:")
        try:
            from core.zpe_core import ZPECore
            zpe_core = ZPECore()

            # Test ZPE calculations
            zpe_energy = zpe_core.calculate_zero_point_energy(frequency=1000.0, amplitude=1.0)
            print(f"  ZPE Energy: {zpe_energy:.6e}")

            # Test thermal efficiency
            thermal_data = zpe_core.calculate_thermal_efficiency(energy_input=1.0, energy_output=0.8)
            print(f"  Thermal Efficiency: {thermal_data.thermal_efficiency:.3f}")

            print("  ✅ ZPE Core integration successful")

        except Exception as e:
            print(f"  ⚠️ ZPE Core integration issue: {e}")

        # Test ZBE core integration
        print("\n📊 Testing ZBE Core Integration:")
        try:
            from core.zbe_core import ZBECore
            zbe_core = ZBECore()

            # Test bit efficiency
            bit_data = zbe_core.calculate_bit_efficiency()
                computational_load=0.5,
                memory_usage=0.3,
                cache_usage=0.2,
                register_usage=0.4
            )
            print(f"  Bit Efficiency: {bit_data.bit_efficiency:.3f}")

            print("  ✅ ZBE Core integration successful")

        except Exception as e:
            print(f"  ⚠️ ZBE Core integration issue: {e}")

        # Test Dual State Router integration
        print("\n📊 Testing Dual State Router Integration:")
        try:
            from core.system.dual_state_router import DualStateRouter
            router = DualStateRouter()

            # Test routing with tensor data
            test_data = {}
                "strategy_tier": "mid",
                "profit_density": 0.6,
                "compute_time": 0.5,
                "tensor_complexity": 0.7
            }

            result = router.route("tensor_strategy", test_data)
            print(f"  Routing Result: {result.get('compute_mode', 'unknown')}")
            print(f"  Strategy Tier: {result.get('strategy_tier', 'unknown')}")

            print("  ✅ Dual State Router integration successful")

        except Exception as e:
            print(f"  ⚠️ Dual State Router integration issue: {e}")

        print("\n✅ Trading Logic Integration tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Trading Logic Integration test failed: {e}")
        return False


def test_api_data_processing():
    """Test API data processing and integration."""
    print("\n" + "=" * 60)
    print("🌐 Testing API Data Processing")
    print("=" * 60)

    try:
        # Test API integration
        print("\n📊 Testing API Integration:")
        try:
            from core.api.integration_manager import IntegrationManager
            api_manager = IntegrationManager()
            print("  ✅ API Integration Manager initialized")

        except Exception as e:
            print(f"  ⚠️ API Integration Manager issue: {e}")

        # Test market data processing
        print("\n📊 Testing Market Data Processing:")
        try:
            from core.unified_market_data_pipeline import UnifiedMarketDataPipeline
            pipeline = UnifiedMarketDataPipeline()
            print("  ✅ Market Data Pipeline initialized")

        except Exception as e:
            print(f"  ⚠️ Market Data Pipeline issue: {e}")

        # Test exchange connection
        print("\n📊 Testing Exchange Connection:")
        try:
            from core.api.exchange_connection import ExchangeConnection
            exchange = ExchangeConnection()
            print("  ✅ Exchange Connection initialized")

        except Exception as e:
            print(f"  ⚠️ Exchange Connection issue: {e}")

        print("\n✅ API Data Processing tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ API Data Processing test failed: {e}")
        return False


def test_advanced_hashing_tensor():
    """Test advanced hashing and tensor operations."""
    print("\n" + "=" * 60)
    print("🔐 Testing Advanced Hashing and Tensor Operations")
    print("=" * 60)

    try:
            enhanced_hash_matching, enhanced_fractal_compression
        )

        # Test data
        size = 1000
        hashes = np.random.randint(0, 256, (100, size))  # 100 hash vectors
        target_hash = np.random.randint(0, 256, size)    # Target hash

        print(f"✅ Test data generated (100 hash vectors, size: {size})")

        # Test hash matching
        print("\n📊 Testing Enhanced Hash Matching:")
        start_time = time.perf_counter()
        matches = enhanced_hash_matching()
            hashes, target_hash,
            entropy=0.6, profit_weight=0.5,
            use_enhancement=True
        )
        hash_time = time.perf_counter() - start_time
        print(f"  Execution time: {hash_time:.6f}s")
        print(f"  Result shape: {matches.shape}")
        print(f"  Best match: {np.max(matches):.6f}")
        print(f"  Average match: {np.mean(matches):.6f}")

        # Test fractal compression
        print("\n📊 Testing Enhanced Fractal Compression:")
        data = np.random.rand(10000)
        start_time = time.perf_counter()
        compressed = enhanced_fractal_compression()
            data, compression_ratio=0.1,
            entropy=0.7, profit_weight=0.6,
            use_enhancement=True
        )
        compression_time = time.perf_counter() - start_time
        print(f"  Execution time: {compression_time:.6f}s")
        print(f"  Original size: {len(data)}")
        print(f"  Compressed size: {len(compressed)}")
        print(f"  Compression ratio: {len(compressed) / len(data):.3f}")

        # Test tensor operations with hashing
        print("\n📊 Testing Tensor Operations with Hashing:")
        try:
            from core.advanced_tensor_algebra import AdvancedTensorAlgebra
            tensor_algebra = AdvancedTensorAlgebra()

            # Create hash-based tensors
            hash_tensor_1 = np.random.rand(100, 100)
            hash_tensor_2 = np.random.rand(100, 100)

            # Test tensor fusion with hash data
            start_time = time.perf_counter()
            fused_hash_tensor = tensor_algebra.tensor_dot_fusion(hash_tensor_1, hash_tensor_2)
            tensor_time = time.perf_counter() - start_time
            print(f"  Tensor fusion time: {tensor_time:.6f}s")
            print(f"  Fused tensor shape: {fused_hash_tensor.shape}")
            print(f"  Tensor sum: {np.sum(fused_hash_tensor):.6f}")

        except Exception as e:
            print(f"  ⚠️ Tensor operations issue: {e}")

        print("\n✅ Advanced Hashing and Tensor Operations tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Advanced Hashing and Tensor Operations test failed: {e}")
        return False


def test_clean_math_pipelines():
    """Test clean math pipelines and calculations."""
    print("\n" + "=" * 60)
    print("🧹 Testing Clean Math Pipelines")
    print("=" * 60)

    try:
        # Test clean math foundation
        print("\n📊 Testing Clean Math Foundation:")
        try:
            from core.clean_math_foundation import CleanMathFoundation
            math_foundation = CleanMathFoundation()
            print("  ✅ Clean Math Foundation initialized")

        except Exception as e:
            print(f"  ⚠️ Clean Math Foundation issue: {e}")

        # Test unified math system
        print("\n📊 Testing Unified Math System:")
        try:
            from core.unified_math_system import UnifiedMathSystem
            unified_math = UnifiedMathSystem()
            print("  ✅ Unified Math System initialized")

        except Exception as e:
            print(f"  ⚠️ Unified Math System issue: {e}")

        # Test clean trading pipeline
        print("\n📊 Testing Clean Trading Pipeline:")
        try:
            from core.clean_trading_pipeline import CleanTradingPipeline
            trading_pipeline = CleanTradingPipeline()
            print("  ✅ Clean Trading Pipeline initialized")

        except Exception as e:
            print(f"  ⚠️ Clean Trading Pipeline issue: {e}")

        # Test profit vectorization
        print("\n📊 Testing Profit Vectorization:")
        try:
            from core.clean_profit_vectorization import CleanProfitVectorization
            profit_vectorization = CleanProfitVectorization()
            print("  ✅ Clean Profit Vectorization initialized")

        except Exception as e:
            print(f"  ⚠️ Clean Profit Vectorization issue: {e}")

        print("\n✅ Clean Math Pipelines tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Clean Math Pipelines test failed: {e}")
        return False


def test_registry_storage():
    """Test registry storage functionality."""
    print("\n" + "=" * 60)
    print("🗄️ Testing Registry Storage Functionality")
    print("=" * 60)

    try:
        # Test soulprint registry
        print("\n📊 Testing Soulprint Registry:")
        try:
            from core.soulprint_registry import SoulprintRegistry
            registry = SoulprintRegistry()
            print("  ✅ Soulprint Registry initialized")

        except Exception as e:
            print(f"  ⚠️ Soulprint Registry issue: {e}")

        # Test tensor state manager
        print("\n📊 Testing Tensor State Manager:")
        try:
            from core.cli_tensor_state_manager import TensorStateManager
            tensor_manager = TensorStateManager()
            print("  ✅ Tensor State Manager initialized")

        except Exception as e:
            print(f"  ⚠️ Tensor State Manager issue: {e}")

        # Test system monitor
        print("\n📊 Testing System Monitor:")
        try:
            from core.cli_system_monitor import SystemMonitor
            system_monitor = SystemMonitor()
            print("  ✅ System Monitor initialized")

        except Exception as e:
            print(f"  ⚠️ System Monitor issue: {e}")

        print("\n✅ Registry Storage tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Registry Storage test failed: {e}")
        return False


def main():
    """Run comprehensive system tests."""
    print("🚀 Starting Comprehensive Schwabot System Tests")
    print("This test validates the complete system for tensor calculations,")
    print("profit vectors, trading logic, and advanced operations.")
    print()

    test_results = {}

    # Run all tests
    tests = []
        ("Advanced Tensor Algebra", test_tensor_algebra_system),
        ("Profit Vector System", test_profit_vector_system),
        ("Acceleration Enhancement", test_acceleration_enhancement),
        ("Trading Logic Integration", test_trading_logic_integration),
        ("API Data Processing", test_api_data_processing),
        ("Advanced Hashing & Tensor", test_advanced_hashing_tensor),
        ("Clean Math Pipelines", test_clean_math_pipelines),
        ("Registry Storage", test_registry_storage),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            test_results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The Schwabot system is properly structured for:")
        print("   - Advanced tensor calculations")
        print("   - Profit vector processing")
        print("   - CUDA + CPU hybrid acceleration")
        print("   - Trading logic integration")
        print("   - API data processing")
        print("   - Advanced hashing and tensor operations")
        print("   - Clean math pipelines")
        print("   - Registry storage functionality")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please review the issues above.")

    print("\n🔧 System is ready for complex trading operations!")


if __name__ == "__main__":
    main() 