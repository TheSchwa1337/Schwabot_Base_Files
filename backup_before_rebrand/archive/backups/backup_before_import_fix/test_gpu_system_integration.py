#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU System Integration Test for Schwabot Trading System
======================================================

Test script to demonstrate the GPU system state profiler and auto-detection
capabilities. This validates the hardware-adaptive GPU acceleration system.

Test Coverage:
- System state profiling (CPU, GPU, memory)
- GPU DNA detection and tier classification  
- Shader configuration optimization
- GPU fit testing
- Performance benchmarking
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


def test_system_state_profiler():
    """Test system state profiling capabilities."""
    logger.info("🔍 Testing System State Profiler...")

    try:
        from core.system_state_profiler import create_system_profiler, get_system_profile

        # Get system profile
        profile = get_system_profile()

        logger.info("✅ System Profile Generated Successfully")
        logger.info(f"🔧 Device: {profile.device_type} ({profile.device_id})")
        logger.info(f"💻 OS: {profile.os_info}")
        logger.info(f"🧠 CPU: {profile.cpu.model} ({profile.cpu.cpu_tier.value})")
        logger.info(f"⚙️  Cores: {profile.cpu.logical_cores} logical, {profile.cpu.physical_cores} physical")
        logger.info(f"⚡ Frequency: {profile.cpu.base_frequency:.2f} GHz (base)")
        logger.info(f"🎮 GPU: {profile.gpu.renderer} ({profile.gpu.gpu_tier.value})")
        logger.info(f"📊 Matrix Size: {profile.gpu.max_matrix_size}x{profile.gpu.max_matrix_size}")
        logger.info(f"🔧 Precision: {'Half' if profile.gpu.use_half_precision else 'Full'}")
        logger.info(f"🌊 Morphing: {'Enabled' if profile.gpu.shader_morph_enabled else 'Disabled'}")
        logger.info(f"💾 RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available")
        logger.info(f"🏆 System Tier: {profile.system_tier.value}")
        logger.info(f"🔐 System Hash: {profile.system_hash[:16]}...")

        return True, profile

    except Exception as e:
        logger.error(f"❌ System State Profiler test failed: {e}")
        return False, None


def test_gpu_dna_detection():
    """Test GPU DNA auto-detection."""
    logger.info("🧬 Testing GPU DNA Auto-Detection...")

    try:
        from core.gpu_dna_autodetect import detect_gpu_dna, get_cosine_similarity_config

        # Detect GPU DNA
        dna_profile = detect_gpu_dna()

        logger.info("✅ GPU DNA Detection Complete")
        logger.info(f"🎮 GPU Fingerprint: {dna_profile['gpu_fingerprint']['renderer']}")
        logger.info(f"🏆 GPU Tier: {dna_profile['gpu_fingerprint']['gpu_tier']}")
        logger.info(f"📊 Shader Config:")
        shader_config = dna_profile['shader_config']
        logger.info(f"   Matrix Size: {shader_config['matrix_size']}x{shader_config['matrix_size']}")
        logger.info(f"   Batch Size: {shader_config['batch_size']}")
        logger.info(f"   Precision: {'Half' if shader_config['use_half_precision'] else 'Full'}")
        logger.info(f"   Morphing: {'Enabled' if shader_config['shader_morph_enabled'] else 'Disabled'}")
        logger.info(f"   Performance Multiplier: {shader_config['performance_multiplier']}x")

        # Test cosine similarity config
        cosine_config = get_cosine_similarity_config()
        logger.info(f"🔢 Cosine Similarity Config:")
        logger.info(f"   Matrix Size: {cosine_config['matrix_size']}")
        logger.info(f"   Precision: {cosine_config['precision']}")
        logger.info(f"   Batch Strategies: {cosine_config['batch_strategies']}")

        return True, dna_profile

    except Exception as e:
        logger.error(f"❌ GPU DNA Detection test failed: {e}")
        return False, None


def test_gpu_fit_test():
    """Test GPU fit testing capabilities."""
    logger.info("🧪 Testing GPU Fit Test...")

    try:
        from core.gpu_dna_autodetect import run_gpu_fit_test

        # Run GPU fit test
        fit_results = run_gpu_fit_test()

        if fit_results['test_passed']:
            logger.info("✅ GPU Fit Test Passed")
            logger.info(f"📊 Maximum Matrix Size: {fit_results['max_matrix_size']}x{fit_results['max_matrix_size']}")
            logger.info(f"🔧 Configured Size: {fit_results['configured_size']}x{fit_results['configured_size']}")
            logger.info(f"💡 Recommended Size: {fit_results['recommended_size']}x{fit_results['recommended_size']}")
            logger.info(f"🎮 GPU Tier: {fit_results['gpu_tier']}")
            logger.info(f"⚡ Performance Multiplier: {fit_results['performance_multiplier']}x")
        else:
            logger.warning(f"⚠️  GPU Fit Test Failed: {fit_results.get('error', 'Unknown error')}")
            logger.info(f"🔄 Fallback Matrix Size: {fit_results['max_matrix_size']}x{fit_results['max_matrix_size']}")

        return fit_results['test_passed'], fit_results

    except Exception as e:
        logger.error(f"❌ GPU Fit Test failed: {e}")
        return False, None


def test_shader_integration():
    """Test GPU shader integration."""
    logger.info("🔧 Testing GPU Shader Integration...")

    try:
        from core.gpu_shader_integration import create_gpu_shader_integration

        # Create shader integration
        integration = create_gpu_shader_integration()

        # Test with sample data
        logger.info("🧮 Testing GPU-accelerated cosine similarity...")

        # Create test vectors
        vector_size = 32
        num_strategies = 10

        tick_vector = np.random.rand(vector_size).astype(np.float32)
        strategy_vectors = np.random.rand(num_strategies, vector_size).astype(np.float32)

        # Compute similarities
        start_time = time.time()
        similarities = integration.compute_strategy_similarity(tick_vector, strategy_vectors)
        execution_time = time.time() - start_time

        logger.info(f"✅ GPU Cosine Similarity Complete")
        logger.info(f"⏱️  Execution Time: {execution_time:.3f}s")
        logger.info(f"📊 Results: {len(similarities)} similarities computed")
        logger.info(f"📈 Similarity Range: [{similarities.min():.3f}, {similarities.max():.3f}]")

        # Get performance metrics
        metrics = integration.get_performance_metrics()
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"   GPU Init Time: {metrics['gpu_init_time']:.2f}s")
        logger.info(f"   Shader Compile Time: {metrics['shader_compile_time']:.2f}s")
        logger.info(f"   Average Execution Time: {metrics['average_execution_time']:.3f}s")
        logger.info(f"   Operations Count: {metrics['operations_count']}")
        logger.info(f"   OpenGL Available: {metrics['opengl_available']}")
        logger.info(f"   OpenGL Initialized: {metrics['opengl_initialized']}")

        # Cleanup
        integration.cleanup()

        return True, metrics

    except Exception as e:
        logger.error(f"❌ GPU Shader Integration test failed: {e}")
        return False, None


def test_core_integration():
    """Test integration with core Schwabot system."""
    logger.info("🚀 Testing Core Schwabot Integration...")

    try:
            get_system_status, 
            create_clean_trading_system,
            initialize_gpu_system
        )

        # Check system status
        status = get_system_status()
        logger.info("✅ System Status Retrieved")
        logger.info(f"🔧 Clean Implementations Available: {status['clean_implementations']}")
        logger.info(f"🎮 GPU System Available: {status['gpu_system']}")
        logger.info(f"🚀 System Operational: {status['system_operational']}")
        logger.info(f"⚡ GPU Acceleration Available: {status['gpu_acceleration_available']}")

        # Initialize GPU system
        if status['gpu_acceleration_available']:
            logger.info("🧬 Initializing GPU System...")
            gpu_system = initialize_gpu_system()

            init_status = gpu_system['initialization_status']
            logger.info(f"📊 GPU System Initialization:")
            logger.info(f"   System Profiler: {'✅' if init_status['profiler'] else '❌'}")
            logger.info(f"   DNA Detection: {'✅' if init_status['dna_detection'] else '❌'}")
            logger.info(f"   Shader Integration: {'✅' if init_status['shader_integration'] else '❌'}")

            if gpu_system['system_profile']:
                profile = gpu_system['system_profile']
                logger.info(f"🎮 GPU: {profile.gpu.renderer} ({profile.gpu.gpu_tier.value})")

        # Create trading system with GPU acceleration
        logger.info("🏪 Creating GPU-Accelerated Trading System...")
        trading_system = create_clean_trading_system()
            initial_capital=100000.0,
            enable_gpu_acceleration=True
        )

        logger.info("✅ GPU-Accelerated Trading System Created")
        logger.info(f"💰 Initial Capital: $100,00")
        logger.info(f"🧮 Components: {list(trading_system.keys())}")

        return True, trading_system

    except Exception as e:
        logger.error(f"❌ Core integration test failed: {e}")
        return False, None


def benchmark_cpu_vs_gpu():
    """Benchmark CPU vs GPU performance for cosine similarity."""
    logger.info("⚡ Benchmarking CPU vs GPU Performance...")

    try:
        import numpy as np

        from core.gpu_shader_integration import compute_strategy_similarities_gpu

        # Test parameters
        vector_sizes = [16, 32, 64]
        num_strategies = [10, 50, 100]

        benchmark_results = {}

        for vec_size in vector_sizes:
            for num_strat in num_strategies:
                if vec_size > 64 or num_strat > 100:
                    continue  # Skip large tests for demo

                logger.info(f"🧮 Testing {vec_size}D vectors, {num_strat} strategies...")

                # Generate test data
                tick_vector = np.random.rand(vec_size).astype(np.float32)
                strategy_vectors = np.random.rand(num_strat, vec_size).astype(np.float32)

                # CPU benchmark
                start_time = time.time()
                cpu_similarities = np.dot(strategy_vectors, tick_vector) / ()
                    np.linalg.norm(tick_vector) * np.linalg.norm(strategy_vectors, axis=1) + 1e-8
                )
                cpu_time = time.time() - start_time

                # GPU benchmark
                start_time = time.time()
                gpu_similarities = compute_strategy_similarities_gpu(tick_vector, strategy_vectors)
                gpu_time = time.time() - start_time

                # Calculate speedup
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0

                test_key = f"{vec_size}D_{num_strat}strat"
                benchmark_results[test_key] = {}
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": speedup,
                    "vector_size": vec_size,
                    "num_strategies": num_strat
                }

                logger.info(f"   CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {speedup:.1f}x")

        # Summary
        logger.info("📊 Benchmark Summary:")
        for test_key, results in benchmark_results.items():
            logger.info(f"   {test_key}: {results['speedup']:.1f}x speedup")

        return True, benchmark_results

    except Exception as e:
        logger.error(f"❌ Benchmark test failed: {e}")
        return False, None


def main():
    """Run all GPU system integration tests."""
    logger.info("🧬 SCHWABOT GPU SYSTEM INTEGRATION TEST")
    logger.info("=" * 60)

    test_results = {}

    # Test 1: System State Profiler
    logger.info("\n" + "="*60)
    success, result = test_system_state_profiler()
    test_results['system_profiler'] = {'success': success, 'result': result}

    # Test 2: GPU DNA Detection
    logger.info("\n" + "="*60)
    success, result = test_gpu_dna_detection()
    test_results['gpu_dna'] = {'success': success, 'result': result}

    # Test 3: GPU Fit Test
    logger.info("\n" + "="*60)
    success, result = test_gpu_fit_test()
    test_results['gpu_fit_test'] = {'success': success, 'result': result}

    # Test 4: Shader Integration
    logger.info("\n" + "="*60)
    success, result = test_shader_integration()
    test_results['shader_integration'] = {'success': success, 'result': result}

    # Test 5: Core Integration
    logger.info("\n" + "="*60)
    success, result = test_core_integration()
    test_results['core_integration'] = {'success': success, 'result': result}

    # Test 6: Performance Benchmark
    logger.info("\n" + "="*60)
    success, result = benchmark_cpu_vs_gpu()
    test_results['benchmark'] = {'success': success, 'result': result}

    # Final Summary
    logger.info("\n" + "="*60)
    logger.info("🎯 FINAL TEST SUMMARY")
    logger.info("="*60)

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['success'])

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")

    logger.info(f"\n🏆 Tests Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        logger.info("🚀 ALL TESTS PASSED - GPU SYSTEM FULLY OPERATIONAL")
        logger.info("💡 Schwabot is ready for GPU-accelerated trading!")
    else:
        logger.warning("⚠️  SOME TESTS FAILED - CHECK SYSTEM CONFIGURATION")
        logger.info("🔄 Schwabot will use CPU fallback mode")

    return test_results


if __name__ == "__main__":
    main() 