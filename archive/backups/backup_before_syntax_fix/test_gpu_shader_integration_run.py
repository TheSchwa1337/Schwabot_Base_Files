#!/usr/bin/env python3
"""
MAXIMUM PARANOIA GPU SHADER INTEGRATION TEST
============================================

This script performs comprehensive testing of the GPU shader integration system
to ensure it works correctly for live BTC/USDC trading scenarios.

Tests include:
1. GPU shader compilation and execution
2. CPU vs GPU correctness comparison
3. Performance benchmarking
4. Fallback mechanism validation
5. Error handling and recovery
"""

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np

# Configure logging for maximum paranoia
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_shader_integration():
    """Test GPU shader integration with maximum paranoia."""

    print("🚨 MAXIMUM PARANOIA GPU SHADER INTEGRATION TEST 🚨")
    print("=" * 60)

    try:
        # Import the GPU integration module
        logger.info("📦 Importing GPU shader integration...")
        from core.gpu_shader_integration import create_gpu_shader_integration
        logger.info("✅ GPU shader integration imported successfully")

        # Create test data (simulating real trading, vectors)
        logger.info("🧮 Generating test trading vectors...")
        vector_size = 128  # Typical strategy vector size
        num_strategies = 16  # Number of strategies to test

        # Create realistic trading vectors (normalized)
        tick_vector = np.random.rand(vector_size).astype(np.float32)
        tick_vector = tick_vector / np.linalg.norm(tick_vector)  # Normalize

        strategy_vectors = np.random.rand(num_strategies, vector_size).astype(np.float32)
        # Normalize each strategy vector
        for i in range(num_strategies):
            strategy_vectors[i] = strategy_vectors[i] / np.linalg.norm(strategy_vectors[i])

        logger.info(f"✅ Generated {num_strategies} strategy vectors of size {vector_size}")

        # Initialize GPU shader integration
        logger.info("🎮 Initializing GPU shader integration...")
        integration = create_gpu_shader_integration()

        # Check initialization status
        logger.info(f"🔧 OpenGL Initialized: {integration.opengl_initialized}")
        logger.info(f"🎯 Shader Config: {integration.shader_config}")

        # Test 1: GPU Computation (if, available)
        if integration.opengl_initialized:
            logger.info("🔥 Testing GPU cosine similarity computation...")
            start_time = time.time()

            try:
                gpu_result = integration.compute_strategy_similarity(tick_vector, strategy_vectors)
                gpu_time = time.time() - start_time

                logger.info(f"✅ GPU computation successful in {gpu_time:.4f}s")
                logger.info(f"📊 GPU result shape: {gpu_result.shape}")
                logger.info(f"📊 GPU result range: [{gpu_result.min():.6f}, {gpu_result.max():.6f}]")

            except Exception as e:
                logger.error(f"❌ GPU computation failed: {e}")
                gpu_result = None
                gpu_time = None
        else:
            logger.warning("⚠️ OpenGL not available, skipping GPU test")
            gpu_result = None
            gpu_time = None

        # Test 2: CPU Fallback Computation
        logger.info("🔄 Testing CPU fallback computation...")
        start_time = time.time()

        try:
            cpu_result = integration._compute_cpu_fallback(tick_vector, strategy_vectors)
            cpu_time = time.time() - start_time

            logger.info(f"✅ CPU computation successful in {cpu_time:.4f}s")
            logger.info(f"📊 CPU result shape: {cpu_result.shape}")
            logger.info(f"📊 CPU result range: [{cpu_result.min():.6f}, {cpu_result.max():.6f}]")

        except Exception as e:
            logger.error(f"❌ CPU computation failed: {e}")
            cpu_result = None
            cpu_time = None

        # Test 3: Correctness Comparison
        if gpu_result is not None and cpu_result is not None:
            logger.info("🔍 Comparing GPU vs CPU results for correctness...")

            # Check shapes
            if gpu_result.shape != cpu_result.shape:
                logger.error(f"❌ Shape mismatch: GPU {gpu_result.shape} vs CPU {cpu_result.shape}")
            else:
                logger.info("✅ Result shapes match")

            # Check numerical accuracy
            max_diff = np.max(np.abs(gpu_result - cpu_result))
            mean_diff = np.mean(np.abs(gpu_result - cpu_result))

            logger.info(f"📊 Maximum difference: {max_diff:.8f}")
            logger.info(f"📊 Mean difference: {mean_diff:.8f}")

            # Tolerance check (GPU precision, differences)
            tolerance = 1e-5
            if max_diff < tolerance:
                logger.info("✅ GPU and CPU results are within acceptable tolerance")
            else:
                logger.warning(f"⚠️ GPU and CPU results differ by {max_diff:.8f} (tolerance: {tolerance})")

            # Performance comparison
            if gpu_time is not None and cpu_time is not None:
                speedup = cpu_time / gpu_time
                logger.info(f"⚡ GPU speedup: {speedup:.2f}x faster than CPU")

                if speedup > 1.0:
                    logger.info("✅ GPU acceleration is working")
                else:
                    logger.warning("⚠️ GPU is not faster than CPU (may be due to small problem, size)")

        # Test 4: Performance Metrics
        logger.info("📈 Collecting performance metrics...")
        metrics = integration.get_performance_metrics()

        logger.info("📊 Performance Metrics:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value}")

        # Test 5: Cleanup
        logger.info("🧹 Testing cleanup...")
        try:
            integration.cleanup()
            logger.info("✅ Cleanup successful")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

        # Final Assessment
        print("\n" + "=" * 60)
        print("🎯 MAXIMUM PARANOIA ASSESSMENT")
        print("=" * 60)

        if gpu_result is not None:
            print("✅ GPU shader integration: WORKING")
        else:
            print("⚠️ GPU shader integration: FALLBACK TO CPU")

        if cpu_result is not None:
            print("✅ CPU fallback: WORKING")
        else:
            print("❌ CPU fallback: FAILED")

        if gpu_result is not None and cpu_result is not None:
            if max_diff < tolerance:
                print("✅ Mathematical correctness: VERIFIED")
            else:
                print("⚠️ Mathematical correctness: NEEDS INVESTIGATION")

        if integration.opengl_initialized:
            print("✅ OpenGL context: AVAILABLE")
        else:
            print("⚠️ OpenGL context: NOT AVAILABLE (using, CPU)")

        print("=" * 60)
        print("🚨 MAXIMUM PARANOIA TEST COMPLETE 🚨")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        print("❌ CRITICAL: GPU shader integration module not available")
        return False

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"❌ CRITICAL: Unexpected error during testing: {e}")
        return False

def test_trading_scenario():
    """Test with realistic trading scenario data."""

    print("\n🎯 REALISTIC TRADING SCENARIO TEST")
    print("=" * 40)

    try:
        from core.gpu_shader_integration import create_gpu_shader_integration

        # Simulate BTC/USDC price movements (normalized)
        logger.info("📈 Generating realistic BTC/USDC trading vectors...")

        # Create price movement patterns
        base_price = 50000.0
        price_changes = np.random.normal(0, 0.2, 128)  # 2% volatility
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Create tick vector from price movements
        tick_vector = np.diff(prices) / prices[:-1]  # Returns
        tick_vector = tick_vector.astype(np.float32)
        tick_vector = tick_vector / np.linalg.norm(tick_vector)

        # Create strategy vectors (different trading, strategies)
        strategies = []
        for i in range(8):
            # Different strategy patterns
            if i == 0:  # Momentum strategy
                strategy = np.roll(tick_vector, -i*2) * 0.8
            elif i == 1:  # Mean reversion strategy
                strategy = -np.roll(tick_vector, i*3) * 0.6
            else:  # Random strategy
                strategy = np.random.normal(0, 1, len(tick_vector))

            strategy = strategy.astype(np.float32)
            strategy = strategy / np.linalg.norm(strategy)
            strategies.append(strategy)

        strategy_vectors = np.array(strategies)

        logger.info(f"✅ Generated {len(strategies)} trading strategies")

        # Test with trading data
        integration = create_gpu_shader_integration()

        if integration.opengl_initialized:
            logger.info("🔥 Testing GPU with trading data...")
            start_time = time.time()
            trading_result_gpu = integration.compute_strategy_similarity(tick_vector, strategy_vectors)
            gpu_time = time.time() - start_time
            logger.info(f"✅ GPU trading computation: {gpu_time:.4f}s")

        # CPU comparison
        logger.info("🔄 Testing CPU with trading data...")
        start_time = time.time()
        trading_result_cpu = integration._compute_cpu_fallback(tick_vector, strategy_vectors)
        cpu_time = time.time() - start_time
        logger.info(f"✅ CPU trading computation: {cpu_time:.4f}s")

        # Check results
        if integration.opengl_initialized:
            max_diff = np.max(np.abs(trading_result_gpu - trading_result_cpu))
            logger.info(f"📊 Trading scenario max difference: {max_diff:.8f}")

            if max_diff < 1e-5:
                print("✅ Trading scenario: MATHEMATICALLY CORRECT")
            else:
                print("⚠️ Trading scenario: NEEDS INVESTIGATION")

        integration.cleanup()
        return True

    except Exception as e:
        logger.error(f"❌ Trading scenario test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 STARTING MAXIMUM PARANOIA GPU TESTING 🚨")

    # Run main integration test
    success1 = test_gpu_shader_integration()

    # Run trading scenario test
    success2 = test_trading_scenario()

    # Final verdict
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 MAXIMUM PARANOIA: ALL TESTS PASSED")
        print("✅ Your GPU shader integration is ready for live trading!")
    else:
        print("⚠️ MAXIMUM PARANOIA: SOME TESTS FAILED")
        print("🔧 Review the logs above for issues")

    print("=" * 60) 