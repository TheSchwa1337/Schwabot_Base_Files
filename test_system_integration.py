#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Integration Test for Schwabot Trading System

Tests the integration between:
- SystemFitProfile and CUDA helper
- Mathematical functions and trading strategies
- GPU-aware scaling and matrix operations
- Core trading modules and mathematical core
"""

import sys
import os
import logging
import numpy as np
import time
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_cuda_helper_integration():
    """Test CUDA helper integration with SystemFitProfile."""
    print("🔧 Testing CUDA Helper Integration")
    print("=" * 50)
    
    try:
        from utils.cuda_helper import (
            FIT_PROFILE, math_core, detector,
            matrix_fit, cosine_match, entropy_of_vector,
            flatness_measure, phantom_score, ideal_tick_time,
            memory_tile_limit, gpu_load_ratio, smooth_gradient_detection
        )
        
        # Test SystemFitProfile
        print(f"✅ SystemFitProfile loaded successfully")
        print(f"   GPU Tier: {FIT_PROFILE.gpu_tier}")
        print(f"   Device Type: {FIT_PROFILE.device_type}")
        print(f"   Matrix Size: {FIT_PROFILE.matrix_size}")
        print(f"   Precision: {FIT_PROFILE.precision}")
        print(f"   Can Run GPU Logic: {FIT_PROFILE.can_run_gpu_logic}")
        print(f"   System Hash: {FIT_PROFILE.system_hash[:12]}...")
        
        # Test mathematical functions
        print(f"\n🧮 Testing Mathematical Functions:")
        
        # Matrix operations
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        C = matrix_fit(A, B)
        print(f"   Matrix Fit: {C.shape} ✅")
        
        # Cosine similarity
        vec1 = np.array([1, 2, 3, 4])
        vec2 = np.array([4, 3, 2, 1])
        similarity = cosine_match(vec1, vec2)
        print(f"   Cosine Similarity: {similarity:.4f} ✅")
        
        # Entropy
        data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4])
        entropy = entropy_of_vector(data)
        print(f"   Entropy: {entropy:.4f} ✅")
        
        # Flatness
        prices = np.array([100, 101, 100, 99, 100, 101, 100, 99, 100])
        flatness = flatness_measure(prices)
        print(f"   Flatness: {flatness:.4f} ✅")
        
        # Phantom score
        phantom = phantom_score(prices)
        print(f"   Phantom Score: {phantom:.4f} ✅")
        
        # Timing calculations
        tick_time = ideal_tick_time(1000, 1.5)
        print(f"   Ideal Tick Time: {tick_time:.6f}s ✅")
        
        # Memory calculations
        tile_limit = memory_tile_limit(2 * 1024**3, 32)  # 2GB, 32x32 matrix
        print(f"   Memory Tile Limit: {tile_limit} ✅")
        
        # Load ratio
        load_ratio = gpu_load_ratio(75, 100)
        print(f"   GPU Load Ratio: {load_ratio:.2f} ✅")
        
        # Smooth gradient
        smooth_grad = smooth_gradient_detection(prices, window=3)
        print(f"   Smooth Gradient: {smooth_grad:.4f} ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA Helper Integration failed: {e}")
        return False


def test_matrix_math_utils():
    """Test matrix math utils integration."""
    print(f"\n📊 Testing Matrix Math Utils")
    print("=" * 50)
    
    try:
        from core.matrix_math_utils import (
            analyze_price_matrix, risk_parity_weights,
            calculate_sharpe_ratio, calculate_max_drawdown,
            calculate_var, calculate_cvar
        )
        
        print("✅ Matrix Math Utils loaded successfully")
        
        # Test price matrix analysis
        price_data = np.random.rand(100, 5) * 100  # 100 samples, 5 assets
        analysis = analyze_price_matrix(price_data)
        print(f"   Price Matrix Analysis: {analysis['num_assets']} assets ✅")
        
        # Test risk parity weights
        cov_matrix = np.random.rand(5, 5)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make it positive definite
        weights, metadata = risk_parity_weights(cov_matrix)
        print(f"   Risk Parity Weights: {weights.shape} ✅")
        
        # Test Sharpe ratio
        returns = np.random.randn(100) * 0.01
        sharpe = calculate_sharpe_ratio(returns)
        print(f"   Sharpe Ratio: {sharpe:.4f} ✅")
        
        # Test max drawdown
        drawdown = calculate_max_drawdown(returns)
        print(f"   Max Drawdown: {drawdown['max_drawdown']:.4f} ✅")
        
        # Test VaR
        var = calculate_var(returns)
        print(f"   VaR: {var:.4f} ✅")
        
        # Test CVaR
        cvar = calculate_cvar(returns)
        print(f"   CVaR: {cvar:.4f} ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Matrix Math Utils failed: {e}")
        return False


def test_core_trading_modules():
    """Test core trading modules integration."""
    print(f"\n📈 Testing Core Trading Modules")
    print("=" * 50)
    
    try:
        # Test unified math system
        from core.unified_math_system import UnifiedMathSystem
        
        math_system = UnifiedMathSystem()
        print("✅ Unified Math System loaded successfully")
        
        # Test matrix operations
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        result = math_system.matrix_multiply(A, B)
        print(f"   Matrix Multiply: {result.shape} ✅")
        
        # Test vector operations
        vec1 = np.random.rand(10)
        vec2 = np.random.rand(10)
        similarity = math_system.cosine_similarity(vec1, vec2)
        print(f"   Cosine Similarity: {similarity:.4f} ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Core Trading Modules failed: {e}")
        return False


def test_system_scaling():
    """Test system-aware scaling functionality."""
    print(f"\n⚙️ Testing System-Aware Scaling")
    print("=" * 50)
    
    try:
        from utils.cuda_helper import FIT_PROFILE, math_core, memory_tile_limit
        
        # Test matrix size scaling
        print(f"Current Matrix Size: {FIT_PROFILE.matrix_size}")
        
        # Test with different matrix sizes
        sizes = [8, 16, 32, 64]
        for size in sizes:
            if size <= FIT_PROFILE.matrix_size:
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)
                start_time = time.time()
                C = math_core.matrix_fit(A, B)
                end_time = time.time()
                print(f"   {size}x{size} Matrix: {(end_time - start_time)*1000:.2f}ms ✅")
            else:
                print(f"   {size}x{size} Matrix: Skipped (exceeds limit) ⏭️")
        
        # Test precision scaling
        print(f"Precision Mode: {FIT_PROFILE.precision}")
        
        # Test memory tile calculations
        for mem_gb in [1, 2, 4, 8]:
            tiles = memory_tile_limit(mem_gb * 1024**3, FIT_PROFILE.matrix_size)
            print(f"   {mem_gb}GB Memory: {tiles} tiles ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ System Scaling failed: {e}")
        return False


def test_performance_benchmarks():
    """Test performance benchmarks."""
    print(f"\n🚀 Testing Performance Benchmarks")
    print("=" * 50)
    
    try:
        from utils.cuda_helper import (
            matrix_fit, cosine_match, entropy_of_vector,
            safe_matrix_multiply, safe_fft, FIT_PROFILE
        )
        
        # Benchmark matrix operations
        sizes = [16, 32, 64]
        for size in sizes:
            if size <= FIT_PROFILE.matrix_size:
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)
                
                # Direct matrix fit
                start_time = time.time()
                C1 = matrix_fit(A, B)
                time1 = (time.time() - start_time) * 1000
                
                # Safe matrix multiply
                start_time = time.time()
                C2 = safe_matrix_multiply(A, B)
                time2 = (time.time() - start_time) * 1000
                
                print(f"   {size}x{size} Matrix:")
                print(f"     Direct: {time1:.2f}ms")
                print(f"     Safe: {time2:.2f}ms ✅")
        
        # Benchmark FFT
        data_sizes = [1024, 2048, 4096]
        for size in data_sizes:
            data = np.random.rand(size)
            start_time = time.time()
            result = safe_fft(data)
            fft_time = (time.time() - start_time) * 1000
            print(f"   FFT {size} points: {fft_time:.2f}ms ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmarks failed: {e}")
        return False


def test_mathematical_core_functions():
    """Test all mathematical core functions."""
    print(f"\n🧮 Testing Mathematical Core Functions")
    print("=" * 50)
    
    try:
        from utils.cuda_helper import math_core
        
        # Test all mathematical functions
        print("Testing matrix_fit...")
        A = np.random.rand(4, 4)
        B = np.random.rand(4, 4)
        C = math_core.matrix_fit(A, B)
        print(f"   Matrix Fit: {C.shape} ✅")
        
        print("Testing cosine_match...")
        vec1 = np.random.rand(10)
        vec2 = np.random.rand(10)
        similarity = math_core.cosine_match(vec1, vec2)
        print(f"   Cosine Match: {similarity:.4f} ✅")
        
        print("Testing entropy_of_vector...")
        data = np.random.randint(0, 5, 100)
        entropy = math_core.entropy_of_vector(data)
        print(f"   Entropy: {entropy:.4f} ✅")
        
        print("Testing flatness_measure...")
        prices = np.random.rand(50) * 100
        flatness = math_core.flatness_measure(prices)
        print(f"   Flatness: {flatness:.4f} ✅")
        
        print("Testing ideal_tick_time...")
        tick_time = math_core.ideal_tick_time(1000)
        print(f"   Ideal Tick Time: {tick_time:.6f}s ✅")
        
        print("Testing memory_tile_limit...")
        tiles = math_core.memory_tile_limit()
        print(f"   Memory Tiles: {tiles} ✅")
        
        print("Testing gpu_load_ratio...")
        load_ratio = math_core.gpu_load_ratio(75)
        print(f"   Load Ratio: {load_ratio:.2f} ✅")
        
        print("Testing smooth_gradient_detection...")
        smooth_grad = math_core.smooth_gradient_detection(prices)
        print(f"   Smooth Gradient: {smooth_grad:.4f} ✅")
        
        print("Testing phantom_score...")
        phantom = math_core.phantom_score(prices)
        print(f"   Phantom Score: {phantom:.4f} ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical Core Functions failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧪 Schwabot System Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("CUDA Helper Integration", test_cuda_helper_integration),
        ("Matrix Math Utils", test_matrix_math_utils),
        ("Core Trading Modules", test_core_trading_modules),
        ("System-Aware Scaling", test_system_scaling),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Mathematical Core Functions", test_mathematical_core_functions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 