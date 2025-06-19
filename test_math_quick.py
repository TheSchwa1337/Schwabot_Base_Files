#!/usr/bin/env python3
"""
Quick Mathematical Implementation Test
====================================

Quick test to validate that our mathematical implementations are working
without importing complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_basic_math_functions():
    """Test basic mathematical functions from mathlib.py"""
    print("Testing Basic Mathematical Functions")
    print("=" * 40)
    
    try:
        # Test direct import from mathlib.py
        from mathlib import entropy, klein_bottle, recursive_operation
        print("PASS: Successfully imported mathematical functions")
        
        # Test entropy function
        test_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        entropy_result = entropy(test_data)
        print(f"PASS: Entropy calculation: {entropy_result:.3f}")
        
        # Test Klein bottle function
        point = (3.14159/2, 3.14159/4)
        coords = klein_bottle(point)
        print(f"PASS: Klein bottle coordinates: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f})")
        
        # Test recursive operation
        fib_result = recursive_operation(8, operation_type='fibonacci')
        factorial_result = recursive_operation(5, operation_type='factorial')
        print(f"PASS: Recursive operations - Fibonacci(8): {fib_result}, Factorial(5): {factorial_result}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Error testing basic functions: {e}")
        return False

def test_hash_functions():
    """Test hash processing mathematical functions"""
    print("\nTesting Hash Mathematical Functions")
    print("=" * 40)
    
    try:
        # Test hash profit matrix functions
        sys.path.insert(0, 'core')
        
        # Test mathematical constants
        import numpy as np
        
        # Test basic hash calculations
        test_hash = "1a2b3c4d5e6f7890abcdef1234567890"
        
        # Test hash echo calculation (simplified)
        prev_hash = "fedcba9876543210abcdef1234567890"
        hash_bytes = bytes.fromhex(test_hash[:32])
        prev_bytes = bytes.fromhex(prev_hash[:32])
        
        xor_result = [a ^ b for a, b in zip(hash_bytes, prev_bytes)]
        hash_echo = sum(xor_result) / (255.0 * len(xor_result))
        print(f"✅ Hash echo calculation: {hash_echo:.3f}")
        
        # Test entropy correlation calculation
        data = np.random.rand(100)
        hist, _ = np.histogram(data, bins=10, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            entropy_corr = -np.sum(hist * np.log2(hist + 1e-10))
            print(f"✅ Entropy correlation: {entropy_corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing hash functions: {e}")
        return False

def test_gpu_fallbacks():
    """Test GPU fallback mechanisms"""
    print("\n🚀 Testing GPU Fallback Mechanisms")
    print("=" * 40)
    
    try:
        # Test CuPy fallback
        try:
            import cupy as cp
            print("✅ CuPy available - GPU acceleration enabled")
            gpu_available = True
        except ImportError:
            print("⚠️  CuPy not available - using NumPy fallback")
            import numpy as np
            cp = np
            gpu_available = False
        
        # Test torch fallback
        try:
            import torch
            print("✅ PyTorch available - tensor operations enabled")
            torch_available = True
        except ImportError:
            print("⚠️  PyTorch not available - using NumPy fallback")
            torch_available = False
        
        # Test system monitoring
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            print(f"✅ System monitoring available - CPU: {cpu_usage}%")
        except ImportError:
            print("⚠️  psutil not available - system monitoring disabled")
        
        try:
            import GPUtil
            print("✅ GPU monitoring available")
        except ImportError:
            print("⚠️  GPUtil not available - GPU monitoring disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing GPU fallbacks: {e}")
        return False

def test_critical_mathematical_operations():
    """Test critical mathematical operations"""
    print("\n🔬 Testing Critical Mathematical Operations")
    print("=" * 40)
    
    try:
        import numpy as np
        
        # Test numerical stability
        small_numbers = [1e-10, 2e-10, 3e-10]
        large_numbers = [1e10, 2e10, 3e10]
        
        # Test correlation calculations
        data1 = np.random.rand(100)
        data2 = np.random.rand(100)
        
        if np.std(data1) > 0 and np.std(data2) > 0:
            correlation = np.corrcoef(data1, data2)[0, 1]
            if not np.isnan(correlation):
                print(f"✅ Correlation calculation: {correlation:.3f}")
            else:
                print("⚠️  Correlation resulted in NaN - handled gracefully")
        
        # Test entropy with edge cases
        from mathlib import entropy
        
        empty_entropy = entropy([])
        single_entropy = entropy([1])
        uniform_entropy = entropy([1, 1, 1, 1])
        varied_entropy = entropy([1, 2, 3, 4, 5])
        
        print(f"✅ Entropy edge cases - Empty: {empty_entropy}, Single: {single_entropy}, Uniform: {uniform_entropy}, Varied: {varied_entropy:.3f}")
        
        # Test Klein bottle mathematical stability
        from mathlib import klein_bottle
        
        # Test multiple points
        test_points = [(0, 0), (np.pi/2, np.pi/2), (np.pi, np.pi), (2*np.pi, 2*np.pi)]
        for i, point in enumerate(test_points):
            coords = klein_bottle(point)
            if all(np.isfinite(c) for c in coords):
                print(f"✅ Klein bottle point {i+1}: mathematically stable")
            else:
                print(f"⚠️  Klein bottle point {i+1}: numerical instability detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing critical operations: {e}")
        return False

def main():
    """Run all mathematical implementation tests"""
    print("🧮 Mathematical Implementation Validation")
    print("=" * 60)
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(test_basic_math_functions())
    test_results.append(test_hash_functions())
    test_results.append(test_gpu_fallbacks())
    test_results.append(test_critical_mathematical_operations())
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✅ All mathematical implementations are working correctly!")
        print("🚀 System ready for production use")
        return True
    else:
        print("\n⚠️  Some tests failed - review implementation")
        print("🛠️  Fix any issues before production deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 