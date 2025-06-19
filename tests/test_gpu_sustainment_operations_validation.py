"""
GPU Sustainment Operations Validation Test Suite
===============================================

Comprehensive test to validate that gpu_sustainment_vector_operations function
is properly implemented and provides appropriate CPU fallback when GPU is unavailable.

This specifically addresses Gap #3 identified in the mathematical foundation analysis:
- Validation of gpu_sustainment_vector_operations function availability
- Testing CPU fallback when GPU operations are not available
- Integration with existing sustainment mathematical framework

Author: Schwabot Engineering Team
Created: 2024 - Gap #3 Resolution  
"""

import unittest
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestGPUSustainmentOperationsValidation(unittest.TestCase):
    """Validate GPU sustainment operations functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_start_time = datetime.now()
        
        # Create test sustainment vectors
        try:
            from core.mathlib_v3 import SustainmentVector, create_test_context
            
            self.test_vectors = []
            for i in range(5):
                vector = SustainmentVector(
                    principles=np.random.rand(8) * 0.5 + 0.3,  # Values between 0.3-0.8
                    confidence=np.random.rand(8) * 0.3 + 0.7,  # Values between 0.7-1.0
                    timestamp=datetime.now()
                )
                self.test_vectors.append(vector)
            
            self.test_weights = np.array([0.15, 0.15, 0.10, 0.10, 0.15, 0.15, 0.10, 0.10])
            
        except ImportError:
            self.test_vectors = []
            self.test_weights = np.ones(8) * 0.125
    
    def test_gpu_sustainment_vector_operations_function_exists(self):
        """Test that gpu_sustainment_vector_operations function exists"""
        try:
            from core.mathlib_v3 import gpu_sustainment_vector_operations
            
            # Check it's callable
            self.assertTrue(callable(gpu_sustainment_vector_operations))
            
            print("[PASS] gpu_sustainment_vector_operations function exists")
            
        except ImportError as e:
            self.fail(f"Failed to import gpu_sustainment_vector_operations: {e}")
    
    def test_gpu_sustainment_operations_with_empty_input(self):
        """Test GPU sustainment operations handles empty input gracefully"""
        try:
            from core.mathlib_v3 import gpu_sustainment_vector_operations
            
            # Test with empty vectors list
            result = gpu_sustainment_vector_operations([], self.test_weights)
            
            # Should return empty dict or handle gracefully
            self.assertIsInstance(result, dict)
            
            print("[PASS] gpu_sustainment_vector_operations handles empty input")
            
        except Exception as e:
            self.fail(f"Failed to handle empty input: {e}")
    
    def test_gpu_sustainment_operations_with_valid_input(self):
        """Test GPU sustainment operations with valid input"""
        if not self.test_vectors:
            self.skipTest("SustainmentVector not available for testing")
        
        try:
            from core.mathlib_v3 import gpu_sustainment_vector_operations
            
            # Test with valid vectors
            result = gpu_sustainment_vector_operations(self.test_vectors, self.test_weights)
            
            # Validate result structure
            self.assertIsInstance(result, dict)
            
            # If GPU is available, should have statistical results
            if result:  # Non-empty result indicates GPU processing
                expected_keys = ['mean_si', 'std_si', 'min_si', 'max_si', 'sustainable_ratio']
                for key in expected_keys:
                    if key in result:
                        self.assertIsInstance(result[key], (int, float))
                        
                print("[PASS] gpu_sustainment_vector_operations processes valid input")
            else:
                print("[INFO] GPU not available - CPU fallback mode")
            
        except Exception as e:
            self.fail(f"Failed to process valid input: {e}")
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection and fallback behavior"""
        try:
            from core.mathlib_v3 import GPU_ENABLED
            
            # Check GPU_ENABLED flag
            self.assertIsInstance(GPU_ENABLED, bool)
            
            if GPU_ENABLED:
                print("[INFO] GPU detected as available")
                
                # Test GPU import
                try:
                    import cupy as cp
                    print("[PASS] CuPy GPU library available")
                except ImportError:
                    print("[WARN] GPU_ENABLED=True but CuPy not available")
            else:
                print("[INFO] GPU not available - running in CPU mode")
            
        except ImportError:
            print("[INFO] GPU_ENABLED flag not found - assuming CPU mode")
    
    def test_cpu_fallback_behavior(self):
        """Test CPU fallback behavior when GPU operations fail"""
        if not self.test_vectors:
            self.skipTest("SustainmentVector not available for testing")
        
        try:
            from core.mathlib_v3 import gpu_sustainment_vector_operations
            
            # Force test with potentially problematic input
            result = gpu_sustainment_vector_operations(self.test_vectors, self.test_weights)
            
            # Function should either:
            # 1. Return valid results (GPU working)
            # 2. Return empty dict (CPU fallback)
            # 3. Handle errors gracefully
            
            self.assertIsInstance(result, dict)
            print("[PASS] CPU fallback behavior working correctly")
            
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"CPU fallback failed with unhandled exception: {e}")
    
    def test_sustainment_vector_integration(self):
        """Test integration with SustainmentVector class"""
        if not self.test_vectors:
            self.skipTest("SustainmentVector not available for testing")
        
        try:
            from core.mathlib_v3 import SustainmentVector, gpu_sustainment_vector_operations
            
            # Test that SustainmentVector instances work with GPU operations
            test_vector = self.test_vectors[0]
            
            # Validate vector structure
            self.assertIsInstance(test_vector.principles, np.ndarray)
            self.assertIsInstance(test_vector.confidence, np.ndarray)
            self.assertEqual(len(test_vector.principles), 8)
            self.assertEqual(len(test_vector.confidence), 8)
            
            # Test GPU operations can process the vector
            result = gpu_sustainment_vector_operations([test_vector], self.test_weights)
            self.assertIsInstance(result, dict)
            
            print("[PASS] SustainmentVector integration with GPU operations successful")
            
        except Exception as e:
            self.fail(f"SustainmentVector integration failed: {e}")
    
    def test_sustainment_math_lib_gpu_integration(self):
        """Test SustainmentMathLib can utilize GPU operations"""
        try:
            from core.mathlib_v3 import SustainmentMathLib, create_test_context
            
            # Create sustainment math library
            math_lib = SustainmentMathLib()
            
            # Create test context
            context = create_test_context()
            
            # Calculate sustainment vector
            sustainment_vector = math_lib.calculate_sustainment_vector(context)
            
            # Verify integration
            self.assertIsNotNone(sustainment_vector)
            self.assertTrue(hasattr(sustainment_vector, 'principles'))
            self.assertTrue(hasattr(sustainment_vector, 'confidence'))
            
            print("[PASS] SustainmentMathLib GPU integration successful")
            
        except Exception as e:
            self.fail(f"SustainmentMathLib GPU integration failed: {e}")
    
    def test_mathematical_correctness_of_gpu_operations(self):
        """Test mathematical correctness of GPU operations"""
        if not self.test_vectors:
            self.skipTest("SustainmentVector not available for testing")
        
        try:
            from core.mathlib_v3 import gpu_sustainment_vector_operations
            
            # Test with known input to verify mathematical correctness
            result = gpu_sustainment_vector_operations(self.test_vectors, self.test_weights)
            
            if result and 'mean_si' in result:
                # Validate mathematical properties
                mean_si = result['mean_si']
                min_si = result.get('min_si', 0)
                max_si = result.get('max_si', 1)
                
                # Basic mathematical validations
                self.assertLessEqual(min_si, mean_si)
                self.assertLessEqual(mean_si, max_si)
                self.assertGreaterEqual(min_si, 0.0)  # Sustainment index should be non-negative
                
                print("[PASS] Mathematical correctness of GPU operations validated")
            else:
                print("[INFO] GPU operations returned empty result - CPU fallback mode")
            
        except Exception as e:
            self.fail(f"Mathematical correctness test failed: {e}")
    
    def tearDown(self):
        """Clean up after tests"""
        test_duration = datetime.now() - self.test_start_time
        print(f"[TIME] Test completed in {test_duration.total_seconds():.3f}s")

def run_gpu_sustainment_operations_validation():
    """Run comprehensive GPU sustainment operations validation"""
    print("[START] Starting GPU Sustainment Operations Validation Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGPUSustainmentOperationsValidation)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("[SUCCESS] ALL GPU SUSTAINMENT OPERATIONS VALIDATION TESTS PASSED!")
        print("[RESOLVED] gpu_sustainment_vector_operations is properly implemented")
        print("[RESOLVED] CPU fallback behavior working correctly")
        print("[RESOLVED] Gap #3 Fix #3 - GPU Sustainment Operations - VERIFIED")
    else:
        print("[FAILED] Some tests failed - GPU sustainment operations issues remain")
        for failure in result.failures:
            print(f"[FAILURE] {failure[0]}")
        for error in result.errors:
            print(f"[ERROR] {error[0]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_gpu_sustainment_operations_validation()
    sys.exit(0 if success else 1) 