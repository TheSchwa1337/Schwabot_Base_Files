#!/usr/bin/env python3
"""
Test script for SchwafitCore mirror analysis functions.
"""

import numpy as np
from core.schwafit_core import SchwafitCore

def test_mirror_functions():
    """Test all mirror analysis functions."""
    print("🧪 Testing SchwafitCore Mirror Analysis Functions...")
    
    # Create test instance
    schwafit = SchwafitCore(window=8, entropy_threshold=2.5, fit_threshold=0.85)
    
    # Test data
    test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    test_inverse = -test_vector
    test_phases = np.array([1.0, 2.0, 3.0, 4.0])
    test_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    print("\n1. Testing ALIF Certainty...")
    try:
        alif_score = schwafit.alif_certainty(test_vector, test_inverse)
        print(f"   ALIF Score (inverse): {alif_score:.4f}")
        assert 0.0 <= alif_score <= 1.0, "ALIF score should be between 0 and 1"
        
        # Test with different vectors
        vector1 = np.array([1.0, 2.0, 3.0, 4.0])
        vector2 = np.array([1.1, 2.1, 3.1, 4.1])
        alif_score2 = schwafit.alif_certainty(vector1, vector2)
        print(f"   ALIF Score (similar): {alif_score2:.4f}")
        
        # Test with identical vectors
        alif_score3 = schwafit.alif_certainty(vector1, vector1)
        print(f"   ALIF Score (identical): {alif_score3:.4f}")
        
        print("   ✅ ALIF test passed")
    except Exception as e:
        print(f"   ❌ ALIF test failed: {e}")
    
    print("\n2. Testing MIR4X Reflection...")
    try:
        mir4x_score = schwafit.mir4x_reflection(test_phases)
        print(f"   MIR4X Score: {mir4x_score:.4f}")
        assert 0.0 <= mir4x_score <= 1.0, "MIR4X score should be between 0 and 1"
        print("   ✅ MIR4X test passed")
    except Exception as e:
        print(f"   ❌ MIR4X test failed: {e}")
    
    print("\n3. Testing PR1SMA Alignment...")
    try:
        pr1sma_score = schwafit.pr1sma_alignment(test_matrix, test_matrix, test_matrix)
        print(f"   PR1SMA Score: {pr1sma_score:.4f}")
        assert -1.0 <= pr1sma_score <= 1.0, "PR1SMA score should be between -1 and 1"
        print("   ✅ PR1SMA test passed")
    except Exception as e:
        print(f"   ❌ PR1SMA test failed: {e}")
    
    print("\n4. Testing Δ-Mirror Envelope...")
    try:
        delta_score = schwafit.delta_mirror_envelope(0.5, 1.0)
        print(f"   Δ-Mirror Score: {delta_score:.4f}")
        assert 0.0 <= delta_score <= 1.0, "Δ-Mirror score should be between 0 and 1"
        print("   ✅ Δ-Mirror test passed")
    except Exception as e:
        print(f"   ❌ Δ-Mirror test failed: {e}")
    
    print("\n5. Testing Z-Matrix Reversal Logic...")
    try:
        z_matrix = np.flipud(test_matrix)
        z_score = schwafit.z_matrix_reversal_logic(test_matrix, z_matrix)
        print(f"   Z-Matrix Score: {z_score:.4f}")
        assert 0.0 <= z_score <= 1.0, "Z-Matrix score should be between 0 and 1"
        print("   ✅ Z-Matrix test passed")
    except Exception as e:
        print(f"   ❌ Z-Matrix test failed: {e}")
    
    print("\n6. Testing Complete Mirror Analysis...")
    try:
        # Create test price series
        price_series = [100.0 + i * 0.1 + np.sin(i * 0.1) for i in range(20)]
        # Calculate correct pattern size: delta2 reduces size by 2, so (window) elements
        pattern_size = schwafit.window  # delta2 on (window+2) gives (window) elements
        pattern_library = [np.random.randn(pattern_size) for _ in range(5)]
        
        mirror_results = schwafit.mirror_analysis(price_series, pattern_library)
        print(f"   Composite Mirror Score: {mirror_results['composite_mirror_score']:.4f}")
        print(f"   Mirror Decision: {mirror_results['mirror_decision']}")
        print("   ✅ Complete mirror analysis test passed")
    except Exception as e:
        print(f"   ❌ Complete mirror analysis test failed: {e}")
    
    print("\n7. Testing Enhanced Fit Vector...")
    try:
        # Create test data
        price_series = [100.0 + i * 0.1 + np.sin(i * 0.1) for i in range(20)]
        # Calculate correct pattern size: delta2 reduces size by 2, so (window) elements
        pattern_size = schwafit.window  # delta2 on (window+2) gives (window) elements
        pattern_library = [np.random.randn(pattern_size) for _ in range(5)]
        profit_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        fit_results = schwafit.fit_vector(price_series, pattern_library, profit_scores)
        print(f"   Traditional Fit Score: {fit_results['fit_score']:.4f}")
        print(f"   Enhanced Fit Score: {fit_results['enhanced_fit_score']:.4f}")
        print(f"   Combined Decision: {fit_results['combined_decision']}")
        print("   ✅ Enhanced fit vector test passed")
    except Exception as e:
        print(f"   ❌ Enhanced fit vector test failed: {e}")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_mirror_functions() 