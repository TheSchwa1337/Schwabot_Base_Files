#!/usr/bin/env python3
"""
Debug script to understand vector sizes in SchwafitCore.
"""

import numpy as np
from core.schwafit_core import SchwafitCore

def debug_sizes():
    """Debug the vector sizes in SchwafitCore."""
    print("🔍 Debugging SchwafitCore Vector Sizes...")
    
    # Create test instance
    schwafit = SchwafitCore(window=8, entropy_threshold=2.5, fit_threshold=0.85)
    
    # Create test price series
    price_series = [100.0 + i * 0.1 + np.sin(i * 0.1) for i in range(20)]
    print(f"Price series length: {len(price_series)}")
    
    # Check what we extract
    window_slice = price_series[-(schwafit.window + 2):]
    print(f"Window slice length (window+2): {len(window_slice)}")
    
    # Check delta2 output
    delta2_output = schwafit.delta2(window_slice)
    print(f"Delta2 output length: {len(delta2_output)}")
    
    # Check normalized output
    normalized = schwafit.normalize(delta2_output)
    print(f"Normalized output length: {len(normalized)}")
    
    print(f"\nExpected pattern library size: {len(normalized)}")
    
    # Test with correct size
    pattern_library = [np.random.randn(len(normalized)) for _ in range(5)]
    print(f"Pattern library vectors size: {len(pattern_library[0])}")
    
    # Test fit_vector
    try:
        fit_results = schwafit.fit_vector(price_series, pattern_library, [0.1, 0.2, 0.3, 0.4, 0.5])
        print("✅ Fit vector works with correct pattern size!")
    except Exception as e:
        print(f"❌ Fit vector failed: {e}")

if __name__ == "__main__":
    debug_sizes() 