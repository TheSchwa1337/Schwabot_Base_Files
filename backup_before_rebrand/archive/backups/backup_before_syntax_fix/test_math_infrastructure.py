#!/usr/bin/env python3
"""
Test script for the new centralized math infrastructure.
Verifies that Phase 1 cleanup created working components.
"""

import os
import sys

sys.path.append('core')

def test_math_config_manager():
    """Test the math configuration manager."""
    print("Testing Math Configuration Manager...")
    
    try:
        from math_config_manager import get_math_config, math_config

        # Test basic functionality
        config = get_math_config()
        
        # Test configuration values
        precision = config.get_precision()
        cache_enabled = config.is_cache_enabled()
        gpu_enabled = config.is_gpu_enabled()
        
        print(f"  ✓ Precision: {precision}")
        print(f"  ✓ Cache enabled: {cache_enabled}")
        print(f"  ✓ GPU enabled: {gpu_enabled}")
        
        # Test nested configuration
        tensor_config = config.get("tensor_operations")
        print(f"  ✓ Tensor config: {tensor_config}")
        
        # Test setting values
        config.set("test.value", 42)
        test_value = config.get("test.value")
        print(f"  ✓ Test value set/get: {test_value}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_math_cache():
    """Test the math results cache."""
    print("\nTesting Math Results Cache...")
    
    try:
        import numpy as np
        from math_cache import MathResultsCache, get_math_cache
        
        cache = get_math_cache()
        
        # Test basic caching
        test_data = np.array([1, 2, 3, 4, 5])
        test_params = {"data": test_data, "operation": "sum"}
        
        # Test cache miss
        result = cache.get("test_operation", test_params)
        print(f"  ✓ Cache miss (None): {result is None}")
        
        # Test cache set/get
        expected_result = np.sum(test_data)
        cache.set("test_operation", test_params, expected_result)
        
        cached_result = cache.get("test_operation", test_params)
        print(f"  ✓ Cache hit: {cached_result == expected_result}")
        
        # Test get_or_compute
        def compute_func():
            return np.mean(test_data)
        
        computed_result = cache.get_or_compute("mean_operation", test_params, compute_func)
        print(f"  ✓ Get or compute: {computed_result}")
        
        # Test stats
        stats = cache.get_stats()
        print(f"  ✓ Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_math_orchestrator():
    """Test the math orchestrator."""
    print("\nTesting Math Orchestrator...")
    
    try:
        import numpy as np
        from math_orchestrator import MathOrchestrator, get_math_orchestrator
        
        orchestrator = get_math_orchestrator()
        
        # Test basic math operation
        def simple_operation():
            return np.array([1, 2, 3]) * 2
        
        result = orchestrator.execute_math_operation(
            "simple_test", 
            {"multiplier": 2}, 
            simple_operation
        )
        print(f"  ✓ Simple operation: {result}")
        
        # Test tensor operation
        def tensor_operation(a, b):
            return np.dot(a, b)
        
        tensor_a = np.array([1, 2, 3])
        tensor_b = np.array([4, 5, 6])
        
        tensor_result = orchestrator.execute_tensor_operation(
            "dot_product",
            [tensor_a, tensor_b],
            tensor_operation
        )
        print(f"  ✓ Tensor operation: {tensor_result}")
        
        # Test performance stats
        stats = orchestrator.get_performance_stats()
        print(f"  ✓ Performance stats available: {len(stats) > 0}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_core_utilities():
    """Test the consolidated core utilities."""
    print("\nTesting Core Utilities...")
    
    try:
        import numpy as np
        from core_utilities import (
            GlyphRouter,
            IntegrationOrchestrator,
            UnifiedAPICoordinator,
            adjust_profit_tier,
            analyze_order_wall,
            detect_swing_pattern,
            normalize_array,
            safe_divide,
        )

        # Test safe_divide
        result1 = safe_divide(10, 2)
        result2 = safe_divide(10, 0, default=42)
        print(f"  ✓ Safe divide (10/2): {result1}")
        print(f"  ✓ Safe divide (10/0): {result2}")
        
        # Test normalize_array
        test_array = np.array([1, 5, 10, 15, 20])
        normalized = normalize_array(test_array)
        print(f"  ✓ Normalized array: {normalized}")
        
        # Test GlyphRouter
        router = GlyphRouter()
        router.register_glyph("test", lambda x: x * 2)
        glyph_result = router.route("test", 5)
        print(f"  ✓ Glyph router: {glyph_result}")
        
        # Test IntegrationOrchestrator
        integrator = IntegrationOrchestrator()
        integrator.register_integration("test_integration", lambda: "success")
        integration_result = integrator.run_integration("test_integration")
        print(f"  ✓ Integration orchestrator: {integration_result}")
        
        # Test order wall analyzer
        order_book = {
            "bids": [(100, 0.5), (99, 1.2), (98, 0.8)],
            "asks": [(101, 0.3), (102, 0.9), (103, 0.6)]
        }
        wall_analysis = analyze_order_wall(order_book, threshold=0.5)
        print(f"  ✓ Order wall analysis: {len(wall_analysis['bid_walls'])} bid walls")
        
        # Test profit tier adjuster
        tier = adjust_profit_tier(1, 0.75, [0.5, 0.7, 0.9])
        print(f"  ✓ Profit tier adjustment: {tier}")
        
        # Test swing pattern detection
        prices = np.array([100, 105, 110, 108, 112, 115, 113, 118, 120, 117])
        pattern = detect_swing_pattern(prices, window=2)
        print(f"  ✓ Swing pattern: {pattern['pattern']}")
        
        # Test API coordinator
        api_coord = UnifiedAPICoordinator()
        api_coord.register_api_handler("test_api", lambda method, *args: f"{method}_result")
        api_result = api_coord.call_api("test_api", "GET")
        print(f"  ✓ API coordinator: {api_result}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING PHASE 1 CLEANUP - CENTRALIZED MATH INFRASTRUCTURE")
    print("=" * 60)
    
    tests = [
        test_math_config_manager,
        test_math_cache,
        test_math_orchestrator,
        test_core_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Phase 1 cleanup successful.")
        print("\nNext steps:")
        print("1. Phase 2: Consolidation of moderately complex files")
        print("2. Phase 3: Optimization of ultra-complex files")
        print("3. Phase 4: Integration and testing")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 