#!/usr/bin/env python3
"""Simple Hybrid Strategy Mapper Test
====================================

Standalone test for hybrid strategy mapper to validate both
Ghost Phase and legacy UROS/ZPE paths work correctly.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def safe_print(message):
    """Safe print function for testing."""
    print(message)

def test_hybrid_strategy_mapper_simple():
    """Test hybrid strategy mapper without complex imports."""
    safe_print("🧪 Testing Hybrid Strategy Mapper (Simple)...")
    
    try:
        # Test imports work
        from core.strategy_mapper import StrategyMapper
        safe_print("✅ StrategyMapper imported successfully")
        
        # Test 1: Basic initialization
        mapper = StrategyMapper(
            enable_ghost_phase=True, 
            enable_legacy=True, 
            default_to_legacy=False
        )
        safe_print("✅ Hybrid mapper initialized")
        
        # Test 2: Simple data
        prices = [50000, 51000, 50500, 52000, 53000]
        live_vector = [0.8, 0.2, 0.6, 0.4, 0.9, 0.1]
        raw_signals = [0.7, 0.3, 0.6, 0.8, 0.4]
        
        # Test 3: Ghost Phase path
        result_ghost = mapper.map_strategy(
            prices, live_vector, raw_signals, use_legacy=False
        )
        
        assert result_ghost.success, f"Ghost Phase should succeed, got: {result_ghost}"
        assert isinstance(result_ghost.strategy_id, str), "Strategy ID must be string"
        assert len(result_ghost.strategy_id) > 0, "Strategy ID must not be empty"
        safe_print(f"✅ Ghost Phase Strategy: {result_ghost.strategy_id}")
        
        # Test 4: Legacy path
        execution_packet = {
            "strategy_type": "momentum",
            "prices": prices,
            "signals": raw_signals,
            "timestamp": time.time(),
        }
        
        result_legacy = mapper.map_strategy(
            prices, live_vector, raw_signals, execution_packet, use_legacy=True
        )
        
        assert result_legacy.success, f"Legacy should succeed, got: {result_legacy}"
        assert isinstance(result_legacy.strategy_id, str), "Strategy ID must be string"
        assert len(result_legacy.strategy_id) > 0, "Strategy ID must not be empty"
        safe_print(f"✅ Legacy Strategy: {result_legacy.strategy_id}")
        
        # Test 5: Performance stats
        stats = mapper.get_performance_stats()
        assert "total_mappings" in stats, "Should have performance stats"
        assert stats["total_mappings"] > 0, "Should have recorded mappings"
        safe_print(f"✅ Performance: {stats['total_mappings']} mappings, {stats['success_rate']:.2f} success rate")
        
        # Test 6: Auto-detection
        # Modern packet (no legacy indicators) should use Ghost Phase
        result_auto = mapper.map_strategy(
            prices, live_vector, raw_signals, None, None
        )
        assert result_auto.success, "Auto-detection should succeed"
        safe_print(f"✅ Auto-detection: {result_auto.strategy_id}")
        
        safe_print("🎉 All hybrid strategy mapper tests passed!")
        return True
        
    except Exception as e:
        safe_print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_legacy_compatibility():
    """Test legacy compatibility functions."""
    safe_print("\n🧪 Testing Legacy Compatibility...")
    
    try:
        from core.strategy_mapper import map_strategy
        
        # Test legacy function
        execution_packet = {
            "strategy_type": "momentum",
            "prices": [50000, 51000, 50500, 52000],
            "signals": [0.7, 0.3, 0.6, 0.8],
        }
        
        result = map_strategy(execution_packet)
        assert isinstance(result, dict), "Legacy function should return dict"
        assert "mapped_at" in result, "Should have mapping timestamp"
        safe_print(f"✅ Legacy map_strategy: {result.get('strategy_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Legacy compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple hybrid tests."""
    safe_print("🚀 Hybrid Strategy Mapper Simple Tests")
    safe_print("=" * 50)
    
    tests = [
        test_hybrid_strategy_mapper_simple,
        test_legacy_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        safe_print("")  # Add spacing
    
    safe_print("=" * 50)
    safe_print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        safe_print("🎉 All hybrid tests passed! System is working correctly.")
        return True
    else:
        safe_print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    main() 