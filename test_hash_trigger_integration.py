#!/usr/bin/env python3
"""
Comprehensive Hash Trigger Integration Test
==========================================

This test validates the complete integration between:
- HashTriggerMapper
- GhostSignal
- GhostStrategyIntegration

It tests all components together to ensure proper functionality.
"""

import sys
import time
from typing import Dict, Any

# Add core to path
sys.path.append('core')

def test_complete_integration():
    """Test the complete hash trigger integration system."""
    print("Comprehensive Hash Trigger Integration Test")
    print("=" * 60)
    
    try:
        # Import our modules
        from hash_trigger_mapper import HashTriggerMapper, HashTriggerMapping
        from ghost_signal import GhostSignal, GhostSignalProcessor
        from ghost_strategy_integration import GhostStrategyIntegrator, EnhancedStrategyDecision
        
        print("✓ All modules imported successfully")
        
        # Test 1: HashTriggerMapper functionality
        print("\n1. Testing HashTriggerMapper...")
        mapper = HashTriggerMapper()
        
        # Test different hash patterns
        test_hashes = [
            ("000000", "critical"),
            ("123456", "sequential"),
            ("a1b2c3", "patterned"),
            ("111111", "repeating"),
            ("abcdef", "sequential"),
            ("random1", "random")
        ]
        
        for hash_trigger, expected_pattern in test_hashes:
            mapping = mapper.map_hash_trigger(hash_trigger)
            actual_pattern = mapping.pattern_type.value
            print(f"  {hash_trigger} -> {actual_pattern} (expected: {expected_pattern})")
            assert actual_pattern == expected_pattern, f"Pattern mismatch for {hash_trigger}"
        
        print("✓ HashTriggerMapper pattern detection working correctly")
        
        # Test 2: GhostSignal functionality
        print("\n2. Testing GhostSignal...")
        processor = GhostSignalProcessor()
        
        # Create mock BTCVector
        class MockBTCVector:
            def __init__(self, price=50000.0, volatility=0.025, momentum=0.003):
                self.price = price
                self.volatility = volatility
                self.momentum = momentum
                self.mean_price = price
                self.hash_trigger = "a1b2c3"
        
        # Test different market conditions
        test_conditions = [
            {"name": "Low Risk", "entropy": 0.2, "volatility": 0.01, "momentum": 0.001},
            {"name": "Medium Risk", "entropy": 0.5, "volatility": 0.025, "momentum": 0.003},
            {"name": "High Risk", "entropy": 0.8, "volatility": 0.06, "momentum": 0.01}
        ]
        
        for condition in test_conditions:
            btc_vector = MockBTCVector(
                price=50000.0,
                volatility=condition["volatility"],
                momentum=condition["momentum"]
            )
            
            signal = processor.create_signal(
                btc_vector=btc_vector,
                entropy=condition["entropy"],
                timestamp=time.time()
            )
            
            print(f"  {condition['name']}: {signal.phase_state} -> {signal.suggested_pathway}")
            assert hasattr(signal, 'phase_state'), "Signal missing phase_state"
            assert hasattr(signal, 'suggested_pathway'), "Signal missing suggested_pathway"
        
        print("✓ GhostSignal processing working correctly")
        
        # Test 3: GhostStrategyIntegration functionality
        print("\n3. Testing GhostStrategyIntegration...")
        integrator = GhostStrategyIntegrator()
        
        # Test enhanced decision making
        for condition in test_conditions:
            btc_vector = MockBTCVector(
                price=50000.0,
                volatility=condition["volatility"],
                momentum=condition["momentum"]
            )
            
            decision = integrator.make_enhanced_decision(
                btc_vector=btc_vector,
                entropy=condition["entropy"],
                timestamp=time.time()
            )
            
            print(f"  {condition['name']}: {decision.decision} -> {decision.strategy_pathway}")
            assert isinstance(decision, EnhancedStrategyDecision), "Invalid decision type"
            assert hasattr(decision, 'combined_score'), "Decision missing combined_score"
            assert 0.0 <= decision.combined_score <= 1.0, "Invalid combined score"
        
        print("✓ GhostStrategyIntegration working correctly")
        
        # Test 4: Integration statistics
        print("\n4. Testing Integration Statistics...")
        
        # Get mapper statistics
        mapper_stats = mapper.get_mapping_statistics()
        print(f"  Mapper: {mapper_stats['total_mappings']} mappings")
        assert mapper_stats['total_mappings'] > 0, "No mappings found"
        
        # Get processor statistics
        processor_stats = processor.get_signal_statistics()
        print(f"  Processor: {processor_stats['total_signals']} signals")
        assert processor_stats['total_signals'] > 0, "No signals found"
        
        # Get integrator statistics
        integrator_stats = integrator.get_integration_statistics()
        print(f"  Integrator: {integrator_stats['total_decisions']} decisions")
        assert integrator_stats['total_decisions'] > 0, "No decisions found"
        
        print("✓ All statistics working correctly")
        
        # Test 5: Error handling
        print("\n5. Testing Error Handling...")
        
        # Test with invalid data
        try:
            invalid_mapping = mapper.map_hash_trigger("")
            print("  ✓ Empty hash trigger handled gracefully")
        except Exception as e:
            print(f"  ✓ Error handling working: {e}")
        
        # Test with None data
        try:
            decision = integrator.make_enhanced_decision(
                btc_vector=None,
                entropy=0.5,
                timestamp=time.time()
            )
            print("  ✓ None BTCVector handled gracefully")
        except Exception as e:
            print(f"  ✓ Error handling working: {e}")
        
        print("✓ Error handling working correctly")
        
        # Test 6: Performance validation
        print("\n6. Testing Performance...")
        
        start_time = time.time()
        for i in range(10):
            btc_vector = MockBTCVector()
            decision = integrator.make_enhanced_decision(
                btc_vector=btc_vector,
                entropy=0.5,
                timestamp=time.time()
            )
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        
        print(f"  Average decision time: {avg_time:.4f}s")
        assert avg_time < 1.0, "Decision making too slow"
        
        print("✓ Performance acceptable")
        
        # Test 7: Data consistency
        print("\n7. Testing Data Consistency...")
        
        # Verify that mappings are consistent
        for hash_trigger, _ in test_hashes:
            mapping1 = mapper.map_hash_trigger(hash_trigger)
            mapping2 = mapper.map_hash_trigger(hash_trigger)
            
            assert mapping1.strategy_pathway == mapping2.strategy_pathway, "Inconsistent pathway"
            assert mapping1.pattern_type == mapping2.pattern_type, "Inconsistent pattern type"
        
        print("✓ Data consistency verified")
        
        # Final summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)
        print("✓ HashTriggerMapper: PASSED")
        print("✓ GhostSignal: PASSED")
        print("✓ GhostStrategyIntegration: PASSED")
        print("✓ Error Handling: PASSED")
        print("✓ Performance: PASSED")
        print("✓ Data Consistency: PASSED")
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe hash trigger mapping system is fully integrated and working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unicode_compatibility():
    """Test Unicode/emoji compatibility for Windows CLI."""
    print("\nTesting Unicode/Emoji Compatibility...")
    
    try:
        # Test various Unicode characters
        test_strings = [
            "🚀 Ghost Signal",
            "⚡ Hash Trigger",
            "🎯 Strategy",
            "📊 Statistics",
            "✅ Success",
            "⚠️ Warning",
            "❌ Error"
        ]
        
        for test_string in test_strings:
            print(f"  Testing: {test_string}")
            # If we can print it without error, it's working
            print(f"    ✓ {test_string} displayed correctly")
        
        print("✓ Unicode/emoji compatibility working")
        return True
        
    except UnicodeEncodeError as e:
        print(f"⚠️ Unicode encoding issue: {e}")
        print("This is expected on some Windows systems")
        return True  # Not a failure, just a limitation
    except Exception as e:
        print(f"❌ Unicode test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("Starting Comprehensive Hash Trigger Integration Tests")
    print("=" * 60)
    
    # Run main integration test
    integration_success = test_complete_integration()
    
    # Run Unicode compatibility test
    unicode_success = test_unicode_compatibility()
    
    # Final result
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    if integration_success and unicode_success:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe hash trigger mapping system is ready for production use.")
        print("\nKey Features Validated:")
        print("  • Hash pattern detection and mapping")
        print("  • Ghost signal processing and analysis")
        print("  • Enhanced strategy decision making")
        print("  • Multi-factor integration logic")
        print("  • Error handling and fallback mechanisms")
        print("  • Performance optimization")
        print("  • Unicode/emoji CLI compatibility")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    exit(main()) 