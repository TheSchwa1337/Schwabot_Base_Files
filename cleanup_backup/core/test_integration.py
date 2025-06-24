#!/usr/bin/env python3
"""
Integration Test Script - Schwabot UROS v1.0
==========================================

Simple test script to validate the integration of all components.
"""

import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_voltage_lane_mapper():
    """Test voltage lane mapper."""
    try:
        from voltage_lane_mapper import VoltageLaneMapper
        
        print("Testing Voltage Lane Mapper...")
        mapper = VoltageLaneMapper()
        
        # Test voltage calculations
        for bit_depth in [4, 8, 42]:
            voltage_mapping = mapper.calculate_voltage_for_bit_depth(bit_depth)
            print(f"  Bit depth {bit_depth}: {voltage_mapping.calculated_voltage:.3f}V ({voltage_mapping.voltage_level.value})")
        
        # Test channel assignment
        voltage_mapping = mapper.calculate_voltage_for_bit_depth(8)
        assignment = mapper.assign_channel_for_voltage(voltage_mapping, priority=2.0)
        print(f"  Channel assignment: {assignment.channel_id} (score: {assignment.assignment_score:.3f})")
        
        print("✓ Voltage Lane Mapper test passed")
        return True
        
    except Exception as e:
        print(f"✗ Voltage Lane Mapper test failed: {e}")
        return False

def test_tensor_path_router():
    """Test tensor path router."""
    try:
        from tensor_path_router import TensorPathRouter
        
        print("Testing Tensor Path Router...")
        router = TensorPathRouter()
        
        # Test routing
        test_prefixes = ["hash_00", "hash_15", "hash_31"]
        for prefix in test_prefixes:
            request_id = router.route_hash_prefix(prefix, bit_depth=8, priority=2.0)
            print(f"  Routing request: {request_id} for {prefix}")
        
        # Wait for processing
        time.sleep(1)
        
        # Check results
        for prefix in test_prefixes:
            routes = router.get_routes_by_hash_prefix(prefix)
            for route in routes:
                print(f"  Route: {route.tensor_path} (score: {route.routing_score:.3f})")
        
        print("✓ Tensor Path Router test passed")
        return True
        
    except Exception as e:
        print(f"✗ Tensor Path Router test failed: {e}")
        return False

def test_tensor_harness_matrix():
    """Test tensor harness matrix."""
    try:
        from tensor_harness_matrix import TensorHarnessMatrix, TensorMode
        
        print("Testing Tensor Harness Matrix...")
        harness = TensorHarnessMatrix()
        
        # Test tensor routing
        test_prefixes = ["hash_00", "hash_15", "hash_31"]
        profit_sensor_data = {"profit_rate": 0.75, "volatility": 0.25, "volume": 0.8}
        
        for prefix in test_prefixes:
            request_id = harness.route_tensor_with_drift_compensation(
                prefix, 
                bit_depth=8, 
                mode=TensorMode.DEMO,
                profit_sensor_data=profit_sensor_data
            )
            print(f"  Tensor harness request: {request_id} for {prefix}")
        
        # Wait for processing
        time.sleep(1)
        
        # Check results
        for prefix in test_prefixes:
            routes = harness.get_routes_by_hash_prefix(prefix)
            for route in routes:
                print(f"  Route: {route.tensor_path} (profit_score: {route.profit_score:.3f})")
        
        print("✓ Tensor Harness Matrix test passed")
        return True
        
    except Exception as e:
        print(f"✗ Tensor Harness Matrix test failed: {e}")
        return False

def test_hash_registry_manager():
    """Test hash registry manager."""
    try:
        from hash_registry_manager import HashRegistryManager
        
        print("Testing Hash Registry Manager...")
        manager = HashRegistryManager()
        
        # Test hash resolution
        test_prefixes = ["hash_00", "hash_15", "hash_31"]
        for prefix in test_prefixes:
            entry = manager.get_hash_entry(prefix)
            if entry:
                print(f"  {prefix}: bit_depth={entry.bit_depth}, priority={entry.priority}")
        
        # Test statistics
        stats = manager.get_registry_statistics()
        print(f"  Registry statistics: {len(stats.get('entries', []))} entries")
        
        print("✓ Hash Registry Manager test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hash Registry Manager test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Schwabot UROS v1.0 - Integration Test")
    print("=" * 60)
    
    tests = [
        ("Hash Registry Manager", test_hash_registry_manager),
        ("Voltage Lane Mapper", test_voltage_lane_mapper),
        ("Tensor Path Router", test_tensor_path_router),
        ("Tensor Harness Matrix", test_tensor_harness_matrix),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All integration tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 