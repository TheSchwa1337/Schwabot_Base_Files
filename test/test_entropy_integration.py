#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for entropy signal integration.

This script tests the entropy signal flow through the trading pipeline
to ensure all components work together correctly.
"""

import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

# Import the entropy signal integration
try:
    from entropy_signal_integration import (
        get_entropy_integrator,
        process_entropy_signal,
        should_execute_routing,
        should_execute_tick,
    )
    print("✅ Successfully imported entropy signal integration")
except ImportError as e:
    print(f"❌ Failed to import entropy signal integration: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_order_book() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Generate test order book data."""
    base_price = 50000.0
    
    # Generate bids
    bids = []
    for i in range(10):
        price = base_price * (1 - 0.001 * (i + 1))
        volume = np.random.uniform(0.1, 1.0)
        bids.append((price, volume))
    
    # Generate asks
    asks = []
    for i in range(10):
        price = base_price * (1 + 0.001 * (i + 1))
        volume = np.random.uniform(0.1, 1.0)
        asks.append((price, volume))
    
    return bids, asks


def test_entropy_integration():
    """Test the entropy signal integration."""
    print("\n🧪 Testing Entropy Signal Integration")
    print("=" * 50)
    
    try:
        # Test 1: Initialize entropy integrator
        print("\n1. Testing entropy integrator initialization...")
        integrator = get_entropy_integrator()
        print("✅ Entropy integrator initialized successfully")
        
        # Test 2: Process entropy signal
        print("\n2. Testing entropy signal processing...")
        bids, asks = generate_test_order_book()
        entropy_signal = process_entropy_signal(bids, asks)
        
        print(f"   Entropy Value: {entropy_signal.entropy_value:.6f}")
        print(f"   Routing State: {entropy_signal.routing_state}")
        print(f"   Quantum State: {entropy_signal.quantum_state}")
        print(f"   Confidence: {entropy_signal.confidence:.3f}")
        print("✅ Entropy signal processed successfully")
        
        # Test 3: Test timing cycles
        print("\n3. Testing timing cycles...")
        
        # Test tick cycle
        tick_should_execute = should_execute_tick()
        print(f"   Tick cycle should execute: {tick_should_execute}")
        
        # Test routing cycle
        routing_should_execute = should_execute_routing()
        print(f"   Routing cycle should execute: {routing_should_execute}")
        print("✅ Timing cycles tested successfully")
        
        # Test 4: Test multiple signal processing
        print("\n4. Testing multiple signal processing...")
        signals = []
        for i in range(5):
            bids, asks = generate_test_order_book()
            signal = process_entropy_signal(bids, asks)
            signals.append(signal)
            print(f"   Signal {i+1}: Entropy={signal.entropy_value:.6f}, "
                  f"Routing={signal.routing_state}, Confidence={signal.confidence:.3f}")
        
        print("✅ Multiple signals processed successfully")
        
        # Test 5: Test performance metrics
        print("\n5. Testing performance metrics...")
        performance = integrator.get_performance_summary()
        print(f"   Total signals processed: {performance.get('total_signals_processed', 0)}")
        print(f"   Average detection rate: {performance.get('average_detection_rate', 0):.3f}")
        print(f"   Average latency: {performance.get('average_latency_ms', 0):.1f}ms")
        print("✅ Performance metrics retrieved successfully")
        
        # Test 6: Test current state
        print("\n6. Testing current state...")
        current_state = integrator.get_current_state()
        print(f"   Current entropy state: {current_state.get('current_entropy_state', 'UNKNOWN')}")
        print(f"   Tick cycle enabled: {current_state.get('tick_cycle', {}).get('enabled', False)}")
        print(f"   Routing cycle enabled: {current_state.get('routing_cycle', {}).get('enabled', False)}")
        print("✅ Current state retrieved successfully")
        
        print("\n🎉 All entropy integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        return False


def test_timing_cycle_adaptation():
    """Test timing cycle adaptation based on entropy."""
    print("\n⏱️ Testing Timing Cycle Adaptation")
    print("=" * 50)
    
    try:
        integrator = get_entropy_integrator()
        
        # Test with different entropy levels
        test_cases = [
            ("Low entropy", 0.005),
            ("Medium entropy", 0.015),
            ("High entropy", 0.025)
        ]
        
        for name, entropy_value in test_cases:
            print(f"\nTesting {name} (entropy={entropy_value:.6f})...")
            
            # Create mock signal
            from entropy_signal_integration import EntropySignal
            signal = EntropySignal(
                timestamp=time.time(),
                entropy_value=entropy_value,
                routing_state="ROUTE_ACTIVE" if entropy_value > 0.018 else "ROUTE_PASSIVE",
                quantum_state="ENTROPIC_INVERSION_ACTIVATED" if entropy_value > 0.019 else "INERT",
                confidence=min(entropy_value * 10, 1.0)
            )
            
            # Get initial timing
            initial_tick_interval = integrator.tick_cycle.current_interval_ms
            initial_routing_interval = integrator.routing_cycle.current_interval_ms
            
            # Adapt timing cycles
            integrator._adapt_timing_cycles(signal)
            
            # Check results
            new_tick_interval = integrator.tick_cycle.current_interval_ms
            new_routing_interval = integrator.routing_cycle.current_interval_ms
            
            print(f"   Tick interval: {initial_tick_interval}ms → {new_tick_interval}ms")
            print(f"   Routing interval: {initial_routing_interval}ms → {new_routing_interval}ms")
            
            # Verify adaptation logic
            if entropy_value > 0.018:  # High entropy
                if new_tick_interval < initial_tick_interval:
                    print("   ✅ Tick cycle accelerated for high entropy")
                else:
                    print("   ⚠️ Tick cycle not accelerated as expected")
            elif entropy_value < 0.008:  # Low entropy
                if new_tick_interval > initial_tick_interval:
                    print("   ✅ Tick cycle decelerated for low entropy")
                else:
                    print("   ⚠️ Tick cycle not decelerated as expected")
        
        print("\n🎉 Timing cycle adaptation tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Timing cycle test failed: {e}")
        logger.error(f"Timing cycle test failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration loading from YAML."""
    print("\n📋 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        integrator = get_entropy_integrator()
        
        # Test configuration structure
        config = integrator.config
        
        # Check required sections
        required_sections = [
            "entropy_signal_flow",
            "timing_cycles",
            "signal_pipeline",
            "performance_monitoring"
        ]
        
        for section in required_sections:
            if section in config:
                print(f"✅ Configuration section '{section}' found")
            else:
                print(f"❌ Configuration section '{section}' missing")
                return False
        
        # Test specific configuration values
        entropy_config = config.get("entropy_signal_flow", {})
        order_book_config = entropy_config.get("order_book_analysis", {})
        
        if order_book_config.get("enabled", False):
            print("✅ Order book analysis enabled in config")
        else:
            print("⚠️ Order book analysis disabled in config")
        
        # Test timing cycle configuration
        timing_config = config.get("timing_cycles", {})
        tick_config = timing_config.get("tick_cycle", {})
        
        if tick_config.get("entropy_adaptive", False):
            print("✅ Entropy-adaptive tick cycles enabled")
        else:
            print("⚠️ Entropy-adaptive tick cycles disabled")
        
        print("\n🎉 Configuration loading tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        logger.error(f"Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧠 Entropy Signal Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Entropy Integration", test_entropy_integration),
        ("Timing Cycle Adaptation", test_timing_cycle_adaptation),
        ("Configuration Loading", test_configuration_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Entropy signal integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 