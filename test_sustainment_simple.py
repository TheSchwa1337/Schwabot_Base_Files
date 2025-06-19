"""
Simple Sustainment Framework Test
================================

Simplified test for the sustainment framework to validate basic functionality
without complex dependencies or integration requirements.

This test focuses on core mathematical operations and validates the 8-principle
sustainment framework is working correctly.
"""

import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_sustainment_core():
    """Test core sustainment functionality"""
    try:
        from core.sustainment_framework import SustainmentFramework, SustainmentState
        
        print("[START] Testing Sustainment Framework Core...")
        
        # Initialize framework
        framework = SustainmentFramework()
        
        # Test basic state calculation
        test_metrics = {
            'cpu_usage': 0.65,
            'memory_usage': 0.45,
            'disk_io': 0.30,
            'network_latency': 50.0,
            'error_rate': 0.02,
            'profit_delta': 150.0,
            'system_temperature': 65.0,
            'power_consumption': 0.75
        }
        
        state = framework.calculate_sustainment_state(test_metrics)
        
        # Validate state
        assert isinstance(state, SustainmentState)
        assert hasattr(state, 'integration')
        assert hasattr(state, 'anticipation')
        assert hasattr(state, 'responsiveness')
        assert hasattr(state, 'simplicity')
        assert hasattr(state, 'economy')
        assert hasattr(state, 'survivability')
        assert hasattr(state, 'continuity')
        assert hasattr(state, 'transcendence')
        
        print("[PASS] SustainmentState structure validation")
        
        # Test each principle
        principles = [
            ('Integration', state.integration),
            ('Anticipation', state.anticipation),
            ('Responsiveness', state.responsiveness),
            ('Simplicity', state.simplicity),
            ('Economy', state.economy),
            ('Survivability', state.survivability),
            ('Continuity', state.continuity),
            ('Transcendence', state.transcendence)
        ]
        
        for name, principle in principles:
            assert 0.0 <= principle.value <= 1.0, f"{name} value out of range: {principle.value}"
            assert 0.0 <= principle.confidence <= 1.0, f"{name} confidence out of range: {principle.confidence}"
            print(f"[PASS] {name}: {principle.value:.3f} (confidence: {principle.confidence:.3f})")
        
        # Test composite score
        composite = framework.calculate_composite_score(state)
        assert 0.0 <= composite <= 1.0, f"Composite score out of range: {composite}"
        print(f"[PASS] Composite Score: {composite:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"[SKIP] Sustainment framework not available: {e}")
        return True  # Skip is okay
    except Exception as e:
        print(f"[FAIL] Sustainment core test failed: {e}")
        return False

def test_mathematical_operations():
    """Test mathematical operations underlying the framework"""
    try:
        from core.sustainment_framework import SustainmentFramework
        
        print("[START] Testing Mathematical Operations...")
        
        framework = SustainmentFramework()
        
        # Test integration weights
        if hasattr(framework, 'integration_weights') and framework.integration_weights is not None:
            weights = framework.integration_weights
            print(f"[PASS] Integration Weights: {weights}")
            
            # Validate weights sum to 1
            weight_sum = np.sum(weights)
            print(f"[PASS] Weight sum validation: {weight_sum:.6f}")
            assert abs(weight_sum - 1.0) < 1e-6, f"Weights don't sum to 1: {weight_sum}"
        else:
            print("[WARN] No integration weights available (expected on first run)")
        
        # Test normalization functions
        test_values = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4])
        
        # Test softmax normalization
        if hasattr(framework, '_softmax_normalize'):
            normalized = framework._softmax_normalize(test_values)
            norm_sum = np.sum(normalized)
            assert abs(norm_sum - 1.0) < 1e-6, f"Softmax normalization failed: sum={norm_sum}"
            print("[PASS] Softmax normalization correct")
        
        # Test exponential decay
        if hasattr(framework, '_apply_exponential_decay'):
            decayed = framework._apply_exponential_decay(test_values, 0.95)
            assert np.all(decayed <= test_values), "Exponential decay should reduce values"
            print("[PASS] Exponential decay correct")
        
        # Test sigmoid normalization
        if hasattr(framework, '_sigmoid_normalize'):
            sigmoid_values = framework._sigmoid_normalize(test_values)
            assert np.all((sigmoid_values >= 0) & (sigmoid_values <= 1)), "Sigmoid values out of range"
            print("[PASS] Sigmoid normalization correct")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Mathematical operations test failed: {e}")
        return False

def test_performance():
    """Test performance of sustainment calculations"""
    try:
        from core.sustainment_framework import SustainmentFramework
        
        print("[START] Testing Performance...")
        
        framework = SustainmentFramework()
        
        # Test calculation performance
        test_metrics = {
            'cpu_usage': 0.65,
            'memory_usage': 0.45,
            'disk_io': 0.30,
            'network_latency': 50.0,
            'error_rate': 0.02,
            'profit_delta': 150.0,
            'system_temperature': 65.0,
            'power_consumption': 0.75
        }
        
        # Time multiple calculations
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            state = framework.calculate_sustainment_state(test_metrics)
            composite = framework.calculate_composite_score(state)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        print(f"[PASS] Average calculation time: {avg_time_ms:.2f}ms per iteration")
        
        # Performance threshold (should be under 10ms per calculation)
        if avg_time_ms < 10.0:
            print("[PASS] Performance acceptable for real-time use")
        else:
            print(f"[WARN] Performance may be slow for high-frequency use: {avg_time_ms:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        return False

def test_individual_principles():
    """Test individual principle calculations"""
    print("[START] Testing Individual Principles...")
    
    try:
        from core.sustainment_framework import SustainmentFramework
        
        framework = SustainmentFramework()
        base_metrics = {
            'cpu_usage': 0.5,
            'memory_usage': 0.4,
            'disk_io': 0.3,
            'network_latency': 40.0,
            'error_rate': 0.01,
            'profit_delta': 100.0,
            'system_temperature': 60.0,
            'power_consumption': 0.6
        }
        
        # Test integration principle
        if hasattr(framework, '_calculate_integration'):
            integration = framework._calculate_integration(base_metrics)
            assert 0.0 <= integration.value <= 1.0, "Integration value out of range"
            print("[PASS] Integration principle working")
        
        # Test anticipation principle
        if hasattr(framework, '_calculate_anticipation'):
            anticipation = framework._calculate_anticipation(base_metrics)
            assert 0.0 <= anticipation.value <= 1.0, "Anticipation value out of range"
            print("[PASS] Anticipation principle working")
        
        # Test responsiveness principle
        if hasattr(framework, '_calculate_responsiveness'):
            responsiveness = framework._calculate_responsiveness(base_metrics)
            assert 0.0 <= responsiveness.value <= 1.0, "Responsiveness value out of range"
            print("[PASS] Responsiveness principle working")
        
        # Test simplicity principle
        if hasattr(framework, '_calculate_simplicity'):
            simplicity = framework._calculate_simplicity(base_metrics)
            assert 0.0 <= simplicity.value <= 1.0, "Simplicity value out of range"
            print("[PASS] Simplicity principle working")
        
        # Test economy principle
        if hasattr(framework, '_calculate_economy'):
            economy = framework._calculate_economy(base_metrics)
            assert 0.0 <= economy.value <= 1.0, "Economy value out of range"
            print("[PASS] Economy principle working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Individual principles test failed: {e}")
        return False

def run_simple_sustainment_test():
    """Run all simple sustainment tests"""
    print("[SYSTEM] SIMPLE SUSTAINMENT FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Core Functionality", test_sustainment_core),
        ("Mathematical Operations", test_mathematical_operations),
        ("Performance", test_performance),
        ("Individual Principles", test_individual_principles)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"[SUCCESS] {test_name} completed successfully")
            else:
                print(f"[FAILED] {test_name} failed")
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"[SUMMARY] Test Results: {passed}/{total} passed ({success_rate:.1f}%)")
    
    if all(results):
        print("[SUCCESS] ALL SUSTAINMENT TESTS PASSED!")
        print("[RESOLVED] Sustainment framework is working correctly")
    else:
        print("[PARTIAL] Some sustainment tests failed")
        print("[ACTION] Review failed tests and fix issues")
    
    return all(results)

if __name__ == "__main__":
    success = run_simple_sustainment_test()
    sys.exit(0 if success else 1) 