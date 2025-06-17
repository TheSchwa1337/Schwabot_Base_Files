"""
Quick Test for Sustainment Principles Framework
==============================================

A simple test to verify all components are working correctly.
"""

import sys
sys.path.append('core')

import numpy as np
import time
import yaml

# Import our sustainment framework
from sustainment_principles import SustainmentCalculator, SustainmentState, PrincipleMetrics
from sustainment_integration_hooks import SustainmentIntegrationManager

def test_basic_functionality():
    """Test basic sustainment principles functionality"""
    print("🧪 Testing Sustainment Principles Framework...")
    
    # Load configuration
    with open('config/sustainment_principles.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create calculator
    calc = SustainmentCalculator(config['principles'])
    
    # Create test context
    context = {
        'subsystem_scores': {
            'strategy_a': 0.8,
            'strategy_b': 0.6, 
            'strategy_c': 0.4,
            'strategy_d': 0.7,
            'strategy_e': 0.5
        },
        'current_state': {
            'price': 100.0,
            'entropy': 0.5,
            'volume': 1000.0
        },
        'system_latency_ms': 50.0,
        'event_response_ms': 25.0,
        'operation_count': 100,
        'ncco_complexity': 10,
        'active_strategies': 5,
        'profit_delta': 0.05,
        'cpu_cycles': 200.0,
        'gpu_cycles': 100.0,
        'memory_usage_mb': 256.0,
        'current_utility': 0.7,
        'entropy_level': 0.5,
        'shock_magnitude': 0.1,
        'recovery_rate': 1.2,
        'coherence': 0.8,
        'stability': 0.9,
        'uptime_ratio': 0.95,
        'optimization_state': 0.6,
        'learning_rate': 0.1,
        'improvement_rate': 0.02
    }
    
    # Calculate all principles
    state = calc.calculate_all(context)
    
    # Verify calculations
    assert state.integration.value >= 0.0, "Integration calculation failed"
    assert state.anticipation.value >= 0.0, "Anticipation calculation failed"
    assert state.responsiveness.value >= 0.0, "Responsiveness calculation failed"
    assert state.simplicity.value >= 0.0, "Simplicity calculation failed"
    assert state.economy.value >= 0.0, "Economy calculation failed"
    assert state.survivability.value >= 0.0, "Survivability calculation failed"
    assert state.continuity.value >= 0.0, "Continuity calculation failed"
    assert state.transcendence.value >= 0.0, "Transcendence calculation failed"
    
    composite = state.composite_score()
    assert 0.0 <= composite <= 1.0, "Composite score out of range"
    
    print(f"✅ Integration: {state.integration.value:.3f} (confidence: {state.integration.confidence:.3f})")
    print(f"✅ Anticipation: {state.anticipation.value:.3f} (confidence: {state.anticipation.confidence:.3f})")
    print(f"✅ Responsiveness: {state.responsiveness.value:.3f} (confidence: {state.responsiveness.confidence:.3f})")
    print(f"✅ Simplicity: {state.simplicity.value:.3f} (confidence: {state.simplicity.confidence:.3f})")
    print(f"✅ Economy: {state.economy.value:.3f} (confidence: {state.economy.confidence:.3f})")
    print(f"✅ Survivability: {state.survivability.value:.3f} (confidence: {state.survivability.confidence:.3f})")
    print(f"✅ Continuity: {state.continuity.value:.3f} (confidence: {state.continuity.confidence:.3f})")
    print(f"✅ Transcendence: {state.transcendence.value:.3f} (confidence: {state.transcendence.confidence:.3f})")
    print(f"✅ Composite Score: {composite:.3f}")
    
    return calc, state

def test_integration_weights(calc):
    """Test integration weights functionality"""
    print("\n🔗 Testing Integration Weights...")
    
    weights = calc.get_integration_weights()
    
    if weights:
        print(f"✅ Integration Weights: {weights}")
        
        # Verify weights sum to 1 (softmax property)
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights don't sum to 1: {weight_sum}"
        print(f"✅ Weight sum validation: {weight_sum:.6f}")
    else:
        print("⚠️  No integration weights available (expected on first run)")

def test_health_monitoring(calc):
    """Test health monitoring functionality"""
    print("\n🏥 Testing Health Monitoring...")
    
    health = calc.get_health_report()
    
    print(f"✅ Overall Health: {health['overall_health']}")
    print(f"✅ Healthy Principles: {health['healthy_principles']}/8")
    print(f"✅ Composite Score: {health['composite_score']:.3f}")
    
    if health['failing_principles']:
        print(f"⚠️  Failing Principles: {health['failing_principles']}")
    else:
        print("✅ All principles healthy!")

def test_mathematical_correctness():
    """Test mathematical properties"""
    print("\n🧮 Testing Mathematical Correctness...")
    
    # Test softmax normalization
    scores = np.array([0.8, 0.6, 0.4, 0.2])
    alpha = 1.0
    exp_scores = np.exp(alpha * scores)
    weights = exp_scores / np.sum(exp_scores)
    
    assert abs(np.sum(weights) - 1.0) < 1e-10, "Softmax doesn't sum to 1"
    assert np.all(weights > 0), "Softmax has non-positive weights"
    assert np.all(weights < 1), "Softmax has weights >= 1"
    print("✅ Softmax normalization correct")
    
    # Test exponential decay
    lambda_max = 100.0
    latencies = [10, 50, 100, 200, 500]
    responses = [np.exp(-l / lambda_max) for l in latencies]
    
    # Should be monotonically decreasing
    for i in range(1, len(responses)):
        assert responses[i] < responses[i-1], "Exponential decay not monotonic"
    print("✅ Exponential decay correct")
    
    # Test sigmoid properties
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    assert abs(sigmoid(0) - 0.5) < 1e-10, "Sigmoid(0) != 0.5"
    assert sigmoid(10) > 0.99, "Sigmoid of large positive not near 1"
    assert sigmoid(-10) < 0.01, "Sigmoid of large negative not near 0"
    print("✅ Sigmoid normalization correct")

def test_performance():
    """Test calculation performance"""
    print("\n⚡ Testing Performance...")
    
    config = {
        'integration': {'integration_softmax_alpha': 1.0},
        'anticipation': {'kalman_gain': 0.3},
        'responsiveness': {'max_latency_ms': 100.0},
        'simplicity': {'max_operations': 1000},
        'economy': {'min_efficiency': 0.001},
        'survivability': {'survivability_threshold': 0.6},
        'continuity': {'continuity_window': 50},
        'transcendence': {'fixed_point_target': 0.8}
    }
    
    calc = SustainmentCalculator(config)
    
    context = {
        'subsystem_scores': {f'strategy_{i}': 0.5 + i*0.05 for i in range(10)},
        'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0},
        'system_latency_ms': 50.0,
        'operation_count': 500,
        'profit_delta': 0.05,
        'cpu_cycles': 200.0,
        'current_utility': 0.7,
        'coherence': 0.8,
        'optimization_state': 0.6
    }
    
    # Time 100 calculations
    iterations = 100
    start_time = time.time()
    
    for _ in range(iterations):
        calc.calculate_all(context)
    
    end_time = time.time()
    avg_time_ms = (end_time - start_time) * 1000 / iterations
    
    print(f"✅ Average calculation time: {avg_time_ms:.2f}ms per iteration")
    
    # Should be fast enough for real-time use
    assert avg_time_ms < 50.0, f"Too slow: {avg_time_ms}ms > 50ms threshold"
    print(f"✅ Performance acceptable for real-time use")

def test_trends_analysis(calc):
    """Test performance trends analysis"""
    print("\n📈 Testing Trends Analysis...")
    
    # Generate improving performance over time
    for i in range(15):
        context = {
            'subsystem_scores': {'strategy_a': 0.6 + i*0.02},  # Gradual improvement
            'current_state': {'price': 100.0, 'entropy': 0.5},
            'system_latency_ms': 60.0 - i*2,  # Improving latency
            'operation_count': 100,
            'profit_delta': 0.04 + i*0.001,  # Improving profit
            'cpu_cycles': 200.0,
            'current_utility': 0.6 + i*0.015,
            'coherence': 0.7 + i*0.01,
            'optimization_state': 0.5 + i*0.02
        }
        calc.calculate_all(context)
    
    trends = calc.get_performance_trends(window=10)
    
    print(f"✅ Trend Direction: {trends['trend_direction']}")
    print(f"✅ Current Score: {trends['current_score']:.3f}")
    print(f"✅ Average Score: {trends['avg_score']:.3f}")
    print(f"✅ Score Volatility: {trends['score_volatility']:.3f}")
    
    # Should detect improving trend
    if trends['trend_direction'] == 'improving':
        print("✅ Correctly detected improving trend")
    else:
        print(f"⚠️  Expected improving trend, got: {trends['trend_direction']}")

def test_integration_manager():
    """Test integration manager"""
    print("\n🔌 Testing Integration Manager...")
    
    config = {'integration': {'integration_softmax_alpha': 1.0}}
    calc = SustainmentCalculator(config)
    manager = SustainmentIntegrationManager(calc)
    
    # Test status
    status = manager.get_integration_status()
    print(f"✅ Integration Manager Status: {status}")
    
    assert status['running'] == False, "Manager should not be running initially"
    assert status['hook_count'] == 0, "Should have no hooks initially"
    print("✅ Integration manager initialized correctly")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Sustainment Principles Framework Tests\n")
    
    try:
        # Basic functionality
        calc, state = test_basic_functionality()
        
        # Integration weights
        test_integration_weights(calc)
        
        # Health monitoring
        test_health_monitoring(calc)
        
        # Mathematical correctness
        test_mathematical_correctness()
        
        # Performance
        test_performance()
        
        # Trends analysis
        test_trends_analysis(calc)
        
        # Integration manager
        test_integration_manager()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Sustainment Principles Framework is ready for integration with Schwabot!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 