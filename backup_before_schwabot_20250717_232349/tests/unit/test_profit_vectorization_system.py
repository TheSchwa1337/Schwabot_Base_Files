#!/usr/bin/env python3
"""
Unified Profit Vectorization System Test Suite 🧪

Comprehensive testing for Schwabot's profit vectorization core:
- High/low entropy trade scenarios
- Drawdown adjustment effects
- Hash registry integration
- Mathematical validation
- Edge cases and error handling
- Integration testing
"""

import numpy as np
import time
import hashlib
from core.unified_profit_vectorization_system import (
    UnifiedProfitVectorizationSystem, 
    ProfitVector, 
    sigmoid,
    profit_vectorization_system
)

def test_sigmoid_function():
    """Test sigmoid function mathematical correctness."""
    print("🧪 Testing Sigmoid Function...")
    
    # Test edge cases
    assert abs(sigmoid(0) - 0.5) < 1e-6, "sigmoid(0) should be 0.5"
    assert sigmoid(100) > 0.99, "sigmoid(100) should be close to 1"
    assert sigmoid(-100) < 0.01, "sigmoid(-100) should be close to 0"
    
    # Test symmetry
    x = 2.5
    assert abs(sigmoid(x) + sigmoid(-x) - 1) < 1e-6, "sigmoid should be symmetric"
    
    # Test monotonicity
    x1, x2 = 1.0, 2.0
    assert sigmoid(x1) < sigmoid(x2), "sigmoid should be monotonically increasing"
    
    print("   ✅ Sigmoid function tests passed")

def test_profit_vector_creation():
    """Test ProfitVector data model creation and validation."""
    print("\n🧪 Testing Profit Vector Creation...")
    
    # Create test vector
    vector = ProfitVector(
        tick=12345,
        profit=0.0271,
        hash="a2b8c3d4e5f6",
        volatility=0.15,
        drawdown=0.05,
        vector_strength=0.9982,
        exit_type="stack_hold",
        risk_profile="low"
    )
    
    # Validate fields
    assert vector.tick == 12345
    assert abs(vector.profit - 0.0271) < 1e-6
    assert vector.hash == "a2b8c3d4e5f6"
    assert abs(vector.volatility - 0.15) < 1e-6
    assert abs(vector.drawdown - 0.05) < 1e-6
    assert abs(vector.vector_strength - 0.9982) < 1e-6
    assert vector.exit_type == "stack_hold"
    assert vector.risk_profile == "low"
    assert vector.timestamp > 0
    assert isinstance(vector.meta, dict)
    
    print("   ✅ Profit vector creation tests passed")

def test_high_entropy_trades():
    """Test high entropy trade scenarios."""
    print("\n🧪 Testing High Entropy Trades...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # High entropy trade (high volatility)
    high_entropy_vector = system.generate_profit_vector(
        entry_tick=1000,
        profit=0.05,
        strategy_hash="high_entropy_hash",
        drawdown=0.02,
        entropy_delta=0.8,  # High volatility
        exit_type="emergency_exit",
        risk_profile="high"
    )
    
    print(f"   High entropy vector strength: {high_entropy_vector.vector_strength:.6f}")
    print(f"   High entropy volatility: {high_entropy_vector.volatility:.6f}")
    
    # Validate high entropy characteristics
    assert high_entropy_vector.volatility == 0.8
    assert high_entropy_vector.risk_profile == "high"
    assert high_entropy_vector.vector_strength < 0.5  # Should be reduced by high volatility
    assert high_entropy_vector.exit_type == "emergency_exit"
    
    print("   ✅ High entropy trade tests passed")

def test_low_entropy_trades():
    """Test low entropy trade scenarios."""
    print("\n🧪 Testing Low Entropy Trades...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Low entropy trade (low volatility)
    low_entropy_vector = system.generate_profit_vector(
        entry_tick=2000,
        profit=0.03,
        strategy_hash="low_entropy_hash",
        drawdown=0.01,
        entropy_delta=0.1,  # Low volatility
        exit_type="stack_hold",
        risk_profile="low"
    )
    
    print(f"   Low entropy vector strength: {low_entropy_vector.vector_strength:.6f}")
    print(f"   Low entropy volatility: {low_entropy_vector.volatility:.6f}")
    
    # Validate low entropy characteristics
    assert low_entropy_vector.volatility == 0.1
    assert low_entropy_vector.risk_profile == "low"
    assert low_entropy_vector.vector_strength > 0.7  # Should be higher due to low volatility
    assert low_entropy_vector.exit_type == "stack_hold"
    
    print("   ✅ Low entropy trade tests passed")

def test_drawdown_adjustment_effects():
    """Test drawdown adjustment effects on vector strength."""
    print("\n🧪 Testing Drawdown Adjustment Effects...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Same profit, different drawdowns
    low_drawdown = system.generate_profit_vector(
        entry_tick=3000,
        profit=0.04,
        strategy_hash="low_dd_hash",
        drawdown=0.01,  # Low drawdown
        entropy_delta=0.2
    )
    
    high_drawdown = system.generate_profit_vector(
        entry_tick=3001,
        profit=0.04,
        strategy_hash="high_dd_hash",
        drawdown=0.03,  # High drawdown
        entropy_delta=0.2
    )
    
    print(f"   Low drawdown vector strength: {low_drawdown.vector_strength:.6f}")
    print(f"   High drawdown vector strength: {high_drawdown.vector_strength:.6f}")
    
    # Higher drawdown should result in lower vector strength
    assert low_drawdown.vector_strength > high_drawdown.vector_strength
    
    print("   ✅ Drawdown adjustment tests passed")

def test_hash_registry_integration():
    """Test hash registry integration and vector retrieval."""
    print("\n🧪 Testing Hash Registry Integration...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Generate multiple vectors with different hashes
    hashes = ["hash_a", "hash_b", "hash_c", "hash_d", "hash_e"]
    for i, hash_val in enumerate(hashes):
        system.generate_profit_vector(
            entry_tick=4000 + i,
            profit=0.02 + i * 0.01,
            strategy_hash=hash_val,
            drawdown=0.01,
            entropy_delta=0.3
        )
    
    # Test retrieval of last vectors
    last_vectors = system.get_last_hash_profit_vectors(n=3)
    assert len(last_vectors) == 3
    assert last_vectors[-1].hash == "hash_e"
    assert last_vectors[-2].hash == "hash_d"
    assert last_vectors[-3].hash == "hash_c"
    
    print(f"   Retrieved {len(last_vectors)} vectors")
    print(f"   Last hash: {last_vectors[-1].hash}")
    
    print("   ✅ Hash registry integration tests passed")

def test_strategy_synchronization():
    """Test synchronization with strategy mapper."""
    print("\n🧪 Testing Strategy Synchronization...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Mock strategy results
    strategy_results = [
        {
            "tick": 5000,
            "profit": 0.025,
            "strategy_hash": "sync_hash_1",
            "drawdown": 0.015,
            "entropy_delta": 0.25,
            "exit_type": "take_profit",
            "risk_profile": "medium"
        },
        {
            "tick": 5001,
            "profit": 0.018,
            "strategy_hash": "sync_hash_2",
            "drawdown": 0.008,
            "entropy_delta": 0.12,
            "exit_type": "stack_hold",
            "risk_profile": "low"
        }
    ]
    
    # Synchronize strategy results
    system.synchronize_with_strategy_mapper(strategy_results)
    
    # Verify synchronization
    vectors = system.get_last_hash_profit_vectors(n=2)
    assert len(vectors) == 2
    assert vectors[0].hash == "sync_hash_1"
    assert vectors[1].hash == "sync_hash_2"
    assert vectors[0].exit_type == "take_profit"
    assert vectors[1].exit_type == "stack_hold"
    
    print(f"   Synchronized {len(vectors)} strategy results")
    
    print("   ✅ Strategy synchronization tests passed")

def test_hash_pipeline_export():
    """Test export to hash pipeline functionality."""
    print("\n🧪 Testing Hash Pipeline Export...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Generate test vectors
    for i in range(5):
        system.generate_profit_vector(
            entry_tick=6000 + i,
            profit=0.02 + i * 0.005,
            strategy_hash=f"export_hash_{i}",
            drawdown=0.01,
            entropy_delta=0.2
        )
    
    # Export to hash pipeline
    export_data = system.export_to_hash_pipeline()
    
    assert len(export_data) == 5
    assert all(isinstance(item, dict) for item in export_data)
    assert all("tick" in item for item in export_data)
    assert all("profit" in item for item in export_data)
    assert all("hash" in item for item in export_data)
    
    print(f"   Exported {len(export_data)} vectors to hash pipeline")
    print(f"   Export format: {list(export_data[0].keys())}")
    
    print("   ✅ Hash pipeline export tests passed")

def test_callback_system():
    """Test callback registration and execution."""
    print("\n🧪 Testing Callback System...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Test callback results
    callback_results = []
    
    def test_callback(data):
        callback_results.append(data)
        return f"processed_{data}"
    
    # Register callback
    system.register_callback("test_callback", test_callback)
    
    # Run callback
    result = system.run_callback("test_callback", "test_data")
    assert result == "processed_test_data"
    assert len(callback_results) == 1
    assert callback_results[0] == "test_data"
    
    # Test non-existent callback
    result = system.run_callback("non_existent", "data")
    assert result is None
    
    print("   ✅ Callback system tests passed")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Test extreme values
    extreme_vector = system.generate_profit_vector(
        entry_tick=7000,
        profit=1.0,  # 100% profit
        strategy_hash="extreme_hash",
        drawdown=0.0,  # No drawdown
        entropy_delta=0.0,  # No volatility
        exit_type="perfect_exit",
        risk_profile="perfect"
    )
    
    assert extreme_vector.vector_strength > 0.9  # Should be very high
    
    # Test negative profit
    negative_vector = system.generate_profit_vector(
        entry_tick=7001,
        profit=-0.1,  # Negative profit
        strategy_hash="negative_hash",
        drawdown=0.05,
        entropy_delta=0.3,
        exit_type="stop_loss",
        risk_profile="high"
    )
    
    assert negative_vector.vector_strength < 0.5  # Should be low
    
    # Test empty strategy results
    system.synchronize_with_strategy_mapper([])
    assert len(system.get_last_hash_profit_vectors()) >= 2  # Should still have previous vectors
    
    print("   ✅ Edge case tests passed")

def test_performance_and_memory():
    """Test performance and memory management."""
    print("\n🧪 Testing Performance and Memory...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Generate many vectors to test memory management
    start_time = time.time()
    for i in range(100):
        system.generate_profit_vector(
            entry_tick=8000 + i,
            profit=0.02,
            strategy_hash=f"perf_hash_{i}",
            drawdown=0.01,
            entropy_delta=0.2
        )
    
    generation_time = time.time() - start_time
    print(f"   Generated 100 vectors in {generation_time:.4f} seconds")
    
    # Test memory limit (max_history = 1000)
    assert len(system.vectors) <= system.max_history
    
    # Test retrieval performance
    start_time = time.time()
    vectors = system.get_last_hash_profit_vectors(n=50)
    retrieval_time = time.time() - start_time
    print(f"   Retrieved 50 vectors in {retrieval_time:.4f} seconds")
    
    assert len(vectors) == 50
    
    print("   ✅ Performance and memory tests passed")

def test_mathematical_validation():
    """Test mathematical correctness of vector strength calculations."""
    print("\n🧪 Testing Mathematical Validation...")
    
    system = UnifiedProfitVectorizationSystem()
    
    # Test vector strength formula: sigmoid(profit - drawdown) * (1 - volatility)
    profit = 0.05
    drawdown = 0.02
    entropy_delta = 0.3
    
    expected_sigmoid = sigmoid(profit - drawdown)
    expected_vector_strength = expected_sigmoid * (1 - entropy_delta)
    
    vector = system.generate_profit_vector(
        entry_tick=9000,
        profit=profit,
        strategy_hash="math_test_hash",
        drawdown=drawdown,
        entropy_delta=entropy_delta
    )
    
    print(f"   Expected sigmoid: {expected_sigmoid:.6f}")
    print(f"   Expected vector strength: {expected_vector_strength:.6f}")
    print(f"   Actual vector strength: {vector.vector_strength:.6f}")
    
    assert abs(vector.vector_strength - expected_vector_strength) < 1e-6
    
    print("   ✅ Mathematical validation tests passed")

def main():
    """Run all tests."""
    print("🚀 Unified Profit Vectorization System Test Suite")
    print("=" * 60)
    
    try:
        test_sigmoid_function()
        test_profit_vector_creation()
        test_high_entropy_trades()
        test_low_entropy_trades()
        test_drawdown_adjustment_effects()
        test_hash_registry_integration()
        test_strategy_synchronization()
        test_hash_pipeline_export()
        test_callback_system()
        test_edge_cases()
        test_performance_and_memory()
        test_mathematical_validation()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Unified Profit Vectorization System is GODMODE READY!")
        print(f"🔗 Next: Integration with clean_unified_math.py and backend_math.py")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 