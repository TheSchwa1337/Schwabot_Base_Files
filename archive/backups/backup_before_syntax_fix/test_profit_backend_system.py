#!/usr/bin/env python3
"""
Test script for the profit-driven backend dispatcher system.
Demonstrates dynamic CPU/GPU selection based on profit metrics.
"""

import os
import sys
import time

import numpy as np

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from profit_backend_dispatcher import (
    dispatch_op,
    elementwise_multiply,
    get_profit_stats,
    matrix_multiply,
    registry,
    reset_stats,
)
from profit_decorators import auto_backend_select, profit_driven_op, profit_tracked


def test_basic_operations():
    """Test basic mathematical operations with profit tracking."""
    print("=== Testing Basic Operations ===")
    
    # Create test data
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    c = np.random.rand(1000)
    d = np.random.rand(1000)
    
    print("Testing matrix multiplication...")
    result1 = matrix_multiply(a, b, profit=1.5)
    print(f"Result shape: {result1.shape}")
    
    print("Testing element-wise multiplication...")
    result2 = elementwise_multiply(c, d, profit=0.8)
    print(f"Result shape: {result2.shape}")
    
    print("Testing FFT...")
    result3 = dispatch_op('fft', c, profit=2.1)
    print(f"Result shape: {result3.shape}")
    
    # Show current stats
    stats = get_profit_stats()
    print(f"\nTotal operations: {stats['total_operations']}")
    for op_name, op_stats in stats['operations'].items():
        print(f"\n{op_name}:")
        print(f"  CPU: {op_stats['cpu']}")
        print(f"  GPU: {op_stats['gpu']}")
        print(f"  Recommended: {op_stats['recommended']}")

def test_learning_behavior():
    """Test how the system learns and adapts over time."""
    print("\n=== Testing Learning Behavior ===")
    
    # Reset stats to start fresh
    reset_stats()
    
    # Create data of different sizes
    small_data = np.random.rand(50, 50)
    large_data = np.random.rand(500, 500)
    
    print("Running operations with different profit levels...")
    
    # Simulate different profit scenarios
    scenarios = [
        (small_data, small_data, 0.1),   # Low profit, small data
        (large_data, large_data, 5.0),   # High profit, large data
        (small_data, small_data, 0.2),   # Low profit, small data
        (large_data, large_data, 8.0),   # Very high profit, large data
    ]
    
    for i, (a, b, profit) in enumerate(scenarios):
        print(f"\nScenario {i+1}: Data size {a.shape}, Profit {profit}")
        
        # Run multiple times to build up statistics
        for j in range(3):
            result = matrix_multiply(a, b, profit=profit)
            time.sleep(0.01)  # Small delay to simulate real processing
        
        # Show current recommendation
        stats = get_profit_stats()
        op_stats = stats['operations']['matrix_multiply']
        print(f"  Current recommendation: {op_stats['recommended']}")
        print(f"  CPU profit rate: {op_stats['cpu']['profit'] / max(op_stats['cpu']['total_time'], 0.001):.3f}")
        print(f"  GPU profit rate: {op_stats['gpu']['profit'] / max(op_stats['gpu']['total_time'], 0.001):.3f}")

def test_decorator_integration():
    """Test integration with existing functions using decorators."""
    print("\n=== Testing Decorator Integration ===")
    
    # Example function that could be in your trading system
    @profit_driven_op('custom_calculation', lambda *args, **kwargs: kwargs.get('profit', 1.0))
    def custom_trading_calculation(data, weights, profit=1.0):
        """Example trading calculation function."""
        # This would normally use the backend directly
        # Now it's automatically profit-driven
        return np.dot(data, weights)
    
    # Test the decorated function
    data = np.random.rand(200, 100)
    weights = np.random.rand(100, 50)
    
    print("Testing decorated trading function...")
    result = custom_trading_calculation(data, weights, profit=3.5)
    print(f"Result shape: {result.shape}")
    
    # Show stats for the new operation
    stats = get_profit_stats()
    if 'custom_calculation' in stats['operations']:
        op_stats = stats['operations']['custom_calculation']
        print(f"Custom calculation recommendation: {op_stats['recommended']}")

def test_trading_simulation():
    """Simulate a realistic trading scenario."""
    print("\n=== Trading Simulation ===")
    
    # Simulate different types of trading calculations
    market_data = np.random.rand(1000, 100)  # 1000 timepoints, 100 features
    strategy_weights = np.random.rand(100, 10)  # 100 features, 10 strategies
    
    # Different calculation types with different profit expectations
    calculations = [
        ('market_analysis', market_data, strategy_weights, 2.5),
        ('risk_assessment', market_data[:100], strategy_weights, 1.8),  # Use same weights
        ('profit_optimization', market_data, strategy_weights, 4.2),
        ('portfolio_rebalance', market_data[:200], strategy_weights, 3.1),  # Use same weights
    ]
    
    print("Running trading calculations...")
    for calc_name, data, weights, profit in calculations:
        print(f"\n{calc_name}: Data {data.shape}, Profit {profit}")
        
        # Use the profit-driven dispatcher
        result = dispatch_op('matrix_multiply', data, weights, profit=profit)
        
        # Show current backend recommendation
        stats = get_profit_stats()
        op_stats = stats['operations']['matrix_multiply']
        print(f"  Backend used: {op_stats['recommended']}")
        print(f"  Result shape: {result.shape}")

def test_performance_comparison():
    """Compare performance between CPU and GPU for different data sizes."""
    print("\n=== Performance Comparison ===")
    
    # Test different data sizes
    sizes = [50, 100, 200, 500, 1000]
    
    for size in sizes:
        print(f"\nTesting size {size}x{size}:")
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # Run with same profit to see which backend is chosen
        profit = size * 0.01  # Profit proportional to size
        
        start_time = time.time()
        result = matrix_multiply(a, b, profit=profit)
        total_time = time.time() - start_time
        
        # Show which backend was used
        stats = get_profit_stats()
        op_stats = stats['operations']['matrix_multiply']
        backend_used = op_stats['recommended']
        
        print(f"  Backend: {backend_used}")
        print(f"  Time: {total_time:.4f}s")
        print(f"  Profit: {profit:.2f}")

def main():
    """Run all tests."""
    print("Profit-Driven Backend Dispatcher Test Suite")
    print("=" * 50)
    
    try:
        test_basic_operations()
        test_learning_behavior()
        test_decorator_integration()
        test_trading_simulation()
        test_performance_comparison()
        
        print("\n" + "=" * 50)
        print("Final Statistics:")
        print("=" * 50)
        
        final_stats = get_profit_stats()
        print(f"Total operations performed: {final_stats['total_operations']}")
        
        for op_name, op_stats in final_stats['operations'].items():
            print(f"\n{op_name}:")
            print(f"  CPU: {op_stats['cpu']['count']} ops, {op_stats['cpu']['profit']:.2f} profit, {op_stats['cpu']['avg_time']:.4f}s avg")
            print(f"  GPU: {op_stats['gpu']['count']} ops, {op_stats['gpu']['profit']:.2f} profit, {op_stats['gpu']['avg_time']:.4f}s avg")
            print(f"  Current recommendation: {op_stats['recommended']}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 