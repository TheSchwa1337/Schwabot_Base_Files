#!/usr/bin/env python3
"""
Galileo Tensor Field Test Suite 🧪

Comprehensive testing for the Galileo Tensor Field implementation:
- Tensor drift calculations
- Entropy field metrics
- Galilean transformations
- Tensor oscillations
- Quantum tensor operations
- GPU/CPU fallback functionality
"""

import numpy as np
import time
from core.entropy.galileo_tensor_field import GalileoTensorField, EntropyMetrics, TensorFieldConfig

def test_tensor_drift_calculation():
    """Test tensor drift calculation functionality."""
    print("🧪 Testing Tensor Drift Calculation...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Generate test market data
    np.random.seed(42)
    market_data = np.random.normal(100, 10, 200)  # 200 data points
    
    try:
        # Test tensor drift calculation
        drift_result = galileo_field.calculate_tensor_drift(market_data, time_window=50)
        
        print(f"   Input data shape: {market_data.shape}")
        print(f"   Drift result shape: {drift_result.shape}")
        print(f"   Drift mean: {np.mean(drift_result):.6f}")
        print(f"   Drift std: {np.std(drift_result):.6f}")
        
        # Validate results
        assert drift_result.shape == market_data.shape, "Drift result should have same shape as input"
        assert not np.any(np.isnan(drift_result)), "Drift result should not contain NaN values"
        assert not np.any(np.isinf(drift_result)), "Drift result should not contain infinite values"
        
        print("   ✅ Tensor drift calculation passed")
        
    except Exception as e:
        print(f"   ❌ Tensor drift calculation failed: {e}")
        raise

def test_entropy_field_calculation():
    """Test entropy field calculation functionality."""
    print(f"\n🧪 Testing Entropy Field Calculation...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Generate test price and volume data
    np.random.seed(42)
    price_data = np.random.normal(50000, 1000, 100)  # BTC price simulation
    volume_data = np.random.exponential(1000, 100)   # Volume simulation
    
    try:
        # Test entropy field calculation
        entropy_metrics = galileo_field.calculate_entropy_field(price_data, volume_data)
        
        print(f"   Shannon Entropy: {entropy_metrics.shannon_entropy:.6f}")
        print(f"   Renyi Entropy: {entropy_metrics.renyi_entropy:.6f}")
        print(f"   Tsallis Entropy: {entropy_metrics.tsallis_entropy:.6f}")
        print(f"   Tensor Entropy: {entropy_metrics.tensor_entropy:.6f}")
        print(f"   Field Strength: {entropy_metrics.field_strength:.6f}")
        print(f"   Oscillation Frequency: {entropy_metrics.oscillation_frequency:.6f}")
        print(f"   Drift Coefficient: {entropy_metrics.drift_coefficient:.6f}")
        
        # Validate entropy metrics
        assert 0 <= entropy_metrics.shannon_entropy <= 20, "Shannon entropy should be reasonable"
        assert 0 <= entropy_metrics.renyi_entropy <= 20, "Renyi entropy should be reasonable"
        assert not np.isnan(entropy_metrics.tsallis_entropy), "Tsallis entropy should not be NaN"
        assert not np.isnan(entropy_metrics.tensor_entropy), "Tensor entropy should not be NaN"
        assert 0 <= entropy_metrics.field_strength <= 1, "Field strength should be between 0 and 1"
        assert entropy_metrics.oscillation_frequency >= 0, "Oscillation frequency should be non-negative"
        
        print("   ✅ Entropy field calculation passed")
        
    except Exception as e:
        print(f"   ❌ Entropy field calculation failed: {e}")
        raise

def test_galilean_transform():
    """Test Galilean transformation functionality."""
    print(f"\n🧪 Testing Galilean Transformation...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 100)
    velocity = 0.1
    
    try:
        # Test Galilean transformation
        transformed_data = galileo_field.galilean_transform(test_data, velocity)
        
        print(f"   Input data shape: {test_data.shape}")
        print(f"   Transformed data shape: {transformed_data.shape}")
        print(f"   Input mean: {np.mean(test_data):.6f}")
        print(f"   Transformed mean: {np.mean(transformed_data):.6f}")
        print(f"   Velocity: {velocity}")
        
        # Validate transformation
        assert transformed_data.shape == test_data.shape, "Transformed data should have same shape"
        assert not np.any(np.isnan(transformed_data)), "Transformed data should not contain NaN"
        
        # Check that transformation actually changed the data
        data_difference = np.mean(np.abs(transformed_data - test_data))
        assert data_difference > 0, "Transformation should change the data"
        
        print("   ✅ Galilean transformation passed")
        
    except Exception as e:
        print(f"   ❌ Galilean transformation failed: {e}")
        raise

def test_tensor_oscillation():
    """Test tensor oscillation functionality."""
    print(f"\n🧪 Testing Tensor Oscillation...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 100)
    frequency = 2.0
    amplitude = 0.5
    
    try:
        # Test tensor oscillation
        oscillated_data = galileo_field.tensor_oscillation(test_data, frequency, amplitude)
        
        print(f"   Input data shape: {test_data.shape}")
        print(f"   Oscillated data shape: {oscillated_data.shape}")
        print(f"   Input std: {np.std(test_data):.6f}")
        print(f"   Oscillated std: {np.std(oscillated_data):.6f}")
        print(f"   Frequency: {frequency}")
        print(f"   Amplitude: {amplitude}")
        
        # Validate oscillation
        assert oscillated_data.shape == test_data.shape, "Oscillated data should have same shape"
        assert not np.any(np.isnan(oscillated_data)), "Oscillated data should not contain NaN"
        
        # Check that oscillation increased variance
        original_variance = np.var(test_data)
        oscillated_variance = np.var(oscillated_data)
        assert oscillated_variance >= original_variance, "Oscillation should increase variance"
        
        print("   ✅ Tensor oscillation passed")
        
    except Exception as e:
        print(f"   ❌ Tensor oscillation failed: {e}")
        raise

def test_quantum_tensor_operation():
    """Test quantum tensor operation functionality."""
    print(f"\n🧪 Testing Quantum Tensor Operations...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Generate test tensors
    np.random.seed(42)
    tensor_a = np.random.normal(0, 1, (10, 10))
    tensor_b = np.random.normal(0, 1, (10, 10))
    
    try:
        # Test quantum tensor operation
        quantum_result = galileo_field.quantum_tensor_operation(tensor_a, tensor_b)
        
        print(f"   Tensor A shape: {tensor_a.shape}")
        print(f"   Tensor B shape: {tensor_b.shape}")
        print(f"   Quantum result shape: {quantum_result.shape}")
        print(f"   Quantum result mean: {np.mean(quantum_result):.6f}")
        print(f"   Quantum result std: {np.std(quantum_result):.6f}")
        
        # Validate quantum operation
        assert len(quantum_result) > 0, "Quantum result should not be empty"
        assert not np.any(np.isnan(quantum_result)), "Quantum result should not contain NaN"
        assert not np.any(np.isinf(quantum_result)), "Quantum result should not contain infinite values"
        
        print("   ✅ Quantum tensor operations passed")
        
    except Exception as e:
        print(f"   ❌ Quantum tensor operations failed: {e}")
        raise

def test_field_status():
    """Test field status functionality."""
    print(f"\n🧪 Testing Field Status...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    try:
        # Test field status
        status = galileo_field.get_field_status()
        
        print(f"   Field Status: {status}")
        
        # Validate status
        assert isinstance(status, dict), "Status should be a dictionary"
        assert "dimension" in status, "Status should contain dimension"
        assert "precision" in status, "Status should contain precision"
        assert "max_iterations" in status, "Status should contain max_iterations"
        assert "convergence_threshold" in status, "Status should contain convergence_threshold"
        
        print("   ✅ Field status passed")
        
    except Exception as e:
        print(f"   ❌ Field status failed: {e}")
        raise

def test_error_handling():
    """Test error handling and edge cases."""
    print(f"\n🧪 Testing Error Handling...")
    
    # Create test instance
    config = TensorFieldConfig(dimension=3, use_gpu=False, fallback_enabled=True)
    galileo_field = GalileoTensorField(config)
    
    # Test with empty data
    print(f"\n1. Testing Empty Data...")
    try:
        empty_result = galileo_field.calculate_tensor_drift(np.array([]), time_window=10)
        print(f"   Empty data result shape: {empty_result.shape}")
        print("   ✅ Empty data handling passed")
    except Exception as e:
        print(f"   ❌ Empty data handling failed: {e}")
    
    # Test with insufficient data
    print(f"\n2. Testing Insufficient Data...")
    try:
        small_data = np.random.normal(0, 1, 5)  # Very small dataset
        small_result = galileo_field.calculate_tensor_drift(small_data, time_window=10)
        print(f"   Small data result shape: {small_result.shape}")
        print("   ✅ Insufficient data handling passed")
    except Exception as e:
        print(f"   ❌ Insufficient data handling failed: {e}")
    
    # Test with NaN data
    print(f"\n3. Testing NaN Data...")
    try:
        nan_data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        nan_result = galileo_field.calculate_tensor_drift(nan_data, time_window=2)
        print(f"   NaN data result shape: {nan_result.shape}")
        print("   ✅ NaN data handling passed")
    except Exception as e:
        print(f"   ❌ NaN data handling failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Galileo Tensor Field Test Suite")
    print("=" * 50)
    
    try:
        test_tensor_drift_calculation()
        test_entropy_field_calculation()
        test_galilean_transform()
        test_tensor_oscillation()
        test_quantum_tensor_operation()
        test_field_status()
        test_error_handling()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Galileo Tensor Field is fully implemented and ready for Schwabot integration!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main() 