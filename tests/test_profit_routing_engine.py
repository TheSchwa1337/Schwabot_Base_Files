# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Test suite for the refactored Profit Routing Engine and Math Utilities.

Tests the enhanced routing engine with shared math utilities, error handling,
and edge case management.
"""

import unittest
import numpy as np
from core.unified_math_system import unified_math
from decimal import Decimal
from typing import List, Tuple

# Import the modules to test
from core.utils.math_utils import (
    calculate_entropy,
    calculate_correlation,
    moving_average,
    exponential_smoothing,
    calculate_true_range,
    calculate_atr,
    calculate_rsi,
    calculate_williams_r,
    calculate_stochastic,
    calculate_gradient,
    calculate_centroid,
    calculate_distance_score,
    calculate_recursive_multiplier,
    calculate_allocation_efficiency,
    calculate_recursive_growth_factor,
    apply_allocation_strategy,
    safe_decimal_operation,
    validate_spatial_dimensions,
    create_spatial_grid,
)

from core.profit_routing_engine import (
    ProfitRoutingEngine,
    ProfitAllocationStrategy,
    VolumeProfile,
    ProfitNode,
    ProfitAllocationResult,
    create_profit_routing_system,
    simulate_profit_allocation,
)


class TestMathUtils(unittest.TestCase):
    """Test cases for shared math utilities."""

    def setUp(self):
        """Set up test data."""
        self.test_array = np.array([1, 2, 3, 4, 5])
        self.test_2d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.test_3d_array = np.zeros((3, 3, 3))
        self.test_3d_array[1, 1, 1] = 1.0

    def test_calculate_entropy(self):
        """Test entropy calculation."""
        # Test with uniform distribution
        uniform_data = np.array([1, 1, 1, 1, 1])
        entropy = calculate_entropy(uniform_data)
        self.assertAlmostEqual(entropy, 0.0, places=5)

        # Test with varied distribution
        varied_data = np.array([1, 2, 3, 4, 5])
        entropy = calculate_entropy(varied_data)
        self.assertGreater(entropy, 0.0)

    def test_calculate_correlation(self):
        """Test correlation calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        corr = calculate_correlation(x, y)
        self.assertAlmostEqual(corr, 1.0, places=5)

        # Test with negative correlation
        y_neg = np.array([5, 4, 3, 2, 1])
        corr_neg = calculate_correlation(x, y_neg)
        self.assertAlmostEqual(corr_neg, -1.0, places=5)

    def test_moving_average(self):
        """Test moving average calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma = moving_average(data, window=3)
        self.assertEqual(len(ma), len(data) - 2)  # Window size affects output length

    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        data = np.array([1, 2, 3, 4, 5])
        smoothed = exponential_smoothing(data, alpha=0.3)
        self.assertEqual(len(smoothed), len(data))

    def test_calculate_gradient(self):
        """Test gradient calculation."""
        # Test 2D gradient
        grad_2d = calculate_gradient(self.test_2d_array)
        self.assertEqual(grad_2d.shape, self.test_2d_array.shape)

        # Test 3D gradient
        grad_3d = calculate_gradient(self.test_3d_array)
        self.assertEqual(grad_3d.shape, self.test_3d_array.shape)

        # Test invalid dimensions
        with self.assertRaises(ValueError):
            calculate_gradient(np.array([1, 2, 3]))  # 1D array

    def test_calculate_centroid(self):
        """Test centroid calculation."""
        # Test 2D centroid
        centroid_2d = calculate_centroid(self.test_2d_array)
        self.assertEqual(len(centroid_2d), 2)

        # Test 3D centroid
        centroid_3d = calculate_centroid(self.test_3d_array)
        self.assertEqual(len(centroid_3d), 3)

        # Test invalid dimensions
        with self.assertRaises(ValueError):
            calculate_centroid(np.array([1, 2, 3]))  # 1D array

    def test_calculate_distance_score(self):
        """Test distance score calculation."""
        pos_a = (1.0, 2.0, 3.0)
        pos_b = (4.0, 5.0, 6.0)
        distance = calculate_distance_score(pos_a, pos_b)
        self.assertGreater(distance, 0.0)

        # Test same position
        distance_same = calculate_distance_score(pos_a, pos_a)
        self.assertEqual(distance_same, 0.0)

        # Test different dimensions
        with self.assertRaises(ValueError):
            calculate_distance_score((1, 2), (1, 2, 3))

    def test_calculate_recursive_multiplier(self):
        """Test recursive multiplier calculation."""
        # Test base case
        multiplier = calculate_recursive_multiplier(1.0, 0)
        self.assertEqual(multiplier, 1.0)

        # Test with depth
        multiplier_depth = calculate_recursive_multiplier(1.0, 2, decay_factor=0.5)
        self.assertEqual(multiplier_depth, 0.25)

        # Test max depth limit
        multiplier_max = calculate_recursive_multiplier(1.0, 15, max_depth=10)
        self.assertEqual(multiplier_max, calculate_recursive_multiplier(1.0, 10, max_depth=10))

    def test_calculate_allocation_efficiency(self):
        """Test allocation efficiency calculation."""
        volume_deltas = [("vol1", 100.0), ("vol2", 200.0), ("vol3", 300.0)]
        efficiency = calculate_allocation_efficiency(volume_deltas)
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)

        # Test empty input
        efficiency_empty = calculate_allocation_efficiency([])
        self.assertEqual(efficiency_empty, 0.0)

    def test_calculate_recursive_growth_factor(self):
        """Test recursive growth factor calculation."""
        profit_history = [100.0, 110.0, 121.0, 133.1]  # 10% growth
        growth_factor = calculate_recursive_growth_factor(profit_history)
        self.assertGreaterEqual(growth_factor, 0.5)
        self.assertLessEqual(growth_factor, 2.0)

        # Test insufficient data
        growth_factor_short = calculate_recursive_growth_factor([100.0])
        self.assertEqual(growth_factor_short, 1.0)

    def test_apply_allocation_strategy(self):
        """Test allocation strategy application."""
        base_value = Decimal("1000.0")

        # Test linear strategy
        linear_result = apply_allocation_strategy(base_value, "LINEAR")
        self.assertEqual(linear_result, base_value)

        # Test exponential strategy
        exp_result = apply_allocation_strategy(base_value, "EXPONENTIAL")
        self.assertNotEqual(exp_result, base_value)

        # Test invalid strategy (should default to linear)
        invalid_result = apply_allocation_strategy(base_value, "INVALID")
        self.assertEqual(invalid_result, base_value)

    def test_safe_decimal_operation(self):
        """Test safe decimal operations."""
        # Test addition
        add_result = safe_decimal_operation("add", 1.0, 2.0, 3.0)
        self.assertEqual(add_result, Decimal("6.0"))

        # Test multiplication
        mult_result = safe_decimal_operation("multiply", 2.0, 3.0, 4.0)
        self.assertEqual(mult_result, Decimal("24.0"))

        # Test division
        div_result = safe_decimal_operation("divide", 10.0, 2.0)
        self.assertEqual(div_result, Decimal("5.0"))

        # Test division by zero
        div_zero_result = safe_decimal_operation("divide", 10.0, 0.0)
        self.assertEqual(div_zero_result, Decimal("0.0"))

        # Test invalid operation
        invalid_result = safe_decimal_operation("invalid", 1.0, 2.0)
        self.assertEqual(invalid_result, Decimal("0.0"))

    def test_validate_spatial_dimensions(self):
        """Test spatial dimension validation."""
        # Test valid 2D dimensions
        self.assertTrue(validate_spatial_dimensions((10, 10)))

        # Test valid 3D dimensions
        self.assertTrue(validate_spatial_dimensions((10, 10, 10)))

        # Test invalid dimensions
        self.assertFalse(validate_spatial_dimensions((10,)))  # 1D
        self.assertFalse(validate_spatial_dimensions((10, 10, 10, 10)))  # 4D
        self.assertFalse(validate_spatial_dimensions((10, -5, 10)))  # Negative
        self.assertFalse(validate_spatial_dimensions("invalid"))  # Not tuple

    def test_create_spatial_grid(self):
        """Test spatial grid creation."""
        # Test 2D grid
        grid_2d = create_spatial_grid((3, 3))
        self.assertEqual(grid_2d.shape, (3, 3))
        self.assertEqual(grid_2d.dtype, np.float64)

        # Test 3D grid
        grid_3d = create_spatial_grid((2, 2, 2))
        self.assertEqual(grid_3d.shape, (2, 2, 2))

        # Test with custom fill value
        grid_custom = create_spatial_grid((2, 2), fill_value=1.0)
        self.assertTrue(np.all(grid_custom == 1.0))

        # Test invalid dimensions
        with self.assertRaises(ValueError):
            create_spatial_grid((10, -5, 10))


class TestProfitRoutingEngine(unittest.TestCase):
    """Test cases for the Profit Routing Engine."""

    def setUp(self):
        """Set up test environment."""
        self.engine = ProfitRoutingEngine(spatial_dimensions=(5, 5, 5))
        self.engine.initialize_profit_space()

    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertEqual(len(self.engine.profit_nodes), 125)  # 5x5x5
        self.assertEqual(len(self.engine.volume_profiles), 125)
        self.assertGreater(len(self.engine.allocation_chains), 0)

    def test_invalid_spatial_dimensions(self):
        """Test engine initialization with invalid dimensions."""
        with self.assertRaises(ValueError):
            ProfitRoutingEngine(spatial_dimensions=(0, 5, 5))

        with self.assertRaises(ValueError):
            ProfitRoutingEngine(spatial_dimensions=(5, -5, 5))

    def test_calculate_volumetric_profit(self):
        """Test volumetric profit calculation."""
        # Create test volume deltas
        volume_deltas = []
        for i in range(3):
            volume_id = f"vol_profit_node_0_0_{i}"
            if volume_id in self.engine.volume_profiles:
                volume_deltas.append((volume_id, 100.0 + i * 50))

        if volume_deltas:
            result = self.engine.calculate_volumetric_profit(
                volume_deltas=volume_deltas,
                price_tick=50000.0,
                strategy=ProfitAllocationStrategy.LINEAR
            )

            self.assertIsInstance(result, ProfitAllocationResult)
            self.assertGreater(float(result.total_profit_allocated), 0.0)
            self.assertGreaterEqual(result.allocation_efficiency, 0.0)
            self.assertLessEqual(result.allocation_efficiency, 1.0)

    def test_calculate_volumetric_profit_empty_input(self):
        """Test volumetric profit calculation with empty input."""
        result = self.engine.calculate_volumetric_profit(
            volume_deltas=[],
            price_tick=50000.0,
            strategy=ProfitAllocationStrategy.LINEAR
        )

        self.assertIsInstance(result, ProfitAllocationResult)
        self.assertEqual(float(result.total_profit_allocated), 0.0)

    def test_calculate_volumetric_profit_invalid_price(self):
        """Test volumetric profit calculation with invalid price."""
        volume_deltas = [("vol_profit_node_0_0_0", 100.0)]

        result = self.engine.calculate_volumetric_profit(
            volume_deltas=volume_deltas,
            price_tick=-1000.0,  # Invalid negative price
            strategy=ProfitAllocationStrategy.LINEAR
        )

        self.assertIsInstance(result, ProfitAllocationResult)
        self.assertEqual(float(result.total_profit_allocated), 0.0)

    def test_create_allocation_chain(self):
        """Test allocation chain creation."""
        start_node_id = "profit_node_0_0_0"
        chain = self.engine.create_allocation_chain(
            chain_id="test_chain",
            start_node_id=start_node_id,
            chain_length=5
        )

        self.assertIsInstance(chain, list)
        self.assertGreater(len(chain), 0)
        self.assertIn(start_node_id, chain)

    def test_create_allocation_chain_invalid_start(self):
        """Test allocation chain creation with invalid start node."""
        chain = self.engine.create_allocation_chain(
            chain_id="test_chain",
            start_node_id="invalid_node",
            chain_length=5
        )

        self.assertEqual(chain, [])

    def test_create_allocation_chain_invalid_length(self):
        """Test allocation chain creation with invalid length."""
        start_node_id = "profit_node_0_0_0"
        chain = self.engine.create_allocation_chain(
            chain_id="test_chain",
            start_node_id=start_node_id,
            chain_length=0
        )

        self.assertEqual(chain, [])

    def test_measure_2d_profit_density(self):
        """Test 2D profit density measurement."""
        density_map = self.engine.measure_2d_profit_density(z_level=0)

        self.assertIsInstance(density_map, np.ndarray)
        self.assertEqual(density_map.shape, (5, 5))

    def test_measure_2d_profit_density_invalid_level(self):
        """Test 2D profit density measurement with invalid z-level."""
        # Test negative z-level
        density_map_neg = self.engine.measure_2d_profit_density(z_level=-1)
        self.assertEqual(density_map_neg.shape, (5, 5))

        # Test z-level beyond dimensions
        density_map_beyond = self.engine.measure_2d_profit_density(z_level=10)
        self.assertEqual(density_map_beyond.shape, (5, 5))

    def test_measure_3d_profit_volume(self):
        """Test 3D profit volume measurement."""
        volume_data = self.engine.measure_3d_profit_volume()

        self.assertIsInstance(volume_data, dict)
        self.assertIn("total_volume", volume_data)
        self.assertIn("max_density", volume_data)
        self.assertIn("mean_density", volume_data)
        self.assertIn("centroid", volume_data)

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        metrics = self.engine.get_performance_metrics()

        self.assertIsInstance(metrics, dict)
        self.assertIn("operation_count", metrics)
        self.assertIn("error_count", metrics)
        self.assertIn("error_rate", metrics)
        self.assertIn("node_count", metrics)
        self.assertIn("chain_count", metrics)
        self.assertIn("history_size", metrics)
        self.assertIn("uptime", metrics)

        self.assertEqual(metrics["node_count"], 125)
        self.assertGreaterEqual(metrics["error_rate"], 0.0)
        self.assertLessEqual(metrics["error_rate"], 1.0)


class TestProfitRoutingIntegration(unittest.TestCase):
    """Integration tests for the complete profit routing system."""

    def test_create_profit_routing_system(self):
        """Test complete profit routing system creation."""
        engine = create_profit_routing_system(dimensions=(3, 3, 3))

        self.assertIsInstance(engine, ProfitRoutingEngine)
        self.assertEqual(len(engine.profit_nodes), 27)  # 3x3x3

    def test_simulate_profit_allocation(self):
        """Test profit allocation simulation."""
        engine = create_profit_routing_system(dimensions=(3, 3, 3))
        results = simulate_profit_allocation(
            engine=engine,
            simulation_steps=5,
            base_profit_per_step=100.0
        )

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)

        for result in results:
            self.assertIsInstance(result, ProfitAllocationResult)

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        engine = create_profit_routing_system(dimensions=(2, 2, 2))

        # Test with various error conditions
        initial_metrics = engine.get_performance_metrics()

        # Perform operations that might cause errors
        for i in range(10):
            try:
                # Mix valid and invalid operations
                if i % 2 == 0:
                    # Valid operation
                    volume_deltas = [("vol_profit_node_0_0_0", 100.0)]
                    engine.calculate_volumetric_profit(
                        volume_deltas=volume_deltas,
                        price_tick=50000.0,
                        strategy=ProfitAllocationStrategy.LINEAR
                    )
                else:
                    # Invalid operation
                    engine.calculate_volumetric_profit(
                        volume_deltas=[],
                        price_tick=-1000.0,
                        strategy=ProfitAllocationStrategy.LINEAR
                    )
            except Exception:
                pass  # Errors should be handled internally

        final_metrics = engine.get_performance_metrics()

        # Engine should still be functional
        self.assertGreater(final_metrics["operation_count"], 0)
        self.assertGreaterEqual(final_metrics["error_count"], 0)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
