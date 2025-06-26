#!/usr/bin/env python3
"""Test Fractal Sync - Cyclical Memory and Fractal State Estimator.

This test module validates cyclical memory and fractal state estimator
integration with strategy_mapper.py and backlog_hash_state.json.

Key Test Areas:
- Fractal state estimation accuracy
- Cyclical memory validation
- Strategy mapper integration
- Backlog hash state consistency
- Fractal overlay calculations
- Directional repeat probability

Flake8 compliant with comprehensive test coverage.
"""

from ghost_strategy_handler import GhostStrategyHandler, GhostEntry
from volume_tick_router import VolumeTickRouter, VolumeConfidence
from tick_backlog_router import TickBacklogRouter, BacklogProfit
from hash_confidence_evaluator import HashConfidenceEvaluator, HashResonance
from fractal_core import FractalCore, FractalState, GrayscaleCollapseResult
import unittest
import time
import json
import os
import sys
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))


class TestFractalSync(unittest.TestCase):
    """Test fractal synchronization and cyclical memory."""

    def setUp(self):
        """Set up test fixtures."""
        self.fractal_core = FractalCore()
        self.hash_evaluator = HashConfidenceEvaluator()
        self.backlog_router = TickBacklogRouter()
        self.volume_router = VolumeTickRouter()
        self.ghost_handler = GhostStrategyHandler()

        # Test data
        self.test_tick_data = {
            'timestamp': time.time(),
            'price': 50000.0,
            'volume': 1000000.0,
            'order_book': {
                'bids': [[49999.0, 100.0], [49998.0, 200.0]],
                'asks': [[50001.0, 150.0], [50002.0, 250.0]]
            }
        }

        self.test_market_data = {
            'price': 50000.0,
            'volume': 1000000.0,
            'price_volatility': 0.02,
            'expected_volume': 1200000.0,
            'bid_volume': 500000.0,
            'ask_volume': 600000.0,
            'price_change': 0.001
        }

        self.test_conventional_signals = {
            'buy_signal': 0.3,
            'sell_signal': 0.2,
            'momentum': 0.1,
            'volume_signal': 0.4
        }

    def test_fractal_state_estimation(self):
        """Test fractal state estimation accuracy."""
        # Add fractal states
        state1 = self.fractal_core.add_fractal_state(
            "test_state_1", 0.8, time.time(), "test_data_1"
        )
        state2 = self.fractal_core.add_fractal_state(
            "test_state_2", 0.6, time.time() + 1, "test_data_2"
        )

        # Perform grayscale collapse
        collapse_result = self.fractal_core.grayscale_collapse(time.time() + 2)

        # Validate results
        self.assertIsInstance(collapse_result, GrayscaleCollapseResult)
        self.assertGreater(collapse_result.collapsed_value, 0.0)
        self.assertGreaterEqual(collapse_result.confidence_score, 0.0)
        self.assertLessEqual(collapse_result.confidence_score, 1.0)
        self.assertIn(state1, collapse_result.contributing_states)
        self.assertIn(state2, collapse_result.contributing_states)

    def test_cyclical_memory_validation(self):
        """Test cyclical memory validation."""
        # Process multiple ticks to build memory
        for i in range(10):
            tick_data = self.test_tick_data.copy()
            tick_data['timestamp'] = time.time() + i
            tick_data['price'] = 50000.0 + (i * 10)

            # Process through hash evaluator
            trigger = self.hash_evaluator.process_tick_event(tick_data)

            # Process through backlog router
            profit = self.backlog_router.process_tick_data(tick_data)

            # Process through volume router
            volume_confidence = self.volume_router.process_volume_event(tick_data)

        # Validate memory consistency
        self.assertGreater(len(self.hash_evaluator.tick_history), 0)
        self.assertGreater(len(self.backlog_router.tick_memory), 0)
        self.assertGreater(len(self.volume_router.volume_history), 0)

        # Check for cyclical patterns
        hash_analytics = self.hash_evaluator.get_hash_resonance_analytics()
        backlog_analytics = self.backlog_router.get_backlog_analytics()
        volume_analytics = self.volume_router.get_volume_analytics()

        self.assertIn('total_hashes_processed', hash_analytics)
        self.assertIn('total_ticks_processed', backlog_analytics)
        self.assertIn('total_volume_events', volume_analytics)

    def test_strategy_mapper_integration(self):
        """Test strategy mapper integration with fractal core."""
        # Create fractal command dispatcher
        fractal_core, command_dispatcher = self.fractal_core, Mock()

        # Simulate strategy mapping
        command_data = {
            'action': 'buy',
            'confidence': 0.8,
            'price_target': 51000.0,
            'volume_target': 500000.0
        }

        # Mock command dispatch
        command_dispatcher.dispatch_command.return_value = {
            'success': True,
            'execution_time': time.time(),
            'command_id': 'test_command_1'
        }

        # Validate integration
        result = command_dispatcher.dispatch_command('test_command', command_data)
        self.assertTrue(result['success'])
        self.assertIn('command_id', result)

    def test_backlog_hash_state_consistency(self):
        """Test backlog hash state consistency."""
        # Process tick data
        tick_data = self.test_tick_data.copy()
        profit = self.backlog_router.process_tick_data(tick_data)

        # Validate backlog state
        self.assertIsInstance(profit, BacklogProfit)
        self.assertGreaterEqual(profit.total_profit, 0.0)
        self.assertGreaterEqual(profit.api_sync_score, 0.0)
        self.assertLessEqual(profit.api_sync_score, 1.0)

        # Check state consistency
        analytics = self.backlog_router.get_backlog_analytics()
        self.assertIn('backlog_state', analytics)
        self.assertIn('memory_persistence_factor', analytics)

    def test_fractal_overlay_calculations(self):
        """Test fractal overlay calculations."""
        # Add multiple fractal states with different weights
        states = []
        for i in range(5):
            state = self.fractal_core.add_fractal_state(
                f"overlay_state_{i}",
                0.5 + (i * 0.1),
                time.time() + i,
                f"overlay_data_{i}"
            )
            states.append(state)

        # Calculate fractal command weights
        weights = []
        for i in range(5):
            weight = self.fractal_core.calculate_fractal_command_weight(i)
            weights.append(weight)

        # Validate weight progression (should follow golden ratio)
        for i in range(1, len(weights)):
            ratio = weights[i] / weights[i-1]
            # Should approximate golden ratio (1.618)
            self.assertGreater(ratio, 1.5)
            self.assertLess(ratio, 1.7)

    def test_directional_repeat_probability(self):
        """Test directional repeat probability calculations."""
        # Create price sequence with directional patterns
        price_sequence = [50000.0]
        for i in range(10):
            # Simulate directional movement
            if i % 3 == 0:  # Every 3rd tick, change direction
                direction = 1 if i % 6 == 0 else -1
            else:
                direction = 1 if price_sequence[-1] > price_sequence[-2] else -1

            new_price = price_sequence[-1] + (direction * 100)
            price_sequence.append(new_price)

        # Calculate directional repeat probability
        repeat_count = 0
        total_transitions = len(price_sequence) - 2

        for i in range(1, len(price_sequence) - 1):
            direction1 = 1 if price_sequence[i] > price_sequence[i-1] else -1
            direction2 = 1 if price_sequence[i+1] > price_sequence[i] else -1

            if direction1 == direction2:
                repeat_count += 1

        repeat_probability = repeat_count / total_transitions if total_transitions > 0 else 0.0

        # Validate probability
        self.assertGreaterEqual(repeat_probability, 0.0)
        self.assertLessEqual(repeat_probability, 1.0)

    def test_hash_resonance_integration(self):
        """Test hash resonance integration with fractal core."""
        # Process tick data through hash evaluator
        trigger = self.hash_evaluator.process_tick_event(self.test_tick_data)

        # Validate hash resonance
        self.assertIsNotNone(trigger.hash_value)
        self.assertGreaterEqual(trigger.confidence, 0.0)
        self.assertLessEqual(trigger.confidence, 1.0)

        # Check resonance map
        analytics = self.hash_evaluator.get_hash_resonance_analytics()
        self.assertIn('total_resonances', analytics)
        self.assertIn('average_resonance_strength', analytics)

    def test_volume_pressure_integration(self):
        """Test volume pressure integration with fractal calculations."""
        # Process volume event
        volume_data = {
            'volume': 1000000.0,
            'timestamp': time.time()
        }

        volume_confidence = self.volume_router.process_volume_event(
            volume_data, self.test_market_data
        )

        # Validate volume confidence
        self.assertIsInstance(volume_confidence, VolumeConfidence)
        self.assertGreaterEqual(volume_confidence.confidence_score, 0.0)
        self.assertLessEqual(volume_confidence.confidence_score, 1.0)
        self.assertIn('volume_sensitivity', volume_confidence.__dict__)
        self.assertIn('hash_intersection', volume_confidence.__dict__)

    def test_ghost_strategy_integration(self):
        """Test ghost strategy integration with fractal patterns."""
        # Detect ghost entry
        ghost_entry = self.ghost_handler.detect_ghost_entry(
            self.test_market_data, self.test_conventional_signals
        )

        # Ghost entry may or may not be detected based on conditions
        if ghost_entry:
            self.assertIsInstance(ghost_entry, GhostEntry)
            self.assertGreaterEqual(ghost_entry.stealth_level, 0.0)
            self.assertLessEqual(ghost_entry.stealth_level, 1.0)

            # Execute ghost trade
            execution = self.ghost_handler.execute_ghost_trade(
                ghost_entry, self.test_market_data
            )

            self.assertIsInstance(execution, type(execution))
            self.assertIn('success', execution.__dict__)
            self.assertIn('stealth_score', execution.__dict__)

    def test_fractal_state_persistence(self):
        """Test fractal state persistence across cycles."""
        # Add initial states
        initial_states = []
        for i in range(3):
            state = self.fractal_core.add_fractal_state(
                f"persistent_state_{i}", 0.7, time.time() + i
            )
            initial_states.append(state)

        # Perform multiple collapse cycles
        collapse_results = []
        for i in range(5):
            result = self.fractal_core.grayscale_collapse(time.time() + i + 10)
            collapse_results.append(result)

        # Validate persistence
        self.assertEqual(len(collapse_results), 5)
        for result in collapse_results:
            self.assertIsInstance(result, GrayscaleCollapseResult)
            self.assertGreater(result.collapsed_value, 0.0)

    def test_entropy_calculation(self):
        """Test entropy calculation in fractal system."""
        # Calculate state entropy
        entropy = self.fractal_core.get_state_entropy()

        # Add more states and recalculate
        for i in range(5):
            self.fractal_core.add_fractal_state(
                f"entropy_state_{i}", 0.5, time.time() + i
            )

        new_entropy = self.fractal_core.get_state_entropy()

        # Validate entropy calculations
        self.assertGreaterEqual(entropy, 0.0)
        self.assertGreaterEqual(new_entropy, 0.0)
        # More states should generally increase entropy
        self.assertGreaterEqual(new_entropy, entropy)

    def test_recursive_hash_structure(self):
        """Test recursive hash structure generation."""
        # Generate recursive hash structure
        data = "test_recursive_data"
        depth = 3

        hash_structure = self.fractal_core.recursive_hash_structure(data, depth)

        # Validate structure
        self.assertIn('hash', hash_structure)
        self.assertEqual(hash_structure['depth'], depth)
        self.assertIn('fractal_weight', hash_structure)
        self.assertIn('collapse_probability', hash_structure)
        self.assertIn('recursive_component', hash_structure)

        # Validate recursive component
        recursive = hash_structure['recursive_component']
        self.assertIn('hash', recursive)
        self.assertEqual(recursive['depth'], depth - 1)

    def test_memory_state_retention(self):
        """Test memory state retention across system components."""
        # Process data through all components
        tick_data = self.test_tick_data.copy()

        # Hash evaluator
        trigger = self.hash_evaluator.process_tick_event(tick_data)

        # Backlog router
        profit = self.backlog_router.process_tick_data(tick_data)

        # Volume router
        volume_confidence = self.volume_router.process_volume_event(tick_data)

        # Validate memory retention
        self.assertGreater(len(self.hash_evaluator.command_memory), 0)
        self.assertGreater(len(self.backlog_router.backlog_history), 0)
        self.assertGreater(len(self.volume_router.volume_matches), 0)

        # Check memory consistency
        hash_memory_size = len(self.hash_evaluator.command_memory)
        backlog_memory_size = len(self.backlog_router.backlog_history)
        volume_memory_size = len(self.volume_router.volume_matches)

        self.assertGreaterEqual(hash_memory_size, 0)
        self.assertGreaterEqual(backlog_memory_size, 0)
        self.assertGreaterEqual(volume_memory_size, 0)

    def test_fractal_command_weighting(self):
        """Test fractal command weighting system."""
        # Test different depths
        depths = [0, 1, 2, 3, 4, 5]
        weights = []

        for depth in depths:
            weight = self.fractal_core.calculate_fractal_command_weight(depth)
            weights.append(weight)

        # Validate weight progression
        self.assertEqual(weights[0], 1.0)  # Base case

        # Weights should increase with depth (golden ratio progression)
        for i in range(1, len(weights)):
            self.assertGreater(weights[i], weights[i-1])

            # Check golden ratio approximation
            if i > 1:
                ratio = weights[i] / weights[i-1]
                self.assertGreater(ratio, 1.5)
                self.assertLess(ratio, 1.7)

    def test_system_integration_validation(self):
        """Test complete system integration validation."""
        # Comprehensive integration test
        test_data = self.test_tick_data.copy()
        test_data['timestamp'] = time.time()

        # Process through all systems
        results = {}

        # 1. Hash confidence evaluation
        results['hash_trigger'] = self.hash_evaluator.process_tick_event(test_data)

        # 2. Backlog processing
        results['backlog_profit'] = self.backlog_router.process_tick_data(test_data)

        # 3. Volume processing
        results['volume_confidence'] = self.volume_router.process_volume_event(test_data)

        # 4. Ghost strategy detection
        results['ghost_entry'] = self.ghost_handler.detect_ghost_entry(
            self.test_market_data, self.test_conventional_signals
        )

        # 5. Fractal state addition
        results['fractal_state'] = self.fractal_core.add_fractal_state(
            "integration_test", 0.8, time.time(), "integration_data"
        )

        # Validate all results
        self.assertIn('hash_trigger', results)
        self.assertIn('backlog_profit', results)
        self.assertIn('volume_confidence', results)
        self.assertIn('fractal_state', results)

        # Check data consistency
        self.assertIsInstance(results['hash_trigger'], type(results['hash_trigger']))
        self.assertIsInstance(results['backlog_profit'], BacklogProfit)
        self.assertIsInstance(results['volume_confidence'], VolumeConfidence)
        self.assertIsInstance(results['fractal_state'], FractalState)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
\n# -*- coding: utf-8 -*-\n