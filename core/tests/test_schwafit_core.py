"""
Test suite for Schwafit core functionality.
Tests the Schwafitting antifragile validation and partitioning system.
"""

import unittest
import numpy as np
from typing import List, Dict, Any
import tempfile
import os
import yaml
from pathlib import Path

from ..schwafit_core import (
    SchwafitManager,
    load_yaml_config,
    validate_config,
    generate_default_config
)

class TestSchwafitCore(unittest.TestCase):
    """Test cases for Schwafit core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test configs
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.test_dir.name) / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
        # Create a test config file
        self.test_config = {
            'meta_tag': 'test',
            'fallback_matrix': 'test_fallback',
            'scoring': {
                'hash_weight': 0.3,
                'volume_weight': 0.2,
                'drift_weight': 0.4,
                'error_weight': 0.1
            }
        }
        self.config_path = self.config_dir / 'test_strategies.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)

        # Initialize SchwafitManager with test parameters
        self.manager = SchwafitManager(
            min_ratio=0.1,
            max_ratio=0.5,
            cycle_period=100,
            noise_scale=0.01
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()

    def test_config_loading(self):
        """Test YAML configuration loading."""
        config = load_yaml_config('test_strategies.yaml')
        self.assertEqual(config['meta_tag'], 'test')
        self.assertEqual(config['fallback_matrix'], 'test_fallback')
        self.assertIn('scoring', config)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        self.assertTrue(validate_config(self.test_config))
        
        # Test invalid config
        invalid_config = self.test_config.copy()
        del invalid_config['meta_tag']
        with self.assertRaises(ValueError):
            validate_config(invalid_config)

    def test_default_config_generation(self):
        """Test default configuration generation."""
        default_config = generate_default_config('default_strategies.yaml')
        self.assertEqual(default_config['meta_tag'], 'default')
        self.assertIn('scoring', default_config)

    def test_dynamic_holdout_ratio(self):
        """Test dynamic holdout ratio calculation."""
        ratios = []
        for _ in range(100):
            ratio = self.manager.dynamic_holdout_ratio()
            ratios.append(ratio)
            self.assertGreaterEqual(ratio, self.manager.min_ratio)
            self.assertLessEqual(ratio, self.manager.max_ratio)
        
        # Test that ratios vary (not constant)
        self.assertGreater(np.std(ratios), 0)

    def test_data_splitting(self):
        """Test data partitioning functionality."""
        data = list(range(100))
        visible, holdout = self.manager.split_data(data)
        
        # Test basic properties
        self.assertEqual(len(visible) + len(holdout), len(data))
        self.assertGreater(len(visible), 0)
        self.assertGreater(len(holdout), 0)
        
        # Test no overlap
        visible_set = set(visible)
        holdout_set = set(holdout)
        self.assertEqual(len(visible_set.intersection(holdout_set)), 0)

    def test_shell_weights(self):
        """Test shell weight computation."""
        holdout = list(range(10))
        shell_states = [np.array([1.0, 0.0]) for _ in range(10)]
        weights = self.manager.compute_shell_weights(holdout, shell_states)
        
        # Test weight properties
        self.assertEqual(len(weights), len(holdout))
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_validation_tensor(self):
        """Test validation tensor generation."""
        def dummy_strategy(x): return x * 2
        
        strategies = [dummy_strategy]
        holdout = list(range(5))
        shell_states = [np.array([1.0]) for _ in range(5)]
        
        T = self.manager.schwafit_validation_tensor(
            strategies, holdout, shell_states
        )
        
        # Test tensor shape
        self.assertEqual(T.shape, (1, 5, 1))

    def test_adaptive_confidence_scores(self):
        """Test adaptive confidence score computation."""
        T = np.ones((2, 5, 1))  # 2 strategies, 5 samples, 1 shell class
        weights = np.ones(5) / 5
        predictions = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                              [1.1, 2.1, 3.1, 4.1, 5.1]])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        scores = self.manager.adaptive_confidence_scores(
            T, weights, predictions, targets
        )
        
        # Test score properties
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_variance_pool_update(self):
        """Test variance pool update mechanism."""
        initial_pool = self.manager.variance_pool
        holdout = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        self.manager.update_variance_pool(holdout)
        
        # Test that variance pool was updated
        self.assertNotEqual(self.manager.variance_pool, initial_pool)
        self.assertGreaterEqual(self.manager.variance_pool, 0)

    def test_memory_key_evolution(self):
        """Test memory key evolution."""
        holdout = list(range(5))
        shell_states = [np.array([1.0, 0.0]) for _ in range(5)]
        
        self.manager.evolve_memory_keys(holdout, shell_states)
        
        # Test that memory keys were updated
        self.assertIn('default', self.manager.memory_keys)
        self.assertEqual(len(self.manager.memory_keys['default']), 2)

    def test_profit_calibration(self):
        """Test profit tier calibration."""
        scores = np.array([0.8, 0.6])
        base_tiers = {'tier1': 1.0, 'tier2': 1.0}
        
        self.manager.calibrate_profits(scores, base_tiers)
        
        # Test that profit tiers were updated
        self.assertIn('tier1', self.manager.profit_tiers)
        self.assertIn('tier2', self.manager.profit_tiers)
        self.assertGreater(self.manager.profit_tiers['tier1'], 0)

    def test_full_schwafit_update(self):
        """Test complete Schwafit update cycle."""
        def dummy_strategy(x): return x * 2
        
        data = list(range(10))
        shell_states = [np.array([1.0]) for _ in range(10)]
        strategies = [dummy_strategy]
        predictions = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]])
        targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        meta_tags = ['test_strategy']
        
        result = self.manager.schwafit_update(
            data, shell_states, strategies, predictions, targets, meta_tags
        )
        
        # Test result structure
        self.assertIn('scores', result)
        self.assertIn('memory_keys', result)
        self.assertIn('profit_tiers', result)
        self.assertIn('variance_pool', result)
        self.assertIn('last_ratio', result)
        self.assertIn('validation_history', result)

    def test_strategy_registration(self):
        """Test strategy registration with meta tags."""
        self.manager.register_strategy('strategy1', 'test_tag')
        self.assertEqual(self.manager.strategy_tags['strategy1'], 'test_tag')

    def test_top_strategies(self):
        """Test top strategy retrieval."""
        # Add some test validation history
        self.manager.validation_history = [
            {'strategy': 's1', 'meta_tag': 'tag1', 'score': 0.9, 'timestamp': 1},
            {'strategy': 's2', 'meta_tag': 'tag1', 'score': 0.8, 'timestamp': 2},
            {'strategy': 's3', 'meta_tag': 'tag2', 'score': 0.7, 'timestamp': 3}
        ]
        
        # Test getting top strategies
        top = self.manager.get_top_strategies(n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]['strategy'], 's1')
        
        # Test filtering by tag
        top_tag1 = self.manager.get_top_strategies(n=2, tag='tag1')
        self.assertEqual(len(top_tag1), 2)
        self.assertEqual(top_tag1[0]['meta_tag'], 'tag1')

if __name__ == '__main__':
    unittest.main() 