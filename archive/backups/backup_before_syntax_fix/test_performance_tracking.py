#!/usr/bin/env python3
"""
Test Suite for Performance Tracking Components

Tests for QuantumPerformanceRegistry and ZPEZBEPerformanceTracker.
"""

import logging
import time
import unittest
from typing import Any, Dict

    QuantumPerformanceRegistry, 
    QuantumPerformanceEntry,
    ZPEZBEPerformanceTracker, 
    ZPEVector, 
    ZBEBalance, 
    QuantumSyncStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestQuantumPerformanceRegistry(unittest.TestCase):
    """Test suite for Quantum Performance Registry."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = QuantumPerformanceRegistry(max_entries=5)
        self.test_entry = QuantumPerformanceEntry()
            strategy_id="test_strategy",
            quantum_sync_status="full_sync",
            zpe_energy=1e-33,
            zbe_status=0.0,
            entry_timestamp=time.time(),
            profit=100.0,
            risk_score=0.3,
            thermal_state="warm",
            bit_phase=16
        )

    def test_add_performance_entry(self):
        """Test adding performance entries."""
        self.assertEqual(len(self.registry.performance_entries), 0)

        self.registry.add_performance_entry(self.test_entry)
        self.assertEqual(len(self.registry.performance_entries), 1)

        # Test max entries limit
        for i in range(10):
            entry = QuantumPerformanceEntry()
                strategy_id=f"strategy_{i}",
                quantum_sync_status="full_sync",
                zpe_energy=1e-33,
                zbe_status=0.0,
                entry_timestamp=time.time(),
                profit=100.0
            )
            self.registry.add_performance_entry(entry)

        self.assertEqual(len(self.registry.performance_entries), 5)

    def test_analyze_quantum_performance(self):
        """Test performance analysis."""
        # Add some test entries
        entries = []
            QuantumPerformanceEntry()
                strategy_id="strategy_1",
                quantum_sync_status="resonance",
                zpe_energy=1e-33,
                zbe_status=0.0,
                entry_timestamp=time.time(),
                profit=100.0,
                thermal_state="hot"
            ),
            QuantumPerformanceEntry()
                strategy_id="strategy_2",
                quantum_sync_status="full_sync",
                zpe_energy=1e-33,
                zbe_status=0.0,
                entry_timestamp=time.time(),
                profit=50.0,
                thermal_state="warm"
            )
        ]

        for entry in entries:
            self.registry.add_performance_entry(entry)

        analysis = self.registry.analyze_quantum_performance()

        self.assertIsInstance(analysis, dict)
        self.assertIn('total_strategies', analysis)
        self.assertIn('average_profit', analysis)
        self.assertIn('performance_by_sync_status', analysis)
        self.assertIn('optimal_thermal_state', analysis)

        self.assertEqual(analysis['total_strategies'], 2)
        self.assertEqual(analysis['average_profit'], 75.0)

    def test_recommend_quantum_strategy_params(self):
        """Test strategy parameter recommendations."""
        # Add test entries first
        self.registry.add_performance_entry(self.test_entry)

        recommendations = self.registry.recommend_quantum_strategy_params()

        self.assertIsInstance(recommendations, dict)
        self.assertIn('recommended_thermal_state', recommendations)
        self.assertIn('best_sync_status', recommendations)
        self.assertIn('recommended_bit_phase', recommendations)
        self.assertIn('risk_tolerance', recommendations)


class TestZPEZBEPerformanceTracker(unittest.TestCase):
    """Test suite for ZPE-ZBE Performance Tracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ZPEZBEPerformanceTracker()
        self.zpe_vector = ZPEVector()
            energy=1e-33,
            frequency=7.83,
            mass_coefficient=1e-6,
            sync_status=QuantumSyncStatus.FULL_SYNC,
            timestamp=time.time()
        )
        self.zbe_balance = ZBEBalance()
            status=0.0,
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            stability_score=0.8
        )

    def test_log_strategy_performance(self):
        """Test logging strategy performance."""
        strategy_metadata = {}
            'strategy_id': 'test_strategy',
            'profit': 100.0,
            'risk_score': 0.3,
            'thermal_state': 'warm',
            'bit_phase': 16
        }

        self.tracker.log_strategy_performance()
            self.zpe_vector, self.zbe_balance, strategy_metadata
        )

        self.assertEqual(len(self.tracker.performance_registry.performance_entries), 1)

    def test_get_quantum_strategy_recommendations(self):
        """Test getting strategy recommendations."""
        # Log some performance first
        strategy_metadata = {}
            'strategy_id': 'test_strategy',
            'profit': 100.0,
            'risk_score': 0.3,
            'thermal_state': 'warm',
            'bit_phase': 16
        }

        self.tracker.log_strategy_performance()
            self.zpe_vector, self.zbe_balance, strategy_metadata
        )

        recommendations = self.tracker.get_quantum_strategy_recommendations()

        self.assertIsInstance(recommendations, dict)
        self.assertIn('recommended_thermal_state', recommendations)
        self.assertIn('best_sync_status', recommendations)

    def test_get_performance_analysis(self):
        """Test getting performance analysis."""
        # Log some performance first
        strategy_metadata = {}
            'strategy_id': 'test_strategy',
            'profit': 100.0,
            'risk_score': 0.3,
            'thermal_state': 'warm',
            'bit_phase': 16
        }

        self.tracker.log_strategy_performance()
            self.zpe_vector, self.zbe_balance, strategy_metadata
        )

        analysis = self.tracker.get_performance_analysis()

        self.assertIsInstance(analysis, dict)
        self.assertIn('total_strategies', analysis)
        self.assertIn('average_profit', analysis)


def run_performance_tests():
    """Run performance tracking tests."""
    logger.info("=" * 60)
    logger.info("📊 TESTING PERFORMANCE TRACKING COMPONENTS")
    logger.info("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestQuantumPerformanceRegistry))
    test_suite.addTest(unittest.makeSuite(TestZPEZBEPerformanceTracker))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_performance_tests()
    if success:
        logger.info("✅ Performance tracking tests passed!")
    else:
        logger.error("❌ Performance tracking tests failed!")
    exit(0 if success else 1) 