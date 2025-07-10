#!/usr/bin/env python3
"""
Comprehensive Test Suite for ZPE-ZBE Core Implementation

This script tests the complete integration of the new ZPE-ZBE core system
with quantum synchronization, performance tracking, and unified math system.
"""

import logging
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestZPEZBECore(unittest.TestCase):
    """Test suite for ZPE-ZBE Core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from core.clean_math_foundation import CleanMathFoundation
        from core.zpe_zbe_core import QuantumSyncStatus, ZBEBalance, ZPEVector, ZPEZBECore

        self.math_foundation = CleanMathFoundation()
        self.zpe_zbe_core = ZPEZBECore(self.math_foundation)

    def test_zpe_calculation(self):
        """Test Zero Point Energy calculation."""
        from core.zpe_zbe_core import QuantumSyncStatus

        # Test with default parameters
        zpe_vector = self.zpe_zbe_core.calculate_zero_point_energy()

        self.assertIsInstance(zpe_vector.energy, float)
        self.assertIsInstance(zpe_vector.frequency, float)
        self.assertIsInstance(zpe_vector.mass_coefficient, float)
        self.assertIsInstance(zpe_vector.sync_status, QuantumSyncStatus)
        self.assertIsInstance(zpe_vector.timestamp, float)
        self.assertIsInstance(zpe_vector.metadata, dict)

        # Test with custom parameters
        custom_zpe = self.zpe_zbe_core.calculate_zero_point_energy()
            frequency=10.0,
            mass_coefficient=1e-5
        )

        self.assertGreater(custom_zpe.energy, 0)
        self.assertEqual(custom_zpe.frequency, 10.0)
        self.assertEqual(custom_zpe.mass_coefficient, 1e-5)

    def test_zbe_balance_calculation(self):
        """Test Zero-Based Equilibrium balance calculation."""
        # Test equilibrium state
        zbe_balance = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        self.assertEqual(zbe_balance.status, 0.0)  # Inside equilibrium range
        self.assertEqual(zbe_balance.entry_price, 100.0)
        self.assertEqual(zbe_balance.current_price, 100.0)
        self.assertIsInstance(zbe_balance.stability_score, float)

        # Test under equilibrium
        under_zbe = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=90.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        self.assertEqual(under_zbe.status, -1.0)  # Under equilibrium

        # Test over equilibrium
        over_zbe = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=110.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        self.assertEqual(over_zbe.status, 1.0)  # Over equilibrium

    def test_quantum_sync_assessment(self):
        """Test quantum synchronization assessment."""
        from core.zpe_zbe_core import QuantumSyncStatus

        # Test different energy levels
        low_energy = self.zpe_zbe_core._assess_quantum_sync(1e-34)
        self.assertEqual(low_energy, QuantumSyncStatus.UNSYNCED)

        # Fix the medium energy test - use the correct threshold
        medium_energy = self.zpe_zbe_core._assess_quantum_sync(3e-33)  # Above threshold
        self.assertEqual(medium_energy, QuantumSyncStatus.FULL_SYNC)

        high_energy = self.zpe_zbe_core._assess_quantum_sync(1e-32)
        self.assertEqual(high_energy, QuantumSyncStatus.RESONANCE)

    def test_dual_matrix_sync_trigger(self):
        """Test dual matrix synchronization trigger."""
        zpe_vector = self.zpe_zbe_core.calculate_zero_point_energy()
        zbe_balance = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        sync_trigger = self.zpe_zbe_core.dual_matrix_sync_trigger(zpe_vector, zbe_balance)

        self.assertIsInstance(sync_trigger, dict)
        self.assertIn('is_synced', sync_trigger)
        self.assertIn('sync_strategy', sync_trigger)
        self.assertIn('zpe_energy', sync_trigger)
        self.assertIn('zbe_status', sync_trigger)
        self.assertIn('recommended_action', sync_trigger)

    def test_quantum_soulprint_vector(self):
        """Test quantum soulprint vector generation."""
        zpe_vector = self.zpe_zbe_core.calculate_zero_point_energy()
        zbe_balance = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        soulprint_vector = self.zpe_zbe_core.generate_quantum_soulprint_vector()
            zpe_vector, zbe_balance
        )

        self.assertIsInstance(soulprint_vector, dict)
        self.assertIn('entropy', soulprint_vector)
        self.assertIn('momentum', soulprint_vector)
        self.assertIn('volatility', soulprint_vector)
        self.assertIn('temporal_variance', soulprint_vector)
        self.assertIn('quantum_sync_status', soulprint_vector)
        self.assertIn('zbe_status', soulprint_vector)

    def test_strategy_confidence(self):
        """Test strategy confidence calculation."""
        zpe_vector = self.zpe_zbe_core.calculate_zero_point_energy()
        zbe_balance = self.zpe_zbe_core.calculate_zbe_balance()
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0
        )

        confidence = self.zpe_zbe_core.assess_quantum_strategy_confidence()
            zpe_vector, zbe_balance
        )

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestQuantumPerformanceRegistry(unittest.TestCase):
    """Test suite for Quantum Performance Registry."""

    def setUp(self):
        """Set up test fixtures."""
        from core.zpe_zbe_core import QuantumPerformanceEntry, QuantumPerformanceRegistry

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
        from core.zpe_zbe_core import ZBEBalance, ZPEVector, ZPEZBEPerformanceTracker

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


class TestUnifiedMathSystem(unittest.TestCase):
    """Test suite for Unified Math System."""

    def setUp(self):
        """Set up test fixtures."""
        from core.unified_math_system import UnifiedMathSystem
        from core.zpe_zbe_core import ZPEZBECore, ZPEZBEPerformanceTracker

        self.unified_math = UnifiedMathSystem()

    def test_quantum_market_analysis(self):
        """Test quantum market analysis."""
        market_data = {}
            'price': 100.0,
            'entry_price': 100.0,
            'lower_bound': 95.0,
            'upper_bound': 105.0,
            'frequency': 7.83,
            'mass_coefficient': 1e-6
        }

        analysis = self.unified_math.quantum_market_analysis(market_data)

        self.assertIsInstance(analysis, dict)
        self.assertIn('is_synced', analysis)
        self.assertIn('sync_strategy', analysis)
        self.assertIn('zpe_energy', analysis)
        self.assertIn('zpe_sync_status', analysis)
        self.assertIn('zbe_status', analysis)
        self.assertIn('zbe_stability_score', analysis)
        self.assertIn('quantum_potential', analysis)
        self.assertIn('resonance_factor', analysis)

    def test_advanced_quantum_decision_router(self):
        """Test advanced quantum decision routing."""
        quantum_analysis = {}
            'is_synced': True,
            'sync_strategy': 'LotusHold_Ω33',
            'zpe_energy': 1e-33,
            'zbe_status': 0.0,
            'quantum_potential': 0.5
        }

        decision = self.unified_math.advanced_quantum_decision_router(quantum_analysis)

        self.assertIsInstance(decision, dict)
        self.assertIn('strategy', decision)
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
        self.assertIn('quantum_potential', decision)
        self.assertIn('risk_adjustment', decision)

    def test_system_entropy(self):
        """Test system entropy calculation."""
        quantum_analysis = {}
            'zpe_energy': 1e-33,
            'zbe_status': 0.0,
            'quantum_potential': 0.5
        }

        entropy = self.unified_math.get_system_entropy(quantum_analysis)

        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

    def test_performance_tracking_integration(self):
        """Test integration with performance tracking."""
        from core.zpe_zbe_core import QuantumSyncStatus, ZBEBalance, ZPEVector

        zpe_vector = ZPEVector()
            energy=1e-33,
            frequency=7.83,
            mass_coefficient=1e-6,
            sync_status=QuantumSyncStatus.FULL_SYNC,
            timestamp=time.time()
        )

        zbe_balance = ZBEBalance()
            status=0.0,
            entry_price=100.0,
            current_price=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            stability_score=0.8
        )

        strategy_metadata = {}
            'strategy_id': 'test_strategy',
            'profit': 100.0,
            'risk_score': 0.3,
            'thermal_state': 'warm',
            'bit_phase': 16
        }

        # Test logging performance
        self.unified_math.log_strategy_performance()
            zpe_vector, zbe_balance, strategy_metadata
        )

        # Test getting recommendations
        recommendations = self.unified_math.get_quantum_strategy_recommendations()
        self.assertIsInstance(recommendations, dict)

        # Test getting analysis
        analysis = self.unified_math.get_performance_analysis()
        self.assertIsInstance(analysis, dict)


def run_basic_tests():
    """Run basic ZPE-ZBE core tests."""
    logger.info("=" * 60)
    logger.info("🧪 TESTING ZPE-ZBE CORE FUNCTIONALITY")
    logger.info("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestZPEZBECore))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_basic_tests()
    if success:
        logger.info("✅ ZPE-ZBE Core tests passed!")
    else:
        logger.error("❌ ZPE-ZBE Core tests failed!")
    exit(0 if success else 1) 