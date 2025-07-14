#!/usr/bin/env python3
"""
Test Suite for Unified Math System

Tests for the unified mathematical system that integrates ZPE-ZBE core.
"""

import logging
import time
import unittest
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            'quantum_potential': 0.5,
            'strategy_confidence': 0.9  # Add high confidence
        }

        decision = self.unified_math.advanced_quantum_decision_router(quantum_analysis)

        self.assertIsInstance(decision, dict)
        self.assertIn('strategy', decision)
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
        self.assertIn('quantum_potential', decision)
        self.assertIn('risk_adjustment', decision)

        # Test synced scenario with high confidence
        self.assertEqual(decision['action'], 'hold')
        self.assertGreaterEqual(decision['confidence'], 0.8)

        # Test unsynced scenario
        unsynced_analysis = {}
            'is_synced': False,
            'zpe_energy': 1e-34,
            'zbe_status': 0.5,
            'quantum_potential': 0.1,
            'strategy_confidence': 0.3
        }

        unsynced_decision = self.unified_math.advanced_quantum_decision_router(unsynced_analysis)
        self.assertIn(unsynced_decision['action'], ['monitor', 'assess', 'wait'])

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

        # Test with higher values
        high_analysis = {}
            'zpe_energy': 1e-32,
            'zbe_status': 1.0,
            'quantum_potential': 1.0
        }

        high_entropy = self.unified_math.get_system_entropy(high_analysis)
        self.assertGreater(high_entropy, entropy)

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

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Market analysis
        market_data = {}
            'price': 100.0,
            'entry_price': 100.0,
            'lower_bound': 95.0,
            'upper_bound': 105.0,
            'frequency': 7.83,
            'mass_coefficient': 1e-6
        }

        quantum_analysis = self.unified_math.quantum_market_analysis(market_data)

        # 2. Decision routing
        decision = self.unified_math.advanced_quantum_decision_router(quantum_analysis)

        # 3. System entropy
        entropy = self.unified_math.get_system_entropy(quantum_analysis)

        # 4. Performance tracking (simulate a, trade)
        from core.zpe_zbe_core import QuantumSyncStatus, ZBEBalance, ZPEVector

        zpe_vector = ZPEVector()
            energy=quantum_analysis['zpe_energy'],
            frequency=7.83,
            mass_coefficient=1e-6,
            sync_status=QuantumSyncStatus(quantum_analysis['zpe_sync_status']),
            timestamp=time.time()
        )

        zbe_balance = ZBEBalance()
            status=quantum_analysis['zbe_status'],
            entry_price=market_data['entry_price'],
            current_price=market_data['price'],
            lower_bound=market_data['lower_bound'],
            upper_bound=market_data['upper_bound'],
            stability_score=quantum_analysis['zbe_stability_score']
        )

        strategy_metadata = {}
            'strategy_id': decision['strategy'],
            'profit': 50.0,  # Simulated profit
            'risk_score': 1.0 - decision['confidence'],
            'thermal_state': 'warm',
            'bit_phase': 16
        }

        self.unified_math.log_strategy_performance()
            zpe_vector, zbe_balance, strategy_metadata
        )

        # 5. Get recommendations based on performance
        recommendations = self.unified_math.get_quantum_strategy_recommendations()

        # Verify all components work together
        self.assertIsInstance(quantum_analysis, dict)
        self.assertIsInstance(decision, dict)
        self.assertIsInstance(entropy, float)
        self.assertIsInstance(recommendations, dict)

        logger.info(f"✅ End-to-end workflow completed successfully")
        logger.info(f"   Quantum Analysis: {quantum_analysis['zpe_sync_status']}")
        logger.info(f"   Decision: {decision['action']} with {decision['confidence']:.2f} confidence")
        logger.info(f"   System Entropy: {entropy:.6f}")
        logger.info(f"   Recommendations: {recommendations.get('recommended_thermal_state', 'N/A')}")


def run_unified_math_tests():
    """Run unified math system tests."""
    logger.info("=" * 60)
    logger.info("🧮 TESTING UNIFIED MATH SYSTEM")
    logger.info("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestUnifiedMathSystem))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_unified_math_tests()
    if success:
        logger.info("✅ Unified Math System tests passed!")
    else:
        logger.error("❌ Unified Math System tests failed!")
    exit(0 if success else 1) 