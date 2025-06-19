"""
Comprehensive Core Systems Integration Test
==========================================

This test suite validates:
1. Mathematical Library Integration (mathlib v1-3)
2. NCCO Core System functionality
3. ALEPH Core System functionality  
4. Cross-system integration and data flow
5. Mathematical correctness and consistency
6. System dependencies and imports

Purpose: Ensure all core file systems work together seamlessly with no errors.
"""

import sys
import os
import unittest
import asyncio
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the necessary paths
sys.path.insert(0, '/c:/Users/maxde/OneDrive/Documents')
sys.path.insert(0, '/c:/Users/maxde/OneDrive/Documents/core')
sys.path.insert(0, '/c:/Users/maxde/OneDrive/Documents/ncco_core')
sys.path.insert(0, '/c:/Users/maxde/OneDrive/Documents/aleph_core')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveCoreIntegrationTest(unittest.TestCase):
    """Comprehensive integration test for all core systems"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize all core systems for testing"""
        cls.test_results = {
            'mathlib_tests': {},
            'ncco_tests': {},
            'aleph_tests': {},
            'integration_tests': {},
            'mathematical_validation': {}
        }
        
        logger.info("🚀 Starting Comprehensive Core Systems Integration Test")
    
    def setUp(self):
        """Set up test environment for each test"""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after each test"""
        test_duration = time.time() - self.start_time
        logger.info(f"Test completed in {test_duration:.3f}s")
    
    # ===== MATHEMATICAL LIBRARY TESTS =====
    
    def test_mathlib_v1_core_functionality(self):
        """Test core mathlib functionality"""
        logger.info("Testing MathLib v1 Core Functionality...")
        
        try:
            from core.mathlib import CoreMathLib, GradedProfitVector
            
            # Initialize mathlib
            mathlib = CoreMathLib()
            
            # Test basic mathematical operations
            test_vector_a = np.array([1.0, 2.0, 3.0])
            test_vector_b = np.array([4.0, 5.0, 6.0])
            
            # Test cosine similarity
            similarity = mathlib.cosine_similarity(test_vector_a, test_vector_b)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, -1.0)
            self.assertLessEqual(similarity, 1.0)
            
            # Test euclidean distance
            distance = mathlib.euclidean_distance(test_vector_a, test_vector_b)
            self.assertIsInstance(distance, float)
            self.assertGreaterEqual(distance, 0.0)
            
            # Test vector normalization
            normalized = mathlib.normalize_vector(test_vector_a)
            np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0, decimal=6)
            
            # Test graded profit vector
            test_trade = {
                'profit': 150.0,
                'volume_allocated': 1000.0,
                'time_held': 3600.0,
                'signal_strength': 0.8,
                'smart_money_score': 0.75
            }
            
            graded_vector = mathlib.grading_vector(test_trade)
            self.assertIsInstance(graded_vector, GradedProfitVector)
            self.assertEqual(graded_vector.profit, 150.0)
            
            # Test entropy calculation
            test_prices = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
            entropy = mathlib.shell_entropy(test_prices / np.sum(test_prices))
            self.assertIsInstance(entropy, float)
            self.assertGreaterEqual(entropy, 0.0)
            
            self.test_results['mathlib_tests']['v1_core'] = True
            logger.info("✅ MathLib v1 Core Functionality: PASSED")
            
        except Exception as e:
            self.test_results['mathlib_tests']['v1_core'] = False
            logger.error(f"❌ MathLib v1 Core Functionality: FAILED - {e}")
            self.fail(f"MathLib v1 test failed: {e}")
    
    def test_mathlib_v2_advanced_functionality(self):
        """Test mathlib v2 advanced features"""
        logger.info("Testing MathLib v2 Advanced Functionality...")
        
        try:
            from core.mathlib_v2 import CoreMathLibV2, SmartStop, klein_bottle_collapse
            
            # Initialize mathlib v2
            mathlib_v2 = CoreMathLibV2()
            
            # Test VWAP calculation
            test_prices = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
            test_volumes = np.array([1000.0, 1500.0, 800.0, 2000.0, 1200.0])
            
            vwap = mathlib_v2.calculate_vwap(test_prices, test_volumes)
            self.assertEqual(len(vwap), len(test_prices))
            self.assertTrue(np.all(np.isfinite(vwap)))
            
            # Test RSI calculation
            rsi = mathlib_v2.calculate_rsi(test_prices)
            self.assertEqual(len(rsi), len(test_prices))
            self.assertTrue(np.all((rsi >= 0) & (rsi <= 100)))
            
            # Test True Range calculation
            test_high = test_prices * 1.02
            test_low = test_prices * 0.98
            test_close = test_prices
            
            tr = mathlib_v2.calculate_true_range(test_high, test_low, test_close)
            self.assertEqual(len(tr), len(test_prices))
            self.assertTrue(np.all(tr >= 0))
            
            # Test ATR calculation
            atr = mathlib_v2.calculate_atr(test_high, test_low, test_close)
            self.assertEqual(len(atr), len(test_prices))
            self.assertTrue(np.all(atr >= 0))
            
            # Test SmartStop functionality
            smart_stop = SmartStop()
            entry_price = 100.0
            current_price = 105.0
            
            stop_result = smart_stop.update(current_price, entry_price)
            self.assertIn('stop_price', stop_result)
            self.assertIn('profit_pct', stop_result)
            self.assertIn('should_exit', stop_result)
            
            # Test Klein bottle collapse
            klein_result = klein_bottle_collapse()
            self.assertIsInstance(klein_result, np.ndarray)
            
            self.test_results['mathlib_tests']['v2_advanced'] = True
            logger.info("✅ MathLib v2 Advanced Functionality: PASSED")
            
        except Exception as e:
            self.test_results['mathlib_tests']['v2_advanced'] = False
            logger.error(f"❌ MathLib v2 Advanced Functionality: FAILED - {e}")
            self.fail(f"MathLib v2 test failed: {e}")
    
    def test_mathlib_v3_sustainment_framework(self):
        """Test mathlib v3 sustainment framework"""
        logger.info("Testing MathLib v3 Sustainment Framework...")
        
        try:
            from core.mathlib_v3 import (
                SustainmentMathLib, SustainmentVector, MathematicalContext,
                SustainmentPrinciple, create_test_context
            )
            
            # Initialize sustainment mathlib
            sustainment_lib = SustainmentMathLib()
            
            # Create test context
            context = create_test_context()
            self.assertIsInstance(context, MathematicalContext)
            
            # Test sustainment vector calculation
            sustainment_vector = sustainment_lib.calculate_sustainment_vector(context)
            self.assertIsInstance(sustainment_vector, SustainmentVector)
            self.assertEqual(len(sustainment_vector.principles), 8)
            self.assertEqual(len(sustainment_vector.confidence), 8)
            
            # Test sustainment index
            si_value = sustainment_vector.sustainment_index()
            self.assertIsInstance(si_value, float)
            self.assertGreaterEqual(si_value, 0.0)
            self.assertLessEqual(si_value, 1.0)
            
            # Test sustainability check
            is_sustainable = sustainment_vector.is_sustainable()
            self.assertIsInstance(is_sustainable, bool)
            
            # Test individual principles
            for principle in SustainmentPrinciple:
                principle_value = sustainment_vector.principles[principle.value]
                self.assertIsInstance(principle_value, (float, np.floating))
                self.assertGreaterEqual(principle_value, 0.0)
                self.assertLessEqual(principle_value, 1.0)
            
            # Test failing principles detection
            failing = sustainment_vector.failing_principles()
            self.assertIsInstance(failing, list)
            
            # Test sustainment-aware trading decision
            test_prices = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
            test_volumes = np.array([1000.0, 1500.0, 800.0, 2000.0, 1200.0])
            
            trading_decision = sustainment_lib.sustainment_aware_trading_decision(
                test_prices, test_volumes, context
            )
            self.assertIn('decision', trading_decision)
            self.assertIn('sustainment_score', trading_decision)
            self.assertIn('confidence', trading_decision)
            
            self.test_results['mathlib_tests']['v3_sustainment'] = True
            logger.info("✅ MathLib v3 Sustainment Framework: PASSED")
            
        except Exception as e:
            self.test_results['mathlib_tests']['v3_sustainment'] = False
            logger.error(f"❌ MathLib v3 Sustainment Framework: FAILED - {e}")
            self.fail(f"MathLib v3 test failed: {e}")
    
    # ===== NCCO CORE TESTS =====
    
    def test_ncco_core_functionality(self):
        """Test NCCO core system functionality"""
        logger.info("Testing NCCO Core Functionality...")
        
        try:
            from ncco_core import NCCO, generate_nccos, score_nccos
            from ncco_core.ferris_rde import FerrisRDE
            from ncco_core.rde_core import RDEEngine
            
            # Test NCCO creation
            ncco = NCCO(
                id=1,
                price_delta=5.0,
                base_price=100.0,
                bit_mode=8,
                score=0.75,
                pre_commit_id="test_commit_123"
            )
            
            self.assertEqual(ncco.id, 1)
            self.assertEqual(ncco.price_delta, 5.0)
            self.assertEqual(ncco.base_price, 100.0)
            self.assertEqual(ncco.bit_mode, 8)
            self.assertEqual(ncco.score, 0.75)
            
            # Test NCCO string representation
            ncco_str = str(ncco)
            self.assertIn("NCCO", ncco_str)
            self.assertIn("id=1", ncco_str)
            
            # Test NCCO generation
            generated_nccos = generate_nccos(5)
            self.assertEqual(len(generated_nccos), 5)
            for generated_ncco in generated_nccos:
                self.assertIsInstance(generated_ncco, NCCO)
            
            # Test NCCO scoring
            scores = score_nccos(generated_nccos)
            self.assertEqual(len(scores), 5)
            for score in scores:
                self.assertIsInstance(score, (int, float))
            
            # Test RDE Engine
            rde_engine = RDEEngine()
            self.assertIsNotNone(rde_engine)
            
            # Test Ferris RDE
            ferris_rde = FerrisRDE()
            self.assertIsNotNone(ferris_rde)
            
            self.test_results['ncco_tests']['core_functionality'] = True
            logger.info("✅ NCCO Core Functionality: PASSED")
            
        except Exception as e:
            self.test_results['ncco_tests']['core_functionality'] = False
            logger.error(f"❌ NCCO Core Functionality: FAILED - {e}")
            self.fail(f"NCCO core test failed: {e}")
    
    # ===== ALEPH CORE TESTS =====
    
    def test_aleph_core_functionality(self):
        """Test Aleph core system functionality"""
        logger.info("Testing Aleph Core Functionality...")
        
        try:
            from aleph_core import (
                AlephUnitizer, TesseractPortal, PatternMatcher, 
                EntropyAnalyzer, BatchIntegrator
            )
            
            # Test AlephUnitizer
            unitizer = AlephUnitizer()
            self.assertIsNotNone(unitizer)
            
            # Test with sample data
            test_data = [
                {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},
                {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5},
            ]
            
            validation_result = unitizer.validate_units(test_data)
            self.assertIn('math_validation', validation_result)
            self.assertIn('dormant_states', validation_result)
            self.assertIn('timestamp', validation_result)
            
            # Test TesseractPortal
            tesseract_portal = TesseractPortal()
            self.assertIsNotNone(tesseract_portal)
            
            # Test PatternMatcher
            pattern_matcher = PatternMatcher()
            self.assertIsNotNone(pattern_matcher)
            
            # Test EntropyAnalyzer  
            entropy_analyzer = EntropyAnalyzer()
            self.assertIsNotNone(entropy_analyzer)
            
            # Test BatchIntegrator
            batch_integrator = BatchIntegrator()
            self.assertIsNotNone(batch_integrator)
            
            self.test_results['aleph_tests']['core_functionality'] = True
            logger.info("✅ Aleph Core Functionality: PASSED")
            
        except Exception as e:
            self.test_results['aleph_tests']['core_functionality'] = False
            logger.error(f"❌ Aleph Core Functionality: FAILED - {e}")
            self.fail(f"Aleph core test failed: {e}")
    
    # ===== INTEGRATION TESTS =====
    
    def test_cross_system_integration(self):
        """Test integration between all core systems"""
        logger.info("Testing Cross-System Integration...")
        
        try:
            # Import components from all systems
            from core.mathlib_v3 import SustainmentMathLib, create_test_context
            from ncco_core import NCCO, generate_nccos
            from aleph_core import AlephUnitizer
            
            # Initialize all systems
            sustainment_lib = SustainmentMathLib()
            context = create_test_context()
            nccos = generate_nccos(3)
            unitizer = AlephUnitizer()
            
            # Test data flow between systems
            test_data = []
            for i, ncco in enumerate(nccos):
                test_data.append({
                    'feature1': ncco.price_delta,
                    'feature2': ncco.base_price / 100.0,
                    'feature3': ncco.score
                })
            
            # Validate through Aleph
            aleph_result = unitizer.validate_units(test_data)
            self.assertIsNotNone(aleph_result)
            
            # Process through sustainment framework
            sustainment_vector = sustainment_lib.calculate_sustainment_vector(context)
            self.assertIsNotNone(sustainment_vector)
            
            # Test mathematical consistency
            test_prices = np.array([ncco.base_price for ncco in nccos])
            test_volumes = np.array([abs(ncco.price_delta) * 100 for ncco in nccos])
            
            trading_decision = sustainment_lib.sustainment_aware_trading_decision(
                test_prices, test_volumes, context
            )
            self.assertIn('decision', trading_decision)
            
            self.test_results['integration_tests']['cross_system'] = True
            logger.info("✅ Cross-System Integration: PASSED")
            
        except Exception as e:
            self.test_results['integration_tests']['cross_system'] = False
            logger.error(f"❌ Cross-System Integration: FAILED - {e}")
            self.fail(f"Cross-system integration test failed: {e}")
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency across all libraries"""
        logger.info("Testing Mathematical Consistency...")
        
        try:
            from core.mathlib import CoreMathLib
            from core.mathlib_v2 import CoreMathLibV2
            from core.mathlib_v3 import SustainmentMathLib
            
            # Initialize all math libraries
            mathlib_v1 = CoreMathLib()
            mathlib_v2 = CoreMathLibV2()
            mathlib_v3 = SustainmentMathLib()
            
            # Test vector operations consistency
            test_vector_a = np.array([1.0, 2.0, 3.0])
            test_vector_b = np.array([4.0, 5.0, 6.0])
            
            # Compare cosine similarity across versions
            sim_v1 = mathlib_v1.cosine_similarity(test_vector_a, test_vector_b)
            sim_v2 = mathlib_v2.cosine_similarity(test_vector_a, test_vector_b)
            
            np.testing.assert_almost_equal(sim_v1, sim_v2, decimal=6)
            
            # Compare distance calculations
            dist_v1 = mathlib_v1.euclidean_distance(test_vector_a, test_vector_b)
            dist_v2 = mathlib_v2.euclidean_distance(test_vector_a, test_vector_b)
            
            np.testing.assert_almost_equal(dist_v1, dist_v2, decimal=6)
            
            # Test entropy calculations
            test_prices = np.array([100.0, 102.0, 98.0, 105.0, 97.0])
            test_distribution = test_prices / np.sum(test_prices)
            
            entropy_v1 = mathlib_v1.shell_entropy(test_distribution)
            entropy_v2 = mathlib_v2.shell_entropy(test_distribution) if hasattr(mathlib_v2, 'shell_entropy') else entropy_v1
            
            self.assertIsInstance(entropy_v1, float)
            self.assertGreaterEqual(entropy_v1, 0.0)
            
            self.test_results['mathematical_validation']['consistency'] = True
            logger.info("✅ Mathematical Consistency: PASSED")
            
        except Exception as e:
            self.test_results['mathematical_validation']['consistency'] = False
            logger.error(f"❌ Mathematical Consistency: FAILED - {e}")
            self.fail(f"Mathematical consistency test failed: {e}")
    
    def test_core_dependencies(self):
        """Test that all core dependencies are properly resolved"""
        logger.info("Testing Core Dependencies...")
        
        try:
            # Test core module imports
            from core.math_core import UnifiedMathematicalProcessor, MATHEMATICAL_CONSTANTS
            from core.dormant_engine import DormantStateLearningEngine, DormantState
            
            # Test math core
            processor = UnifiedMathematicalProcessor()
            self.assertIsNotNone(processor)
            
            # Test mathematical constants
            self.assertIsNotNone(MATHEMATICAL_CONSTANTS)
            self.assertIsInstance(MATHEMATICAL_CONSTANTS, dict)
            
            # Test dormant engine
            dormant_engine = DormantStateLearningEngine()
            self.assertIsNotNone(dormant_engine)
            
            # Test dormant state creation
            dormant_state = DormantState(
                state_id=1,
                features=np.array([1.0, 2.0, 3.0]),
                label="test_state",
                confidence=0.95,
                timestamp=time.time()
            )
            self.assertIsNotNone(dormant_state)
            self.assertEqual(dormant_state.state_id, 1)
            
            self.test_results['integration_tests']['dependencies'] = True
            logger.info("✅ Core Dependencies: PASSED")
            
        except Exception as e:
            self.test_results['integration_tests']['dependencies'] = False
            logger.error(f"❌ Core Dependencies: FAILED - {e}")
            self.fail(f"Core dependencies test failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Generate final test report"""
        cls._generate_test_report()
    
    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CORE SYSTEMS INTEGRATION TEST REPORT")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for category, tests in cls.test_results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            for test_name, result in tests.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"  {test_name}: {status}")
                total_tests += 1
                if result:
                    passed_tests += 1
        
        print(f"\n{'='*80}")
        print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"SUCCESS RATE: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("🎉 EXCELLENT: All core systems are functioning optimally!")
        elif success_rate >= 80:
            print("✅ GOOD: Core systems are mostly functional with minor issues")
        elif success_rate >= 60:
            print("⚠️  WARNING: Some core systems have significant issues")
        else:
            print("❌ CRITICAL: Major issues detected in core systems")
        
        print("="*80)

if __name__ == '__main__':
    # Run the comprehensive test suite
    unittest.main(verbosity=2) 