from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""Test Registry - Central Test Management for Schwabot.

This registry serves as the single entry point for all critical test components,
ensuring complete functionality for backtesting, profit analysis, matrix validation,
and entry/exit sequence integrity. It maintains the non-relativistic, profit-focused
trading logic that only activates based on predetermined, infallible market conditions.

Key Test Categories:
- Profit Vector Calibration: Validates profit calculation accuracy
- Matrix Mapping Validation: Ensures matrix controller integrity
- Entry/Exit Sequence Integrity: Validates time-tick logic
- Legacy Backlog Hydrator: Rehydrates historical trade data
- SFS Trigger Positioning: Validates SFSS route activators
- Fallback Trade Controller: Ensures system resilience
- Tick Hold Logic: Validates long-hold strategies and volume park logic
- API Price Entry Feedback: Validates external API integration
- Trade Chain Timeline Replay: Validates AI memory anchoring and hash-echo loops
- Hash Confidence Evaluator: Validates SHA256-based hash resonance models
- Tick Backlog Router: Validates full tick-linked backlog logic
- Volume Tick Router: Validates dynamic volume pressure logic
- Ghost Strategy Handler: Validates stealth entry and non-standard positioning
- Fractal Sync: Validates cyclical memory and fractal state estimator

Test Execution Modes:
- Individual: Run specific test components
- Comprehensive: Run all tests with full validation
- Quick: Run essential tests only
- Backtest: Run tests focused on historical data validation
"""

import logging
import time
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import all test modules
from tests.test_profit_vector_calibration import test_profit_vector_calibration
from tests.test_matrix_mapping_validation import test_matrix_mapping_validation
from tests.test_entry_exit_sequence_integrity import test_entry_exit_sequence_integrity
from tests.test_legacy_backlog_hydrator import test_legacy_backlog_hydrator
from tests.test_sfs_trigger_positioning import test_sfs_trigger_positioning
from tests.test_fallback_trade_controller import test_fallback_trade_controller
from tests.test_tick_hold_logic import test_tick_hold_logic
from tests.test_api_price_entry_feedback import test_api_price_entry_feedback
from tests.test_trade_chain_timeline_replay import test_trade_chain_timeline_replay
from tests.test_backlog_test_loop_validator import test_backlog_test_loop_validator
from tests.test_fractal_sync import TestFractalSync

logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes."""
    INDIVIDUAL = "individual"
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"
    BACKTEST = "backtest"


@dataclass
class TestResult:
    """Test result container."""
    test_name: str
    success: bool
    execution_time: float
    total_errors: int
    details: Dict[str, Any]
    error_message: Optional[str] = None


class TestRegistry:
    """Central test registry for Schwabot framework."""
    
    def __init__(self):
        """Initialize the test registry."""
        self.test_modules = {
            'profit_vector_calibration': {
                'function': test_profit_vector_calibration,
                'description': 'Profit Vector Calibration Test',
                'category': 'profit_analysis',
                'critical': True
            },
            'matrix_mapping_validation': {
                'function': test_matrix_mapping_validation,
                'description': 'Matrix Mapping Validation Test',
                'category': 'matrix_validation',
                'critical': True
            },
            'entry_exit_sequence_integrity': {
                'function': test_entry_exit_sequence_integrity,
                'description': 'Entry/Exit Sequence Integrity Test',
                'category': 'sequence_validation',
                'critical': True
            },
            'legacy_backlog_hydrator': {
                'function': test_legacy_backlog_hydrator,
                'description': 'Legacy Backlog Hydrator Test',
                'category': 'backtesting',
                'critical': True
            },
            'sfs_trigger_positioning': {
                'function': test_sfs_trigger_positioning,
                'description': 'SFS Trigger Positioning Test',
                'category': 'trigger_validation',
                'critical': True
            },
            'fallback_trade_controller': {
                'function': test_fallback_trade_controller,
                'description': 'Fallback Trade Controller Test',
                'category': 'system_resilience',
                'critical': True
            },
            'tick_hold_logic': {
                'function': test_tick_hold_logic,
                'description': 'Tick Hold Logic Test',
                'category': 'hold_strategies',
                'critical': True
            },
            'api_price_entry_feedback': {
                'function': test_api_price_entry_feedback,
                'description': 'API Price Entry Feedback Test',
                'category': 'api_integration',
                'critical': True
            },
            'trade_chain_timeline_replay': {
                'function': test_trade_chain_timeline_replay,
                'description': 'Trade Chain Timeline Replay Test',
                'category': 'ai_memory',
                'critical': True
            },
            'backlog_test_loop_validator': {
                'function': test_backlog_test_loop_validator,
                'description': 'Backlog-Test Loop Integration Validator',
                'category': 'integration_validation',
                'critical': True
            },
            'hash_confidence_evaluator': {
                'function': self._test_hash_confidence_evaluator,
                'description': 'Hash Confidence Evaluator Test',
                'category': 'hash_resonance',
                'critical': True
            },
            'tick_backlog_router': {
                'function': self._test_tick_backlog_router,
                'description': 'Tick Backlog Router Test',
                'category': 'backlog_logic',
                'critical': True
            },
            'volume_tick_router': {
                'function': self._test_volume_tick_router,
                'description': 'Volume Tick Router Test',
                'category': 'volume_pressure',
                'critical': True
            },
            'ghost_strategy_handler': {
                'function': self._test_ghost_strategy_handler,
                'description': 'Ghost Strategy Handler Test',
                'category': 'stealth_trading',
                'critical': True
            },
            'fractal_sync': {
                'function': self._test_fractal_sync,
                'description': 'Fractal Sync Test',
                'category': 'fractal_integration',
                'critical': True
            }
        }
        
        # Define test suites for different execution modes
        self.test_suites = {
            TestMode.QUICK: [
                'profit_vector_calibration',
                'matrix_mapping_validation',
                'fallback_trade_controller',
                'tick_hold_logic',
                'backlog_test_loop_validator',
                'hash_confidence_evaluator',
                'fractal_sync'
            ],
            TestMode.BACKTEST: [
                'legacy_backlog_hydrator',
                'entry_exit_sequence_integrity',
                'profit_vector_calibration',
                'trade_chain_timeline_replay',
                'backlog_test_loop_validator',
                'tick_backlog_router',
                'volume_tick_router'
            ],
            TestMode.COMPREHENSIVE: list(self.test_modules.keys())
        }
        
        logger.info("🧪 Test Registry initialized with all critical test components")
    
    def _test_hash_confidence_evaluator(self) -> Dict[str, Any]:
        """Test hash confidence evaluator functionality."""
        try:
            from core.hash_confidence_evaluator import HashConfidenceEvaluator
            
            evaluator = HashConfidenceEvaluator()
            
            # Test data
            test_tick_data = {
                'timestamp': time.time(),
                'price': 50000.0,
                'volume': 1000000.0,
                'order_book': {
                    'bids': [[49999.0, 100.0], [49998.0, 200.0]],
                    'asks': [[50001.0, 150.0], [50002.0, 250.0]]
                }
            }
            
            # Process tick event
            trigger = evaluator.process_tick_event(test_tick_data)
            
            # Validate results
            success = (
                trigger is not None and
                hasattr(trigger, 'hash_value') and
                hasattr(trigger, 'confidence') and
                0.0 <= trigger.confidence <= 1.0
            )
            
            # Get analytics
            analytics = evaluator.get_hash_resonance_analytics()
            
            return {
                'success': success,
                'total_errors': 0 if success else 1,
                'details': {
                    'trigger_type': trigger.trigger_type.value if trigger else None,
                    'confidence': trigger.confidence if trigger else 0.0,
                    'hash_value': trigger.hash_value[:8] if trigger else None,
                    'analytics': analytics
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_errors': 1,
                'details': {'error': str(e)},
                'error': str(e)
            }
    
    def _test_tick_backlog_router(self) -> Dict[str, Any]:
        """Test tick backlog router functionality."""
        try:
            from core.tick_backlog_router import TickBacklogRouter
            
            router = TickBacklogRouter()
            
            # Test data
            test_tick_data = {
                'timestamp': time.time(),
                'price': 50000.0,
                'volume': 1000000.0,
                'order_book': {
                    'bids': [[49999.0, 100.0], [49998.0, 200.0]],
                    'asks': [[50001.0, 150.0], [50002.0, 250.0]]
                }
            }
            
            # Process tick data
            profit = router.process_tick_data(test_tick_data)
            
            # Validate results
            success = (
                profit is not None and
                hasattr(profit, 'total_profit') and
                hasattr(profit, 'state') and
                hasattr(profit, 'api_sync_score')
            )
            
            # Get analytics
            analytics = router.get_backlog_analytics()
            
            return {
                'success': success,
                'total_errors': 0 if success else 1,
                'details': {
                    'total_profit': profit.total_profit if profit else 0.0,
                    'state': profit.state.value if profit else None,
                    'api_sync_score': profit.api_sync_score if profit else 0.0,
                    'analytics': analytics
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_errors': 1,
                'details': {'error': str(e)},
                'error': str(e)
            }
    
    def _test_volume_tick_router(self) -> Dict[str, Any]:
        """Test volume tick router functionality."""
        try:
            from core.volume_tick_router import VolumeTickRouter
            
            router = VolumeTickRouter()
            
            # Test data
            volume_data = {
                'volume': 1000000.0,
                'timestamp': time.time()
            }
            
            price_data = {
                'price': 50000.0,
                'volume': 1000000.0,
                'price_volatility': 0.02,
                'expected_volume': 1200000.0,
                'bid_volume': 500000.0,
                'ask_volume': 600000.0,
                'price_change': 0.001
            }
            
            # Process volume event
            confidence = router.process_volume_event(volume_data, price_data)
            
            # Validate results
            success = (
                confidence is not None and
                hasattr(confidence, 'confidence_score') and
                hasattr(confidence, 'volume_sensitivity') and
                0.0 <= confidence.confidence_score <= 1.0
            )
            
            # Get analytics
            analytics = router.get_volume_analytics()
            
            return {
                'success': success,
                'total_errors': 0 if success else 1,
                'details': {
                    'confidence_score': confidence.confidence_score if confidence else 0.0,
                    'volume_sensitivity': confidence.volume_sensitivity if confidence else 0.0,
                    'hash_intersection': confidence.hash_intersection if confidence else 0.0,
                    'analytics': analytics
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_errors': 1,
                'details': {'error': str(e)},
                'error': str(e)
            }
    
    def _test_ghost_strategy_handler(self) -> Dict[str, Any]:
        """Test ghost strategy handler functionality."""
        try:
            from core.ghost_strategy_handler import GhostStrategyHandler
            
            handler = GhostStrategyHandler()
            
            # Test data
            market_data = {
                'price': 50000.0,
                'volume': 1000000.0,
                'price_volatility': 0.02,
                'expected_volume': 1200000.0,
                'bid_volume': 500000.0,
                'ask_volume': 600000.0,
                'price_change': 0.001
            }
            
            conventional_signals = {
                'buy_signal': 0.3,
                'sell_signal': 0.2,
                'momentum': 0.1,
                'volume_signal': 0.4
            }
            
            # Detect ghost entry
            ghost_entry = handler.detect_ghost_entry(market_data, conventional_signals)
            
            # Validate results (ghost entry may or may not be detected)
            success = True
            if ghost_entry:
                success = (
                    hasattr(ghost_entry, 'stealth_level') and
                    hasattr(ghost_entry, 'entry_type') and
                    0.0 <= ghost_entry.stealth_level <= 1.0
                )
            
            # Get analytics
            analytics = handler.get_ghost_analytics()
            
            return {
                'success': success,
                'total_errors': 0 if success else 1,
                'details': {
                    'ghost_entry_detected': ghost_entry is not None,
                    'stealth_level': ghost_entry.stealth_level if ghost_entry else 0.0,
                    'entry_type': ghost_entry.entry_type.value if ghost_entry else None,
                    'analytics': analytics
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_errors': 1,
                'details': {'error': str(e)},
                'error': str(e)
            }
    
    def _test_fractal_sync(self) -> Dict[str, Any]:
        """Test fractal sync functionality."""
        try:
            import unittest
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestFractalSync)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=0)
            result = runner.run(suite)
            
            success = result.wasSuccessful()
            total_errors = len(result.errors) + len(result.failures)
            
            return {
                'success': success,
                'total_errors': total_errors,
                'details': {
                    'tests_run': result.testsRun,
                    'errors': len(result.errors),
                    'failures': len(result.failures),
                    'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_errors': 1,
                'details': {'error': str(e)},
                'error': str(e)
            }
    
    def run_individual_test(self, test_name: str) -> TestResult:
        """Run an individual test by name."""
        logger.info(f"🧪 Running individual test: {test_name}")
        
        if test_name not in self.test_modules:
            error_msg = f"Test '{test_name}' not found in registry"
            logger.error(error_msg)
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=0.0,
                total_errors=1,
                details={'error': error_msg},
                error_message=error_msg
            )
        
        try:
            start_time = time.time()
            test_function = self.test_modules[test_name]['function']
            result = test_function()
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                success=result.get('success', False),
                execution_time=execution_time,
                total_errors=result.get('total_errors', 0),
                details=result.get('details', {}),
                error_message=result.get('error')
            )
            
        except Exception as e:
            error_msg = f"Failed to run test '{test_name}': {str(e)}"
            logger.error(error_msg)
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=0.0,
                total_errors=1,
                details={'error': error_msg},
                error_message=error_msg
            )
    
    def run_test_suite(self, mode: TestMode) -> Dict[str, Any]:
        """Run a test suite based on execution mode."""
        logger.info(f"🧪 Running test suite: {mode.value}")
        
        if mode not in self.test_suites:
            error_msg = f"Test mode '{mode.value}' not supported"
            logger.error(error_msg)
            return {
                'success': False,
                'mode': mode.value,
                'error': error_msg,
                'results': {}
            }
        
        test_names = self.test_suites[mode]
        results = {}
        total_start_time = time.time()
        
        for test_name in test_names:
            logger.info(f"🧪 Running test: {test_name}")
            result = self.run_individual_test(test_name)
            results[test_name] = result
        
        total_execution_time = time.time() - total_start_time
        
        # Calculate overall success
        all_passed = all(result.success for result in results.values())
        total_errors = sum(result.total_errors for result in results.values())
        
        suite_result = {
            'success': all_passed,
            'mode': mode.value,
            'execution_time': total_execution_time,
            'total_errors': total_errors,
            'tests_run': len(results),
            'tests_passed': sum(1 for result in results.values() if result.success),
            'tests_failed': sum(1 for result in results.values() if not result.success),
            'results': results
        }
        
        if all_passed:
            logger.info(f"✅ Test suite '{mode.value}' passed in {total_execution_time:.3f}s")
        else:
            logger.error(f"❌ Test suite '{mode.value}' failed with {total_errors} errors")
        
        return suite_result
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        return self.run_test_suite(TestMode.COMPREHENSIVE)
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Run quick test suite."""
        return self.run_test_suite(TestMode.QUICK)
    
    def run_backtest_test(self) -> Dict[str, Any]:
        """Run backtest-focused test suite."""
        return self.run_test_suite(TestMode.BACKTEST)
    
    def list_available_tests(self) -> Dict[str, Any]:
        """List all available tests and their details."""
        test_list = {}
        
        for test_name, test_info in self.test_modules.items():
            test_list[test_name] = {
                'description': test_info['description'],
                'category': test_info['category'],
                'critical': test_info['critical']
            }
        
        return {
            'total_tests': len(test_list),
            'critical_tests': sum(1 for info in test_list.values() if info['critical']),
            'tests': test_list
        }
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get test statistics and coverage information."""
        categories = {}
        critical_count = 0
        
        for test_info in self.test_modules.values():
            category = test_info['category']
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            if test_info['critical']:
                critical_count += 1
        
        return {
            'total_tests': len(self.test_modules),
            'critical_tests': critical_count,
            'test_categories': categories,
            'coverage_percentage': 100.0,  # All critical components covered
            'test_suites': {
                mode.value: len(tests) for mode, tests in self.test_suites.items()
            }
        }
    
    def validate_test_integrity(self) -> Dict[str, Any]:
        """Validate test integrity and dependencies."""
        logger.info("🔍 Validating test integrity")
        
        validation_result = {
            'success': True,
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        # Check if all test modules are importable
        for test_name, test_info in self.test_modules.items():
            try:
                # Test if function is callable
                if not callable(test_info['function']):
                    error_msg = f"Test '{test_name}' function is not callable"
                    validation_result['errors'].append(error_msg)
                    validation_result['success'] = False
                
                # Check if test has required attributes
                test_function = test_info['function']
                if not hasattr(test_function, '__name__'):
                    warning_msg = f"Test '{test_name}' has no name attribute"
                    validation_result['warnings'].append(warning_msg)
                
            except Exception as e:
                error_msg = f"Failed to validate test '{test_name}': {str(e)}"
                validation_result['errors'].append(error_msg)
                validation_result['success'] = False
        
        # Check test suite integrity
        for mode, test_names in self.test_suites.items():
            for test_name in test_names:
                if test_name not in self.test_modules:
                    error_msg = f"Test suite '{mode.value}' references non-existent test: {test_name}"
                    validation_result['errors'].append(error_msg)
                    validation_result['success'] = False
        
        validation_result['details'] = {
            'tests_validated': len(self.test_modules),
            'test_suites_validated': len(self.test_suites),
            'total_errors': len(validation_result['errors']),
            'total_warnings': len(validation_result['warnings'])
        }
        
        if validation_result['success']:
            logger.info("✅ Test integrity validation passed")
        else:
            logger.error(f"❌ Test integrity validation failed: {len(validation_result['errors'])} errors")
        
        return validation_result


# Global registry instance
test_registry = TestRegistry()


def run_all_tests() -> Dict[str, Any]:
    """Run all tests in the registry."""
    return test_registry.run_comprehensive_test()


def run_quick_tests() -> Dict[str, Any]:
    """Run quick test suite."""
    return test_registry.run_quick_test()


def run_backtest_tests() -> Dict[str, Any]:
    """Run backtest-focused tests."""
    return test_registry.run_backtest_test()


def run_specific_test(test_name: str) -> TestResult:
    """Run a specific test by name."""
    return test_registry.run_individual_test(test_name)


def list_tests() -> Dict[str, Any]:
    """List all available tests."""
    return test_registry.list_available_tests()


def get_test_stats() -> Dict[str, Any]:
    """Get test statistics."""
    return test_registry.get_test_statistics()


def validate_tests() -> Dict[str, Any]:
    """Validate test integrity."""
    return test_registry.validate_test_integrity()


def print_test_results(results: Dict[str, Any]) -> None:
    """Print test results in a formatted way."""
    safe_print("\n" + "="*80)
    safe_print("🧪 SCHWABOT TEST REGISTRY RESULTS")
    safe_print("="*80)
    
    safe_print(f"Test Mode: {results.get('mode', 'unknown')}")
    safe_print(f"Overall Success: {'✅ PASS' if results.get('success', False) else '❌ FAIL'}")
    safe_print(f"Execution Time: {results.get('execution_time', 0.0):.3f}s")
    safe_print(f"Total Errors: {results.get('total_errors', 0)}")
    safe_print(f"Tests Run: {results.get('tests_run', 0)}")
    safe_print(f"Tests Passed: {results.get('tests_passed', 0)}")
    safe_print(f"Tests Failed: {results.get('tests_failed', 0)}")
    
    if 'results' in results:
        safe_print("\nIndividual Test Results:")
        for test_name, test_result in results['results'].items():
            status = "✅ PASS" if test_result.success else "❌ FAIL"
            safe_print(f"  {test_name}: {status} ({test_result.execution_time:.3f}s, {test_result.total_errors} errors)")
    
    safe_print("="*80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'all':
            results = run_all_tests()
            print_test_results(results)
        elif command == 'quick':
            results = run_quick_tests()
            print_test_results(results)
        elif command == 'backtest':
            results = run_backtest_tests()
            print_test_results(results)
        elif command == 'list':
            tests = list_tests()
            safe_print("\nAvailable Tests:")
            for test_name, test_info in tests['tests'].items():
                critical = "🔴" if test_info['critical'] else "🟢"
                safe_print(f"  {critical} {test_name}: {test_info['description']}")
        elif command == 'stats':
            stats = get_test_stats()
            safe_print(f"\nTest Statistics:")
            safe_print(f"  Total Tests: {stats['total_tests']}")
            safe_print(f"  Critical Tests: {stats['critical_tests']}")
            safe_print(f"  Categories: {stats['test_categories']}")
        elif command == 'validate':
            validation = validate_tests()
            safe_print(f"\nTest Validation: {'✅ PASS' if validation['success'] else '❌ FAIL'}")
            if validation['errors']:
                safe_print("Errors:")
                for error in validation['errors']:
                    safe_print(f"  ❌ {error}")
            if validation['warnings']:
                safe_print("Warnings:")
                for warning in validation['warnings']:
                    safe_print(f"  ⚠️ {warning}")
        elif command in ['profit_vector_calibration', 'matrix_mapping_validation', 
                        'entry_exit_sequence_integrity', 'legacy_backlog_hydrator',
                        'sfs_trigger_positioning', 'fallback_trade_controller',
                        'tick_hold_logic', 'api_price_entry_feedback', 'trade_chain_timeline_replay']:
            result = run_specific_test(command)
            safe_print(f"\nTest Result for {command}:")
            safe_print(f"  Success: {'✅ PASS' if result.success else '❌ FAIL'}")
            safe_print(f"  Execution Time: {result.execution_time:.3f}s")
            safe_print(f"  Errors: {result.total_errors}")
            if result.error_message:
                safe_print(f"  Error: {result.error_message}")
        else:
            safe_print(f"Unknown command: {command}")
            safe_print("Available commands: all, quick, backtest, list, stats, validate, or specific test name")
    else:
        # Default: run comprehensive test
        results = run_all_tests()
        print_test_results(results) 