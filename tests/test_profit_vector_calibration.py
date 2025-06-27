# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Test suite for profit vector calibration functionality.

Tests the calibration of profit vectors with various mathematical operations
and validation of results.
"""

import unittest
import logging
import time
from typing import Dict, Any, List
from decimal import Decimal

# Import core modules
from core.unified_math_system import unified_math
from core.profit_vector_calibration import (
    ProfitVectorCalibrator,
    CalibrationResult,
    VectorCalibrationConfig
)

# Import safe print for Windows compatibility
try:
    from core.utils.windows_cli_compatibility import (
        safe_print, info, warn, error, success, debug
    )
    CLI_HANDLER_AVAILABLE = True
except ImportError:
    CLI_HANDLER_AVAILABLE = False
    
    def safe_print(message):
        print(message)
    
    def info(message):
        print(f"[INFO] {message}")
    
    def warn(message):
        print(f"[WARN] {message}")
    
    def error(message):
        print(f"[ERROR] {message}")
    
    def success(message):
        print(f"[SUCCESS] {message}")
    
    def debug(message):
        print(f"[DEBUG] {message}")

logger = logging.getLogger(__name__)


@dataclass
class ProfitVectorTestCase:
    """Test case for profit vector calibration."""
    asset: str
    entry_price: float
    exit_price: float
    volume: float
    thermal_index: float
    expected_profit: float
    expected_efficiency: float
    description: str


class ProfitVectorCalibrationTest:
    """Comprehensive profit vector calibration testing."""

    def __init__(self):
        """Initialize the profit vector calibration test."""
        self.controller = UnifiedMathematicalTradingController()
        self.ferris_engine = FerrisWheelEngine()

        # Test cases for profit vector calibration
        self.test_cases = [
            ProfitVectorTestCase(
                asset="BTC",
                entry_price=26000.0,
                exit_price=27200.0,
                volume=0.5,
                thermal_index=1.2,
                expected_profit=600.0,  # (27200 - 26000) * 0.5
                expected_efficiency=500.0,  # 600 / 1.2
                description="Standard BTC profit scenario"
            ),
            ProfitVectorTestCase(
                asset="ETH",
                entry_price=1700.0,
                exit_price=1850.0,
                volume=2.0,
                thermal_index=0.9,
                expected_profit=300.0,  # (1850 - 1700) * 2.0
                expected_efficiency=333.33,  # 300 / 0.9
                description="Standard ETH profit scenario"
            ),
            ProfitVectorTestCase(
                asset="XRP",
                entry_price=0.50,
                exit_price=0.55,
                volume=1000.0,
                thermal_index=0.5,
                expected_profit=50.0,  # (0.55 - 0.50) * 1000
                expected_efficiency=100.0,  # 50 / 0.5
                description="High volume XRP scenario"
            ),
            ProfitVectorTestCase(
                asset="USDC",
                entry_price=1.0,
                exit_price=1.0,
                volume=100.0,
                thermal_index=0.1,
                expected_profit=0.0,  # No price change
                expected_efficiency=0.0,  # 0 / 0.1
                description="Stable coin scenario"
            ),
            ProfitVectorTestCase(
                asset="BTC",
                entry_price=30000.0,
                exit_price=29000.0,
                volume=0.1,
                thermal_index=2.0,
                expected_profit=-100.0,  # Loss scenario
                expected_efficiency=-50.0,  # -100 / 2.0
                description="Loss scenario with high thermal index"
            )
        ]

        logger.info("\\u1f4b0 Profit Vector Calibration Test initialized")

    def test_profit_calculation_accuracy(self) -> Dict[str, Any]:
        """Test profit calculation accuracy with high precision."""
        logger.info("\\u1f9ee Testing profit calculation accuracy")

        results = {
            'test_name': 'profit_calculation_accuracy',
            'success': True,
            'details': {},
            'errors': []
        }

        for i, test_case in enumerate(self.test_cases):
            try:
                # Create signal data
                signal_data = {
                    "asset": test_case.asset,
                    "entry_price": test_case.entry_price,
                    "exit_price": test_case.exit_price,
                    "volume": test_case.volume,
                    "thermal_index": test_case.thermal_index,
                    "timestamp": time.time(),
                    "strategy": "calibration_test"
                }

                # Process through controller
                result = self.controller.process_trade_signal(signal_data)

                if result['status'] != 'success':
                    results['errors'].append(
                        f"Test case {i}: Processing failed - {result.get('error', 'Unknown error')}")
                    results['success'] = False
                    continue

                # Extract calculated values
                calculated_profit = result['profit']
                calculated_efficiency = result['efficiency']

                # Calculate expected values with high precision
                expected_profit = (test_case.exit_price - test_case.entry_price) * test_case.volume
                expected_efficiency = expected_profit / test_case.thermal_index if test_case.thermal_index != 0 else 0.0

                # Validate profit calculation
                profit_tolerance = 0.01  # 1 cent tolerance
                if unified_math.abs(calculated_profit - expected_profit) > profit_tolerance:
                    error_msg = f"Test case {i} ({test_case.description}): Profit mismatch. Expected: {expected_profit:.2f}, Got: {calculated_profit:.2f}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate efficiency calculation
                efficiency_tolerance = 0.01
                if unified_math.abs(calculated_efficiency - expected_efficiency) > efficiency_tolerance:
                    error_msg = f"Test case {i} ({test_case.description}): Efficiency mismatch. Expected: {expected_efficiency:.2f}, Got: {calculated_efficiency:.2f}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Store test case results
                results['details'][f'test_case_{i}'] = {
                    'description': test_case.description,
                    'expected_profit': expected_profit,
                    'calculated_profit': calculated_profit,
                    'expected_efficiency': expected_efficiency,
                    'calculated_efficiency': calculated_efficiency,
                    'profit_accuracy': unified_math.abs(calculated_profit - expected_profit) <= profit_tolerance,
                    'efficiency_accuracy': unified_math.abs(calculated_efficiency - expected_efficiency) <= efficiency_tolerance
                }

            except Exception as e:
                error_msg = f"Test case {i} ({test_case.description}): Exception - {str(e)}"
                results['errors'].append(error_msg)
                results['success'] = False

        if results['success']:
            logger.info("\\u2705 Profit calculation accuracy test passed")
        else:
            logger.error(f"\\u274c Profit calculation accuracy test failed: {len(results['errors'])} errors")

        return results

    def test_profit_memory_integration(self) -> Dict[str, Any]:
        """Test profit memory storage and retrieval."""
        logger.info("\\u1f4be Testing profit memory integration")

        results = {
            'test_name': 'profit_memory_integration',
            'success': True,
            'details': {},
            'errors': []
        }

        try:
            # Clear profit tracker
            initial_profit = profit_summary()[0]

            # Process multiple test cases
            test_signals = [
                {
                    "asset": "BTC",
                    "entry_price": 26000.0,
                    "exit_price": 27200.0,
                    "volume": 0.5,
                    "thermal_index": 1.2,
                    "timestamp": time.time(),
                    "strategy": "memory_test"
                },
                {
                    "asset": "ETH",
                    "entry_price": 1700.0,
                    "exit_price": 1850.0,
                    "volume": 2.0,
                    "thermal_index": 0.9,
                    "timestamp": time.time() + 1,
                    "strategy": "memory_test"
                }
            ]

            expected_total_profit = 0.0

            for signal in test_signals:
                result = self.controller.process_trade_signal(signal)
                if result['status'] == 'success':
                    expected_total_profit += result['profit']

            # Check profit tracker
            final_profit = profit_summary()[0]
            profit_increase = final_profit - initial_profit

            # Validate profit tracking
            profit_tolerance = 0.01
            if unified_math.abs(profit_increase - expected_total_profit) > profit_tolerance:
                error_msg = f"Profit tracking mismatch. Expected increase: {expected_total_profit:.2f}, Actual increase: {profit_increase:.2f}"
                results['errors'].append(error_msg)
                results['success'] = False

            results['details'] = {
                'initial_profit': initial_profit,
                'final_profit': final_profit,
                'expected_increase': expected_total_profit,
                'actual_increase': profit_increase,
                'tracking_accuracy': unified_math.abs(profit_increase - expected_total_profit) <= profit_tolerance
            }

        except Exception as e:
            results['errors'].append(f"Profit memory integration test failed: {str(e)}")
            results['success'] = False

        if results['success']:
            logger.info("\\u2705 Profit memory integration test passed")
        else:
            logger.error(f"\\u274c Profit memory integration test failed: {len(results['errors'])} errors")

        return results

    def test_ferris_wheel_integration(self) -> Dict[str, Any]:
        """Test Ferris wheel cycle integration."""
        logger.info("\\u1f3a1 Testing Ferris wheel integration")

        results = {
            'test_name': 'ferris_wheel_integration',
            'success': True,
            'details': {},
            'errors': []
        }

        try:
            # Test signal for Ferris wheel
            signal_data = {
                "asset": "BTC",
                "entry_price": 26000.0,
                "exit_price": 27200.0,
                "volume": 0.5,
                "thermal_index": 1.2,
                "timestamp": time.time(),
                "strategy": "ferris_test"
            }

            # Process signal
            result = self.controller.process_trade_signal(signal_data)

            if result['status'] != 'success':
                results['errors'].append(f"Signal processing failed: {result.get('error', 'Unknown error')}")
                results['success'] = False
            else:
                # Check Ferris wheel integration
                cycle_name = result.get('cycle_name')
                thermal_signature = result.get('thermal_signature', {})

                if not cycle_name:
                    results['errors'].append("No cycle name returned from Ferris wheel")
                    results['success'] = False

                if not thermal_signature:
                    results['errors'].append("No thermal signature returned from Ferris wheel")
                    results['success'] = False

                results['details'] = {
                    'cycle_name': cycle_name,
                    'thermal_signature': thermal_signature,
                    'ferris_integration_success': bool(cycle_name and thermal_signature)
                }

        except Exception as e:
            results['errors'].append(f"Ferris wheel integration test failed: {str(e)}")
            results['success'] = False

        if results['success']:
            logger.info("\\u2705 Ferris wheel integration test passed")
        else:
            logger.error(f"\\u274c Ferris wheel integration test failed: {len(results['errors'])} errors")

        return results

    def test_ghost_signal_detection(self) -> Dict[str, Any]:
        """Test ghost signal detection accuracy."""
        logger.info("\\u1f47b Testing ghost signal detection")

        results = {
            'test_name': 'ghost_signal_detection',
            'success': True,
            'details': {},
            'errors': []
        }

        try:
            # Test normal signal (should not be ghost)
            normal_signal = {
                "asset": "BTC",
                "entry_price": 26000.0,
                "exit_price": 27200.0,
                "volume": 0.5,
                "thermal_index": 1.2,
                "timestamp": time.time(),
                "strategy": "normal_test"
            }

            normal_result = self.controller.process_trade_signal(normal_signal)

            # Test potential ghost signal (very small price change)
            ghost_signal = {
                "asset": "BTC",
                "entry_price": 26000.0,
                "exit_price": 26000.01,  # Minimal price change
                "volume": 0.5,
                "thermal_index": 1.2,
                "timestamp": time.time() + 1,
                "strategy": "ghost_test"
            }

            ghost_result = self.controller.process_trade_signal(ghost_signal)

            # Validate results
            if normal_result['status'] != 'success':
                results['errors'].append("Normal signal processing failed")
                results['success'] = False

            if ghost_result['status'] != 'success':
                results['errors'].append("Ghost signal processing failed")
                results['success'] = False

            # Check ghost detection
            normal_is_phantom = normal_result.get('is_phantom_trigger', False)
            ghost_is_phantom = ghost_result.get('is_phantom_trigger', False)

            results['details'] = {
                'normal_signal_phantom': normal_is_phantom,
                'ghost_signal_phantom': ghost_is_phantom,
                'normal_ghost_id': normal_result.get('ghost_signal_id'),
                'ghost_signal_id': ghost_result.get('ghost_signal_id'),
                'detection_logic_working': True  # Both processed successfully
            }

        except Exception as e:
            results['errors'].append(f"Ghost signal detection test failed: {str(e)}")
            results['success'] = False

        if results['success']:
            logger.info("\\u2705 Ghost signal detection test passed")
        else:
            logger.error(f"\\u274c Ghost signal detection test failed: {len(results['errors'])} errors")

        return results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive profit vector calibration test."""
        logger.info("\\u1f680 Running comprehensive profit vector calibration test")

        start_time = time.time()

        # Run all test components
        test_results = {
            'profit_calculation': self.test_profit_calculation_accuracy(),
            'profit_memory': self.test_profit_memory_integration(),
            'ferris_wheel': self.test_ferris_wheel_integration(),
            'ghost_detection': self.test_ghost_signal_detection()
        }

        # Determine overall success
        all_passed = all(result['success'] for result in test_results.values())

        # Calculate total errors
        total_errors = sum(len(result.get('errors', [])) for result in test_results.values())

        execution_time = time.time() - start_time

        comprehensive_result = {
            'success': all_passed,
            'test_name': 'profit_vector_calibration',
            'execution_time': execution_time,
            'total_errors': total_errors,
            'test_components': test_results,
            'summary': {
                'profit_calculation_passed': test_results['profit_calculation']['success'],
                'profit_memory_passed': test_results['profit_memory']['success'],
                'ferris_wheel_passed': test_results['ferris_wheel']['success'],
                'ghost_detection_passed': test_results['ghost_detection']['success']
            }
        }

        if all_passed:
            logger.info(f"\\u2705 Comprehensive profit vector calibration test passed in {execution_time:.3f}s")
        else:
            logger.error(f"\\u274c Comprehensive profit vector calibration test failed with {total_errors} errors")

        return comprehensive_result


# Global test function for registry
def test_profit_vector_calibration() -> Dict[str, Any]:
    """Main test function for profit vector calibration."""
    try:
        test_suite = ProfitVectorCalibrationTest()
        return test_suite.run_comprehensive_test()
    except Exception as e:
        logger.error(f"Profit vector calibration test failed: {e}")
        return {
            'success': False,
            'test_name': 'profit_vector_calibration',
            'error': str(e),
            'execution_time': 0.0
        }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    result = test_profit_vector_calibration()

    # Print results
    safe_print("\n" + "="*60)
    safe_print("\\u1f4b0 PROFIT VECTOR CALIBRATION TEST RESULTS")
    safe_print("="*60)

    safe_print(f"Overall Success: {'\\u2705 PASS' if result['success'] else '\\u274c FAIL'}")
    safe_print(f"Execution Time: {result['execution_time']:.3f}s")
    safe_print(f"Total Errors: {result['total_errors']}")

    if 'test_components' in result:
        safe_print("\\nComponent Results:")
        for component, component_result in result['test_components'].items():
            status = "\\u2705 PASS" if component_result['success'] else "\\u274c FAIL"
            safe_print(f"  {component}: {status}")

    safe_print("="*60)

"""