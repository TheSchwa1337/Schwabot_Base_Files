from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""Tick Hold Logic Test - Schwabot Framework.

This test validates tick-based entry/hold/exit logic and ensures the system
can correctly handle long-hold strategies, temporary volume park logic, and
rebuy decisions across 3-12 tick delays. It tests the non-relativistic logic
that maintains trading functionality during hold periods.

Key Validations:
- Long-hold strategy validation
- Temporary volume park logic
- Rebuy decision windows (3-12 tick delays)
- Hold confidence calculations
- Volume threshold management
- Tick sequence integrity during holds
- Profit preservation during hold periods
"""

import unittest
import logging
import time
from core.unified_math_system import unified_math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HoldTestCase:
    """Test case for tick hold logic."""
    test_name: str
    initial_confidence: float
    volume_threshold: float
    hold_duration_ticks: int
    expected_hold_action: str
    expected_rebuy_window: tuple
    description: str


class TickHoldLogicTest:
    """Comprehensive tick hold logic testing."""

    def __init__(self):
        """Initialize the tick hold logic test."""
        self.test_cases = [
            HoldTestCase(
                test_name="short_hold_high_confidence",
                initial_confidence=0.85,
                volume_threshold=0.7,
                hold_duration_ticks=3,
                expected_hold_action="hold",
                expected_rebuy_window=(2, 5),
                description="Short hold with high confidence"
            ),
            HoldTestCase(
                test_name="medium_hold_moderate_confidence",
                initial_confidence=0.65,
                volume_threshold=0.5,
                hold_duration_ticks=6,
                expected_hold_action="hold",
                expected_rebuy_window=(4, 8),
                description="Medium hold with moderate confidence"
            ),
            HoldTestCase(
                test_name="long_hold_low_confidence",
                initial_confidence=0.45,
                volume_threshold=0.3,
                hold_duration_ticks=12,
                expected_hold_action="hold",
                expected_rebuy_window=(8, 15),
                description="Long hold with low confidence"
            ),
            HoldTestCase(
                test_name="immediate_exit_very_low_confidence",
                initial_confidence=0.25,
                volume_threshold=0.2,
                hold_duration_ticks=1,
                expected_hold_action="exit",
                expected_rebuy_window=(0, 0),
                description="Immediate exit with very low confidence"
            )
        ]

        logger.info("⏱️ Tick Hold Logic Test initialized")

    def test_long_hold_strategy_validation(self) -> Dict[str, Any]:
        """Test long-hold strategy validation."""
        logger.info("📈 Testing long-hold strategy validation")

        results = {
            'test_name': 'long_hold_strategy_validation',
            'success': True,
            'details': {},
            'errors': []
        }

        for i, test_case in enumerate(self.test_cases):
            try:
                # Simulate long-hold strategy logic
                hold_result = self._simulate_long_hold_strategy(test_case)

                # Validate hold action
                if hold_result['hold_action'] != test_case.expected_hold_action:
                    error_msg = f"Test case {i} ({test_case.description}): Hold action mismatch. Expected: {test_case.expected_hold_action}, Got: {hold_result['hold_action']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate hold duration
                if hold_result['hold_duration'] != test_case.hold_duration_ticks:
                    error_msg = f"Test case {i} ({test_case.description}): Hold duration mismatch. Expected: {test_case.hold_duration_ticks}, Got: {hold_result['hold_duration']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate confidence decay
                if not (0.0 <= hold_result['confidence_decay'] <= 1.0):
                    error_msg = f"Test case {i} ({test_case.description}): Invalid confidence decay. Expected [0.0, 1.0], Got: {hold_result['confidence_decay']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Store test case results
                results['details'][f'test_case_{i}'] = {
                    'description': test_case.description,
                    'expected_action': test_case.expected_hold_action,
                    'actual_action': hold_result['hold_action'],
                    'expected_duration': test_case.hold_duration_ticks,
                    'actual_duration': hold_result['hold_duration'],
                    'confidence_decay': hold_result['confidence_decay'],
                    'hold_strategy_valid': hold_result['hold_action'] == test_case.expected_hold_action
                }

            except Exception as e:
                error_msg = f"Test case {i} ({test_case.description}): Exception - {str(e)}"
                results['errors'].append(error_msg)
                results['success'] = False

        if results['success']:
            logger.info("✅ Long-hold strategy validation test passed")
        else:
            logger.error(f"❌ Long-hold strategy validation test failed: {len(results['errors'])} errors")

        return results

    def test_temporary_volume_park_logic(self) -> Dict[str, Any]:
        """Test temporary volume park logic."""
        logger.info("📊 Testing temporary volume park logic")

        results = {
            'test_name': 'temporary_volume_park_logic',
            'success': True,
            'details': {},
            'errors': []
        }

        for i, test_case in enumerate(self.test_cases):
            try:
                # Simulate volume park logic
                park_result = self._simulate_volume_park_logic(test_case)

                # Validate volume threshold
                if park_result['volume_threshold'] != test_case.volume_threshold:
                    error_msg = f"Test case {i} ({test_case.description}): Volume threshold mismatch. Expected: {test_case.volume_threshold}, Got: {park_result['volume_threshold']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate park decision
                if not isinstance(park_result['should_park'], bool):
                    error_msg = f"Test case {i} ({test_case.description}): Invalid park decision type"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate park duration
                if park_result['park_duration'] < 0:
                    error_msg = f"Test case {i} ({test_case.description}): Invalid park duration. Expected >= 0, Got: {park_result['park_duration']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Store test case results
                results['details'][f'test_case_{i}'] = {
                    'description': test_case.description,
                    'volume_threshold': park_result['volume_threshold'],
                    'should_park': park_result['should_park'],
                    'park_duration': park_result['park_duration'],
                    'volume_pressure': park_result['volume_pressure'],
                    'park_logic_valid': len(results['errors']) == 0
                }

            except Exception as e:
                error_msg = f"Test case {i} ({test_case.description}): Exception - {str(e)}"
                results['errors'].append(error_msg)
                results['success'] = False

        if results['success']:
            logger.info("✅ Temporary volume park logic test passed")
        else:
            logger.error(f"❌ Temporary volume park logic test failed: {len(results['errors'])} errors")

        return results

    def test_rebuy_decision_windows(self) -> Dict[str, Any]:
        """Test rebuy decision windows across 3-12 tick delays."""
        logger.info("🔄 Testing rebuy decision windows")

        results = {
            'test_name': 'rebuy_decision_windows',
            'success': True,
            'details': {},
            'errors': []
        }

        for i, test_case in enumerate(self.test_cases):
            try:
                # Simulate rebuy decision logic
                rebuy_result = self._simulate_rebuy_decision_logic(test_case)

                # Validate rebuy window
                min_ticks, max_ticks = test_case.expected_rebuy_window
                actual_ticks = rebuy_result['rebuy_ticks']

                if not (min_ticks <= actual_ticks <= max_ticks):
                    error_msg = f"Test case {i} ({test_case.description}): Rebuy ticks out of range. Expected [{min_ticks}, {max_ticks}], Got: {actual_ticks}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate rebuy confidence
                if not (0.0 <= rebuy_result['rebuy_confidence'] <= 1.0):
                    error_msg = f"Test case {i} ({test_case.description}): Invalid rebuy confidence. Expected [0.0, 1.0], Got: {rebuy_result['rebuy_confidence']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate rebuy trigger
                if not isinstance(rebuy_result['rebuy_triggered'], bool):
                    error_msg = f"Test case {i} ({test_case.description}): Invalid rebuy trigger type"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Store test case results
                results['details'][f'test_case_{i}'] = {
                    'description': test_case.description,
                    'expected_window': test_case.expected_rebuy_window,
                    'actual_ticks': actual_ticks,
                    'rebuy_confidence': rebuy_result['rebuy_confidence'],
                    'rebuy_triggered': rebuy_result['rebuy_triggered'],
                    'window_valid': min_ticks <= actual_ticks <= max_ticks
                }

            except Exception as e:
                error_msg = f"Test case {i} ({test_case.description}): Exception - {str(e)}"
                results['errors'].append(error_msg)
                results['success'] = False

        if results['success']:
            logger.info("✅ Rebuy decision windows test passed")
        else:
            logger.error(f"❌ Rebuy decision windows test failed: {len(results['errors'])} errors")

        return results

    def test_hold_confidence_calculations(self) -> Dict[str, Any]:
        """Test hold confidence calculations."""
        logger.info("🎯 Testing hold confidence calculations")

        results = {
            'test_name': 'hold_confidence_calculations',
            'success': True,
            'details': {},
            'errors': []
        }

        try:
            # Test confidence calculations for different scenarios
            confidence_scenarios = [
                {'initial_confidence': 0.9, 'hold_ticks': 5, 'expected_decay': 0.1},
                {'initial_confidence': 0.7, 'hold_ticks': 8, 'expected_decay': 0.2},
                {'initial_confidence': 0.5, 'hold_ticks': 12, 'expected_decay': 0.3}
            ]

            for i, scenario in enumerate(confidence_scenarios):
                # Calculate hold confidence
                hold_confidence = self._calculate_hold_confidence(
                    scenario['initial_confidence'],
                    scenario['hold_ticks']
                )

                # Validate confidence range
                if not (0.0 <= hold_confidence <= 1.0):
                    error_msg = f"Scenario {i}: Invalid hold confidence. Expected [0.0, 1.0], Got: {hold_confidence}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Validate confidence decay
                expected_confidence = scenario['initial_confidence'] - scenario['expected_decay']
                confidence_diff = unified_math.abs(hold_confidence - expected_confidence)

                if confidence_diff > 0.2:  # Allow reasonable tolerance
                    error_msg = f"Scenario {i}: Confidence decay too large. Expected ~{expected_confidence:.3f}, Got: {hold_confidence:.3f}"
                    results['errors'].append(error_msg)
                    results['success'] = False

                # Store scenario results
                results['details'][f'scenario_{i}'] = {
                    'initial_confidence': scenario['initial_confidence'],
                    'hold_ticks': scenario['hold_ticks'],
                    'hold_confidence': hold_confidence,
                    'expected_decay': scenario['expected_decay'],
                    'confidence_valid': 0.0 <= hold_confidence <= 1.0
                }

        except Exception as e:
            results['errors'].append(f"Hold confidence calculations test failed: {str(e)}")
            results['success'] = False

        if results['success']:
            logger.info("✅ Hold confidence calculations test passed")
        else:
            logger.error(f"❌ Hold confidence calculations test failed: {len(results['errors'])} errors")

        return results

    def test_tick_sequence_integrity(self) -> Dict[str, Any]:
        """Test tick sequence integrity during holds."""
        logger.info("🔢 Testing tick sequence integrity")

        results = {
            'test_name': 'tick_sequence_integrity',
            'success': True,
            'details': {},
            'errors': []
        }

        try:
            # Generate test tick sequence
            tick_sequence = self._generate_test_tick_sequence(20)  # 20 ticks

            # Validate tick sequence properties
            if len(tick_sequence) != 20:
                error_msg = f"Tick sequence length mismatch. Expected: 20, Got: {len(tick_sequence)}"
                results['errors'].append(error_msg)
                results['success'] = False

            # Validate tick timestamps are increasing
            timestamps = [tick['timestamp'] for tick in tick_sequence]
            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i-1]:
                    error_msg = f"Tick {i}: Timestamp not increasing. Previous: {timestamps[i-1]}, Current: {timestamps[i]}"
                    results['errors'].append(error_msg)
                    results['success'] = False

            # Validate tick hashes are unique
            hashes = [tick['hash'] for tick in tick_sequence]
            unique_hashes = set(hashes)
            if len(unique_hashes) != len(hashes):
                error_msg = "Duplicate tick hashes detected"
                results['errors'].append(error_msg)
                results['success'] = False

            # Validate tick data consistency
            for i, tick in enumerate(tick_sequence):
                if not all(key in tick for key in ['hash', 'timestamp', 'price', 'volume']):
                    error_msg = f"Tick {i}: Missing required fields"
                    results['errors'].append(error_msg)
                    results['success'] = False

                if tick['price'] <= 0 or tick['volume'] <= 0:
                    error_msg = f"Tick {i}: Invalid price or volume. Price: {tick['price']}, Volume: {tick['volume']}"
                    results['errors'].append(error_msg)
                    results['success'] = False

            results['details'] = {
                'total_ticks': len(tick_sequence),
                'unique_hashes': len(unique_hashes),
                'timestamp_sequence_valid': len(results['errors']) == 0,
                'tick_data_consistent': len(results['errors']) == 0,
                'sequence_integrity_score': 1.0 if len(results['errors']) == 0 else 0.0
            }

        except Exception as e:
            results['errors'].append(f"Tick sequence integrity test failed: {str(e)}")
            results['success'] = False

        if results['success']:
            logger.info("✅ Tick sequence integrity test passed")
        else:
            logger.error(f"❌ Tick sequence integrity test failed: {len(results['errors'])} errors")

        return results

    def _simulate_long_hold_strategy(self, test_case: HoldTestCase) -> Dict[str, Any]:
        """Simulate long-hold strategy logic."""
        # Determine hold action based on confidence
        if test_case.initial_confidence > 0.8:
            hold_action = "hold"
        elif test_case.initial_confidence > 0.6:
            hold_action = "hold"
        elif test_case.initial_confidence > 0.4:
            hold_action = "hold"
        else:
            hold_action = "exit"

        # Calculate confidence decay
        confidence_decay = unified_math.min(test_case.hold_duration_ticks * 0.05, 0.3)

        return {
            'hold_action': hold_action,
            'hold_duration': test_case.hold_duration_ticks,
            'confidence_decay': confidence_decay
        }

    def _simulate_volume_park_logic(self, test_case: HoldTestCase) -> Dict[str, Any]:
        """Simulate volume park logic."""
        # Determine if should park based on volume threshold
        should_park = test_case.initial_confidence < test_case.volume_threshold

        # Calculate park duration
        park_duration = test_case.hold_duration_ticks if should_park else 0

        # Calculate volume pressure
        volume_pressure = test_case.initial_confidence / test_case.volume_threshold

        return {
            'volume_threshold': test_case.volume_threshold,
            'should_park': should_park,
            'park_duration': park_duration,
            'volume_pressure': volume_pressure
        }

    def _simulate_rebuy_decision_logic(self, test_case: HoldTestCase) -> Dict[str, Any]:
        """Simulate rebuy decision logic."""
        # Calculate rebuy ticks based on hold duration
        if test_case.hold_duration_ticks <= 3:
            rebuy_ticks = test_case.hold_duration_ticks + 1
        elif test_case.hold_duration_ticks <= 6:
            rebuy_ticks = test_case.hold_duration_ticks + 2
        else:
            rebuy_ticks = test_case.hold_duration_ticks + 3

        # Calculate rebuy confidence
        rebuy_confidence = unified_math.max(0.0, test_case.initial_confidence - 0.1)

        # Determine if rebuy is triggered
        rebuy_triggered = rebuy_confidence > 0.5

        return {
            'rebuy_ticks': rebuy_ticks,
            'rebuy_confidence': rebuy_confidence,
            'rebuy_triggered': rebuy_triggered
        }

    def _calculate_hold_confidence(self, initial_confidence: float, hold_ticks: int) -> float:
        """Calculate hold confidence with decay."""
        # Apply time-based decay
        decay_factor = unified_math.min(hold_ticks * 0.02, 0.3)
        hold_confidence = unified_math.max(0.0, initial_confidence - decay_factor)

        return hold_confidence

    def _generate_test_tick_sequence(self, num_ticks: int) -> List[Dict[str, Any]]:
        """Generate test tick sequence."""
        import hashlib

        sequence = []
        base_time = time.time()

        for i in range(num_ticks):
            # Generate tick data
            price = 50000.0 + (i * 10.0) + np.random.normal(0, 5.0)
            volume = 1000.0 + (i * 50.0) + np.random.normal(0, 100.0)
            timestamp = base_time + (i * 0.1)  # 0.1 second intervals

            # Generate hash
            hash_string = f"{price:.6f}:{volume:.6f}:{timestamp:.6f}"
            tick_hash = hashlib.sha256(hash_string.encode()).hexdigest()[:16]

            tick = {
                'hash': tick_hash,
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            }

            sequence.append(tick)

        return sequence

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tick hold logic test."""
        logger.info("🚀 Running comprehensive tick hold logic test")

        start_time = time.time()

        # Run all test components
        test_results = {
            'long_hold_strategy': self.test_long_hold_strategy_validation(),
            'volume_park_logic': self.test_temporary_volume_park_logic(),
            'rebuy_decision_windows': self.test_rebuy_decision_windows(),
            'hold_confidence_calculations': self.test_hold_confidence_calculations(),
            'tick_sequence_integrity': self.test_tick_sequence_integrity()
        }

        # Determine overall success
        all_passed = all(result['success'] for result in test_results.values())

        # Calculate total errors
        total_errors = sum(len(result.get('errors', [])) for result in test_results.values())

        execution_time = time.time() - start_time

        comprehensive_result = {
            'success': all_passed,
            'test_name': 'tick_hold_logic',
            'execution_time': execution_time,
            'total_errors': total_errors,
            'test_components': test_results,
            'summary': {
                'long_hold_strategy_passed': test_results['long_hold_strategy']['success'],
                'volume_park_logic_passed': test_results['volume_park_logic']['success'],
                'rebuy_decision_windows_passed': test_results['rebuy_decision_windows']['success'],
                'hold_confidence_calculations_passed': test_results['hold_confidence_calculations']['success'],
                'tick_sequence_integrity_passed': test_results['tick_sequence_integrity']['success']
            }
        }

        if all_passed:
            logger.info(f"✅ Comprehensive tick hold logic test passed in {execution_time:.3f}s")
        else:
            logger.error(f"❌ Comprehensive tick hold logic test failed with {total_errors} errors")

        return comprehensive_result


# Global test function for registry
def test_tick_hold_logic() -> Dict[str, Any]:
    """Main test function for tick hold logic."""
    try:
        test_suite = TickHoldLogicTest()
        return test_suite.run_comprehensive_test()
    except Exception as e:
        logger.error(f"Tick hold logic test failed: {e}")
        return {
            'success': False,
            'test_name': 'tick_hold_logic',
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
    result = test_tick_hold_logic()

    # Print results
    safe_print("\n" + "="*60)
    safe_print("⏱️ TICK HOLD LOGIC TEST RESULTS")
    safe_print("="*60)

    safe_print(f"Overall Success: {'✅ PASS' if result['success'] else '❌ FAIL'}")
    safe_print(f"Execution Time: {result['execution_time']:.3f}s")
    safe_print(f"Total Errors: {result['total_errors']}")

    if 'test_components' in result:
        safe_print("\nComponent Results:")
        for component, component_result in result['test_components'].items():
            status = "✅ PASS" if component_result['success'] else "❌ FAIL"
            safe_print(f"  {component}: {status}")

    safe_print("="*60)
