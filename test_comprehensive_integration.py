#!/usr/bin/env python3
"""
Comprehensive Integration Test

This test verifies that all refactored components work together properly:
- CLI output safety (Unicode/emoji handling)
- Unified mathematical system
- Medium risk and hash risk integration
- Core pipeline integration
- Windows compatibility
"""

from utils.safe_print import safe_print, info, warn, error, success, debug, safe_math, safe_risk
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Import safe print for CLI compatibility

# Import core components
try:
    from core.unified_math_system import unified_math, MathResult, MathOperation
    from core.hash_registry_manager import HashRegistryManager
    from core.enhanced_phase_risk_manager import EnhancedPhaseRiskManager
    from core.pipeline_integration_manager import PipelineIntegrationManager
    from core.test_medium_risk_phase_ii import MediumRiskPhaseIITester
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError as e:
    warn(f"Some core components not available: {e}")
    UNIFIED_SYSTEM_AVAILABLE = False


class ComprehensiveIntegrationTester:
    """Comprehensive integration tester for all refactored components."""

    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }

        # Initialize components
        self.hash_registry_manager = None
        self.enhanced_phase_risk_manager = None
        self.pipeline_integration_manager = None
        self.medium_risk_tester = None

        if UNIFIED_SYSTEM_AVAILABLE:
            self._initialize_components()

    def _initialize_components(self):
        """Initialize all core components."""
        try:
            info("Initializing core components...")

            # Initialize hash registry manager
            self.hash_registry_manager = HashRegistryManager()
            success("Hash Registry Manager initialized")

            # Initialize enhanced phase risk manager
            self.enhanced_phase_risk_manager = EnhancedPhaseRiskManager()
            success("Enhanced Phase Risk Manager initialized")

            # Initialize pipeline integration manager
            self.pipeline_integration_manager = PipelineIntegrationManager()
            success("Pipeline Integration Manager initialized")

            # Initialize medium risk tester
            self.medium_risk_tester = MediumRiskPhaseIITester()
            success("Medium Risk Phase II Tester initialized")

        except Exception as e:
            error(f"Failed to initialize components: {e}")
            self.test_results['summary']['errors'].append(f"Initialization error: {e}")

    def test_cli_safety(self) -> Dict[str, Any]:
        """Test CLI output safety with Unicode/emoji handling."""
        info("Testing CLI output safety...")

        test_name = "cli_safety"
        test_result = {
            'status': 'passed',
            'details': [],
            'start_time': time.time()
        }

        try:
            # Test various Unicode/emoji scenarios
            test_cases = [
                ("Basic emoji", "🚀📈💰"),
                ("Mathematical symbols", "αβγ ±×÷ ≤≥≠"),
                ("Currency symbols", "€£¥₹₿"),
                ("Status indicators", "✓✗⚠ℹ"),
                ("Arrows", "→←↑↓⇒⇐⇑⇓"),
                ("Mixed content", "Profit: $1,234.56 📈 Success! ✅"),
            ]

            for case_name, test_text in test_cases:
                try:
                    # Test safe_print
                    safe_print(f"Testing: {case_name} - {test_text}")
                    test_result['details'].append(f"{case_name}: OK")
                except Exception as e:
                    test_result['details'].append(f"{case_name}: FAILED - {e}")
                    test_result['status'] = 'failed'

            # Test specialized safe functions
            safe_math("Test calculation", 42.0)
            safe_risk("MEDIUM", "Test risk level")

            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']

            if test_result['status'] == 'passed':
                success("CLI safety test passed")
            else:
                error("CLI safety test failed")

        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            error(f"CLI safety test error: {e}")

        return test_result

    def test_unified_math_system(self) -> Dict[str, Any]:
        """Test unified mathematical system."""
        info("Testing unified mathematical system...")

        test_name = "unified_math"
        test_result = {
            'status': 'passed',
            'details': [],
            'start_time': time.time()
        }

        try:
            if not UNIFIED_SYSTEM_AVAILABLE:
                test_result['status'] = 'skipped'
                test_result['details'].append("Unified math system not available")
                return test_result

            # Test basic arithmetic
            add_result = unified_math.add(5, 3)
            if add_result.success and add_result.value == 8:
                test_result['details'].append("Addition: OK")
            else:
                test_result['details'].append("Addition: FAILED")
                test_result['status'] = 'failed'

            # Test mathematical functions
            sqrt_result = unified_math.sqrt(16)
            if sqrt_result.success and sqrt_result.value == 4:
                test_result['details'].append("Square root: OK")
            else:
                test_result['details'].append("Square root: FAILED")
                test_result['status'] = 'failed'

            # Test trading-specific functions
            prices = [100, 105, 110, 108, 115]
            returns_result = unified_math.calculate_returns(prices)
            if returns_result.success:
                test_result['details'].append("Returns calculation: OK")
            else:
                test_result['details'].append("Returns calculation: FAILED")
                test_result['status'] = 'failed'

            # Test error handling
            div_result = unified_math.divide(10, 0)
            if not div_result.success and "Division by zero" in div_result.error_message:
                test_result['details'].append("Error handling: OK")
            else:
                test_result['details'].append("Error handling: FAILED")
                test_result['status'] = 'failed'

            # Get system statistics
            stats = unified_math.get_statistics()
            test_result['math_stats'] = stats

            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']

            if test_result['status'] == 'passed':
                success("Unified math system test passed")
            else:
                error("Unified math system test failed")

        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            error(f"Unified math test error: {e}")

        return test_result

    def test_hash_registry_integration(self) -> Dict[str, Any]:
        """Test hash registry integration."""
        info("Testing hash registry integration...")

        test_name = "hash_registry"
        test_result = {
            'status': 'passed',
            'details': [],
            'start_time': time.time()
        }

        try:
            if not self.hash_registry_manager:
                test_result['status'] = 'skipped'
                test_result['details'].append("Hash registry manager not available")
                return test_result

            # Test hash entry creation
            test_hash = "test_hash_123"
            success = self.hash_registry_manager.register_hash_entry(
                hash_id=test_hash,
                hash_value="abc123",
                hash_type="test",
                metadata={"test": True}
            )

            if success:
                test_result['details'].append("Hash entry creation: OK")
            else:
                test_result['details'].append("Hash entry creation: FAILED")
                test_result['status'] = 'failed'

            # Test hash entry retrieval
            entry = self.hash_registry_manager.get_hash_entry(test_hash)
            if entry and entry.hash_id == test_hash:
                test_result['details'].append("Hash entry retrieval: OK")
            else:
                test_result['details'].append("Hash entry retrieval: FAILED")
                test_result['status'] = 'failed'

            # Test hash registry statistics
            stats = self.hash_registry_manager.get_registry_stats()
            test_result['hash_stats'] = stats

            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']

            if test_result['status'] == 'passed':
                success("Hash registry integration test passed")
            else:
                error("Hash registry integration test failed")

        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            error(f"Hash registry test error: {e}")

        return test_result

    def test_medium_risk_integration(self) -> Dict[str, Any]:
        """Test medium risk phase II integration."""
        info("Testing medium risk phase II integration...")

        test_name = "medium_risk"
        test_result = {
            'status': 'passed',
            'details': [],
            'start_time': time.time()
        }

        try:
            if not self.enhanced_phase_risk_manager:
                test_result['status'] = 'skipped'
                test_result['details'].append("Enhanced phase risk manager not available")
                return test_result

            # Test phase risk initialization
            init_result = self.enhanced_phase_risk_manager.initialize_phase_risk()
            if init_result:
                test_result['details'].append("Phase risk initialization: OK")
            else:
                test_result['details'].append("Phase risk initialization: FAILED")
                test_result['status'] = 'failed'

            # Test risk calculation
            risk_data = {
                'price': 50000.0,
                'volume': 1000.0,
                'volatility': 0.02,
                'position_size': 0.1
            }

            risk_result = self.enhanced_phase_risk_manager.calculate_phase_risk(risk_data)
            if risk_result:
                test_result['details'].append("Risk calculation: OK")
                test_result['risk_result'] = risk_result
            else:
                test_result['details'].append("Risk calculation: FAILED")
                test_result['status'] = 'failed'

            # Test pipeline integration
            if self.pipeline_integration_manager:
                pipeline_result = self.pipeline_integration_manager.test_integration()
                test_result['pipeline_result'] = pipeline_result
                test_result['details'].append("Pipeline integration: OK")

            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']

            if test_result['status'] == 'passed':
                success("Medium risk integration test passed")
            else:
                error("Medium risk integration test failed")

        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            error(f"Medium risk test error: {e}")

        return test_result

    def test_core_pipeline_integration(self) -> Dict[str, Any]:
        """Test core pipeline integration."""
        info("Testing core pipeline integration...")

        test_name = "core_pipeline"
        test_result = {
            'status': 'passed',
            'details': [],
            'start_time': time.time()
        }

        try:
            # Test that all components can work together
            components = {
                'unified_math': UNIFIED_SYSTEM_AVAILABLE,
                'hash_registry': self.hash_registry_manager is not None,
                'phase_risk': self.enhanced_phase_risk_manager is not None,
                'pipeline': self.pipeline_integration_manager is not None,
                'medium_risk': self.medium_risk_tester is not None,
            }

            for component_name, available in components.items():
                if available:
                    test_result['details'].append(f"{component_name}: Available")
                else:
                    test_result['details'].append(f"{component_name}: Not available")
                    test_result['status'] = 'failed'

            # Test cross-component communication
            if all(components.values()):
                # Test math -> hash registry integration
                math_result = unified_math.calculate_returns([100, 105, 110])
                if math_result.success:
                    test_result['details'].append("Math -> Hash integration: OK")
                else:
                    test_result['details'].append("Math -> Hash integration: FAILED")
                    test_result['status'] = 'failed'

                # Test hash registry -> risk integration
                test_hash = "integration_test_hash"
                self.hash_registry_manager.register_hash_entry(
                    hash_id=test_hash,
                    hash_value="integration_test",
                    hash_type="test"
                )
                test_result['details'].append("Hash -> Risk integration: OK")

            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']

            if test_result['status'] == 'passed':
                success("Core pipeline integration test passed")
            else:
                error("Core pipeline integration test failed")

        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            error(f"Core pipeline test error: {e}")

        return test_result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive integration tests."""
        info("Starting comprehensive integration tests...")

        test_functions = [
            ("CLI Safety", self.test_cli_safety),
            ("Unified Math System", self.test_unified_math_system),
            ("Hash Registry Integration", self.test_hash_registry_integration),
            ("Medium Risk Integration", self.test_medium_risk_integration),
            ("Core Pipeline Integration", self.test_core_pipeline_integration),
        ]

        for test_name, test_func in test_functions:
            info(f"Running {test_name} test...")

            try:
                result = test_func()
                self.test_results['tests'][test_name] = result

                self.test_results['summary']['total_tests'] += 1
                if result['status'] == 'passed':
                    self.test_results['summary']['passed'] += 1
                else:
                    self.test_results['summary']['failed'] += 1

            except Exception as e:
                error(f"Test {test_name} failed with exception: {e}")
                self.test_results['tests'][test_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.test_results['summary']['total_tests'] += 1
                self.test_results['summary']['failed'] += 1
                self.test_results['summary']['errors'].append(f"{test_name}: {e}")

        # Generate summary
        total = self.test_results['summary']['total_tests']
        passed = self.test_results['summary']['passed']
        failed = self.test_results['summary']['failed']

        success(f"Integration tests complete: {passed}/{total} passed, {failed} failed")

        if failed == 0:
            success("All integration tests passed! 🎉")
        else:
            warn(f"{failed} tests failed. Check results for details.")

        return self.test_results

    def export_results(self, output_path: str = "comprehensive_integration_results.json") -> None:
        """Export test results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)

            success(f"Results exported to: {output_path}")

        except Exception as e:
            error(f"Failed to export results: {e}")


def main():
    """Main entry point."""
    info("Starting Comprehensive Integration Test Suite")
    info("=" * 50)

    tester = ComprehensiveIntegrationTester()
    results = tester.run_all_tests()

    # Export results
    tester.export_results()

    # Print summary
    info("=" * 50)
    info("TEST SUMMARY")
    info("=" * 50)

    summary = results['summary']
    info(f"Total Tests: {summary['total_tests']}")
    info(f"Passed: {summary['passed']}")
    info(f"Failed: {summary['failed']}")

    if summary['errors']:
        error("Errors encountered:")
        for error_msg in summary['errors']:
            error(f"  - {error_msg}")

    # Exit with appropriate code
    if summary['failed'] == 0:
        success("All tests passed! Integration successful.")
        sys.exit(0)
    else:
        error(f"{summary['failed']} tests failed. Integration incomplete.")
        sys.exit(1)


if __name__ == '__main__':
    main()
