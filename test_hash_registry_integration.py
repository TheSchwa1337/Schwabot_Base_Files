from core.math.tensor_algebra import UnifiedTensorAlgebra
from core.bit_resolution_engine import BitResolutionEngine
from core.tensor_matcher import TensorMatcher
from core.matrix_mapper import MatrixMapper, BitPhase
from core.matrix_basket_loader import MatrixBasketLoader, BasketLoadTrigger
from core.hash_registry_manager import HashRegistryManager, HashRegistryEntry
from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Hash Registry Integration Test - Schwabot UROS v1.0
==================================================

Comprehensive integration test for the 32-entry hash registry scaffold system.
Tests all components: hash registry manager, matrix basket loader, and integration points.

Mathematical Foundation:
- 4-bit to 42-bit range logic validation
- Hash ID naming structure (hash_00 to hash_31)
- Basket IDs (0-31) mapping validation
- Route logic (route_0 to route_4) validation
- Bit prioritization (0.1 to 3.2) validation
- Enabled/disabled switch validation
"""

import json
import time
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add core directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HashRegistryIntegrationTester:
    """
    Comprehensive integration tester for hash registry system.

    Tests all aspects of the 32-entry scaffold:
    - Hash registry structure validation
    - Matrix basket loading functionality
    - Integration with core components
    - Mathematical foundation validation
    - Performance and reliability testing
    """

    def __init__(self):
        """Initialize integration tester."""
        self.test_results = {}
        self.start_time = time.time()

        # Initialize components
        self.hash_registry_manager = HashRegistryManager()
        self.matrix_basket_loader = MatrixBasketLoader(self.hash_registry_manager)
        self.matrix_mapper = MatrixMapper()
        self.tensor_matcher = TensorMatcher()
        self.bit_resolution_engine = BitResolutionEngine()
        self.tensor_algebra = UnifiedTensorAlgebra()

        logger.info("Hash Registry Integration Tester initialized")

    def run_complete_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        safe_print("🧮 Hash Registry Integration Test - Schwabot UROS v1.0")
        safe_print("=" * 60)

        test_suites = [
            ("Hash Registry Structure", self.test_hash_registry_structure),
            ("Bit Depth Range Logic", self.test_bit_depth_range_logic),
            ("Hash ID Naming Structure", self.test_hash_id_naming_structure),
            ("Basket ID Mapping", self.test_basket_id_mapping),
            ("Route Logic Validation", self.test_route_logic_validation),
            ("Bit Prioritization", self.test_bit_prioritization),
            ("Enabled/Disabled Switch", self.test_enabled_disabled_switch),
            ("Matrix Basket Loading", self.test_matrix_basket_loading),
            ("Core Component Integration", self.test_core_component_integration),
            ("Mathematical Foundation", self.test_mathematical_foundation),
            ("Performance Testing", self.test_performance),
            ("Reliability Testing", self.test_reliability)
        ]

        for test_name, test_func in test_suites:
            safe_print(f"\n🔍 Running {test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
                safe_print(f"  {test_name}: {status}")
                if not result.get('success', False):
                    safe_print(f"    Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                self.test_results[test_name] = {'success': False, 'error': str(e)}
                safe_print(f"  {test_name}: ❌ FAILED")
                safe_print(f"    Exception: {e}")

        # Calculate overall results
        total_tests = len(test_suites)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        failed_tests = total_tests - passed_tests

        overall_success = failed_tests == 0

        safe_print(f"\n📊 Integration Test Results:")
        safe_print(f"  Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        safe_print(f"  Total Tests: {total_tests}")
        safe_print(f"  Passed: {passed_tests}")
        safe_print(f"  Failed: {failed_tests}")
        safe_print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        safe_print(f"  Total Execution Time: {time.time() - self.start_time:.2f}s")

        return {
            "overall_success": overall_success,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests/total_tests)*100 if total_tests > 0 else 0,
            "execution_time": time.time() - self.start_time,
            "test_results": self.test_results
        }

    def test_hash_registry_structure(self) -> Dict[str, Any]:
        """Test hash registry structure validation."""
        try:
            # Check total entries
            total_entries = len(self.hash_registry_manager.hash_entries)
            if total_entries != 32:
                return {'success': False, 'error': f'Expected 32 entries, got {total_entries}'}

            # Check hash ID format
            for i in range(32):
                expected_hash_id = f"hash_{i:02d}"
                if expected_hash_id not in self.hash_registry_manager.hash_entries:
                    return {'success': False, 'error': f'Missing hash ID: {expected_hash_id}'}

            # Check required fields
            required_fields = ['bit_depth', 'tensor_route', 'matrix_basket_id', 'priority', 'enabled']
            for hash_id, entry in self.hash_registry_manager.hash_entries.items():
                for field in required_fields:
                    if not hasattr(entry, field):
                        return {'success': False, 'error': f'Missing field {field} in {hash_id}'}

            return {'success': True, 'total_entries': total_entries}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_bit_depth_range_logic(self) -> Dict[str, Any]:
        """Test 4-bit to 42-bit range logic."""
        try:
            bit_depths = set()
            for entry in self.hash_registry_manager.hash_entries.values():
                bit_depths.unified_math.add(entry.bit_depth)

            expected_depths = {4, 8, 42}
            if bit_depths != expected_depths:
                return {'success': False, 'error': f'Expected bit depths {expected_depths}, got {bit_depths}'}

            # Test bit depth distribution
            depth_counts = {}
            for entry in self.hash_registry_manager.hash_entries.values():
                depth_counts[entry.bit_depth] = depth_counts.get(entry.bit_depth, 0) + 1

            # Validate reasonable distribution (should have some of each)
            for depth in expected_depths:
                if depth_counts.get(depth, 0) == 0:
                    return {'success': False, 'error': f'No entries with bit depth {depth}'}

            return {
                'success': True,
                'bit_depths': list(bit_depths),
                'depth_distribution': depth_counts
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_hash_id_naming_structure(self) -> Dict[str, Any]:
        """Test hash ID naming structure (hash_00 to hash_31)."""
        try:
            # Test all hash IDs from 00 to 31
            for i in range(32):
                expected_hash_id = f"hash_{i:02d}"
                entry = self.hash_registry_manager.get_hash_entry(expected_hash_id)

                if not entry:
                    return {'success': False, 'error': f'Missing hash ID: {expected_hash_id}'}

                if entry.hash_id != expected_hash_id:
                    return {'success': False, 'error': f'Hash ID mismatch: expected {expected_hash_id}, got {entry.hash_id}'}

            return {'success': True, 'hash_ids_tested': 32}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_basket_id_mapping(self) -> Dict[str, Any]:
        """Test basket ID mapping (0-31)."""
        try:
            basket_ids = set()
            for entry in self.hash_registry_manager.hash_entries.values():
                basket_ids.unified_math.add(entry.matrix_basket_id)

            # Check all basket IDs from 0 to 31
            expected_basket_ids = set(range(32))
            if basket_ids != expected_basket_ids:
                return {'success': False, 'error': f'Expected basket IDs {expected_basket_ids}, got {basket_ids}'}

            # Test basket ID uniqueness
            if len(basket_ids) != 32:
                return {'success': False, 'error': f'Duplicate basket IDs found: {len(basket_ids)} unique IDs'}

            return {'success': True, 'basket_ids': list(basket_ids)}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_route_logic_validation(self) -> Dict[str, Any]:
        """Test route logic (route_0 to route_4)."""
        try:
            routes = set()
            for entry in self.hash_registry_manager.hash_entries.values():
                routes.unified_math.add(entry.tensor_route)

            expected_routes = {f'route_{i}' for i in range(5)}
            if routes != expected_routes:
                return {'success': False, 'error': f'Expected routes {expected_routes}, got {routes}'}

            # Test route distribution
            route_counts = {}
            for entry in self.hash_registry_manager.hash_entries.values():
                route_counts[entry.tensor_route] = route_counts.get(entry.tensor_route, 0) + 1

            # Validate reasonable distribution
            for route in expected_routes:
                if route_counts.get(route, 0) == 0:
                    return {'success': False, 'error': f'No entries with route {route}'}

            return {
                'success': True,
                'routes': list(routes),
                'route_distribution': route_counts
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_bit_prioritization(self) -> Dict[str, Any]:
        """Test bit prioritization (0.1 to 3.2)."""
        try:
            priorities = []
            for entry in self.hash_registry_manager.hash_entries.values():
                priorities.append(entry.priority)

            # Check priority range
            min_priority = unified_math.min(priorities)
            max_priority = unified_math.max(priorities)

            if min_priority < 0.1 or max_priority > 3.2:
                return {'success': False, 'error': f'Priority out of range: min={min_priority}, max={max_priority}'}

            # Check priority uniqueness (should be unique for each entry)
            if len(set(priorities)) != len(priorities):
                return {'success': False, 'error': 'Duplicate priorities found'}

            # Check priority progression (should be increasing)
            sorted_priorities = sorted(priorities)
            if sorted_priorities != priorities:
                return {'success': False, 'error': 'Priorities not in ascending order'}

            return {
                'success': True,
                'min_priority': min_priority,
                'max_priority': max_priority,
                'priority_count': len(priorities)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_enabled_disabled_switch(self) -> Dict[str, Any]:
        """Test enabled/disabled switch functionality."""
        try:
            enabled_count = 0
            disabled_count = 0

            for entry in self.hash_registry_manager.hash_entries.values():
                if entry.enabled:
                    enabled_count += 1
                else:
                    disabled_count += 1

            # Test enable/disable functionality
            test_hash_id = "hash_00"
            original_enabled = self.hash_registry_manager.get_hash_entry(test_hash_id).enabled

            # Test disable
            success = self.hash_registry_manager.disable_entry(test_hash_id)
            if not success:
                return {'success': False, 'error': 'Failed to disable entry'}

            disabled_entry = self.hash_registry_manager.get_hash_entry(test_hash_id)
            if disabled_entry.enabled:
                return {'success': False, 'error': 'Entry still enabled after disable'}

            # Test enable
            success = self.hash_registry_manager.enable_entry(test_hash_id)
            if not success:
                return {'success': False, 'error': 'Failed to enable entry'}

            enabled_entry = self.hash_registry_manager.get_hash_entry(test_hash_id)
            if not enabled_entry.enabled:
                return {'success': False, 'error': 'Entry still disabled after enable'}

            # Restore original state
            if not original_enabled:
                self.hash_registry_manager.disable_entry(test_hash_id)

            return {
                'success': True,
                'enabled_count': enabled_count,
                'disabled_count': disabled_count,
                'enable_disable_test': 'passed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_matrix_basket_loading(self) -> Dict[str, Any]:
        """Test matrix basket loading functionality."""
        try:
            # Test loading by bit depth
            results_4bit = self.matrix_basket_loader.load_baskets_by_bit_depth(4)
            results_8bit = self.matrix_basket_loader.load_baskets_by_bit_depth(8)
            results_42bit = self.matrix_basket_loader.load_baskets_by_bit_depth(42)

            if not results_4bit or not results_8bit or not results_42bit:
                return {'success': False, 'error': 'Failed to load baskets by bit depth'}

            # Test loading by route
            results_route_0 = self.matrix_basket_loader.load_baskets_by_route("route_0")
            if not results_route_0:
                return {'success': False, 'error': 'Failed to load baskets by route'}

            # Test individual basket loading
            result = self.matrix_basket_loader.load_basket_from_registry("hash_10")
            if not result.success:
                return {'success': False, 'error': f'Failed to load individual basket: {result.error_message}'}

            # Test basket properties
            basket = result.basket
            if not basket:
                return {'success': False, 'error': 'Basket object is None'}

            if basket.bit_phase.value != 42:  # hash_10 should be 42-bit
                return {'success': False, 'error': f'Wrong bit depth: expected 42, got {basket.bit_phase.value}'}

            return {
                'success': True,
                '4bit_baskets_loaded': len(results_4bit),
                '8bit_baskets_loaded': len(results_8bit),
                '42bit_baskets_loaded': len(results_42bit),
                'route_0_baskets_loaded': len(results_route_0),
                'individual_basket_test': 'passed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_core_component_integration(self) -> Dict[str, Any]:
        """Test integration with core components."""
        try:
            # Test matrix mapper integration
            self.hash_registry_manager.integrate_with_matrix_mapper(self.matrix_mapper)

            # Test hash resolution
            test_hash = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
            basket_id = self.hash_registry_manager.resolve_hash_to_basket(test_hash)
            if not basket_id:
                return {'success': False, 'error': 'Failed to resolve hash to basket'}

            # Test tensor matcher integration
            self.tensor_matcher.set_bit_phase_engine(self.bit_resolution_engine)
            self.tensor_matcher.set_matrix_mapper(self.matrix_mapper)

            # Test bit resolution engine integration
            bit_result = self.bit_resolution_engine.resolve_bit_phase(test_hash, "auto")
            if not bit_result:
                return {'success': False, 'error': 'Failed to resolve bit phase'}

            return {
                'success': True,
                'matrix_mapper_integration': 'passed',
                'hash_resolution': 'passed',
                'tensor_matcher_integration': 'passed',
                'bit_resolution_integration': 'passed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_mathematical_foundation(self) -> Dict[str, Any]:
        """Test mathematical foundation validation."""
        try:
            # Test tensor algebra integration
            strategy_id = "0x123456789abcdef"
            bit_result = self.tensor_algebra.resolve_bit_phases(strategy_id)
            if not bit_result:
                return {'success': False, 'error': 'Failed to resolve bit phases'}

            # Test bit phase calculations
            if not (0 <= bit_result.phi_4 <= 15):
                return {'success': False, 'error': f'Invalid 4-bit phase: {bit_result.phi_4}'}
            if not (0 <= bit_result.phi_8 <= 255):
                return {'success': False, 'error': f'Invalid 8-bit phase: {bit_result.phi_8}'}
            if not (0 <= bit_result.phi_42 <= 0x3FFFFFFFFFF):
                return {'success': False, 'error': f'Invalid 42-bit phase: {bit_result.phi_42}'}

            # Test tensor contraction
            from core.unified_math_system import unified_math
            matrix_a = np.random.random((4, 4))
            matrix_b = np.random.random((4, 4))
            tensor_result = self.tensor_algebra.perform_tensor_contraction(matrix_a, matrix_b)
            if not tensor_result:
                return {'success': False, 'error': 'Failed to perform tensor contraction'}

            return {
                'success': True,
                'bit_phase_calculations': 'passed',
                'tensor_contraction': 'passed',
                'mathematical_integrity': 'verified'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        try:
            start_time = time.time()

            # Test bulk loading performance
            all_results = self.matrix_basket_loader.load_all_enabled_baskets()
            bulk_load_time = time.time() - start_time

            if bulk_load_time > 5.0:  # Should complete within 5 seconds
                return {'success': False, 'error': f'Bulk loading too slow: {bulk_load_time:.2f}s'}

            # Test individual loading performance
            start_time = time.time()
            for i in range(10):
                result = self.matrix_basket_loader.load_basket_from_registry(f"hash_{i:02d}")
                if not result.success:
                    return {'success': False, 'error': f'Failed to load hash_{i:02d}'}

            individual_load_time = time.time() - start_time
            avg_load_time = individual_load_time / 10

            if avg_load_time > 0.1:  # Should average less than 0.1 seconds per basket
                return {'success': False, 'error': f'Individual loading too slow: {avg_load_time:.3f}s average'}

            return {
                'success': True,
                'bulk_load_time': bulk_load_time,
                'individual_load_time': individual_load_time,
                'avg_load_time': avg_load_time,
                'baskets_loaded': len(all_results)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_reliability(self) -> Dict[str, Any]:
        """Test reliability and error handling."""
        try:
            # Test with invalid hash IDs
            invalid_hash_ids = ["hash_99", "invalid_hash", "hash_32", "hash_-1"]
            for invalid_id in invalid_hash_ids:
                entry = self.hash_registry_manager.get_hash_entry(invalid_id)
                if entry is not None:
                    return {'success': False, 'error': f'Should not find entry for invalid ID: {invalid_id}'}

            # Test with invalid bit depths
            invalid_bit_depths = [0, 1, 2, 3, 5, 6, 7, 9, 10, 41, 43, 100]
            for invalid_depth in invalid_bit_depths:
                entries = self.hash_registry_manager.get_entries_by_bit_depth(invalid_depth)
                if entries:
                    return {'success': False, 'error': f'Should not find entries for invalid bit depth: {invalid_depth}'}

            # Test with invalid routes
            invalid_routes = ["route_5", "route_10", "invalid_route", "route_-1"]
            for invalid_route in invalid_routes:
                entries = self.hash_registry_manager.get_entries_by_route(invalid_route)
                if entries:
                    return {'success': False, 'error': f'Should not find entries for invalid route: {invalid_route}'}

            # Test error handling in basket loading
            result = self.matrix_basket_loader.load_basket_from_registry("invalid_hash")
            if result.success:
                return {'success': False, 'error': 'Should not successfully load invalid hash'}

            return {
                'success': True,
                'invalid_hash_handling': 'passed',
                'invalid_bit_depth_handling': 'passed',
                'invalid_route_handling': 'passed',
                'error_handling': 'passed'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def export_test_results(self, output_path: str = "hash_registry_integration_test_results.json") -> None:
        """Export test results to JSON file."""
        try:
            results = self.run_complete_integration_test()

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Integration test results exported to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting test results: {e}")


def main():
    """Main function for hash registry integration testing."""
    safe_print("🧮 Hash Registry Integration Test - Schwabot UROS v1.0")
    safe_print("=" * 60)

    # Initialize tester
    tester = HashRegistryIntegrationTester()

    # Run complete integration test
    results = tester.run_complete_integration_test()

    # Export results
    tester.export_test_results()

    # Print detailed results
    safe_print(f"\n📋 Detailed Test Results:")
    for test_name, result in results['test_results'].items():
        status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
        safe_print(f"  {test_name}: {status}")
        if result.get('metadata'):
            for key, value in result['metadata'].items():
                safe_print(f"    {key}: {value}")

    # Return exit code
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit(main())
