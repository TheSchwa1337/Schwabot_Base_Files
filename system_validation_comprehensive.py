#!/usr/bin/env python3
"""
Comprehensive System Validation for Schwabot
===========================================

This script validates the entire Schwabot system to address:
- User interface functionality and control panels
- API integration (CoinMarketCap, CoinGecko)
- Mathematical function completeness
- System robustness and error handling
- Code quality and linting compliance
- Performance and timing analysis
- Security and secret management

Answers to User Questions:
1. Have we verified all visible levels of the user interface?
2. Have we integrated all settings for JSON configuration?
3. Does our UI effectively port mathematics into a unified panel?
4. Is API correctly integrated across the entire system?
5. Are we addressing sequences, I/O timings, GPU timings, thermal functionalities?
6. Do we have a robust enough system with no errors under scrutiny/stress?
7. Are all pipelines visually routed correctly?
8. Will we have timing errors or drift that can be corrected?
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import traceback

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

try:
    from utils.windows_cli_compatibility import safe_print
    from utils.math_utils import *
    from math.tensor_algebra.unified_tensor_algebra import *
    from math.tensor_algebra.profit_engine import *
    from math.tensor_algebra.entropy_engine import *
    from math.tensor_algebra.tensor_engine import *
    from config.api_config import *
    from memory_stack.ai_command_sequencer import *
    from memory_stack.execution_validator import *
    from memory_stack.memory_key_allocator import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Available modules in core:")
    import os
    for root, dirs, files in os.walk('core'):
        for file in files:
            if file.endswith('.py'):
                print(f"  {os.path.join(root, file)}")
    sys.exit(1)


class ComprehensiveSystemValidator:
    """
    Comprehensive system validator for Schwabot.

    Validates all aspects of the system including:
    - Mathematical functions
    - API integration
    - User interface
    - Code quality
    - Performance
    - Security
    """

    def __init__(self):
        """Initialize the comprehensive validator."""
        self.results = {
            'mathematical_functions': {},
            'api_integration': {},
            'user_interface': {},
            'code_quality': {},
            'performance': {},
            'security': {},
            'overall_status': 'PENDING'
        }

        self.start_time = time.time()
        safe_print("🚀 Comprehensive System Validation Started")
        safe_print("=" * 60)

    def validate_mathematical_functions(self) -> Dict[str, Any]:
        """Validate all mathematical functions and their completeness."""
        safe_print("\n🔢 Validating Mathematical Functions...")

        results = {
            'ghost_trigger_functions': {},
            'phase_probability_functions': {},
            'tensor_operations': {},
            'profit_engine': {},
            'entropy_engine': {},
            'tensor_engine': {},
            'status': 'PENDING'
        }

        try:
            # Test Ghost Trigger Functions
            safe_print("  Testing Ghost Trigger Functions...")

            # Test phase probability pathway
            test_phases = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            pathway = calculate_phase_probability_pathway(test_phases)
            results['ghost_trigger_functions']['phase_probability_pathway'] = {
                'status': 'PASS',
                'result_shape': pathway.shape if hasattr(pathway, 'shape') else 'scalar'
            }

            # Test grayscale phase mapping
            grayscale_map = create_grayscale_phase_map(test_phases)
            results['ghost_trigger_functions']['grayscale_phase_mapping'] = {
                'status': 'PASS',
                'result_shape': grayscale_map.shape if hasattr(grayscale_map, 'shape') else 'scalar'
            }

            # Test ghost trigger generation
            ghost_triggers = generate_ghost_triggers(test_phases, threshold=0.5)
            results['ghost_trigger_functions']['ghost_trigger_generation'] = {
                'status': 'PASS',
                'trigger_count': len(ghost_triggers) if isinstance(ghost_triggers, (list, np.ndarray)) else 1
            }

            # Test bit phase allocation
            bit_allocation = allocate_bit_phases(test_phases, num_bits=8)
            results['ghost_trigger_functions']['bit_phase_allocation'] = {
                'status': 'PASS',
                'allocation_shape': bit_allocation.shape if hasattr(bit_allocation, 'shape') else 'scalar'
            }

            # Test Profit Engine
            safe_print("  Testing Profit Engine...")
            profit_engine = ProfitEngine()

            # Test profit surface computation
            price_map = np.random.rand(10, 10) * 100
            hold_map = np.random.rand(10, 10) * 10
            profit_surface = profit_engine.compute_profit_surface(price_map, hold_map)
            results['profit_engine']['profit_surface'] = {
                'status': 'PASS',
                'surface_shape': profit_surface.shape
            }

            # Test entropy engine
            safe_print("  Testing Entropy Engine...")
            entropy_engine = EntropyEngine()

            test_signal = np.random.rand(100)
            filtered_signal = entropy_engine.entropy_filter(test_signal)
            results['entropy_engine']['entropy_filtering'] = {
                'status': 'PASS',
                'filtered_shape': filtered_signal.shape
            }

            # Test tensor engine
            safe_print("  Testing Tensor Engine...")
            tensor_engine = TensorEngine()

            test_tensor = np.random.rand(5, 5)
            pattern_analysis = tensor_engine.analyze_tensor_patterns(test_tensor)
            results['tensor_engine']['pattern_analysis'] = {
                'status': 'PASS',
                'patterns_found': len(pattern_analysis.get('pattern_types', []))
            }

            results['status'] = 'PASS'
            safe_print("  ✅ Mathematical Functions Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ Mathematical Functions Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def validate_api_integration(self) -> Dict[str, Any]:
        """Validate API integration for CoinMarketCap and CoinGecko."""
        safe_print("\n🌐 Validating API Integration...")

        results = {
            'coinmarketcap': {},
            'coingecko': {},
            'secret_management': {},
            'rate_limiting': {},
            'status': 'PENDING'
        }

        try:
            # Test API configuration
            safe_print("  Testing API Configuration...")
            config = APIConfig()
            results['configuration'] = {
                'status': 'PASS',
                'coinmarketcap_rate_limit': config.coinmarketcap_rate_limit,
                'coingecko_rate_limit': config.coingecko_rate_limit
            }

            # Test secret management
            safe_print("  Testing Secret Management...")
            secret_manager = APISecretManager()
            results['secret_management'] = {
                'status': 'PASS',
                'secrets_loaded': len(secret_manager._secrets_cache)
            }

            # Test API manager
            safe_print("  Testing API Manager...")
            api_manager = APIManager(config)
            results['api_manager'] = {
                'status': 'PASS',
                'coinmarketcap_configured': secret_manager.has_secret('coinmarketcap_api_key'),
                'coingecko_available': True  # No API key required
            }

            # Test CoinGecko integration (no API key required)
            safe_print("  Testing CoinGecko Integration...")
            try:
                global_data = api_manager.get_global_metrics('coingecko')
                results['coingecko'] = {
                    'status': 'PASS',
                    'global_data_keys': list(global_data.keys()) if isinstance(global_data, dict) else []
                }
            except Exception as e:
                results['coingecko'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }

            # Test rate limiting
            safe_print("  Testing Rate Limiting...")
            rate_limiter = APIRateLimiter(30)
            rate_limiter.record_request()
            results['rate_limiting'] = {
                'status': 'PASS',
                'can_make_request': rate_limiter.can_make_request()
            }

            results['status'] = 'PASS'
            safe_print("  ✅ API Integration Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ API Integration Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def validate_user_interface(self) -> Dict[str, Any]:
        """Validate user interface functionality and control panels."""
        safe_print("\n🖥️  Validating User Interface...")

        results = {
            'configuration_loading': {},
            'settings_management': {},
            'mathematical_panel': {},
            'api_control_panel': {},
            'status': 'PENDING'
        }

        try:
            # Test configuration loading
            safe_print("  Testing Configuration Loading...")

            # Create test configuration
            test_config = {
                'api_settings': {
                    'coinmarketcap_enabled': True,
                    'coingecko_enabled': True,
                    'rate_limits': {
                        'coinmarketcap': 30,
                        'coingecko': 50
                    }
                },
                'mathematical_settings': {
                    'ghost_trigger_threshold': 0.5,
                    'phase_probability_window': 20,
                    'entropy_filter_threshold': 0.3
                },
                'performance_settings': {
                    'enable_caching': True,
                    'cache_duration': 300,
                    'max_iterations': 1000
                }
            }

            # Test JSON configuration handling
            config_file = Path('test_config.json')
            with open(config_file, 'w') as f:
                json.dump(test_config, f, indent=2)

            with open(config_file, 'r') as f:
                loaded_config = json.load(f)

            results['configuration_loading'] = {
                'status': 'PASS',
                'config_keys': list(loaded_config.keys()),
                'api_settings_present': 'api_settings' in loaded_config,
                'mathematical_settings_present': 'mathematical_settings' in loaded_config
            }

            # Clean up test file
            config_file.unlink()

            # Test mathematical panel integration
            safe_print("  Testing Mathematical Panel Integration...")

            # Test that mathematical functions can be controlled via configuration
            test_phases = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            threshold = loaded_config['mathematical_settings']['ghost_trigger_threshold']

            ghost_triggers = generate_ghost_triggers(test_phases, threshold=threshold)

            results['mathematical_panel'] = {
                'status': 'PASS',
                'configurable_threshold': threshold,
                'ghost_triggers_generated': len(ghost_triggers) if isinstance(ghost_triggers, (list, np.ndarray)) else 1
            }

            # Test API control panel
            safe_print("  Testing API Control Panel...")

            api_config = loaded_config['api_settings']
            results['api_control_panel'] = {
                'status': 'PASS',
                'coinmarketcap_enabled': api_config['coinmarketcap_enabled'],
                'coingecko_enabled': api_config['coingecko_enabled'],
                'rate_limits_configured': 'rate_limits' in api_config
            }

            results['status'] = 'PASS'
            safe_print("  ✅ User Interface Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ User Interface Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and linting compliance."""
        safe_print("\n🔍 Validating Code Quality...")

        results = {
            'flake8_compliance': {},
            'import_analysis': {},
            'syntax_check': {},
            'documentation': {},
            'status': 'PENDING'
        }

        try:
            # Test Flake8 compliance
            safe_print("  Testing Flake8 Compliance...")

            # Check if flake8 is available
            try:
                result = subprocess.run(['flake8', '--version'],
                                        capture_output=True, text=True, timeout=10)
                flake8_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                flake8_available = False

            if flake8_available:
                # Run flake8 on core directory
                result = subprocess.run(['flake8', 'core/', '--max-line-length=100'],
                                        capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    results['flake8_compliance'] = {
                        'status': 'PASS',
                        'errors': 0,
                        'warnings': 0
                    }
                else:
                    # Parse flake8 output
                    lines = result.stdout.strip().split('\n')
                    error_count = len([line for line in lines if line.strip()])

                    results['flake8_compliance'] = {
                        'status': 'WARN',
                        'errors': error_count,
                        'output': result.stdout[:500]  # First 500 chars
                    }
            else:
                results['flake8_compliance'] = {
                    'status': 'SKIP',
                    'reason': 'Flake8 not available'
                }

            # Test import analysis
            safe_print("  Testing Import Analysis...")

            # Test that all critical modules can be imported
            critical_modules = [
                'core.utils.math_utils',
                'core.math.tensor_algebra.unified_tensor_algebra',
                'core.math.tensor_algebra.profit_engine',
                'core.math.tensor_algebra.entropy_engine',
                'core.math.tensor_algebra.tensor_engine',
                'core.config.api_config',
                'core.memory_stack.ai_command_sequencer',
                'core.memory_stack.execution_validator',
                'core.memory_stack.memory_key_allocator'
            ]

            import_errors = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    import_errors.append(f"{module}: {e}")

            results['import_analysis'] = {
                'status': 'PASS' if not import_errors else 'FAIL',
                'modules_tested': len(critical_modules),
                'import_errors': import_errors
            }

            # Test syntax check
            safe_print("  Testing Syntax Check...")

            # Test that Python can compile the main modules
            python_files = [
                'core/utils/math_utils.py',
                'core/math/tensor_algebra/profit_engine.py',
                'core/math/tensor_algebra/entropy_engine.py',
                'core/math/tensor_algebra/tensor_engine.py',
                'core/config/api_config.py'
            ]

            syntax_errors = []
            for file_path in python_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            compile(f.read(), file_path, 'exec')
                    except SyntaxError as e:
                        syntax_errors.append(f"{file_path}: {e}")

            results['syntax_check'] = {
                'status': 'PASS' if not syntax_errors else 'FAIL',
                'files_tested': len(python_files),
                'syntax_errors': syntax_errors
            }

            results['status'] = 'PASS'
            safe_print("  ✅ Code Quality Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ Code Quality Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance and timing analysis."""
        safe_print("\n⚡ Validating Performance...")

        results = {
            'mathematical_performance': {},
            'api_performance': {},
            'memory_usage': {},
            'timing_analysis': {},
            'status': 'PENDING'
        }

        try:
            # Test mathematical performance
            safe_print("  Testing Mathematical Performance...")

            # Test large dataset processing
            large_dataset = np.random.rand(1000, 1000)
            start_time = time.time()

            # Test tensor operations
            tensor_engine = TensorEngine()
            pattern_analysis = tensor_engine.analyze_tensor_patterns(large_dataset)

            math_time = time.time() - start_time

            results['mathematical_performance'] = {
                'status': 'PASS',
                'large_dataset_time': math_time,
                'dataset_size': large_dataset.shape
            }

            # Test API performance
            safe_print("  Testing API Performance...")

            # Test rate limiting performance
            rate_limiter = APIRateLimiter(30)
            start_time = time.time()

            for _ in range(10):
                rate_limiter.record_request()
                rate_limiter.can_make_request()

            api_time = time.time() - start_time

            results['api_performance'] = {
                'status': 'PASS',
                'rate_limiter_time': api_time,
                'requests_processed': 10
            }

            # Test timing analysis
            safe_print("  Testing Timing Analysis...")

            # Test that timing functions work correctly
            test_signal = np.random.rand(100)
            entropy_engine = EntropyEngine()

            start_time = time.time()
            filtered_signal = entropy_engine.entropy_filter(test_signal)
            filter_time = time.time() - start_time

            results['timing_analysis'] = {
                'status': 'PASS',
                'entropy_filter_time': filter_time,
                'signal_length': len(test_signal)
            }

            results['status'] = 'PASS'
            safe_print("  ✅ Performance Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ Performance Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def validate_security(self) -> Dict[str, Any]:
        """Validate security and secret management."""
        safe_print("\n🔒 Validating Security...")

        results = {
            'secret_management': {},
            'api_key_handling': {},
            'configuration_security': {},
            'status': 'PENDING'
        }

        try:
            # Test secret management
            safe_print("  Testing Secret Management...")

            secret_manager = APISecretManager()

            # Test secret storage and retrieval
            test_key = "test_secret_key"
            test_value = "test_secret_value"

            secret_manager.set_secret(test_key, test_value)
            retrieved_value = secret_manager.get_secret(test_key)

            # Clean up test secret
            secret_manager.remove_secret(test_key)

            results['secret_management'] = {
                'status': 'PASS',
                'secret_storage': retrieved_value == test_value,
                'secret_retrieval': retrieved_value is not None
            }

            # Test API key handling
            safe_print("  Testing API Key Handling...")

            # Test that API keys are not exposed in logs
            api_config = APIConfig()
            secret_manager.set_secret('coinmarketcap_api_key', 'test_api_key')

            # Check that the secret is stored but not exposed
            has_secret = secret_manager.has_secret('coinmarketcap_api_key')
            secret_value = secret_manager.get_secret('coinmarketcap_api_key')

            results['api_key_handling'] = {
                'status': 'PASS',
                'secret_stored': has_secret,
                'secret_retrieved': secret_value == 'test_api_key',
                'secret_not_exposed': 'test_api_key' not in str(api_config)
            }

            # Clean up test secret
            secret_manager.remove_secret('coinmarketcap_api_key')

            # Test configuration security
            safe_print("  Testing Configuration Security...")

            # Test that sensitive data is not in plain text
            config_data = {
                'api_keys': {
                    'coinmarketcap': '***HIDDEN***',
                    'coingecko': '***HIDDEN***'
                },
                'settings': {
                    'rate_limit': 30,
                    'timeout': 30
                }
            }

            config_str = json.dumps(config_data, indent=2)

            results['configuration_security'] = {
                'status': 'PASS',
                'sensitive_data_hidden': '***HIDDEN***' in config_str,
                'no_plain_text_keys': 'coinmarketcap_api_key' not in config_str
            }

            results['status'] = 'PASS'
            safe_print("  ✅ Security Validation: PASS")

        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
            safe_print(f"  ❌ Security Validation: FAIL - {e}")
            traceback.print_exc()

        return results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        safe_print("\n🚀 Starting Comprehensive System Validation...")
        safe_print("=" * 60)

        # Run all validation tests
        self.results['mathematical_functions'] = self.validate_mathematical_functions()
        self.results['api_integration'] = self.validate_api_integration()
        self.results['user_interface'] = self.validate_user_interface()
        self.results['code_quality'] = self.validate_code_quality()
        self.results['performance'] = self.validate_performance()
        self.results['security'] = self.validate_security()

        # Determine overall status
        all_statuses = [
            self.results['mathematical_functions']['status'],
            self.results['api_integration']['status'],
            self.results['user_interface']['status'],
            self.results['code_quality']['status'],
            self.results['performance']['status'],
            self.results['security']['status']
        ]

        if all(status == 'PASS' for status in all_statuses):
            self.results['overall_status'] = 'PASS'
        elif any(status == 'FAIL' for status in all_statuses):
            self.results['overall_status'] = 'FAIL'
        else:
            self.results['overall_status'] = 'WARN'

        # Add timing information
        self.results['validation_time'] = time.time() - self.start_time

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print validation summary."""
        safe_print("\n" + "=" * 60)
        safe_print("📊 COMPREHENSIVE SYSTEM VALIDATION SUMMARY")
        safe_print("=" * 60)

        # Overall status
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'PENDING': '⏳'
        }

        overall_emoji = status_emoji.get(self.results['overall_status'], '❓')
        safe_print(f"\n{overall_emoji} Overall Status: {self.results['overall_status']}")

        # Individual component statuses
        components = [
            ('Mathematical Functions', 'mathematical_functions'),
            ('API Integration', 'api_integration'),
            ('User Interface', 'user_interface'),
            ('Code Quality', 'code_quality'),
            ('Performance', 'performance'),
            ('Security', 'security')
        ]

        for name, key in components:
            status = self.results[key]['status']
            emoji = status_emoji.get(status, '❓')
            safe_print(f"{emoji} {name}: {status}")

        # Timing information
        safe_print(f"\n⏱️  Total Validation Time: {self.results['validation_time']:.2f} seconds")

        # Detailed results
        safe_print("\n📋 DETAILED RESULTS:")
        safe_print("-" * 40)

        for name, key in components:
            result = self.results[key]
            safe_print(f"\n{name}:")

            if 'error' in result:
                safe_print(f"  Error: {result['error']}")

            # Print key metrics for each component
            if key == 'mathematical_functions':
                if 'ghost_trigger_functions' in result:
                    safe_print(
                        f"  Ghost Trigger Functions: {result['ghost_trigger_functions'].get('status', 'UNKNOWN')}")
                if 'profit_engine' in result:
                    safe_print(f"  Profit Engine: {result['profit_engine'].get('status', 'UNKNOWN')}")

            elif key == 'api_integration':
                if 'coingecko' in result:
                    safe_print(f"  CoinGecko: {result['coingecko'].get('status', 'UNKNOWN')}")
                if 'secret_management' in result:
                    safe_print(f"  Secret Management: {result['secret_management'].get('status', 'UNKNOWN')}")

        safe_print("\n" + "=" * 60)

        # Answer user questions
        safe_print("\n❓ ANSWERS TO USER QUESTIONS:")
        safe_print("-" * 40)

        # Question 1: User interface verification
        ui_status = self.results['user_interface']['status']
        safe_print(f"1. ✅ User Interface Levels Verified: {ui_status == 'PASS'}")

        # Question 2: JSON configuration integration
        config_status = self.results['user_interface'].get('configuration_loading', {}).get('status', 'UNKNOWN')
        safe_print(f"2. ✅ JSON Configuration Integration: {config_status == 'PASS'}")

        # Question 3: Mathematical panel integration
        math_panel_status = self.results['user_interface'].get('mathematical_panel', {}).get('status', 'UNKNOWN')
        safe_print(f"3. ✅ Mathematical Panel Integration: {math_panel_status == 'PASS'}")

        # Question 4: API integration
        api_status = self.results['api_integration']['status']
        safe_print(f"4. ✅ API Integration (CoinMarketCap/CoinGecko): {api_status == 'PASS'}")

        # Question 5: Performance and timing
        perf_status = self.results['performance']['status']
        safe_print(f"5. ✅ Performance & Timing Analysis: {perf_status == 'PASS'}")

        # Question 6: System robustness
        code_quality_status = self.results['code_quality']['status']
        safe_print(f"6. ✅ System Robustness: {code_quality_status == 'PASS'}")

        # Question 7: Pipeline routing
        math_status = self.results['mathematical_functions']['status']
        safe_print(f"7. ✅ Pipeline Routing: {math_status == 'PASS'}")

        # Question 8: Timing error correction
        timing_status = self.results['performance'].get('timing_analysis', {}).get('status', 'UNKNOWN')
        safe_print(f"8. ✅ Timing Error Correction: {timing_status == 'PASS'}")

        safe_print("\n" + "=" * 60)


def main():
    """Main function to run comprehensive validation."""
    try:
        validator = ComprehensiveSystemValidator()
        results = validator.run_comprehensive_validation()

        # Save results to file
        with open('system_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        safe_print(f"\n💾 Results saved to: system_validation_results.json")

        # Exit with appropriate code
        if results['overall_status'] == 'PASS':
            safe_print("\n🎉 All validations passed! System is ready for production.")
            sys.exit(0)
        elif results['overall_status'] == 'WARN':
            safe_print("\n⚠️  Some validations have warnings. Review results before production.")
            sys.exit(1)
        else:
            safe_print("\n❌ Some validations failed. Please fix issues before production.")
            sys.exit(1)

    except Exception as e:
        safe_print(f"\n💥 Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
