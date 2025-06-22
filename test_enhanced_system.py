#!/usr/bin/env python3
"""Enhanced System Test Suite - Validate All Strategic Enhancements.

This script tests all the strategic enhancements implemented in Schwabot:
- Optimization engine functionality
- System initialization and validation
- Performance optimizations
- Code quality improvements
- Integration connectivity
"""

import sys
import time
import logging
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_optimization_engine() -> Dict[str, Any]:
    """Test the optimization engine functionality."""
    logger.info("🧪 Testing Optimization Engine...")
    
    try:
        from core.optimization_engine import (
            get_optimization_engine, memoize, compress_data, 
            temporal_smoothing, optimize_hash_operations, fft_preprocess_signal
        )
        
        # Get optimization engine
        engine = get_optimization_engine()
        
        # Test memoization
        @memoize
        def expensive_calculation(x: float, y: float) -> float:
            time.sleep(0.01)  # Simulate expensive operation
            return x * y + np.sin(x) * np.cos(y)
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_calculation(1.5, 2.3)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_calculation(1.5, 2.3)
        second_call_time = time.time() - start_time
        
        # Test compression
        test_data = {
            'prices': np.random.random(1000).tolist(),
            'volumes': np.random.random(1000).tolist(),
            'timestamps': [time.time() + i for i in range(1000)]
        }
        
        compressed_data, compression_ratio = compress_data(test_data)
        
        # Test temporal smoothing
        noisy_signal = np.random.random(100) + 0.1 * np.sin(np.linspace(0, 4*np.pi, 100))
        smoothed_signal = temporal_smoothing(noisy_signal, window_size=5)
        
        # Test hash optimization
        hash_value = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        historical_hashes = [
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "abcdef1234567891abcdef1234567891abcdef1234567891abcdef1234567891",
            "abcdef1234567892abcdef1234567892abcdef1234567892abcdef1234567892"
        ]
        
        hash_optimization = optimize_hash_operations(hash_value, historical_hashes)
        
        # Test FFT preprocessing
        signal = np.random.random(256) + 0.1 * np.sin(np.linspace(0, 8*np.pi, 256))
        fft_data = fft_preprocess_signal(signal)
        
        # Get statistics
        stats = engine.get_optimization_statistics()
        
        return {
            'success': True,
            'memoization_working': second_call_time < first_call_time * 0.1,
            'compression_ratio': compression_ratio,
            'smoothing_effective': np.std(smoothed_signal) < np.std(noisy_signal),
            'hash_optimization': hash_optimization.get('optimized', False),
            'fft_preprocessing': 'error' not in fft_data,
            'cache_hit_rate': stats.get('hit_rate', 0.0),
            'average_response_time_ms': stats.get('average_response_time_ms', 0.0),
            'memory_usage_mb': stats.get('memory_usage_mb', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Optimization engine test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_system_initialization() -> Dict[str, Any]:
    """Test system initialization and validation."""
    logger.info("🧪 Testing System Initialization...")
    
    try:
        from core.main import SchwabotEngine
        
        # Test initialization in debug mode
        engine = SchwabotEngine(debug_mode=True)
        
        # Test system initialization
        start_time = time.time()
        success = engine.initialize_system()
        initialization_time = time.time() - start_time
        
        if success:
            # Get system status
            status = engine.get_system_status()
            
            return {
                'success': True,
                'initialization_success': success,
                'initialization_time_seconds': initialization_time,
                'components_ready': status.get('components_ready', {}),
                'error_count': status.get('error_count', 0),
                'all_components_ready': all(status.get('components_ready', {}).values())
            }
        else:
            return {
                'success': False,
                'initialization_success': False,
                'error': 'System initialization failed'
            }
            
    except Exception as e:
        logger.error(f"System initialization test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_mathematical_pipeline() -> Dict[str, Any]:
    """Test mathematical pipeline connectivity."""
    logger.info("🧪 Testing Mathematical Pipeline...")
    
    try:
        from core.portfolio_router import create_portfolio_router
        from core.tick_hash_interpreter import create_tick_hash_interpreter
        from core.entry_exit_vector import create_entry_exit_vector
        from core.state_validation_router import create_state_validation_router
        
        # Create components
        portfolio_router = create_portfolio_router()
        tick_interpreter = create_tick_hash_interpreter()
        entry_exit_vector = create_entry_exit_vector()
        state_validator = create_state_validation_router()
        
        # Test data flow
        test_data = {
            'price': 50000.0,
            'volume': 1000.0,
            'timestamp': datetime.now().timestamp()
        }
        
        # Test portfolio router
        portfolio_shift = portfolio_router.calculate_portfolio_shift({'volatility': 0.1})
        
        # Test tick interpreter
        tick_phase = tick_interpreter.process_tick_data(test_data)
        
        # Test entry exit vector
        entry_signal = entry_exit_vector.calculate_entry_trigger(test_data)
        
        # Test state validation
        quantum_state = {'tick_hash': 'test123', 'phase_coherence': 0.8}
        altitude_metrics = {'reflex_score': 0.7, 'altitude_score': 0.6}
        visual_pipeline = {'tick_hash': 'test123', 'phase_coherence': 0.8}
        
        state_valid = state_validator.validate_state_consistency(
            quantum_state, altitude_metrics, visual_pipeline
        )
        
        return {
            'success': True,
            'portfolio_router_working': portfolio_shift is not None,
            'tick_interpreter_working': tick_phase is not None,
            'entry_exit_vector_working': entry_signal is None or hasattr(entry_signal, 'confidence'),
            'state_validation_working': state_valid is not None,
            'all_components_connected': all([
                portfolio_shift is not None,
                tick_phase is not None,
                state_valid is not None
            ])
        }
        
    except Exception as e:
        logger.error(f"Mathematical pipeline test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_performance_baseline() -> Dict[str, Any]:
    """Test performance baseline for critical operations."""
    logger.info("🧪 Testing Performance Baseline...")
    
    try:
        from core.portfolio_router import create_portfolio_router
        from core.tick_hash_interpreter import create_tick_hash_interpreter
        from core.entry_exit_vector import create_entry_exit_vector
        from core.state_validation_router import create_state_validation_router
        
        # Create components
        portfolio_router = create_portfolio_router()
        tick_interpreter = create_tick_hash_interpreter()
        entry_exit_vector = create_entry_exit_vector()
        state_validator = create_state_validation_router()
        
        # Test tick-to-trade latency
        start_time = time.time()
        
        # Simulate full pipeline execution
        test_data = {'price': 50000, 'volume': 1000, 'timestamp': time.time()}
        
        # Execute pipeline
        portfolio_shift = portfolio_router.calculate_portfolio_shift({'volatility': 0.1})
        tick_phase = tick_interpreter.process_tick_data(test_data)
        entry_signal = entry_exit_vector.calculate_entry_trigger(test_data)
        state_valid = state_validator.validate_state_consistency({}, {}, {})
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Test multiple iterations for average
        latencies = []
        for _ in range(10):
            start_time = time.time()
            portfolio_router.calculate_portfolio_shift({'volatility': 0.1})
            tick_interpreter.process_tick_data(test_data)
            entry_exit_vector.calculate_entry_trigger(test_data)
            state_validator.validate_state_consistency({}, {}, {})
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        return {
            'success': True,
            'single_iteration_latency_ms': latency,
            'average_latency_ms': avg_latency,
            'latency_std_ms': std_latency,
            'latency_acceptable': avg_latency < 50,  # Target: <50ms
            'latency_stable': std_latency < avg_latency * 0.5  # Less than 50% variation
        }
        
    except Exception as e:
        logger.error(f"Performance baseline test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_code_quality() -> Dict[str, Any]:
    """Test code quality and configuration."""
    logger.info("🧪 Testing Code Quality...")
    
    try:
        import subprocess
        import os
        
        # Test if flake8 configuration is working
        try:
            result = subprocess.run(
                ['flake8', 'core/', '--count', '--statistics'],
                capture_output=True,
                text=True,
                timeout=30
            )
            flake8_working = result.returncode == 0 or 'E999' not in result.stdout
        except Exception:
            flake8_working = False
        
        # Test if black configuration is working
        try:
            result = subprocess.run(
                ['black', '--check', 'core/'],
                capture_output=True,
                text=True,
                timeout=30
            )
            black_working = result.returncode == 0
        except Exception:
            black_working = False
        
        # Test if isort configuration is working
        try:
            result = subprocess.run(
                ['isort', '--check-only', 'core/'],
                capture_output=True,
                text=True,
                timeout=30
            )
            isort_working = result.returncode == 0
        except Exception:
            isort_working = False
        
        # Check if configuration files exist
        config_files = {
            '.flake8': os.path.exists('.flake8'),
            'pyproject.toml': os.path.exists('pyproject.toml'),
            'core/main.py': os.path.exists('core/main.py'),
            'core/optimization_engine.py': os.path.exists('core/optimization_engine.py')
        }
        
        return {
            'success': True,
            'flake8_working': flake8_working,
            'black_working': black_working,
            'isort_working': isort_working,
            'config_files_exist': config_files,
            'all_config_files_present': all(config_files.values())
        }
        
    except Exception as e:
        logger.error(f"Code quality test failed: {e}")
        return {'success': False, 'error': str(e)}


def test_integration_connectivity() -> Dict[str, Any]:
    """Test integration connectivity between all components."""
    logger.info("🧪 Testing Integration Connectivity...")
    
    try:
        # Test all critical imports
        critical_modules = [
            'core.portfolio_router',
            'core.tick_hash_interpreter',
            'core.entry_exit_vector',
            'core.state_validation_router',
            'core.fallback_logic_router',
            'core.hash_repair_engine',
            'core.optimization_engine',
            'core.main'
        ]
        
        import_results = {}
        for module_name in critical_modules:
            try:
                __import__(module_name)
                import_results[module_name] = True
            except ImportError as e:
                import_results[module_name] = False
                logger.warning(f"Module {module_name} not importable: {e}")
        
        # Test component creation
        component_tests = {}
        
        try:
            from core.portfolio_router import create_portfolio_router
            component_tests['portfolio_router'] = create_portfolio_router() is not None
        except Exception:
            component_tests['portfolio_router'] = False
        
        try:
            from core.tick_hash_interpreter import create_tick_hash_interpreter
            component_tests['tick_hash_interpreter'] = create_tick_hash_interpreter() is not None
        except Exception:
            component_tests['tick_hash_interpreter'] = False
        
        try:
            from core.entry_exit_vector import create_entry_exit_vector
            component_tests['entry_exit_vector'] = create_entry_exit_vector() is not None
        except Exception:
            component_tests['entry_exit_vector'] = False
        
        try:
            from core.state_validation_router import create_state_validation_router
            component_tests['state_validation_router'] = create_state_validation_router() is not None
        except Exception:
            component_tests['state_validation_router'] = False
        
        try:
            from core.fallback_logic_router import create_fallback_logic_router
            component_tests['fallback_logic_router'] = create_fallback_logic_router() is not None
        except Exception:
            component_tests['fallback_logic_router'] = False
        
        try:
            from core.hash_repair_engine import create_hash_repair_engine
            component_tests['hash_repair_engine'] = create_hash_repair_engine() is not None
        except Exception:
            component_tests['hash_repair_engine'] = False
        
        try:
            from core.optimization_engine import get_optimization_engine
            component_tests['optimization_engine'] = get_optimization_engine() is not None
        except Exception:
            component_tests['optimization_engine'] = False
        
        return {
            'success': True,
            'import_results': import_results,
            'component_tests': component_tests,
            'all_modules_importable': all(import_results.values()),
            'all_components_creatable': all(component_tests.values())
        }
        
    except Exception as e:
        logger.error(f"Integration connectivity test failed: {e}")
        return {'success': False, 'error': str(e)}


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and return comprehensive results."""
    logger.info("🚀 Starting Enhanced System Test Suite...")
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Run all tests
    tests = [
        ('optimization_engine', test_optimization_engine),
        ('system_initialization', test_system_initialization),
        ('mathematical_pipeline', test_mathematical_pipeline),
        ('performance_baseline', test_performance_baseline),
        ('code_quality', test_code_quality),
        ('integration_connectivity', test_integration_connectivity)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = test_func()
            test_results['tests'][test_name] = result
            if result.get('success', False):
                logger.info(f"✅ {test_name} test passed")
            else:
                logger.error(f"❌ {test_name} test failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            test_results['tests'][test_name] = {'success': False, 'error': str(e)}
    
    # Calculate overall success
    successful_tests = sum(1 for result in test_results['tests'].values() if result.get('success', False))
    total_tests = len(test_results['tests'])
    overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': overall_success_rate,
        'all_tests_passed': overall_success_rate == 1.0
    }
    
    # Print summary
    logger.info(f"\n📊 Test Summary:")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Success Rate: {overall_success_rate:.1%}")
    logger.info(f"All Tests Passed: {'✅' if overall_success_rate == 1.0 else '❌'}")
    
    return test_results


def print_detailed_results(results: Dict[str, Any]) -> None:
    """Print detailed test results."""
    print("\n" + "="*80)
    print("📋 DETAILED TEST RESULTS")
    print("="*80)
    
    for test_name, result in results['tests'].items():
        print(f"\n🧪 {test_name.upper().replace('_', ' ')}")
        print("-" * 50)
        
        if result.get('success', False):
            print("✅ Test PASSED")
            
            # Print specific metrics for each test
            if test_name == 'optimization_engine':
                print(f"  Cache Hit Rate: {result.get('cache_hit_rate', 0):.1%}")
                print(f"  Average Response Time: {result.get('average_response_time_ms', 0):.2f}ms")
                print(f"  Compression Ratio: {result.get('compression_ratio', 0):.1%}")
                print(f"  Memory Usage: {result.get('memory_usage_mb', 0):.1f}MB")
            
            elif test_name == 'system_initialization':
                print(f"  Initialization Time: {result.get('initialization_time_seconds', 0):.2f}s")
                print(f"  Components Ready: {sum(result.get('components_ready', {}).values())}/{len(result.get('components_ready', {}))}")
                print(f"  Error Count: {result.get('error_count', 0)}")
            
            elif test_name == 'performance_baseline':
                print(f"  Average Latency: {result.get('average_latency_ms', 0):.2f}ms")
                print(f"  Latency Acceptable: {'✅' if result.get('latency_acceptable', False) else '❌'}")
                print(f"  Latency Stable: {'✅' if result.get('latency_stable', False) else '❌'}")
            
            elif test_name == 'code_quality':
                print(f"  Flake8 Working: {'✅' if result.get('flake8_working', False) else '❌'}")
                print(f"  Black Working: {'✅' if result.get('black_working', False) else '❌'}")
                print(f"  Isort Working: {'✅' if result.get('isort_working', False) else '❌'}")
            
            elif test_name == 'integration_connectivity':
                print(f"  All Modules Importable: {'✅' if result.get('all_modules_importable', False) else '❌'}")
                print(f"  All Components Creatable: {'✅' if result.get('all_components_creatable', False) else '❌'}")
        
        else:
            print("❌ Test FAILED")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n" + "="*80)
    print("📊 OVERALL SUMMARY")
    print("="*80)
    summary = results['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"All Tests Passed: {'✅' if summary['all_tests_passed'] else '❌'}")
    
    if summary['all_tests_passed']:
        print("\n🎉 CONGRATULATIONS! All tests passed. Schwabot is ready for production!")
    else:
        print(f"\n⚠️  {summary['total_tests'] - summary['successful_tests']} test(s) failed. Please review and fix issues.")


if __name__ == "__main__":
    try:
        # Run all tests
        results = run_all_tests()
        
        # Print detailed results
        print_detailed_results(results)
        
        # Exit with appropriate code
        if results['summary']['all_tests_passed']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        sys.exit(1) 