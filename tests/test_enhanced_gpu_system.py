#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎮 Test Enhanced GPU Auto-Detection System

This script demonstrates the comprehensive GPU auto-detection system
that can handle any GPU type and automatically switch to more advanced systems.

Usage:
    python test_enhanced_gpu_system.py                    # Run basic GPU detection
    python test_enhanced_gpu_system.py --detailed         # Run detailed GPU analysis
    python test_enhanced_gpu_system.py --strategy-test    # Test strategy mapping
    python test_enhanced_gpu_system.py --fallback-test    # Test fallback system
"""

import argparse
import logging
import sys
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_gpu_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Import enhanced GPU auto-detection system
try:
    from core.enhanced_gpu_auto_detector import (
        create_enhanced_gpu_auto_detector,
        create_enhanced_gpu_logic_mapper,
        EnhancedGPUAutoDetector,
        EnhancedGPULogicMapper
    )
    ENHANCED_GPU_AVAILABLE = True
except ImportError:
    ENHANCED_GPU_AVAILABLE = False
    logger.error("❌ Enhanced GPU auto-detection system not available")
    sys.exit(1)


class EnhancedGPUTestSuite:
    """Test suite for enhanced GPU auto-detection system."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.detector = None
        self.mapper = None
        self.test_results = {}
        
    def run_basic_detection(self) -> Dict[str, Any]:
        """Run basic GPU detection test."""
        logger.info("🎮 Running Basic GPU Detection Test")
        logger.info("=" * 50)
        
        try:
            # Create detector
            self.detector = create_enhanced_gpu_auto_detector()
            
            # Run detection
            start_time = time.time()
            results = self.detector.detect_all_gpus()
            detection_time = time.time() - start_time
            
            # Analyze results
            total_gpus = (
                len(results['cuda_gpus']) + 
                len(results['opencl_gpus']) + 
                len(results['integrated_graphics'])
            )
            
            test_result = {
                'test_name': 'basic_detection',
                'status': 'passed',
                'detection_time': detection_time,
                'total_gpus_detected': total_gpus,
                'cuda_gpus': len(results['cuda_gpus']),
                'opencl_gpus': len(results['opencl_gpus']),
                'integrated_gpus': len(results['integrated_graphics']),
                'available_backends': results['available_backends'],
                'optimal_config': results['optimal_config'],
                'fallback_chain_length': len(results['fallback_chain'])
            }
            
            logger.info(f"✅ Basic detection completed in {detection_time:.3f}s")
            logger.info(f"   Total GPUs detected: {total_gpus}")
            logger.info(f"   Available backends: {results['available_backends']}")
            logger.info(f"   Optimal GPU: {results['optimal_config']['gpu_name']}")
            logger.info(f"   GPU Tier: {results['optimal_config']['gpu_tier']}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ Basic detection test failed: {e}")
            return {
                'test_name': 'basic_detection',
                'status': 'failed',
                'error': str(e)
            }
    
    def run_detailed_analysis(self) -> Dict[str, Any]:
        """Run detailed GPU analysis test."""
        logger.info("🎮 Running Detailed GPU Analysis Test")
        logger.info("=" * 50)
        
        try:
            if not self.detector:
                self.detector = create_enhanced_gpu_auto_detector()
            
            results = self.detector.detect_all_gpus()
            
            # Detailed analysis
            detailed_analysis = {
                'test_name': 'detailed_analysis',
                'status': 'passed',
                'cuda_gpus': [],
                'opencl_gpus': [],
                'integrated_gpus': [],
                'performance_analysis': {},
                'backend_compatibility': {}
            }
            
            # Analyze CUDA GPUs
            for gpu in results['cuda_gpus']:
                gpu_analysis = {
                    'name': gpu['name'],
                    'memory_gb': gpu.get('memory_gb', 0),
                    'cuda_cores': gpu.get('cuda_cores', 0),
                    'compute_capability': gpu.get('compute_capability', ''),
                    'tier': gpu.get('tier', 'unknown'),
                    'backend': gpu.get('backend', 'unknown')
                }
                detailed_analysis['cuda_gpus'].append(gpu_analysis)
            
            # Analyze OpenCL GPUs
            for gpu in results['opencl_gpus']:
                gpu_analysis = {
                    'name': gpu['name'],
                    'memory_gb': gpu.get('memory_gb', 0),
                    'cuda_cores': gpu.get('cuda_cores', 0),
                    'platform': gpu.get('platform', ''),
                    'tier': gpu.get('tier', 'unknown'),
                    'backend': gpu.get('backend', 'unknown')
                }
                detailed_analysis['opencl_gpus'].append(gpu_analysis)
            
            # Analyze integrated GPUs
            for gpu in results['integrated_graphics']:
                gpu_analysis = {
                    'name': gpu['name'],
                    'memory_gb': gpu.get('memory_gb', 0),
                    'cuda_cores': gpu.get('cuda_cores', 0),
                    'tier': gpu.get('tier', 'unknown'),
                    'backend': gpu.get('backend', 'unknown')
                }
                detailed_analysis['integrated_gpus'].append(gpu_analysis)
            
            # Performance analysis
            all_gpus = (
                results['cuda_gpus'] + 
                results['opencl_gpus'] + 
                results['integrated_graphics']
            )
            performance_scores = results.get('ranked_gpus', [])
            if all_gpus:
                # Calculate performance scores if not already present
                if not performance_scores:
                    performance_scores = []
                    for gpu in all_gpus:
                        memory_score = gpu.get('memory_gb', 0) * 10
                        compute_score = gpu.get('cuda_cores', 0) / 1000
                        tier_score = {
                            'extreme': 100,
                            'ultra': 80,
                            'high_end': 60,
                            'mid_range': 40,
                            'low_end': 20,
                            'integrated': 5
                        }.get(gpu.get('tier', 'integrated'), 5)
                        total_score = memory_score + compute_score + tier_score
                        performance_scores.append({
                            'name': gpu['name'],
                            'total_score': total_score,
                            'memory_score': memory_score,
                            'compute_score': compute_score,
                            'tier_score': tier_score
                        })
                    performance_scores.sort(key=lambda x: x['total_score'], reverse=True)
                detailed_analysis['performance_analysis'] = {
                    'ranked_gpus': performance_scores,
                    'best_gpu': performance_scores[0] if performance_scores else None,
                    'total_gpus': len(performance_scores)
                }
            else:
                # Always provide ranked_gpus, even if empty
                detailed_analysis['performance_analysis'] = {
                    'ranked_gpus': [],
                    'best_gpu': None,
                    'total_gpus': 0
                }
            
            # Backend compatibility
            detailed_analysis['backend_compatibility'] = {
                'cupy_available': 'cupy' in results['available_backends'],
                'torch_available': 'torch' in results['available_backends'],
                'opencl_available': 'opencl' in results['available_backends'],
                'numpy_available': 'numpy' in results['available_backends'],
                'recommended_backend': results['optimal_config']['backend']
            }
            
            logger.info("✅ Detailed analysis completed")
            logger.info(f"   CUDA GPUs: {len(detailed_analysis['cuda_gpus'])}")
            logger.info(f"   OpenCL GPUs: {len(detailed_analysis['opencl_gpus'])}")
            logger.info(f"   Integrated GPUs: {len(detailed_analysis['integrated_gpus'])}")
            
            if detailed_analysis['performance_analysis']['ranked_gpus']:
                best_gpu = detailed_analysis['performance_analysis']['best_gpu']
                logger.info(f"   Best GPU: {best_gpu['name']} (Score: {best_gpu['total_score']:.1f})")
            
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"❌ Detailed analysis test failed: {e}")
            return {
                'test_name': 'detailed_analysis',
                'status': 'failed',
                'error': str(e)
            }
    
    def run_strategy_mapping_test(self) -> Dict[str, Any]:
        """Run strategy mapping test."""
        logger.info("🎮 Running Strategy Mapping Test")
        logger.info("=" * 50)
        
        try:
            # Create mapper
            self.mapper = create_enhanced_gpu_logic_mapper()
            
            # Test strategy hashes
            test_hashes = [
                "test_strategy_hash_1",
                "quantum_entanglement_strategy",
                "tensor_analysis_strategy",
                "phantom_math_strategy",
                "entropy_drift_strategy"
            ]
            
            mapping_results = []
            
            for strategy_hash in test_hashes:
                start_time = time.time()
                result = self.mapper.map_strategy_to_gpu(strategy_hash)
                mapping_time = time.time() - start_time
                
                mapping_result = {
                    'strategy_hash': strategy_hash,
                    'status': result['status'],
                    'backend_used': result['backend_used'],
                    'gpu_name': result['gpu_name'],
                    'matrix_size': result['matrix_size'],
                    'mapping_time': mapping_time,
                    'tensor_analysis': 'eigenvalues' in result.get('tensor_analysis_results', {})
                }
                
                mapping_results.append(mapping_result)
                
                logger.info(f"   Strategy: {strategy_hash[:20]}...")
                logger.info(f"     Backend: {result['backend_used']}")
                logger.info(f"     GPU: {result['gpu_name']}")
                logger.info(f"     Time: {mapping_time:.3f}s")
            
            test_result = {
                'test_name': 'strategy_mapping',
                'status': 'passed',
                'total_strategies': len(test_hashes),
                'successful_mappings': len([r for r in mapping_results if r['status'] == 'success']),
                'average_mapping_time': sum(r['mapping_time'] for r in mapping_results) / len(mapping_results),
                'mapping_results': mapping_results
            }
            
            logger.info("✅ Strategy mapping test completed")
            logger.info(f"   Total strategies: {test_result['total_strategies']}")
            logger.info(f"   Successful mappings: {test_result['successful_mappings']}")
            logger.info(f"   Average mapping time: {test_result['average_mapping_time']:.3f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ Strategy mapping test failed: {e}")
            return {
                'test_name': 'strategy_mapping',
                'status': 'failed',
                'error': str(e)
            }
    
    def run_fallback_test(self) -> Dict[str, Any]:
        """Run fallback system test."""
        logger.info("🎮 Running Fallback System Test")
        logger.info("=" * 50)
        
        try:
            if not self.mapper:
                self.mapper = create_enhanced_gpu_logic_mapper()
            
            # Get fallback chain
            gpu_info = self.mapper.get_gpu_info()
            fallback_chain = gpu_info['fallback_chain']
            
            fallback_analysis = {
                'test_name': 'fallback_system',
                'status': 'passed',
                'total_fallbacks': len(fallback_chain),
                'current_fallback_index': gpu_info['current_fallback_index'],
                'current_backend': gpu_info['current_backend'],
                'fallback_chain': fallback_chain,
                'fallback_analysis': []
            }
            
            # Analyze each fallback level
            for i, fallback in enumerate(fallback_chain):
                fallback_info = {
                    'index': i,
                    'type': fallback['type'],
                    'backend': fallback['backend'],
                    'gpu_name': fallback['gpu_name'],
                    'gpu_tier': fallback['gpu_tier'],
                    'memory_limit_gb': fallback['memory_limit_gb'],
                    'status': '🟢 ACTIVE' if i == gpu_info['current_fallback_index'] else '⚪ FALLBACK'
                }
                fallback_analysis['fallback_analysis'].append(fallback_info)
            
            logger.info("✅ Fallback system test completed")
            logger.info(f"   Total fallback levels: {fallback_analysis['total_fallbacks']}")
            logger.info(f"   Current fallback index: {fallback_analysis['current_fallback_index']}")
            logger.info(f"   Current backend: {fallback_analysis['current_backend']}")
            
            return fallback_analysis
            
        except Exception as e:
            logger.error(f"❌ Fallback system test failed: {e}")
            return {
                'test_name': 'fallback_system',
                'status': 'failed',
                'error': str(e)
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("🎮 Running Comprehensive Enhanced GPU Test Suite")
        logger.info("=" * 60)
        
        test_suite_results = {
            'timestamp': time.time(),
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0
            }
        }
        
        # Run all tests
        tests = [
            self.run_basic_detection,
            self.run_detailed_analysis,
            self.run_strategy_mapping_test,
            self.run_fallback_test
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                test_suite_results['tests'].append(result)
                test_suite_results['summary']['total_tests'] += 1
                
                if result['status'] == 'passed':
                    test_suite_results['summary']['passed_tests'] += 1
                else:
                    test_suite_results['summary']['failed_tests'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed: {e}")
                test_suite_results['tests'].append({
                    'test_name': test_func.__name__,
                    'status': 'failed',
                    'error': str(e)
                })
                test_suite_results['summary']['total_tests'] += 1
                test_suite_results['summary']['failed_tests'] += 1
        
        # Calculate success rate
        success_rate = (
            test_suite_results['summary']['passed_tests'] / 
            test_suite_results['summary']['total_tests'] * 100
        ) if test_suite_results['summary']['total_tests'] > 0 else 0
        
        test_suite_results['summary']['success_rate'] = success_rate
        
        # Print summary
        logger.info("🎮 COMPREHENSIVE TEST SUITE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Tests: {test_suite_results['summary']['total_tests']}")
        logger.info(f"Passed: {test_suite_results['summary']['passed_tests']}")
        logger.info(f"Failed: {test_suite_results['summary']['failed_tests']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        return test_suite_results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test Enhanced GPU Auto-Detection System')
    parser.add_argument('--detailed', action='store_true', help='Run detailed GPU analysis')
    parser.add_argument('--strategy-test', action='store_true', help='Test strategy mapping')
    parser.add_argument('--fallback-test', action='store_true', help='Test fallback system')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    # Check if enhanced GPU system is available
    if not ENHANCED_GPU_AVAILABLE:
        logger.error("❌ Enhanced GPU auto-detection system not available")
        sys.exit(1)
    
    # Create test suite
    test_suite = EnhancedGPUTestSuite()
    
    try:
        if args.comprehensive:
            # Run comprehensive test suite
            results = test_suite.run_comprehensive_test()
            print("\n" + "="*60)
            print("🎮 COMPREHENSIVE TEST RESULTS")
            print("="*60)
            import json
            print(json.dumps(results, indent=2, default=str))
            
        elif args.detailed:
            # Run detailed analysis
            results = test_suite.run_detailed_analysis()
            print("\n" + "="*50)
            print("🎮 DETAILED GPU ANALYSIS")
            print("="*50)
            import json
            print(json.dumps(results, indent=2, default=str))
            
        elif args.strategy_test:
            # Run strategy mapping test
            results = test_suite.run_strategy_mapping_test()
            print("\n" + "="*50)
            print("🎮 STRATEGY MAPPING TEST")
            print("="*50)
            import json
            print(json.dumps(results, indent=2, default=str))
            
        elif args.fallback_test:
            # Run fallback test
            results = test_suite.run_fallback_test()
            print("\n" + "="*50)
            print("🎮 FALLBACK SYSTEM TEST")
            print("="*50)
            import json
            print(json.dumps(results, indent=2, default=str))
            
        else:
            # Run basic detection
            results = test_suite.run_basic_detection()
            print("\n" + "="*50)
            print("🎮 BASIC GPU DETECTION")
            print("="*50)
            import json
            print(json.dumps(results, indent=2, default=str))
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 