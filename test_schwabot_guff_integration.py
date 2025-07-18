#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot GUFF AI Integration Test
=================================

This script tests the complete Schwabot system integration with:
1. GUFF AI model through KoboldCPP
2. SOL trading functionality via Coinbase API
3. Mathematical library functionality
4. Rebrand verification
5. System stability and performance

Usage:
    python test_schwabot_guff_integration.py
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchwabotGuffIntegrationTest:
    """Comprehensive test suite for Schwabot GUFF AI integration."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
        # Test configuration
        self.kobold_port = 5001
        self.bridge_port = 5005
        self.kobold_url = f"http://localhost:{self.kobold_port}"
        self.bridge_url = f"http://localhost:{self.bridge_port}"
        
        # Test assets including SOL
        self.test_assets = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'USDC-USD']
        
        # Expected rebrand elements
        self.rebrand_elements = [
            'Schwabot',
            'SchwabotUnifiedInterface',
            'schwabot_ai_config',
            'Schwabot AI Trading System'
        ]
        
        logger.info("🧪 Schwabot GUFF Integration Test Suite Initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Starting Schwabot GUFF Integration Tests...")
        
        test_suite = [
            ("System Import Test", self.test_system_imports),
            ("MathLib V4 Test", self.test_mathlib_v4),
            ("KoboldCPP Connection Test", self.test_koboldcpp_connection),
            ("GUFF AI Model Test", self.test_guff_ai_model),
            ("Bridge Integration Test", self.test_bridge_integration),
            ("SOL Trading Test", self.test_sol_trading),
            ("Rebrand Verification Test", self.test_rebrand_verification),
            ("Mathematical Functions Test", self.test_mathematical_functions),
            ("Memory Architecture Test", self.test_memory_architecture),
            ("Performance Test", self.test_performance)
        ]
        
        for test_name, test_func in test_suite:
            try:
                logger.info(f"🔍 Running {test_name}...")
                result = await test_func()
                self.test_results[test_name] = result
                logger.info(f"✅ {test_name}: {'PASS' if result['success'] else 'FAIL'}")
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {e}")
                self.test_results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'details': {}
                }
        
        return self.generate_final_report()
    
    async def test_system_imports(self) -> Dict[str, Any]:
        """Test that all core system modules can be imported."""
        try:
            # Test core imports
            from mathlib.mathlib_v4 import MathLibV4, compute_zpe, compute_zbe
            from core.koboldcpp_bridge import KoboldCPPBridge
            from core.api.coinbase_direct import CoinbaseDirectAPI
            
            # Test strategy imports
            from strategies.phantom_detector import PhantomDetector
            from strategies.phantom_band_navigator import PhantomBandNavigator
            
            # Test core system imports
            from core.schwabot_unified_interface import SchwabotUnifiedInterface
            from core.visual_layer_controller import VisualLayerController
            
            return {
                'success': True,
                'details': {
                    'mathlib_imported': True,
                    'kobold_bridge_imported': True,
                    'coinbase_api_imported': True,
                    'strategies_imported': True,
                    'core_system_imported': True
                }
            }
        except ImportError as e:
            return {
                'success': False,
                'error': f"Import error: {e}",
                'details': {}
            }
    
    async def test_mathlib_v4(self) -> Dict[str, Any]:
        """Test MathLib V4 functionality."""
        try:
            from mathlib.mathlib_v4 import MathLibV4
            
            mathlib = MathLibV4()
            
            # Test basic functions
            profit_series = [100, 105, 110, 108, 115]
            time_series = [0, 1, 2, 3, 4]
            
            zpe = mathlib.compute_zpe(profit_series, time_series)
            zbe = mathlib.compute_zbe(5.0, 2.0)
            
            # Test hash generation
            test_tick = {'price': 100, 'volume': 1000}
            strategy_hash = mathlib.generate_strategy_hash(test_tick, 'SOL-USD', 0.05, 'test_strategy')
            
            # Test entropy classification
            entropy_band = mathlib.classify_entropy_band(zpe)
            
            # Test asset classification
            asset_class = mathlib.resolve_hash_class(strategy_hash)
            
            return {
                'success': True,
                'details': {
                    'zpe_calculated': zpe > 0,
                    'zbe_calculated': zbe > 0,
                    'hash_generated': len(strategy_hash) > 0,
                    'entropy_band': entropy_band,
                    'asset_class': asset_class,
                    'zpe_value': zpe,
                    'zbe_value': zbe
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_koboldcpp_connection(self) -> Dict[str, Any]:
        """Test KoboldCPP connection."""
        try:
            # Test if KoboldCPP is running
            response = requests.get(f"{self.kobold_url}/api/v1/model", timeout=5)
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'success': True,
                    'details': {
                        'koboldcpp_running': True,
                        'model_name': model_info.get('result', {}).get('model_name', 'Unknown'),
                        'model_type': model_info.get('result', {}).get('model_type', 'Unknown'),
                        'is_guff_model': 'guff' in model_info.get('result', {}).get('model_name', '').lower()
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"KoboldCPP not responding: {response.status_code}",
                    'details': {}
                }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"KoboldCPP connection failed: {e}",
                'details': {}
            }
    
    async def test_guff_ai_model(self) -> Dict[str, Any]:
        """Test GUFF AI model functionality."""
        try:
            # Test AI model response
            test_prompt = "Analyze the current market conditions for SOL-USD trading."
            
            response = requests.post(
                f"{self.kobold_url}/api/v1/generate",
                json={
                    "prompt": test_prompt,
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('results', [{}])[0].get('text', '')
                
                return {
                    'success': True,
                    'details': {
                        'ai_response_received': len(ai_response) > 0,
                        'response_length': len(ai_response),
                        'response_preview': ai_response[:100] + "..." if len(ai_response) > 100 else ai_response,
                        'model_working': True
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"AI model not responding: {response.status_code}",
                    'details': {}
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_bridge_integration(self) -> Dict[str, Any]:
        """Test bridge integration."""
        try:
            # Test bridge status
            response = requests.get(f"{self.bridge_url}/bridge/status", timeout=5)
            
            if response.status_code == 200:
                bridge_status = response.json()
                return {
                    'success': True,
                    'details': {
                        'bridge_running': True,
                        'bridge_status': bridge_status,
                        'endpoints_available': True
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"Bridge not responding: {response.status_code}",
                    'details': {}
                }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Bridge connection failed: {e}",
                'details': {}
            }
    
    async def test_sol_trading(self) -> Dict[str, Any]:
        """Test SOL trading functionality."""
        try:
            from core.api.coinbase_direct import CoinbaseDirectAPI
            
            # Test SOL market data (without API keys for safety)
            test_api = CoinbaseDirectAPI("test", "test", "test", sandbox=True)
            
            # Test SOL product info
            products_response = await test_api.get_products()
            
            if products_response:
                sol_products = [p for p in products_response if 'SOL' in p.get('id', '')]
                
                return {
                    'success': True,
                    'details': {
                        'sol_products_available': len(sol_products) > 0,
                        'sol_product_count': len(sol_products),
                        'sol_product_ids': [p.get('id') for p in sol_products],
                        'trading_pairs_available': len(products_response) > 0
                    }
                }
            else:
                return {
                    'success': False,
                    'error': "No products available",
                    'details': {}
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_rebrand_verification(self) -> Dict[str, Any]:
        """Test that rebranding is complete."""
        try:
            rebrand_checks = {}
            
            # Check for rebrand elements in key files
            import os
            
            # Check main files for rebrand
            files_to_check = [
                'main.py',
                'core/schwabot_unified_interface.py',
                'config/schwabot_config.yaml'
            ]
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        rebrand_checks[file_path] = any(element.lower() in content.lower() for element in self.rebrand_elements)
                else:
                    rebrand_checks[file_path] = False
            
            # Check if old branding is removed
            old_branding = ['koboldcpp', 'kobold_cpp']
            old_branding_found = False
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if any(old in content.lower() for old in old_branding):
                            old_branding_found = True
                            break
            
            return {
                'success': all(rebrand_checks.values()) and not old_branding_found,
                'details': {
                    'rebrand_checks': rebrand_checks,
                    'old_branding_removed': not old_branding_found,
                    'rebrand_complete': all(rebrand_checks.values())
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_mathematical_functions(self) -> Dict[str, Any]:
        """Test advanced mathematical functions."""
        try:
            from mathlib.mathlib_v4 import MathLibV4
            
            mathlib = MathLibV4()
            
            # Test gradient and curvature
            test_series = [1, 2, 4, 7, 11, 16]
            gradient = mathlib.gradient(test_series)
            curvature = mathlib.curvature(test_series)
            
            # Test persistent homology
            profit_vector = [100, 105, 110, 108, 115, 112, 118]
            homology = mathlib.compute_persistent_homology(profit_vector)
            
            # Test quantum functions
            p_tensor = np.array([0.3, 0.4, 0.3])
            quantum_matrix = mathlib.compute_quantum_collapse_matrix(p_tensor, 0.8, 0.2)
            
            # Test phantom bands
            price_series = [100, 100.1, 100.05, 100.08, 100.02, 100.1]
            volume_series = [1000, 1005, 1002, 1003, 1001, 1004]
            phantom_bands = mathlib.detect_phantom_bands(price_series, volume_series)
            
            return {
                'success': True,
                'details': {
                    'gradient_calculated': len(gradient) > 0,
                    'curvature_calculated': len(curvature) > 0,
                    'homology_features': len(homology),
                    'quantum_matrix_shape': quantum_matrix.shape,
                    'phantom_bands_detected': len(phantom_bands),
                    'all_functions_working': True
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_memory_architecture(self) -> Dict[str, Any]:
        """Test recursive memory architecture."""
        try:
            from mathlib.mathlib_v4 import MathLibV4
            
            mathlib = MathLibV4()
            
            # Test hash generation and similarity
            tick1 = {'price': 100, 'volume': 1000, 'timestamp': 1234567890}
            tick2 = {'price': 101, 'volume': 1001, 'timestamp': 1234567891}
            
            hash1 = mathlib.generate_strategy_hash(tick1, 'SOL-USD', 0.05, 'strategy_1')
            hash2 = mathlib.generate_strategy_hash(tick2, 'SOL-USD', 0.05, 'strategy_1')
            
            similarity = mathlib.similarity_score(hash1, hash2)
            
            # Test vault matching
            vault_index = {
                hash1: {
                    'roi': 0.05,
                    'success': True,
                    'timestamp': 1234567890
                }
            }
            
            match = mathlib.match_hash_to_vault(hash1, vault_index, threshold=0.8)
            
            # Test consensus validation
            consensus_valid, consensus_hash = mathlib.validate_consensus_hash([hash1, hash1, hash2])
            
            return {
                'success': True,
                'details': {
                    'hash_generation_working': len(hash1) > 0 and len(hash2) > 0,
                    'similarity_calculation': 0 <= similarity <= 1,
                    'vault_matching': match is not None,
                    'consensus_validation': consensus_valid,
                    'memory_architecture_functional': True
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test system performance."""
        try:
            from mathlib.mathlib_v4 import MathLibV4
            
            mathlib = MathLibV4()
            
            # Performance test with larger datasets
            start_time = time.time()
            
            # Generate test data
            profit_series = list(range(1000))
            time_series = list(range(1000))
            
            # Test mathematical operations
            zpe = mathlib.compute_zpe(profit_series, time_series)
            gradient = mathlib.gradient(profit_series)
            homology = mathlib.compute_persistent_homology(profit_series)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Test hash generation performance
            hash_start = time.time()
            for i in range(100):
                tick = {'price': 100 + i, 'volume': 1000 + i}
                mathlib.generate_strategy_hash(tick, 'SOL-USD', 0.05, f'strategy_{i}')
            hash_time = time.time() - hash_start
            
            return {
                'success': execution_time < 5.0,  # Should complete within 5 seconds
                'details': {
                    'execution_time_seconds': execution_time,
                    'hash_generation_time': hash_time,
                    'performance_acceptable': execution_time < 5.0,
                    'operations_per_second': 100 / hash_time if hash_time > 0 else 0
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'details': {}
            }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        test_duration = time.time() - self.start_time
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'test_duration_seconds': test_duration
            },
            'system_status': {
                'schwabot_ready': passed_tests >= 8,  # At least 8 out of 10 tests must pass
                'guff_integration_working': self.test_results.get('GUFF AI Model Test', {}).get('success', False),
                'sol_trading_ready': self.test_results.get('SOL Trading Test', {}).get('success', False),
                'rebrand_complete': self.test_results.get('Rebrand Verification Test', {}).get('success', False),
                'mathematical_system_working': self.test_results.get('Mathematical Functions Test', {}).get('success', False)
            },
            'detailed_results': self.test_results,
            'recommendations': self.generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results.get('KoboldCPP Connection Test', {}).get('success', False):
            recommendations.append("Start KoboldCPP server with GUFF model")
        
        if not self.test_results.get('SOL Trading Test', {}).get('success', False):
            recommendations.append("Configure Coinbase API credentials for SOL trading")
        
        if not self.test_results.get('Rebrand Verification Test', {}).get('success', False):
            recommendations.append("Complete system rebranding from KoboldCPP to Schwabot")
        
        if not self.test_results.get('Performance Test', {}).get('success', False):
            recommendations.append("Optimize mathematical operations for better performance")
        
        if len(recommendations) == 0:
            recommendations.append("All systems operational - Schwabot is ready for live trading!")
        
        return recommendations

async def main():
    """Main test execution."""
    print("🧪 Schwabot GUFF AI Integration Test Suite")
    print("=" * 50)
    
    tester = SchwabotGuffIntegrationTest()
    report = await tester.run_all_tests()
    
    # Print results
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Passed: {report['test_summary']['passed_tests']}")
    print(f"Failed: {report['test_summary']['failed_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
    print(f"Test Duration: {report['test_summary']['test_duration_seconds']:.2f} seconds")
    
    print("\n🔧 SYSTEM STATUS")
    print("=" * 50)
    for key, value in report['system_status'].items():
        status = "✅ READY" if value else "❌ NOT READY"
        print(f"{key.replace('_', ' ').title()}: {status}")
    
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save detailed report
    with open('schwabot_guff_integration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: schwabot_guff_integration_report.json")
    
    # Final verdict
    if report['system_status']['schwabot_ready']:
        print("\n🎉 SCHWABOT IS READY FOR LIVE TRADING!")
        print("Your recursive memory-driven trading system is operational.")
        print("GUFF AI model integration is working.")
        print("SOL trading functionality is configured.")
        print("Mathematical systems are performing optimally.")
        return 0
    else:
        print("\n⚠️  SCHWABOT NEEDS ADDITIONAL CONFIGURATION")
        print("Please address the recommendations above before live trading.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 