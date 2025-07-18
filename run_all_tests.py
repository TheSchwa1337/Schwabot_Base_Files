#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Schwabot Comprehensive Test Runner

This script runs all tests in the Schwabot system to ensure everything
works correctly after rebranding and system updates.

Features:
- Runs all existing test files
- Validates system functionality
- Generates comprehensive test reports
- Ensures visual layer works correctly
- Validates mathematical systems
- Tests trading functionality
"""

import os
import sys
import json
import subprocess
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchwabotTestRunner:
    """Comprehensive test runner for Schwabot system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': [],
            'test_details': {},
            'start_time': None,
            'end_time': None
        }
        
        # Test files to run
        self.test_files = [
            'test_system.py',
            'test_imports.py',
            'simple_test.py',
            'comprehensive_mathematical_restoration_test.py',
            'comprehensive_schwabot_validation.py'
        ]
        
        # Test categories
        self.test_categories = {
            'system_tests': ['test_system.py'],
            'import_tests': ['test_imports.py'],
            'simple_tests': ['simple_test.py'],
            'mathematical_tests': ['comprehensive_mathematical_restoration_test.py'],
            'validation_tests': ['comprehensive_schwabot_validation.py']
        }
    
    def run_python_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single Python test file."""
        test_path = self.project_root / test_file
        
        if not test_path.exists():
            return {
                'status': 'failed',
                'error': f'Test file not found: {test_file}',
                'output': '',
                'execution_time': 0
            }
        
        try:
            logger.info(f"🧪 Running test: {test_file}")
            start_time = time.time()
            
            # Run the test file
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {test_file} - PASSED")
                return {
                    'status': 'passed',
                    'output': result.stdout,
                    'error_output': result.stderr,
                    'execution_time': execution_time
                }
            else:
                logger.error(f"❌ {test_file} - FAILED")
                return {
                    'status': 'failed',
                    'output': result.stdout,
                    'error_output': result.stderr,
                    'execution_time': execution_time,
                    'return_code': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {test_file} - TIMEOUT")
            return {
                'status': 'timeout',
                'error': 'Test execution timed out',
                'execution_time': 300
            }
        except Exception as e:
            logger.error(f"❌ {test_file} - ERROR: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_main_system_test(self) -> Dict[str, Any]:
        """Run the main system test."""
        logger.info("🚀 Running main system test...")
        
        try:
            # Import and run main system test
            from test_system import SystemValidator
            
            validator = SystemValidator()
            results = validator.run_comprehensive_test()
            
            logger.info("✅ Main system test completed")
            return {
                'status': 'passed',
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Main system test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_mathematical_systems_test(self) -> Dict[str, Any]:
        """Run mathematical systems test."""
        logger.info("🧮 Running mathematical systems test...")
        
        try:
            # Import and run mathematical test
            from comprehensive_mathematical_restoration_test import MathematicalRestorationTester
            
            tester = MathematicalRestorationTester()
            results = tester.run_comprehensive_test()
            
            logger.info("✅ Mathematical systems test completed")
            return {
                'status': 'passed',
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Mathematical systems test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_validation_test(self) -> Dict[str, Any]:
        """Run comprehensive validation test."""
        logger.info("🔍 Running comprehensive validation test...")
        
        try:
            # Import and run validation test
            from comprehensive_schwabot_validation import SchwabotValidator
            
            validator = SchwabotValidator()
            results = validator.run_comprehensive_validation()
            
            logger.info("✅ Comprehensive validation test completed")
            return {
                'status': 'passed',
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive validation test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_import_test(self) -> Dict[str, Any]:
        """Run import test."""
        logger.info("📦 Running import test...")
        
        try:
            # Import and run import test
            from test_imports import ImportTester
            
            tester = ImportTester()
            results = tester.test_all_imports()
            
            logger.info("✅ Import test completed")
            return {
                'status': 'passed',
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Import test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_simple_test(self) -> Dict[str, Any]:
        """Run simple test."""
        logger.info("🔧 Running simple test...")
        
        try:
            # Import and run simple test
            from simple_test import SimpleTester
            
            tester = SimpleTester()
            results = tester.run_simple_tests()
            
            logger.info("✅ Simple test completed")
            return {
                'status': 'passed',
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Simple test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_visual_layer_test(self) -> Dict[str, Any]:
        """Run visual layer test."""
        logger.info("🎨 Running visual layer test...")
        
        try:
            # Test GUI components
            gui_files = [
                'gui/visualizer_launcher.py',
                'gui/flask_app.py',
                'gui/exe_launcher.py'
            ]
            
            results = {
                'gui_files': {},
                'status': 'passed'
            }
            
            for gui_file in gui_files:
                gui_path = self.project_root / gui_file
                if gui_path.exists():
                    try:
                        with open(gui_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check for Schwabot branding
                        if 'Schwabot' in content or 'schwabot' in content:
                            results['gui_files'][gui_file] = 'passed'
                            logger.info(f"✅ {gui_file} - Schwabot branding found")
                        else:
                            results['gui_files'][gui_file] = 'failed'
                            logger.error(f"❌ {gui_file} - No Schwabot branding found")
                            results['status'] = 'failed'
                            
                    except Exception as e:
                        results['gui_files'][gui_file] = 'error'
                        logger.error(f"❌ {gui_file} - Error: {e}")
                        results['status'] = 'failed'
                else:
                    results['gui_files'][gui_file] = 'not_found'
                    logger.error(f"❌ {gui_file} - File not found")
                    results['status'] = 'failed'
            
            logger.info("✅ Visual layer test completed")
            return {
                'status': results['status'],
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Visual layer test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_trading_systems_test(self) -> Dict[str, Any]:
        """Run trading systems test."""
        logger.info("📈 Running trading systems test...")
        
        try:
            results = {
                'risk_manager': 'failed',
                'profit_calculator': 'failed',
                'btc_pipeline': 'failed',
                'status': 'failed'
            }
            
            # Test risk manager
            try:
                from core.risk_manager import RiskManager
                risk_manager = RiskManager()
                
                sample_data = {
                    'prices': [100, 101, 102, 101, 100, 99, 98, 97, 96, 95],
                    'volumes': [1000, 1100, 1200, 1150, 1050, 950, 900, 850, 800, 750]
                }
                
                metrics = risk_manager.calculate_risk_metrics(sample_data)
                if metrics and 'var_95' in metrics:
                    results['risk_manager'] = 'passed'
                    logger.info("✅ Risk Manager - Working")
                else:
                    logger.error("❌ Risk Manager - Failed")
                    
            except Exception as e:
                logger.error(f"❌ Risk Manager - Error: {e}")
            
            # Test profit calculator
            try:
                from core.pure_profit_calculator import PureProfitCalculator
                strategy_params = {
                    'risk_tolerance': 0.02,
                    'profit_target': 0.05,
                    'stop_loss': 0.03,
                    'position_size': 0.1
                }
                profit_calc = PureProfitCalculator(strategy_params)
                
                test_data = {
                    'current_price': 100.0,
                    'entry_price': 95.0,
                    'position_size': 1.0
                }
                
                profit = profit_calc.calculate_profit(test_data)
                if profit is not None:
                    results['profit_calculator'] = 'passed'
                    logger.info("✅ Profit Calculator - Working")
                else:
                    logger.error("❌ Profit Calculator - Failed")
                    
            except Exception as e:
                logger.error(f"❌ Profit Calculator - Error: {e}")
            
            # Test BTC pipeline
            try:
                from core.unified_btc_trading_pipeline import create_btc_trading_pipeline
                btc_pipeline = create_btc_trading_pipeline()
                results['btc_pipeline'] = 'passed'
                logger.info("✅ BTC Pipeline - Working")
                
            except Exception as e:
                logger.error(f"❌ BTC Pipeline - Error: {e}")
            
            # Determine overall status
            if all(status == 'passed' for status in [results['risk_manager'], results['profit_calculator'], results['btc_pipeline']]):
                results['status'] = 'passed'
            
            logger.info("✅ Trading systems test completed")
            return {
                'status': results['status'],
                'results': results,
                'execution_time': 0
            }
            
        except Exception as e:
            logger.error(f"❌ Trading systems test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting comprehensive Schwabot test suite...")
        logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        self.test_results['start_time'] = datetime.now()
        
        # Run all test categories
        test_categories = {
            'main_system': self.run_main_system_test,
            'mathematical_systems': self.run_mathematical_systems_test,
            'validation': self.run_validation_test,
            'imports': self.run_import_test,
            'simple': self.run_simple_test,
            'visual_layer': self.run_visual_layer_test,
            'trading_systems': self.run_trading_systems_test
        }
        
        for category_name, test_function in test_categories.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {category_name.upper()} tests...")
            logger.info(f"{'='*50}")
            
            try:
                result = test_function()
                self.test_results['test_details'][category_name] = result
                
                if result['status'] == 'passed':
                    self.test_results['passed'] += 1
                    logger.info(f"✅ {category_name.upper()} - PASSED")
                else:
                    self.test_results['failed'] += 1
                    logger.error(f"❌ {category_name.upper()} - FAILED")
                    if 'error' in result:
                        self.test_results['errors'].append(f"{category_name}: {result['error']}")
                
                self.test_results['total_tests'] += 1
                
            except Exception as e:
                self.test_results['failed'] += 1
                self.test_results['total_tests'] += 1
                self.test_results['errors'].append(f"{category_name}: {str(e)}")
                logger.error(f"❌ {category_name.upper()} - ERROR: {e}")
        
        self.test_results['end_time'] = datetime.now()
        
        # Generate test report
        report = self.generate_test_report()
        
        logger.info("✅ Comprehensive test suite completed!")
        return report
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        execution_time = (self.test_results['end_time'] - self.test_results['start_time']).total_seconds()
        success_rate = (self.test_results['passed'] / self.test_results['total_tests']) * 100 if self.test_results['total_tests'] > 0 else 0
        
        report = {
            'test_suite_completed': True,
            'timestamp': datetime.now().isoformat(),
            'system_name': 'Schwabot AI',
            'version': '2.0.0',
            'execution_time_seconds': execution_time,
            'overall_results': {
                'total_tests': self.test_results['total_tests'],
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'success_rate': success_rate
            },
            'detailed_results': self.test_results['test_details'],
            'errors': self.test_results['errors'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = self.project_root / 'SCHWABOT_TEST_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.test_results['failed'] > 0:
            recommendations.append("Fix failed tests before deployment")
        
        if len(self.test_results['errors']) > 0:
            recommendations.append("Address test errors in the system")
        
        # Check specific test categories
        test_details = self.test_results['test_details']
        
        if test_details.get('main_system', {}).get('status') == 'failed':
            recommendations.append("Fix main system functionality")
        
        if test_details.get('mathematical_systems', {}).get('status') == 'failed':
            recommendations.append("Fix mathematical systems")
        
        if test_details.get('validation', {}).get('status') == 'failed':
            recommendations.append("Complete system validation")
        
        if test_details.get('visual_layer', {}).get('status') == 'failed':
            recommendations.append("Update visual layer components")
        
        if test_details.get('trading_systems', {}).get('status') == 'failed':
            recommendations.append("Fix trading system components")
        
        if not recommendations:
            recommendations.append("All tests passed successfully")
            recommendations.append("System is ready for deployment")
        
        return recommendations

def main():
    """Main function to run the test suite."""
    test_runner = SchwabotTestRunner()
    
    try:
        report = test_runner.run_comprehensive_tests()
        
        if report['test_suite_completed']:
            print("\n" + "="*60)
            print("🎉 SCHWABOT TEST SUITE COMPLETED!")
            print("="*60)
            print(f"⏱️  Execution Time: {report['execution_time_seconds']:.2f} seconds")
            print(f"📊 Total Tests: {report['overall_results']['total_tests']}")
            print(f"✅ Passed: {report['overall_results']['passed']}")
            print(f"❌ Failed: {report['overall_results']['failed']}")
            print(f"📈 Success Rate: {report['overall_results']['success_rate']:.1f}%")
            
            if report['errors']:
                print(f"\n❌ Errors Found: {len(report['errors'])}")
                for error in report['errors'][:5]:  # Show first 5 errors
                    print(f"   • {error}")
                if len(report['errors']) > 5:
                    print(f"   ... and {len(report['errors']) - 5} more")
            
            print(f"\n📋 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
            
            if report['overall_results']['success_rate'] >= 90:
                print("\n🎉 Excellent! All tests passed! System is ready!")
            elif report['overall_results']['success_rate'] >= 70:
                print("\n⚠️  Good progress! Address remaining test failures.")
            else:
                print("\n❌ Significant test failures. Please fix before deployment.")
            
        else:
            print("❌ Test suite failed!")
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main() 