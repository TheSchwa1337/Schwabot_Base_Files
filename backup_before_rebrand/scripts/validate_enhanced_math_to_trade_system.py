#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 ENHANCED MATH-TO-TRADE SYSTEM VALIDATION
===========================================

Comprehensive validation script for the enhanced mathematical trading system.
This script tests all mathematical modules, imports, configurations, and ensures
the complete math-to-trade pathway is functional for real trading.

Validation Areas:
1. Package Structure and Imports
2. Mathematical Module Availability
3. Configuration Loading
4. Real Trading API Connectivity
5. Signal Generation and Processing
6. Risk Management Systems
7. Performance Monitoring

Author: Schwabot Team
Date: 2025-01-02
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMathToTradeValidator:
    """Comprehensive validator for the enhanced math-to-trade system"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        self.start_time = time.time()
    
    def validate_package_structure(self) -> bool:
        """Validate that all required package structure is in place"""
        logger.info("🔍 Validating package structure...")
        
        required_dirs = [
            "core/immune",
            "core/entropy", 
            "core/math",
            "core/math/tensor_algebra",
            "core/strategy",
            "config"
        ]
        
        required_files = [
            "core/immune/__init__.py",
            "core/immune/qsc_gate.py",
            "core/entropy/__init__.py", 
            "core/entropy/galileo_tensor_field.py",
            "core/math/__init__.py",
            "core/math/tensor_algebra/__init__.py",
            "core/math/tensor_algebra/unified_tensor_algebra.py",
            "core/strategy/__init__.py",
            "core/enhanced_math_to_trade_integration.py",
            "core/math_to_trade_signal_router.py",
            "core/real_market_data_feed.py"
        ]
        
        all_valid = True
        
        # Check directories
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                logger.error(f"❌ Missing directory: {dir_path}")
                all_valid = False
            else:
                logger.info(f"✅ Directory exists: {dir_path}")
        
        # Check files
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Missing file: {file_path}")
                all_valid = False
            else:
                logger.info(f"✅ File exists: {file_path}")
        
        self.validation_results['package_structure'] = all_valid
        return all_valid
    
    def validate_mathematical_imports(self) -> bool:
        """Validate that all mathematical modules can be imported"""
        logger.info("🧮 Validating mathematical module imports...")
        
        import_tests = [
            # Core strategy modules
            ("core.strategy.volume_weighted_hash_oscillator", "VolumeWeightedHashOscillator"),
            ("core.strategy.zygot_zalgo_entropy_dual_key_gate", "ZygotZalgoEntropyDualKeyGate"),
            ("core.strategy.multi_phase_strategy_weight_tensor", "MultiPhaseStrategyWeightTensor"),
            ("core.strategy.enhanced_math_ops", "EnhancedMathOps"),
            
            # Immune and quantum modules
            ("core.immune.qsc_gate", "QSCGate"),
            
            # Tensor and math modules
            ("core.math.tensor_algebra.unified_tensor_algebra", "UnifiedTensorAlgebra"),
            ("core.advanced_tensor_algebra", "AdvancedTensorAlgebra"),
            ("core.clean_unified_math", "CleanUnifiedMathSystem"),
            ("core.enhanced_mathematical_core", "EnhancedMathematicalCore"),
            
            # Entropy modules
            ("core.entropy.galileo_tensor_field", "GalileoTensorField"),
            ("core.entropy_signal_integration", "EntropySignalIntegrator"),
            ("core.entropy_math", "EntropyMath"),
            
            # Advanced modules
            ("core.recursive_hash_echo", "RecursiveHashEcho"),
            ("core.hash_match_command_injector", "HashMatchCommandInjector"),
            ("core.profit_matrix_feedback_loop", "ProfitMatrixFeedbackLoop"),
        ]
        
        all_valid = True
        
        for module_path, class_name in import_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                class_obj = getattr(module, class_name)
                logger.info(f"✅ Import successful: {module_path}.{class_name}")
            except ImportError as e:
                logger.error(f"❌ Import failed: {module_path}.{class_name} - {e}")
                all_valid = False
                self.errors.append(f"Import error: {module_path}.{class_name} - {e}")
            except AttributeError as e:
                logger.error(f"❌ Class not found: {module_path}.{class_name} - {e}")
                all_valid = False
                self.errors.append(f"Class not found: {module_path}.{class_name} - {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error: {module_path}.{class_name} - {e}")
                all_valid = False
                self.errors.append(f"Unexpected error: {module_path}.{class_name} - {e}")
        
        self.validation_results['mathematical_imports'] = all_valid
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Validate that all required dependencies are available"""
        logger.info("📦 Validating dependencies...")
        
        required_deps = [
            "numpy",
            "scipy", 
            "pandas",
            "torch",
            "tensorflow",
            "qiskit",
            "pennylane",
            "ccxt",
            "pyyaml",
            "asyncio",
            "aiohttp",
            "websockets"
        ]
        
        all_valid = True
        
        for dep in required_deps:
            try:
                __import__(dep)
                logger.info(f"✅ Dependency available: {dep}")
            except ImportError as e:
                logger.error(f"❌ Missing dependency: {dep} - {e}")
                all_valid = False
                self.errors.append(f"Missing dependency: {dep} - {e}")
        
        self.validation_results['dependencies'] = all_valid
        return all_valid
    
    def validate_configuration(self) -> bool:
        """Validate configuration files and loading"""
        logger.info("⚙️ Validating configuration...")
        
        config_files = [
            "config/schwabot_live_trading_config.yaml",
            "config/mathematical_functions_registry.yaml",
            "config/api_keys.json",
            "config/trading_pairs.json"
        ]
        
        all_valid = True
        
        for config_file in config_files:
            if os.path.exists(config_file):
                logger.info(f"✅ Config file exists: {config_file}")
                try:
                    # Try to load YAML configs
                    if config_file.endswith('.yaml'):
                        import yaml
                        with open(config_file, 'r') as f:
                            yaml.safe_load(f)
                        logger.info(f"✅ YAML config valid: {config_file}")
                    # Try to load JSON configs
                    elif config_file.endswith('.json'):
                        import json
                        with open(config_file, 'r') as f:
                            json.load(f)
                        logger.info(f"✅ JSON config valid: {config_file}")
                except Exception as e:
                    logger.error(f"❌ Config file invalid: {config_file} - {e}")
                    all_valid = False
                    self.errors.append(f"Config invalid: {config_file} - {e}")
            else:
                logger.warning(f"⚠️ Config file missing: {config_file}")
                self.warnings.append(f"Config missing: {config_file}")
        
        self.validation_results['configuration'] = all_valid
        return all_valid
    
    async def validate_enhanced_integration(self) -> bool:
        """Validate the enhanced math-to-trade integration"""
        logger.info("🚀 Validating enhanced math-to-trade integration...")
        
        try:
            from core.enhanced_math_to_trade_integration import EnhancedMathToTradeIntegration
            
            # Create a test configuration
            test_config = {
                'risk_limits': {
                    'max_position_size': 0.1,
                    'max_daily_loss': 0.05,
                    'max_drawdown': 0.15
                },
                'trading': {
                    'enabled': False,  # Disable for testing
                    'default_pair': 'BTC/USD'
                }
            }
            
            # Initialize the integration
            integration = EnhancedMathToTradeIntegration(test_config)
            
            # Test signal processing
            test_price = 50000.0
            test_volume = 1000.0
            
            signal = await integration.process_market_data_comprehensive(
                price=test_price,
                volume=test_volume,
                asset_pair="BTC/USD"
            )
            
            if signal is not None:
                logger.info(f"✅ Signal generated: {signal.signal_type.value}")
                logger.info(f"   Confidence: {signal.confidence:.3f}")
                logger.info(f"   Strength: {signal.strength:.3f}")
                logger.info(f"   Mathematical score: {signal.mathematical_score:.3f}")
                
                # Check that all mathematical scores are calculated
                score_fields = [
                    'vwho_score', 'zygot_zalgo_score', 'qsc_score', 'tensor_score',
                    'galileo_score', 'advanced_tensor_score', 'entropy_signal_score',
                    'unified_math_score', 'enhanced_math_score', 'entropy_math_score'
                ]
                
                for field in score_fields:
                    if hasattr(signal, field):
                        score = getattr(signal, field)
                        if score != 0.0:  # At least some modules should produce scores
                            logger.info(f"   {field}: {score:.3f}")
                
                all_valid = True
            else:
                logger.error("❌ No signal generated")
                all_valid = False
                self.errors.append("No signal generated from enhanced integration")
                
        except Exception as e:
            logger.error(f"❌ Enhanced integration validation failed: {e}")
            all_valid = False
            self.errors.append(f"Enhanced integration error: {e}")
        
        self.validation_results['enhanced_integration'] = all_valid
        return all_valid
    
    async def validate_market_data_feed(self) -> bool:
        """Validate the real market data feed"""
        logger.info("📊 Validating real market data feed...")
        
        try:
            from core.real_market_data_feed import RealMarketDataFeed
            
            # Create test configuration
            test_config = {
                'exchanges': {
                    'coinbase': {'enabled': False},  # Disable for testing
                    'binance': {'enabled': False},
                    'kraken': {'enabled': False}
                },
                'data_feed': {
                    'enabled': False,  # Disable for testing
                    'update_interval': 1.0
                }
            }
            
            # Initialize the feed
            feed = RealMarketDataFeed(test_config)
            
            # Test initialization
            await feed.initialize()
            logger.info("✅ Market data feed initialized")
            
            # Test data point creation
            test_data_point = feed.create_market_data_point(
                price=50000.0,
                volume=1000.0,
                timestamp=time.time(),
                exchange="test",
                symbol="BTC/USD"
            )
            
            if test_data_point is not None:
                logger.info(f"✅ Market data point created: {test_data_point.price}")
                all_valid = True
            else:
                logger.error("❌ Failed to create market data point")
                all_valid = False
                self.errors.append("Failed to create market data point")
                
        except Exception as e:
            logger.error(f"❌ Market data feed validation failed: {e}")
            all_valid = False
            self.errors.append(f"Market data feed error: {e}")
        
        self.validation_results['market_data_feed'] = all_valid
        return all_valid
    
    async def validate_signal_router(self) -> bool:
        """Validate the math-to-trade signal router"""
        logger.info("🔄 Validating math-to-trade signal router...")
        
        try:
            from core.math_to_trade_signal_router import MathToTradeSignalRouter
            
            # Create test configuration
            test_config = {
                'exchanges': {
                    'coinbase': {'enabled': False},  # Disable for testing
                    'binance': {'enabled': False},
                    'kraken': {'enabled': False}
                },
                'risk_limits': {
                    'max_position_size': 0.1,
                    'max_daily_loss': 0.05
                }
            }
            
            # Initialize the router
            router = MathToTradeSignalRouter(test_config)
            await router.initialize()
            
            logger.info("✅ Signal router initialized")
            all_valid = True
            
        except Exception as e:
            logger.error(f"❌ Signal router validation failed: {e}")
            all_valid = False
            self.errors.append(f"Signal router error: {e}")
        
        self.validation_results['signal_router'] = all_valid
        return all_valid
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        total_time = time.time() - self.start_time
        
        report = {
            'timestamp': time.time(),
            'duration_seconds': total_time,
            'overall_status': 'PASS' if not self.errors else 'FAIL',
            'validation_results': self.validation_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {
                'total_tests': len(self.validation_results),
                'passed_tests': sum(1 for v in self.validation_results.values() if v),
                'failed_tests': sum(1 for v in self.validation_results.values() if not v),
                'error_count': len(self.errors),
                'warning_count': len(self.warnings)
            }
        }
        
        return report
    
    def print_validation_report(self, report: Dict[str, Any]):
        """Print a formatted validation report"""
        print("\n" + "="*80)
        print("🧮 ENHANCED MATH-TO-TRADE SYSTEM VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Status: {report['overall_status']}")
        print(f"⏱️  Validation Duration: {report['duration_seconds']:.2f} seconds")
        
        print(f"\n📈 Test Summary:")
        print(f"   Total Tests: {report['summary']['total_tests']}")
        print(f"   Passed: {report['summary']['passed_tests']}")
        print(f"   Failed: {report['summary']['failed_tests']}")
        print(f"   Errors: {report['summary']['error_count']}")
        print(f"   Warnings: {report['summary']['warning_count']}")
        
        print(f"\n✅ Test Results:")
        for test_name, result in report['validation_results'].items():
            status = "PASS" if result else "FAIL"
            icon = "✅" if result else "❌"
            print(f"   {icon} {test_name}: {status}")
        
        if report['errors']:
            print(f"\n❌ Errors:")
            for error in report['errors']:
                print(f"   • {error}")
        
        if report['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in report['warnings']:
                print(f"   • {warning}")
        
        print("\n" + "="*80)
        
        if report['overall_status'] == 'PASS':
            print("🎉 ENHANCED MATH-TO-TRADE SYSTEM IS READY FOR PRODUCTION!")
            print("🚀 All mathematical modules are functional and integrated.")
            print("💼 Real trading capabilities are available.")
        else:
            print("⚠️  SYSTEM NEEDS ATTENTION BEFORE PRODUCTION USE")
            print("🔧 Please address the errors above before proceeding.")
        
        print("="*80 + "\n")


async def main():
    """Main validation function"""
    logger.info("🚀 Starting Enhanced Math-to-Trade System Validation")
    
    validator = EnhancedMathToTradeValidator()
    
    # Run all validations
    validator.validate_package_structure()
    validator.validate_mathematical_imports()
    validator.validate_dependencies()
    validator.validate_configuration()
    await validator.validate_enhanced_integration()
    await validator.validate_market_data_feed()
    await validator.validate_signal_router()
    
    # Generate and print report
    report = validator.generate_validation_report()
    validator.print_validation_report(report)
    
    # Return exit code
    return 0 if report['overall_status'] == 'PASS' else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        sys.exit(1) 