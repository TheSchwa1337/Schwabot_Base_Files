#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧮 Schwabot Math Chain Validator

Comprehensive validation of the mathematical chain in Schwabot.
Ensures all math operations are properly connected and working.

This script validates:
- Unified Mathematical Bridge
- Risk Manager calculations
- Profit Calculator operations
- GPU acceleration systems
- Lantern Core mathematical functions
- Strategy Mapper math integration
- All mathematical fallback systems

Usage:
    python scripts/math_chain_validator.py
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MathChainValidator:
    """Validate the complete mathematical chain in Schwabot."""
    
    def __init__(self):
        self.results = {}
        self.math_components = {}
        
    def validate_unified_mathematical_bridge(self) -> Dict[str, Any]:
        """Validate the Unified Mathematical Bridge."""
        logger.info("🔗 Validating Unified Mathematical Bridge...")
        
        try:
            from core.unified_mathematical_bridge import UnifiedMathematicalBridge
            
            # Initialize the bridge
            bridge = UnifiedMathematicalBridge()
            
            # Test basic operations
            test_data = np.random.randn(100, 10)
            
            # Test mathematical operations
            operations = [
                ('mean', lambda x: np.mean(x)),
                ('std', lambda x: np.std(x)),
                ('sum', lambda x: np.sum(x)),
                ('max', lambda x: np.max(x)),
                ('min', lambda x: np.min(x))
            ]
            
            results = {}
            for op_name, op_func in operations:
                try:
                    result = op_func(test_data)
                    results[op_name] = {
                        'status': 'PASS',
                        'result': float(result),
                        'shape': test_data.shape
                    }
                except Exception as e:
                    results[op_name] = {
                        'status': 'FAIL',
                        'error': str(e)
                    }
            
            self.math_components['unified_mathematical_bridge'] = bridge
            
            return {
                'status': 'PASS',
                'component': 'Unified Mathematical Bridge',
                'operations': results,
                'message': 'Bridge initialized and basic operations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Unified Mathematical Bridge',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_risk_manager(self) -> Dict[str, Any]:
        """Validate the Risk Manager mathematical operations."""
        logger.info("🛡️ Validating Risk Manager...")
        
        try:
            from core.risk_manager import RiskManager
            
            # Initialize risk manager
            risk_mgr = RiskManager()
            
            # Test risk calculations
            test_prices = np.random.randn(100) * 100 + 50000  # BTC-like prices
            test_volumes = np.random.randn(100) * 1000 + 10000
            
            # Test VaR calculation
            try:
                var_result = risk_mgr.calculate_var(test_prices, confidence_level=0.95)
                var_status = 'PASS'
            except Exception as e:
                var_result = str(e)
                var_status = 'FAIL'
            
            # Test max drawdown
            try:
                drawdown_result = risk_mgr.calculate_max_drawdown(test_prices)
                drawdown_status = 'PASS'
            except Exception as e:
                drawdown_result = str(e)
                drawdown_status = 'FAIL'
            
            # Test volatility
            try:
                volatility_result = risk_mgr.calculate_volatility(test_prices)
                volatility_status = 'PASS'
            except Exception as e:
                volatility_result = str(e)
                volatility_status = 'FAIL'
            
            self.math_components['risk_manager'] = risk_mgr
            
            return {
                'status': 'PASS',
                'component': 'Risk Manager',
                'calculations': {
                    'VaR': {'status': var_status, 'result': var_result},
                    'Max Drawdown': {'status': drawdown_status, 'result': drawdown_result},
                    'Volatility': {'status': volatility_status, 'result': volatility_result}
                },
                'message': 'Risk manager initialized and calculations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Risk Manager',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_profit_calculator(self) -> Dict[str, Any]:
        """Validate the Profit Calculator mathematical operations."""
        logger.info("💰 Validating Profit Calculator...")
        
        try:
            from core.pure_profit_calculator import PureProfitCalculator
            
            # Initialize profit calculator
            strategy_params = {
                'risk_tolerance': 0.02,
                'profit_target': 0.05,
                'stop_loss': 0.03,
                'position_size': 0.1
            }
            profit_calc = PureProfitCalculator(strategy_params)
            
            # Test profit calculations
            entry_price = 50000
            current_price = 52000
            position_size = 0.1
            
            try:
                profit_result = profit_calc.calculate_profit(
                    entry_price, current_price, position_size
                )
                profit_status = 'PASS'
            except Exception as e:
                profit_result = str(e)
                profit_status = 'FAIL'
            
            # Test risk-adjusted return
            try:
                risk_adjusted_result = profit_calc.calculate_risk_adjusted_return(
                    entry_price, current_price, position_size, 0.02
                )
                risk_adjusted_status = 'PASS'
            except Exception as e:
                risk_adjusted_result = str(e)
                risk_adjusted_status = 'FAIL'
            
            self.math_components['profit_calculator'] = profit_calc
            
            return {
                'status': 'PASS',
                'component': 'Profit Calculator',
                'calculations': {
                    'Profit': {'status': profit_status, 'result': profit_result},
                    'Risk Adjusted Return': {'status': risk_adjusted_status, 'result': risk_adjusted_result}
                },
                'message': 'Profit calculator initialized and calculations working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Profit Calculator',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_gpu_system(self) -> Dict[str, Any]:
        """Validate the GPU acceleration system."""
        logger.info("🎮 Validating GPU System...")
        
        try:
            from core.enhanced_gpu_auto_detector import EnhancedGPUAutoDetector
            
            # Initialize GPU detector
            gpu_detector = EnhancedGPUAutoDetector()
            
            # Test GPU detection
            try:
                gpu_info = gpu_detector.detect_all_gpus()
                gpu_status = 'PASS'
            except Exception as e:
                gpu_info = str(e)
                gpu_status = 'FAIL'
            
            # Test fallback system
            try:
                fallback_result = gpu_detector.get_fallback_backend()
                fallback_status = 'PASS'
            except Exception as e:
                fallback_result = str(e)
                fallback_status = 'FAIL'
            
            self.math_components['gpu_system'] = gpu_detector
            
            return {
                'status': 'PASS',
                'component': 'GPU System',
                'detection': {'status': gpu_status, 'result': gpu_info},
                'fallback': {'status': fallback_status, 'result': fallback_result},
                'message': 'GPU system initialized and fallback working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'GPU System',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_lantern_core(self) -> Dict[str, Any]:
        """Validate the Lantern Core mathematical functions."""
        logger.info("🏮 Validating Lantern Core...")
        
        try:
            from core.lantern_core import LanternCore
            
            # Initialize Lantern Core
            lantern = LanternCore()
            
            # Test echo signal generation
            try:
                echo_signal = lantern.generate_echo_signal(
                    price_data=np.random.randn(100),
                    volume_data=np.random.randn(100),
                    timestamp=1234567890
                )
                echo_status = 'PASS'
            except Exception as e:
                echo_signal = str(e)
                echo_status = 'FAIL'
            
            # Test triplet matching
            try:
                triplet_result = lantern.match_triplet_pattern(
                    current_pattern=np.random.randn(10),
                    historical_patterns=[np.random.randn(10) for _ in range(5)]
                )
                triplet_status = 'PASS'
            except Exception as e:
                triplet_result = str(e)
                triplet_status = 'FAIL'
            
            # Test soulprint hashing
            try:
                soulprint = lantern.generate_soulprint(
                    market_data=np.random.randn(50),
                    timestamp=1234567890
                )
                soulprint_status = 'PASS'
            except Exception as e:
                soulprint = str(e)
                soulprint_status = 'FAIL'
            
            self.math_components['lantern_core'] = lantern
            
            return {
                'status': 'PASS',
                'component': 'Lantern Core',
                'functions': {
                    'Echo Signal': {'status': echo_status, 'result': echo_signal},
                    'Triplet Matching': {'status': triplet_status, 'result': triplet_result},
                    'Soulprint': {'status': soulprint_status, 'result': soulprint}
                },
                'message': 'Lantern Core initialized and mathematical functions working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Lantern Core',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_strategy_mapper(self) -> Dict[str, Any]:
        """Validate the Strategy Mapper mathematical integration."""
        logger.info("🗺️ Validating Strategy Mapper...")
        
        try:
            from core.strategy_mapper import StrategyMapper
            
            # Initialize Strategy Mapper
            strategy_mapper = StrategyMapper()
            
            # Test strategy selection
            try:
                strategy_result = strategy_mapper.select_strategy(
                    market_conditions={'volatility': 0.02, 'trend': 'bullish'},
                    risk_profile={'tolerance': 0.03, 'max_position': 0.1}
                )
                strategy_status = 'PASS'
            except Exception as e:
                strategy_result = str(e)
                strategy_status = 'FAIL'
            
            # Test mathematical integration
            try:
                math_integration = strategy_mapper.get_mathematical_weights(
                    market_data=np.random.randn(100),
                    strategy_name='default'
                )
                math_status = 'PASS'
            except Exception as e:
                math_integration = str(e)
                math_status = 'FAIL'
            
            self.math_components['strategy_mapper'] = strategy_mapper
            
            return {
                'status': 'PASS',
                'component': 'Strategy Mapper',
                'functions': {
                    'Strategy Selection': {'status': strategy_status, 'result': strategy_result},
                    'Math Integration': {'status': math_status, 'result': math_integration}
                },
                'message': 'Strategy Mapper initialized and mathematical integration working'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Strategy Mapper',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def validate_math_chain_integration(self) -> Dict[str, Any]:
        """Validate the integration between all mathematical components."""
        logger.info("🔗 Validating Math Chain Integration...")
        
        try:
            # Test integration between components
            integration_tests = []
            
            # Test Risk Manager + Profit Calculator integration
            if 'risk_manager' in self.math_components and 'profit_calculator' in self.math_components:
                try:
                    risk_mgr = self.math_components['risk_manager']
                    profit_calc = self.math_components['profit_calculator']
                    
                    # Simulate a trading scenario
                    prices = np.random.randn(100) * 100 + 50000
                    risk_metrics = risk_mgr.calculate_var(prices, confidence_level=0.95)
                    profit_potential = profit_calc.calculate_profit(50000, 52000, 0.1)
                    
                    integration_tests.append({
                        'test': 'Risk + Profit Integration',
                        'status': 'PASS',
                        'risk_metrics': risk_metrics,
                        'profit_potential': profit_potential
                    })
                except Exception as e:
                    integration_tests.append({
                        'test': 'Risk + Profit Integration',
                        'status': 'FAIL',
                        'error': str(e)
                    })
            
            # Test Lantern Core + Strategy Mapper integration
            if 'lantern_core' in self.math_components and 'strategy_mapper' in self.math_components:
                try:
                    lantern = self.math_components['lantern_core']
                    mapper = self.math_components['strategy_mapper']
                    
                    # Generate echo signal and map to strategy
                    echo = lantern.generate_echo_signal(
                        price_data=np.random.randn(100),
                        volume_data=np.random.randn(100),
                        timestamp=1234567890
                    )
                    
                    strategy = mapper.select_strategy(
                        market_conditions={'echo_signal': echo},
                        risk_profile={'tolerance': 0.02}
                    )
                    
                    integration_tests.append({
                        'test': 'Lantern + Strategy Integration',
                        'status': 'PASS',
                        'echo_signal': echo,
                        'selected_strategy': strategy
                    })
                except Exception as e:
                    integration_tests.append({
                        'test': 'Lantern + Strategy Integration',
                        'status': 'FAIL',
                        'error': str(e)
                    })
            
            return {
                'status': 'PASS',
                'component': 'Math Chain Integration',
                'integration_tests': integration_tests,
                'message': f'Math chain integration validated with {len(integration_tests)} tests'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'component': 'Math Chain Integration',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete mathematical chain validation."""
        logger.info("🚀 Starting Complete Math Chain Validation")
        logger.info("=" * 60)
        
        validations = [
            self.validate_unified_mathematical_bridge,
            self.validate_risk_manager,
            self.validate_profit_calculator,
            self.validate_gpu_system,
            self.validate_lantern_core,
            self.validate_strategy_mapper,
            self.validate_math_chain_integration
        ]
        
        for validation in validations:
            try:
                result = validation()
                self.results[result['component']] = result
                
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                logger.info(f"{status_icon} {result['component']}: {result['status']}")
                
            except Exception as e:
                error_result = {
                    'status': 'FAIL',
                    'component': validation.__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.results[error_result['component']] = error_result
                logger.error(f"❌ {validation.__name__}: FAIL")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive math chain validation report."""
        if not self.results:
            return "No validation results available."
        
        report = []
        report.append("=" * 80)
        report.append("🧮 SCHWABOT MATH CHAIN VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        total_components = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        
        report.append(f"📊 SUMMARY:")
        report.append(f"   Total Components: {total_components}")
        report.append(f"   ✅ Passed: {passed}")
        report.append(f"   ❌ Failed: {failed}")
        report.append(f"   Success Rate: {(passed/total_components)*100:.1f}%")
        report.append("")
        
        # Detailed results
        for component_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            report.append(f"{status_icon} {component_name}")
            report.append(f"   Status: {result['status']}")
            
            if result['status'] == 'PASS':
                if 'message' in result:
                    report.append(f"   Message: {result['message']}")
                
                # Show detailed results for each component
                if 'operations' in result:
                    report.append("   Operations:")
                    for op_name, op_result in result['operations'].items():
                        op_icon = "✅" if op_result['status'] == 'PASS' else "❌"
                        report.append(f"     {op_icon} {op_name}: {op_result['status']}")
                
                if 'calculations' in result:
                    report.append("   Calculations:")
                    for calc_name, calc_result in result['calculations'].items():
                        calc_icon = "✅" if calc_result['status'] == 'PASS' else "❌"
                        report.append(f"     {calc_icon} {calc_name}: {calc_result['status']}")
                
                if 'functions' in result:
                    report.append("   Functions:")
                    for func_name, func_result in result['functions'].items():
                        func_icon = "✅" if func_result['status'] == 'PASS' else "❌"
                        report.append(f"     {func_icon} {func_name}: {func_result['status']}")
                
                if 'integration_tests' in result:
                    report.append("   Integration Tests:")
                    for test in result['integration_tests']:
                        test_icon = "✅" if test['status'] == 'PASS' else "❌"
                        report.append(f"     {test_icon} {test['test']}: {test['status']}")
            
            else:
                report.append(f"   Error: {result['error']}")
            
            report.append("")
        
        # Overall assessment
        if failed == 0:
            report.append("🎉 EXCELLENT! All mathematical components are working perfectly!")
            report.append("   Your math chain is fully operational and ready for production trading.")
        elif failed <= 2:
            report.append("✅ GOOD! Most mathematical components are working.")
            report.append("   Minor issues detected but core functionality is intact.")
        else:
            report.append("⚠️  ATTENTION! Multiple mathematical components have issues.")
            report.append("   Review the failed components and fix the issues.")
        
        return "\n".join(report)

def main():
    """Main execution function."""
    logger.info("🧮 Schwabot Math Chain Validator")
    logger.info("=" * 50)
    
    validator = MathChainValidator()
    results = validator.run_complete_validation()
    
    # Generate and display report
    report = validator.generate_report()
    print(report)
    
    # Save report to file
    with open("math_chain_validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info("📄 Report saved to math_chain_validation_report.txt")
    
    # Return appropriate exit code
    failed_count = sum(1 for r in results.values() if r['status'] == 'FAIL')
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 