#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot System Validation Test
===============================
Comprehensive test script to validate all core systems are working.
"""

import sys
import os
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validator for Schwabot."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def test_core_imports(self) -> Dict[str, Any]:
        """Test all core module imports."""
        logger.info("🔍 Testing core module imports...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Test core modules
        core_modules = [
            'core.hash_config_manager',
            'core.unified_mathematical_bridge',
            'core.lantern_core_integration',
            'core.vault_orbital_bridge',
            'core.tcell_survival_engine',
            'core.symbolic_registry',
            'core.phantom_registry',
            'core.risk_manager',
            'core.pure_profit_calculator',
            'core.unified_btc_trading_pipeline',
            'core.enhanced_gpu_auto_detector',
            'mathlib',
            'mathlib.quantum_strategy',
            'mathlib.persistent_homology',
            'strategies.phantom_band_navigator'
        ]
        
        for module_name in core_modules:
            try:
                __import__(module_name)
                results['passed'] += 1
                logger.info(f"✅ {module_name} - OK")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{module_name}: {str(e)}")
                logger.error(f"❌ {module_name} - FAILED: {e}")
        
        return results
    
    def test_mathematical_systems(self) -> Dict[str, Any]:
        """Test mathematical systems."""
        logger.info("🧮 Testing mathematical systems...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Test hash config manager
            from core.hash_config_manager import HashConfigManager, get_hash_settings
            hash_manager = HashConfigManager()
            hash_settings = get_hash_settings()
            
            # Test hash generation
            test_hash = hash_manager.generate_hash_from_string("test_data")
            if test_hash and len(test_hash) > 0:
                results['passed'] += 1
                logger.info("✅ Hash Config Manager - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Hash generation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Hash Config Manager: {str(e)}")
            logger.error(f"❌ Hash Config Manager - FAILED: {e}")
        
        try:
            # Test symbolic registry
            from core.symbolic_registry import SymbolicRegistry
            registry = SymbolicRegistry()
            symbols = registry.list_all_symbols()
            if symbols:
                results['passed'] += 1
                logger.info("✅ Symbolic Registry - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Symbolic registry empty")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Symbolic Registry: {str(e)}")
            logger.error(f"❌ Symbolic Registry - FAILED: {e}")
        
        try:
            # Test mathlib
            from mathlib import MathLib, MathLibV2, MathLibV3
            math_lib = MathLib()
            result = math_lib.add(1.0, 2.0)
            if result == 3.0:
                results['passed'] += 1
                logger.info("✅ MathLib - OK")
            else:
                results['failed'] += 1
                results['errors'].append("MathLib calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"MathLib: {str(e)}")
            logger.error(f"❌ MathLib - FAILED: {e}")
        
        return results
    
    def test_trading_systems(self) -> Dict[str, Any]:
        """Test trading systems."""
        logger.info("📈 Testing trading systems...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Test risk manager
            from core.risk_manager import RiskManager
            risk_manager = RiskManager()
            
            # Test with sample data
            sample_data = {
                'prices': [100, 101, 102, 101, 100, 99, 98, 97, 96, 95],
                'volumes': [1000, 1100, 1200, 1150, 1050, 950, 900, 850, 800, 750]
            }
            
            metrics = risk_manager.calculate_risk_metrics(sample_data)
            if metrics and 'var_95' in metrics:
                results['passed'] += 1
                logger.info("✅ Risk Manager - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Risk calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Risk Manager: {str(e)}")
            logger.error(f"❌ Risk Manager - FAILED: {e}")
        
        try:
            # Test profit calculator
            from core.pure_profit_calculator import PureProfitCalculator
            strategy_params = {
                'risk_tolerance': 0.02,
                'profit_target': 0.05,
                'stop_loss': 0.03,
                'position_size': 0.1
            }
            profit_calc = PureProfitCalculator(strategy_params)
            
            # Test profit calculation
            test_data = {
                'current_price': 100.0,
                'entry_price': 95.0,
                'position_size': 1.0
            }
            
            profit = profit_calc.calculate_profit(test_data)
            if profit is not None:
                results['passed'] += 1
                logger.info("✅ Profit Calculator - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Profit calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Profit Calculator: {str(e)}")
            logger.error(f"❌ Profit Calculator - FAILED: {e}")
        
        return results
    
    def test_advanced_systems(self) -> Dict[str, Any]:
        """Test advanced systems."""
        logger.info("🚀 Testing advanced systems...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # Test quantum strategy
            from mathlib.quantum_strategy import QuantumStrategyEngine
            qse = QuantumStrategyEngine()
            
            # Test superposition strategy
            strategy = qse.create_superposition_strategy(
                "test_strategy",
                ["BTC", "ETH", "ADA"],
                [0.4, 0.4, 0.2]
            )
            
            if strategy and strategy.strategy_id == "test_strategy":
                results['passed'] += 1
                logger.info("✅ Quantum Strategy Engine - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Quantum strategy creation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Quantum Strategy: {str(e)}")
            logger.error(f"❌ Quantum Strategy Engine - FAILED: {e}")
        
        try:
            # Test persistent homology
            from mathlib.persistent_homology import PersistentHomology
            import numpy as np
            
            ph = PersistentHomology()
            
            # Test with sample data
            points = np.random.rand(10, 2)
            simplices = ph.build_simplicial_complex(points, 0.5)
            
            if simplices and len(simplices) > 0:
                results['passed'] += 1
                logger.info("✅ Persistent Homology - OK")
            else:
                results['failed'] += 1
                results['errors'].append("Persistent homology computation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Persistent Homology: {str(e)}")
            logger.error(f"❌ Persistent Homology - FAILED: {e}")
        
        return results
    
    def test_gpu_system(self) -> Dict[str, Any]:
        """Test GPU auto-detection system."""
        logger.info("🎮 Testing GPU system...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            from core.enhanced_gpu_auto_detector import create_enhanced_gpu_auto_detector
            detector = create_enhanced_gpu_auto_detector()
            gpu_info = detector.detect_all_gpus()
            
            if gpu_info and 'optimal_config' in gpu_info:
                results['passed'] += 1
                logger.info(f"✅ GPU Auto-Detection - OK (Found: {gpu_info['optimal_config']['gpu_name']})")
            else:
                results['failed'] += 1
                results['errors'].append("GPU detection failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"GPU Auto-Detection: {str(e)}")
            logger.error(f"❌ GPU Auto-Detection - FAILED: {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        logger.info("🚀 STARTING COMPREHENSIVE SCHWABOT SYSTEM TEST")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_results['core_imports'] = self.test_core_imports()
        self.test_results['mathematical_systems'] = self.test_mathematical_systems()
        self.test_results['trading_systems'] = self.test_trading_systems()
        self.test_results['advanced_systems'] = self.test_advanced_systems()
        self.test_results['gpu_system'] = self.test_gpu_system()
        
        # Calculate totals
        total_passed = sum(result['passed'] for result in self.test_results.values())
        total_failed = sum(result['failed'] for result in self.test_results.values())
        
        # Compile all errors
        all_errors = []
        for result in self.test_results.values():
            all_errors.extend(result['errors'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_tests_passed': total_passed,
            'total_tests_failed': total_failed,
            'success_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
            'all_errors': all_errors,
            'test_details': self.test_results
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Duration: {duration:.2f} seconds")
        logger.info(f"✅ Tests Passed: {total_passed}")
        logger.info(f"❌ Tests Failed: {total_failed}")
        logger.info(f"📈 Success Rate: {final_results['success_rate']:.1%}")
        
        if all_errors:
            logger.info("\n❌ ERRORS FOUND:")
            for error in all_errors:
                logger.error(f"  - {error}")
        else:
            logger.info("\n🎉 ALL SYSTEMS OPERATIONAL!")
        
        return final_results

def main():
    """Main test function."""
    validator = SystemValidator()
    results = validator.run_comprehensive_test()
    
    # Save results to file
    import json
    with open('system_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Results saved to: system_validation_results.json")
    
    # Return exit code
    if results['total_tests_failed'] == 0:
        logger.info("🎉 SYSTEM VALIDATION: SUCCESS")
        return 0
    else:
        logger.error("❌ SYSTEM VALIDATION: FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 