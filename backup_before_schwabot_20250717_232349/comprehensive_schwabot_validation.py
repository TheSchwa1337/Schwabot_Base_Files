#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Schwabot Comprehensive System Validation

This script validates all systems after rebranding to ensure everything
works correctly with the new Schwabot AI branding and functionality.

Features:
- Import validation for all core modules
- Configuration file validation
- Test system validation
- Visual layer validation
- API endpoint validation
- Documentation validation
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchwabotValidator:
    """Comprehensive validator for Schwabot system after rebranding."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.validation_results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'errors': [],
            'warnings_list': [],
            'test_results': {}
        }
        
        # Schwabot branding verification
        self.expected_branding = {
            'system_name': 'Schwabot AI',
            'system_description': 'Advanced AI-Powered Trading System',
            'version': '2.0.0',
            'author': 'Schwabot Development Team'
        }
        
        # Core modules to validate
        self.core_modules = [
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
        
        # Files that should NOT contain koboldcpp references
        self.files_to_check_for_old_references = [
            'main.py',
            'test_system.py',
            'test_imports.py',
            'simple_test.py',
            'config/schwabot_config.yaml',
            'config/schwabot_config.json',
            'config/integrations.yaml',
            'README.md',
            'SYSTEM_STATUS_REPORT.md'
        ]
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate all core module imports."""
        logger.info("🔍 Validating core module imports...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        for module_name in self.core_modules:
            try:
                __import__(module_name)
                results['passed'] += 1
                logger.info(f"✅ {module_name} - Import successful")
            except Exception as e:
                results['failed'] += 1
                error_msg = f"{module_name}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"❌ {module_name} - Import failed: {e}")
        
        self.validation_results['test_results']['imports'] = results
        return results
    
    def validate_branding_files(self) -> Dict[str, Any]:
        """Validate Schwabot branding files."""
        logger.info("🎨 Validating Schwabot branding files...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Check for branding configuration file
        branding_file = self.project_root / 'config' / 'schwabot_branding.json'
        if branding_file.exists():
            try:
                with open(branding_file, 'r', encoding='utf-8') as f:
                    branding_config = json.load(f)
                
                # Validate branding configuration
                for key, expected_value in self.expected_branding.items():
                    if key in branding_config and branding_config[key] == expected_value:
                        results['passed'] += 1
                        logger.info(f"✅ Branding {key} - Correct")
                    else:
                        results['failed'] += 1
                        error_msg = f"Branding {key} mismatch"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ Branding {key} - Incorrect")
                
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Branding file error: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"❌ Branding file error: {e}")
        else:
            results['failed'] += 1
            error_msg = "Branding file not found"
            results['errors'].append(error_msg)
            logger.error("❌ Branding file not found")
        
        # Check for logo file
        logo_file = self.project_root / 'static' / 'schwabot_logo.txt'
        if logo_file.exists():
            results['passed'] += 1
            logger.info("✅ Schwabot logo file - Found")
        else:
            results['failed'] += 1
            error_msg = "Logo file not found"
            results['errors'].append(error_msg)
            logger.error("❌ Logo file not found")
        
        self.validation_results['test_results']['branding'] = results
        return results
    
    def validate_no_old_references(self) -> Dict[str, Any]:
        """Validate that no files contain old koboldcpp references."""
        logger.info("🔍 Checking for old koboldcpp references...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        old_references = ['koboldcpp', 'KoboldCPP', 'kobold', 'Kobold']
        
        for file_path in self.files_to_check_for_old_references:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    found_old_refs = []
                    for old_ref in old_references:
                        if old_ref in content:
                            found_old_refs.append(old_ref)
                    
                    if found_old_refs:
                        results['failed'] += 1
                        error_msg = f"{file_path}: Contains old references: {found_old_refs}"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ {file_path} - Contains old references: {found_old_refs}")
                    else:
                        results['passed'] += 1
                        logger.info(f"✅ {file_path} - No old references found")
                        
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"{file_path}: Error reading file: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(f"❌ {file_path} - Error reading file: {e}")
            else:
                results['failed'] += 1
                error_msg = f"{file_path}: File not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {file_path} - File not found")
        
        self.validation_results['test_results']['no_old_references'] = results
        return results
    
    def validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files."""
        logger.info("⚙️  Validating configuration files...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        config_files = [
            'config/schwabot_config.yaml',
            'config/schwabot_config.json',
            'config/integrations.yaml',
            'config/master_integration.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    if config_path.suffix.lower() == '.json':
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    elif config_path.suffix.lower() in ['.yaml', '.yml']:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                    
                    # Check for old configuration keys
                    old_keys = ['koboldcpp', 'kobold_path', 'kobold_port', 'kobold_model']
                    found_old_keys = self._find_old_keys_in_config(config, old_keys)
                    
                    if found_old_keys:
                        results['failed'] += 1
                        error_msg = f"{config_file}: Contains old keys: {found_old_keys}"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ {config_file} - Contains old keys: {found_old_keys}")
                    else:
                        results['passed'] += 1
                        logger.info(f"✅ {config_file} - Valid configuration")
                        
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"{config_file}: Error reading config: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(f"❌ {config_file} - Error reading config: {e}")
            else:
                results['failed'] += 1
                error_msg = f"{config_file}: Config file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {config_file} - Config file not found")
        
        self.validation_results['test_results']['configuration'] = results
        return results
    
    def _find_old_keys_in_config(self, config: Any, old_keys: List[str]) -> List[str]:
        """Recursively find old keys in configuration."""
        found_keys = []
        
        if isinstance(config, dict):
            for key, value in config.items():
                for old_key in old_keys:
                    if old_key in key:
                        found_keys.append(key)
                found_keys.extend(self._find_old_keys_in_config(value, old_keys))
        elif isinstance(config, list):
            for item in config:
                found_keys.extend(self._find_old_keys_in_config(item, old_keys))
        elif isinstance(config, str):
            for old_key in old_keys:
                if old_key in config:
                    found_keys.append(config)
        
        return found_keys
    
    def validate_test_systems(self) -> Dict[str, Any]:
        """Validate test systems."""
        logger.info("🧪 Validating test systems...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        test_files = [
            'test_system.py',
            'test_imports.py',
            'simple_test.py',
            'comprehensive_mathematical_restoration_test.py'
        ]
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                try:
                    with open(test_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for Schwabot references
                    schwabot_refs = ['Schwabot', 'schwabot', 'SchwabotAI', 'schwabot_ai']
                    found_schwabot_refs = []
                    for ref in schwabot_refs:
                        if ref in content:
                            found_schwabot_refs.append(ref)
                    
                    if found_schwabot_refs:
                        results['passed'] += 1
                        logger.info(f"✅ {test_file} - Contains Schwabot references: {found_schwabot_refs}")
                    else:
                        results['failed'] += 1
                        error_msg = f"{test_file}: No Schwabot references found"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ {test_file} - No Schwabot references found")
                        
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"{test_file}: Error reading file: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(f"❌ {test_file} - Error reading file: {e}")
            else:
                results['failed'] += 1
                error_msg = f"{test_file}: Test file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {test_file} - Test file not found")
        
        self.validation_results['test_results']['test_systems'] = results
        return results
    
    def validate_mathematical_systems(self) -> Dict[str, Any]:
        """Validate mathematical systems."""
        logger.info("🧮 Validating mathematical systems...")
        
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
                logger.info("✅ Hash Config Manager - Working")
            else:
                results['failed'] += 1
                results['errors'].append("Hash generation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Hash Config Manager: {str(e)}")
            logger.error(f"❌ Hash Config Manager - Failed: {e}")
        
        try:
            # Test symbolic registry
            from core.symbolic_registry import SymbolicRegistry
            registry = SymbolicRegistry()
            symbols = registry.list_all_symbols()
            if symbols:
                results['passed'] += 1
                logger.info("✅ Symbolic Registry - Working")
            else:
                results['failed'] += 1
                results['errors'].append("Symbolic registry empty")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Symbolic Registry: {str(e)}")
            logger.error(f"❌ Symbolic Registry - Failed: {e}")
        
        try:
            # Test mathlib
            from mathlib import MathLib, MathLibV2, MathLibV3
            math_lib = MathLib()
            result = math_lib.add(1.0, 2.0)
            if result == 3.0:
                results['passed'] += 1
                logger.info("✅ MathLib - Working")
            else:
                results['failed'] += 1
                results['errors'].append("MathLib calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"MathLib: {str(e)}")
            logger.error(f"❌ MathLib - Failed: {e}")
        
        self.validation_results['test_results']['mathematical_systems'] = results
        return results
    
    def validate_trading_systems(self) -> Dict[str, Any]:
        """Validate trading systems."""
        logger.info("📈 Validating trading systems...")
        
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
                logger.info("✅ Risk Manager - Working")
            else:
                results['failed'] += 1
                results['errors'].append("Risk calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Risk Manager: {str(e)}")
            logger.error(f"❌ Risk Manager - Failed: {e}")
        
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
                logger.info("✅ Profit Calculator - Working")
            else:
                results['failed'] += 1
                results['errors'].append("Profit calculation failed")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Profit Calculator: {str(e)}")
            logger.error(f"❌ Profit Calculator - Failed: {e}")
        
        self.validation_results['test_results']['trading_systems'] = results
        return results
    
    def validate_visual_layer(self) -> Dict[str, Any]:
        """Validate visual layer components."""
        logger.info("🎨 Validating visual layer...")
        
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Check GUI components
        gui_files = [
            'gui/visualizer_launcher.py',
            'gui/flask_app.py',
            'gui/exe_launcher.py'
        ]
        
        for gui_file in gui_files:
            gui_path = self.project_root / gui_file
            if gui_path.exists():
                try:
                    with open(gui_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for Schwabot branding
                    if 'Schwabot' in content or 'schwabot' in content:
                        results['passed'] += 1
                        logger.info(f"✅ {gui_file} - Contains Schwabot branding")
                    else:
                        results['failed'] += 1
                        error_msg = f"{gui_file}: No Schwabot branding found"
                        results['errors'].append(error_msg)
                        logger.error(f"❌ {gui_file} - No Schwabot branding found")
                        
                except Exception as e:
                    results['failed'] += 1
                    error_msg = f"{gui_file}: Error reading file: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(f"❌ {gui_file} - Error reading file: {e}")
            else:
                results['failed'] += 1
                error_msg = f"{gui_file}: GUI file not found"
                results['errors'].append(error_msg)
                logger.error(f"❌ {gui_file} - GUI file not found")
        
        self.validation_results['test_results']['visual_layer'] = results
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of all systems."""
        logger.info("🚀 Starting comprehensive Schwabot validation...")
        logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Run all validation tests
        self.validate_imports()
        self.validate_branding_files()
        self.validate_no_old_references()
        self.validate_configuration_files()
        self.validate_test_systems()
        self.validate_mathematical_systems()
        self.validate_trading_systems()
        self.validate_visual_layer()
        
        # Calculate overall statistics
        for test_name, test_results in self.validation_results['test_results'].items():
            self.validation_results['passed'] += test_results['passed']
            self.validation_results['failed'] += test_results['failed']
            self.validation_results['errors'].extend(test_results['errors'])
        
        # Generate validation report
        report = self.generate_validation_report()
        
        logger.info("✅ Comprehensive validation completed!")
        return report
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'validation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'system_name': 'Schwabot AI',
            'version': '2.0.0',
            'overall_results': {
                'total_tests': self.validation_results['passed'] + self.validation_results['failed'],
                'passed': self.validation_results['passed'],
                'failed': self.validation_results['failed'],
                'success_rate': (self.validation_results['passed'] / (self.validation_results['passed'] + self.validation_results['failed'])) * 100 if (self.validation_results['passed'] + self.validation_results['failed']) > 0 else 0
            },
            'detailed_results': self.validation_results['test_results'],
            'errors': self.validation_results['errors'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_file = self.project_root / 'SCHWABOT_VALIDATION_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if self.validation_results['failed'] > 0:
            recommendations.append("Fix failed validation tests before deployment")
        
        if len(self.validation_results['errors']) > 0:
            recommendations.append("Address validation errors in the system")
        
        if self.validation_results['test_results'].get('no_old_references', {}).get('failed', 0) > 0:
            recommendations.append("Remove remaining koboldcpp references from files")
        
        if self.validation_results['test_results'].get('branding', {}).get('failed', 0) > 0:
            recommendations.append("Complete Schwabot branding setup")
        
        if self.validation_results['test_results'].get('configuration', {}).get('failed', 0) > 0:
            recommendations.append("Update configuration files with proper Schwabot settings")
        
        if self.validation_results['test_results'].get('visual_layer', {}).get('failed', 0) > 0:
            recommendations.append("Update visual layer components with Schwabot branding")
        
        if not recommendations:
            recommendations.append("System is ready for deployment")
            recommendations.append("All tests passed successfully")
        
        return recommendations

def main():
    """Main function to run the validation process."""
    validator = SchwabotValidator()
    
    try:
        report = validator.run_comprehensive_validation()
        
        if report['validation_completed']:
            print("\n" + "="*60)
            print("🎉 SCHWABOT VALIDATION COMPLETED!")
            print("="*60)
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
                print("\n🎉 Excellent! System is ready for deployment!")
            elif report['overall_results']['success_rate'] >= 70:
                print("\n⚠️  Good progress! Address remaining issues before deployment.")
            else:
                print("\n❌ Significant issues found. Please fix before deployment.")
            
        else:
            print("❌ Validation failed!")
            
    except Exception as e:
        logger.error(f"❌ Validation process failed: {e}")
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    main() 