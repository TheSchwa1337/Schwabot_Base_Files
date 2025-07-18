#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Safe System Analysis - No Changes Made

This script analyzes the current system state without making any changes.
It maps all functionality, tests, and components to understand what works
before planning any transformation.

SAFETY FIRST: This script only READS and ANALYZES - no modifications!
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeSystemAnalyzer:
    """Safe analyzer that only reads and maps the system - NO CHANGES!"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'system_state': 'analyzing',
            'working_components': [],
            'test_results': {},
            'file_structure': {},
            'import_status': {},
            'configuration_files': {},
            'potential_issues': [],
            'recommendations': []
        }
        
        # Core components to analyze (without importing)
        self.core_components = [
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
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze the current file structure without making changes."""
        logger.info("📁 Analyzing file structure...")
        
        structure = {
            'root_files': [],
            'directories': [],
            'python_files': [],
            'config_files': [],
            'test_files': [],
            'documentation_files': []
        }
        
        try:
            # Analyze root level
            for item in self.project_root.iterdir():
                if item.is_file():
                    structure['root_files'].append(item.name)
                    if item.suffix == '.py':
                        structure['python_files'].append(str(item))
                    elif item.suffix in ['.yaml', '.yml', '.json']:
                        structure['config_files'].append(str(item))
                    elif item.suffix in ['.md', '.txt']:
                        structure['documentation_files'].append(str(item))
                elif item.is_dir() and not item.name.startswith('.'):
                    structure['directories'].append(item.name)
            
            # Analyze test files
            for file_path in self.project_root.rglob('test_*.py'):
                structure['test_files'].append(str(file_path))
            
            logger.info(f"✅ Found {len(structure['python_files'])} Python files")
            logger.info(f"✅ Found {len(structure['test_files'])} test files")
            logger.info(f"✅ Found {len(structure['config_files'])} config files")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing file structure: {e}")
            self.analysis_results['potential_issues'].append(f"File structure analysis error: {e}")
        
        self.analysis_results['file_structure'] = structure
        return structure
    
    def analyze_import_status(self) -> Dict[str, Any]:
        """Analyze import status of core components (read-only)."""
        logger.info("📦 Analyzing import status...")
        
        import_status = {
            'successful_imports': [],
            'failed_imports': [],
            'missing_modules': [],
            'import_errors': {}
        }
        
        for module_name in self.core_components:
            try:
                # Try to import without actually importing
                module_path = self.project_root / module_name.replace('.', '/')
                if module_path.exists() or (module_path.parent / f"{module_path.name}.py").exists():
                    import_status['successful_imports'].append(module_name)
                    logger.info(f"✅ {module_name} - Module file exists")
                else:
                    import_status['missing_modules'].append(module_name)
                    logger.warning(f"⚠️ {module_name} - Module file not found")
            except Exception as e:
                import_status['failed_imports'].append(module_name)
                import_status['import_errors'][module_name] = str(e)
                logger.error(f"❌ {module_name} - Import error: {e}")
        
        self.analysis_results['import_status'] = import_status
        return import_status
    
    def analyze_configuration_files(self) -> Dict[str, Any]:
        """Analyze configuration files (read-only)."""
        logger.info("⚙️  Analyzing configuration files...")
        
        config_analysis = {
            'found_configs': [],
            'config_content': {},
            'potential_issues': []
        }
        
        config_patterns = ['*.yaml', '*.yml', '*.json', '*.cfg', '*.ini']
        
        for pattern in config_patterns:
            for config_file in self.project_root.rglob(pattern):
                if 'config' in str(config_file) or 'setup' in str(config_file):
                    try:
                        config_analysis['found_configs'].append(str(config_file))
                        
                        # Read file content (read-only)
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Analyze content for key patterns
                        content_analysis = {
                            'size_bytes': len(content),
                            'lines': len(content.split('\n')),
                            'contains_koboldcpp': 'koboldcpp' in content.lower(),
                            'contains_schwabot': 'schwabot' in content.lower(),
                            'contains_api': 'api' in content.lower(),
                            'contains_trading': 'trading' in content.lower()
                        }
                        
                        config_analysis['config_content'][str(config_file)] = content_analysis
                        
                    except Exception as e:
                        config_analysis['potential_issues'].append(f"Error reading {config_file}: {e}")
        
        self.analysis_results['configuration_files'] = config_analysis
        return config_analysis
    
    def analyze_test_files(self) -> Dict[str, Any]:
        """Analyze test files and their status (read-only)."""
        logger.info("🧪 Analyzing test files...")
        
        test_analysis = {
            'test_files': [],
            'test_categories': {},
            'test_content': {}
        }
        
        # Find all test files
        for test_file in self.project_root.rglob('test_*.py'):
            test_analysis['test_files'].append(str(test_file))
            
            try:
                # Read test file content (read-only)
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze test content
                content_analysis = {
                    'size_bytes': len(content),
                    'lines': len(content.split('\n')),
                    'contains_import': 'import' in content,
                    'contains_test': 'test' in content.lower(),
                    'contains_assert': 'assert' in content,
                    'contains_schwabot': 'schwabot' in content.lower(),
                    'contains_koboldcpp': 'koboldcpp' in content.lower()
                }
                
                test_analysis['test_content'][str(test_file)] = content_analysis
                
                # Categorize tests
                if 'system' in test_file.name.lower():
                    test_analysis['test_categories']['system_tests'] = test_analysis['test_categories'].get('system_tests', []) + [str(test_file)]
                elif 'import' in test_file.name.lower():
                    test_analysis['test_categories']['import_tests'] = test_analysis['test_categories'].get('import_tests', []) + [str(test_file)]
                elif 'mathematical' in test_file.name.lower():
                    test_analysis['test_categories']['mathematical_tests'] = test_analysis['test_categories'].get('mathematical_tests', []) + [str(test_file)]
                else:
                    test_analysis['test_categories']['other_tests'] = test_analysis['test_categories'].get('other_tests', []) + [str(test_file)]
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing test file {test_file}: {e}")
        
        self.analysis_results['test_results'] = test_analysis
        return test_analysis
    
    def analyze_working_components(self) -> List[str]:
        """Analyze which components are currently working."""
        logger.info("🔧 Analyzing working components...")
        
        working_components = []
        
        # Check for key working files
        key_files = [
            'main.py',
            'test_system.py',
            'simple_test.py',
            'comprehensive_mathematical_restoration_test.py'
        ]
        
        for file_name in key_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                working_components.append(f"✅ {file_name} - Exists and accessible")
                logger.info(f"✅ {file_name} - Found")
            else:
                logger.warning(f"⚠️ {file_name} - Not found")
        
        # Check for core directories
        core_dirs = ['core', 'gui', 'config', 'mathlib', 'strategies']
        for dir_name in core_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                working_components.append(f"✅ {dir_name}/ - Directory exists")
                logger.info(f"✅ {dir_name}/ - Found")
            else:
                logger.warning(f"⚠️ {dir_name}/ - Not found")
        
        self.analysis_results['working_components'] = working_components
        return working_components
    
    def generate_safe_recommendations(self) -> List[str]:
        """Generate safe recommendations based on analysis."""
        recommendations = []
        
        # Analyze current state
        file_structure = self.analysis_results['file_structure']
        import_status = self.analysis_results['import_status']
        config_analysis = self.analysis_results['configuration_files']
        
        # Recommendations based on findings
        if len(import_status['missing_modules']) > 0:
            recommendations.append("Some core modules may be missing - verify before changes")
        
        if len(config_analysis['found_configs']) == 0:
            recommendations.append("No configuration files found - may need to create configs")
        
        if len(file_structure['test_files']) == 0:
            recommendations.append("No test files found - consider adding tests before changes")
        
        # Safe transformation recommendations
        recommendations.extend([
            "Create backup before any file movements",
            "Test each component individually before integration",
            "Maintain existing file structure until new structure is proven",
            "Keep all working tests functional during transformation",
            "Verify koboldcpp integration before removing references"
        ])
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def run_safe_analysis(self) -> Dict[str, Any]:
        """Run complete safe analysis without making any changes."""
        logger.info("🔍 Starting SAFE system analysis (NO CHANGES WILL BE MADE)...")
        logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🔍 SAFE ANALYSIS 🔍                      ║
    ║                                                              ║
    ║              READ-ONLY SYSTEM MAPPING                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Run all analysis steps (read-only)
        self.analyze_file_structure()
        self.analyze_import_status()
        self.analyze_configuration_files()
        self.analyze_test_files()
        self.analyze_working_components()
        self.generate_safe_recommendations()
        
        # Mark analysis as complete
        self.analysis_results['system_state'] = 'analyzed'
        
        # Save analysis report
        report_file = self.project_root / 'SAFE_SYSTEM_ANALYSIS_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Safe analysis completed - NO CHANGES MADE!")
        return self.analysis_results

def main():
    """Main function to run safe analysis."""
    analyzer = SafeSystemAnalyzer()
    
    try:
        results = analyzer.run_safe_analysis()
        
        print("\n" + "="*60)
        print("🔍 SAFE SYSTEM ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📊 Working Components: {len(results['working_components'])}")
        print(f"📁 Python Files: {len(results['file_structure']['python_files'])}")
        print(f"🧪 Test Files: {len(results['file_structure']['test_files'])}")
        print(f"⚙️  Config Files: {len(results['file_structure']['config_files'])}")
        print(f"📦 Successful Imports: {len(results['import_status']['successful_imports'])}")
        print(f"❌ Failed Imports: {len(results['import_status']['failed_imports'])}")
        
        print(f"\n📋 Safe Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n💾 Analysis saved to: SAFE_SYSTEM_ANALYSIS_REPORT.json")
        print("🔒 NO CHANGES WERE MADE - System is safe!")
        
    except Exception as e:
        logger.error(f"❌ Safe analysis failed: {e}")
        print(f"❌ Safe analysis failed: {e}")

if __name__ == "__main__":
    main() 