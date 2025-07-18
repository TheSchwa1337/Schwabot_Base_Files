#!/usr/bin/env python3
"""
Complete Component Analysis and Repair System
Identifies and fixes all failing components to achieve 100% success rate
"""

import ast
import importlib
import inspect
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComponentAnalyzer:
    """Comprehensive component analysis and repair system"""

    def __init__(self):
        self.core_dir = "core"
        self.failed_components = []
        self.successful_components = []
        self.repair_recommendations = []

    def analyze_all_components(self) -> Dict:
        """Analyze every component in the core directory"""
        print("🔍 COMPREHENSIVE COMPONENT ANALYSIS")
        print("=" * 60)

        # Get all Python files in core
        core_files = [f for f in os.listdir(self.core_dir)]
                     if f.endswith('.py') and f != '__init__.py']

        total_files = len(core_files)

        for i, file in enumerate(core_files, 1):
            print(f"[{i:2d}/{total_files}] Testing {file:<45}", end=" ")

            try:
                result = self._test_component_import(file)
                if result['success']:
                    print("✅ SUCCESS")
                    self.successful_components.append(result)
                else:
                    print(f"❌ FAILED: {result['error'][:50]}")
                    self.failed_components.append(result)

            except Exception as e:
                print(f"❌ CRITICAL: {str(e)[:50]}")
                self.failed_components.append({)}
                    'file': file,
                    'success': False,
                    'error': str(e),
                    'error_type': 'Critical'
                })

        return self._generate_analysis_report()

    def _test_component_import(self, filename: str) -> Dict:
        """Test individual component import and basic functionality"""
        module_name = filename[:-3]  # Remove .py
        module_path = f"core.{module_name}"

        try:
            # Try to import the module
            module = importlib.import_module(module_path)

            # Test basic attributes
            has_classes = self._count_classes(module)
            has_functions = self._count_functions(module)

            return {}
                'file': filename,
                'module': module_name,
                'success': True,
                'classes': has_classes,
                'functions': has_functions,
                'module_obj': module
            }

        except ImportError as e:
            return {}
                'file': filename,
                'module': module_name,
                'success': False,
                'error': str(e),
                'error_type': 'ImportError',
                'fixable': True
            }
        except SyntaxError as e:
            return {}
                'file': filename,
                'module': module_name,
                'success': False,
                'error': str(e),
                'error_type': 'SyntaxError',
                'fixable': True,
                'line_number': e.lineno
            }
        except Exception as e:
            return {}
                'file': filename,
                'module': module_name,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'fixable': False
            }

    def _count_classes(self, module) -> int:
        """Count classes in module"""
        return len([obj for name, obj in inspect.getmembers(module, inspect.isclass)])

    def _count_functions(self, module) -> int:
        """Count functions in module"""
        return len([obj for name, obj in inspect.getmembers(module, inspect.isfunction)])

    def _generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        total = len(self.successful_components) + len(self.failed_components)
        success_rate = (len(self.successful_components) / total * 100) if total > 0 else 0

        print("\n" + "=" * 60)
        print("📊 COMPONENT ANALYSIS REPORT")
        print("=" * 60)
        print(f"✅ Successful: {len(self.successful_components)}")
        print(f"❌ Failed: {len(self.failed_components)}")
        print(f"🎯 Success Rate: {success_rate:.1f}%")

        if self.failed_components:
            print(f"\n❌ FAILING COMPONENTS:")
            print("-" * 40)

            error_types = {}
            for comp in self.failed_components:
                error_type = comp.get('error_type', 'Unknown')
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(comp['file'])

            for error_type, files in error_types.items():
                print(f"\n{error_type}:")
                for file in files:
                    print(f"  - {file}")

        return {}
            'total_components': total,
            'successful': len(self.successful_components),
            'failed': len(self.failed_components),
            'success_rate': success_rate,
            'failed_components': self.failed_components,
            'successful_components': self.successful_components,
            'error_breakdown': self._categorize_errors()
        }

    def _categorize_errors(self) -> Dict:
        """Categorize errors for targeted fixing"""
        categories = {}
            'import_errors': [],
            'syntax_errors': [],
            'missing_dependencies': [],
            'circular_imports': [],
            'other_errors': []
        }

        for comp in self.failed_components:
            error_type = comp.get('error_type', '')
            error_msg = comp.get('error', '').lower()

            if 'import' in error_type.lower() or 'modulenotfound' in error_type.lower():
                if 'circular' in error_msg:
                    categories['circular_imports'].append(comp)
                else:
                    categories['import_errors'].append(comp)
            elif 'syntax' in error_type.lower():
                categories['syntax_errors'].append(comp)
            elif 'no module named' in error_msg:
                categories['missing_dependencies'].append(comp)
            else:
                categories['other_errors'].append(comp)

        return categories

    def generate_repair_plan(self, analysis_report: Dict) -> List[Dict]:
        """Generate systematic repair plan to achieve 100% success"""
        print("\n🔧 GENERATING REPAIR PLAN")
        print("=" * 60)

        repair_plan = []
        error_breakdown = analysis_report['error_breakdown']

        # Priority 1: Fix syntax errors (easiest, fixes)
        if error_breakdown['syntax_errors']:
            repair_plan.append({)}
                'priority': 1,
                'category': 'Syntax Errors',
                'components': error_breakdown['syntax_errors'],
                'strategy': 'Fix Python syntax issues',
                'auto_fixable': True
            })

        # Priority 2: Fix import errors
        if error_breakdown['import_errors']:
            repair_plan.append({)}
                'priority': 2,
                'category': 'Import Errors',
                'components': error_breakdown['import_errors'],
                'strategy': 'Resolve missing imports and dependencies',
                'auto_fixable': True
            })

        # Priority 3: Fix circular imports
        if error_breakdown['circular_imports']:
            repair_plan.append({)}
                'priority': 3,
                'category': 'Circular Imports',
                'components': error_breakdown['circular_imports'],
                'strategy': 'Restructure imports to avoid circular dependencies',
                'auto_fixable': False
            })

        # Priority 4: Handle other errors
        if error_breakdown['other_errors']:
            repair_plan.append({)}
                'priority': 4,
                'category': 'Other Errors',
                'components': error_breakdown['other_errors'],
                'strategy': 'Manual review and fixing required',
                'auto_fixable': False
            })

        # Display repair plan
        for plan_item in repair_plan:
            print(f"\nPriority {plan_item['priority']}: {plan_item['category']}")
            print(f"Strategy: {plan_item['strategy']}")
            print(f"Components ({len(plan_item['components'])}):")
            for comp in plan_item['components']:
                print(f"  - {comp['file']}: {comp['error'][:60]}")

        return repair_plan


def main():
    """Main analysis function"""
    print("🚀 SCHWABOT COMPONENT ANALYZER")
    print("Building path to 100% success rate...")
    print("")

    analyzer = ComponentAnalyzer()

    # Run comprehensive analysis
    analysis_report = analyzer.analyze_all_components()

    # Generate repair plan
    repair_plan = analyzer.generate_repair_plan(analysis_report)

    print(f"\n🎯 TARGET: {analysis_report['failed']} components need fixing")
    print(f"🎯 GOAL: Achieve 41/41 = 100% success rate")
    print(f"🎯 CURRENT: {analysis_report['successful']}/41 = {analysis_report['success_rate']:.1f}%")

    return analysis_report, repair_plan


if __name__ == "__main__":
    analysis_report, repair_plan = main() 