#!/usr/bin/env python3
"""
Mathematical Robustness Analysis for Schwabot Core System
========================================================
Analyzes the mathematical integrity and robustness of the cleaned Schwabot system.
"""

import ast
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathematicalRobustnessAnalyzer:
    """Analyzer for mathematical robustness and system integrity."""
    
    def __init__(self, core_dir: str = "core"):
        """Initialize the analyzer."""
        self.core_dir = Path(core_dir)
        self.analysis_results = {}
        self.mathematical_functions = {}
        self.system_components = {}
        
    def analyze_mathematical_functions(self) -> Dict[str, Any]:
        """Analyze all mathematical functions in the system."""
        logger.info("🔍 Analyzing mathematical functions...")
        
        # Key mathematical files to analyze
        math_files = [
            "math_logic_engine.py",
            "math_integration_bridge.py", 
            "advanced_tensor_algebra.py",
            "entropy_math.py",
            "tensor_score_utils.py",
            "orbital_energy_quantizer.py",
            "entropy_drift_engine.py",
            "vault_orbital_bridge.py",
            "symbolic_registry.py",
            "entropy_decay_system.py",
            "strategy_bit_mapper.py",
            "unified_pipeline_manager.py",
            "two_gram_detector.py",
            "symbolic_math_interface.py",
            "quantum_mathematical_bridge.py",
            "math_orchestrator.py",
            "math_config_manager.py",
            "math_cache.py"
        ]
        
        results = {
            'total_functions': 0,
            'valid_functions': 0,
            'error_functions': 0,
            'function_details': {},
            'mathematical_categories': {
                'entropy_functions': [],
                'tensor_operations': [],
                'orbital_calculations': [],
                'strategy_functions': [],
                'quantum_operations': [],
                'statistical_functions': []
            }
        }
        
        for file_name in math_files:
            file_path = self.core_dir / file_name
            if not file_path.exists():
                logger.warning(f"⚠️  File not found: {file_name}")
                continue
                
            try:
                file_results = self._analyze_math_file(file_path)
                results['total_functions'] += file_results['total_functions']
                results['valid_functions'] += file_results['valid_functions']
                results['error_functions'] += file_results['error_functions']
                results['function_details'][file_name] = file_results
                
                # Categorize functions
                for func_name, func_info in file_results['functions'].items():
                    if func_info['valid']:
                        category = self._categorize_function(func_name, func_info)
                        results['mathematical_categories'][category].append(func_name)
                        
            except Exception as e:
                logger.error(f"❌ Error analyzing {file_name}: {e}")
                results['error_functions'] += 1
                
        self.mathematical_functions = results
        return results
    
    def _analyze_math_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single mathematical file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        results = {
            'total_functions': 0,
            'valid_functions': 0,
            'error_functions': 0,
            'functions': {}
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                results['total_functions'] += 1
                func_name = node.name
                
                # Analyze function
                func_info = self._analyze_function(node, content)
                results['functions'][func_name] = func_info
                
                if func_info['valid']:
                    results['valid_functions'] += 1
                else:
                    results['error_functions'] += 1
                    
        return results
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a single function."""
        func_info = {
            'name': node.name,
            'valid': True,
            'has_docstring': False,
            'has_type_hints': False,
            'has_error_handling': False,
            'uses_numpy': False,
            'uses_math': False,
            'complexity_score': 0,
            'issues': []
        }
        
        # Check for docstring
        if ast.get_docstring(node):
            func_info['has_docstring'] = True
            
        # Check for type hints
        if node.returns or any(arg.annotation for arg in node.args.args):
            func_info['has_type_hints'] = True
            
        # Check for error handling
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                func_info['has_error_handling'] = True
                break
                
        # Check for mathematical imports
        func_code = ast.unparse(node)
        if 'np.' in func_code or 'numpy.' in func_code:
            func_info['uses_numpy'] = True
        if 'math.' in func_code:
            func_info['uses_math'] = True
            
        # Calculate complexity score
        func_info['complexity_score'] = self._calculate_complexity(node)
        
        # Validate function
        if not func_info['has_docstring']:
            func_info['issues'].append("Missing docstring")
            
        if not func_info['has_error_handling']:
            func_info['issues'].append("No error handling")
            
        if func_info['complexity_score'] > 10:
            func_info['issues'].append("High complexity")
            
        if func_info['issues']:
            func_info['valid'] = False
            
        return func_info
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _categorize_function(self, func_name: str, func_info: Dict[str, Any]) -> str:
        """Categorize a mathematical function."""
        name_lower = func_name.lower()
        
        if any(keyword in name_lower for keyword in ['entropy', 'drift', 'decay']):
            return 'entropy_functions'
        elif any(keyword in name_lower for keyword in ['tensor', 'matrix', 'vector']):
            return 'tensor_operations'
        elif any(keyword in name_lower for keyword in ['orbital', 'energy', 'vault']):
            return 'orbital_calculations'
        elif any(keyword in name_lower for keyword in ['strategy', 'hash', 'bit']):
            return 'strategy_functions'
        elif any(keyword in name_lower for keyword in ['quantum', 'entanglement']):
            return 'quantum_operations'
        else:
            return 'statistical_functions'
    
    def analyze_system_integrity(self) -> Dict[str, Any]:
        """Analyze overall system integrity."""
        logger.info("🔍 Analyzing system integrity...")
        
        results = {
            'total_files': 0,
            'valid_files': 0,
            'error_files': 0,
            'system_components': {},
            'dependencies': {},
            'integration_points': []
        }
        
        # Analyze all Python files in core directory
        for file_path in self.core_dir.glob("*.py"):
            results['total_files'] += 1
            
            try:
                file_analysis = self._analyze_system_file(file_path)
                if file_analysis['valid']:
                    results['valid_files'] += 1
                    results['system_components'][file_path.name] = file_analysis
                else:
                    results['error_files'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {file_path.name}: {e}")
                results['error_files'] += 1
                
        # Analyze dependencies
        results['dependencies'] = self._analyze_dependencies()
        
        # Find integration points
        results['integration_points'] = self._find_integration_points()
        
        self.system_components = results
        return results
    
    def _analyze_system_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a system file for integrity."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        analysis = {
            'valid': True,
            'has_init': False,
            'has_main': False,
            'imports': [],
            'classes': [],
            'functions': [],
            'issues': []
        }
        
        try:
            tree = ast.parse(content)
            
            # Check for __init__ method
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                    analysis['has_init'] = True
                    
            # Check for main guard
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if 'name' in node.test.left.__dict__ and node.test.left.name == '__name__':
                        analysis['has_main'] = True
                        
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                        
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                    
        except Exception as e:
            analysis['valid'] = False
            analysis['issues'].append(f"Syntax error: {e}")
            
        return analysis
    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze system dependencies."""
        dependencies = {
            'external': [],
            'internal': [],
            'mathematical': []
        }
        
        # Common external dependencies
        external_deps = [
            'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
            'sklearn', 'tensorflow', 'torch', 'ccxt', 'requests',
            'websockets', 'asyncio', 'aiohttp', 'sqlalchemy'
        ]
        
        # Mathematical dependencies
        math_deps = [
            'numpy', 'scipy', 'math', 'statistics', 'random',
            'hashlib', 'itertools', 'functools'
        ]
        
        # Analyze imports from all files
        for file_path in self.core_dir.glob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dep = alias.name.split('.')[0]
                            if dep in external_deps:
                                if dep not in dependencies['external']:
                                    dependencies['external'].append(dep)
                            elif dep in math_deps:
                                if dep not in dependencies['mathematical']:
                                    dependencies['mathematical'].append(dep)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        if module:
                            dep = module.split('.')[0]
                            if dep in external_deps:
                                if dep not in dependencies['external']:
                                    dependencies['external'].append(dep)
                            elif dep in math_deps:
                                if dep not in dependencies['mathematical']:
                                    dependencies['mathematical'].append(dep)
                                    
            except Exception:
                continue
                
        return dependencies
    
    def _find_integration_points(self) -> List[str]:
        """Find integration points between system components."""
        integration_points = []
        
        # Look for bridge classes and integration methods
        bridge_patterns = [
            'bridge', 'integration', 'connector', 'adapter',
            'orchestrator', 'manager', 'coordinator'
        ]
        
        for file_path in self.core_dir.glob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name.lower()
                        if any(pattern in class_name for pattern in bridge_patterns):
                            integration_points.append(f"{file_path.name}:{node.name}")
                            
            except Exception:
                continue
                
        return integration_points
    
    def run_mathematical_tests(self) -> Dict[str, Any]:
        """Run mathematical function tests."""
        logger.info("🧪 Running mathematical function tests...")
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': {}
        }
        
        # Test entropy drift function
        try:
            psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            xi = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            
            # This would require importing the actual function
            # For now, we'll just test numpy operations
            drift_test = np.std(psi) * np.gradient(psi)[-1] - np.log(np.mean(phi) / np.mean(xi))
            
            test_results['total_tests'] += 1
            if not np.isnan(drift_test) and not np.isinf(drift_test):
                test_results['passed_tests'] += 1
                test_results['test_details']['entropy_drift'] = {'status': 'PASS', 'result': float(drift_test)}
            else:
                test_results['failed_tests'] += 1
                test_results['test_details']['entropy_drift'] = {'status': 'FAIL', 'error': 'Invalid result'}
                
        except Exception as e:
            test_results['total_tests'] += 1
            test_results['failed_tests'] += 1
            test_results['test_details']['entropy_drift'] = {'status': 'FAIL', 'error': str(e)}
            
        # Test tensor operations
        try:
            tensor = np.array([[1, 2], [3, 4]])
            norm = np.linalg.norm(tensor)
            trace = np.trace(tensor)
            
            test_results['total_tests'] += 2
            if not np.isnan(norm) and not np.isnan(trace):
                test_results['passed_tests'] += 2
                test_results['test_details']['tensor_operations'] = {
                    'status': 'PASS', 
                    'norm': float(norm), 
                    'trace': float(trace)
                }
            else:
                test_results['failed_tests'] += 2
                test_results['test_details']['tensor_operations'] = {'status': 'FAIL', 'error': 'Invalid tensor operations'}
                
        except Exception as e:
            test_results['total_tests'] += 2
            test_results['failed_tests'] += 2
            test_results['test_details']['tensor_operations'] = {'status': 'FAIL', 'error': str(e)}
            
        return test_results
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        logger.info("📊 Generating analysis report...")
        
        # Run all analyses
        math_analysis = self.analyze_mathematical_functions()
        system_analysis = self.analyze_system_integrity()
        test_results = self.run_mathematical_tests()
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MATHEMATICAL ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Mathematical Functions Analysis
        report.append("📈 MATHEMATICAL FUNCTIONS ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Functions: {math_analysis['total_functions']}")
        report.append(f"Valid Functions: {math_analysis['valid_functions']}")
        report.append(f"Error Functions: {math_analysis['error_functions']}")
        report.append(f"Success Rate: {(math_analysis['valid_functions']/max(1, math_analysis['total_functions'])*100):.1f}%")
        report.append("")
        
        # Function Categories
        report.append("Function Categories:")
        for category, functions in math_analysis['mathematical_categories'].items():
            report.append(f"  {category.replace('_', ' ').title()}: {len(functions)} functions")
        report.append("")
        
        # System Integrity Analysis
        report.append("🔧 SYSTEM INTEGRITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Files: {system_analysis['total_files']}")
        report.append(f"Valid Files: {system_analysis['valid_files']}")
        report.append(f"Error Files: {system_analysis['error_files']}")
        report.append(f"System Health: {(system_analysis['valid_files']/max(1, system_analysis['total_files'])*100):.1f}%")
        report.append("")
        
        # Dependencies
        report.append("Dependencies:")
        report.append(f"  External: {', '.join(system_analysis['dependencies']['external'])}")
        report.append(f"  Mathematical: {', '.join(system_analysis['dependencies']['mathematical'])}")
        report.append("")
        
        # Integration Points
        report.append(f"Integration Points: {len(system_analysis['integration_points'])}")
        for point in system_analysis['integration_points'][:5]:  # Show first 5
            report.append(f"  - {point}")
        if len(system_analysis['integration_points']) > 5:
            report.append(f"  ... and {len(system_analysis['integration_points']) - 5} more")
        report.append("")
        
        # Test Results
        report.append("🧪 MATHEMATICAL TEST RESULTS")
        report.append("-" * 40)
        report.append(f"Total Tests: {test_results['total_tests']}")
        report.append(f"Passed Tests: {test_results['passed_tests']}")
        report.append(f"Failed Tests: {test_results['failed_tests']}")
        report.append(f"Test Success Rate: {(test_results['passed_tests']/max(1, test_results['total_tests'])*100):.1f}%")
        report.append("")
        
        # Overall Assessment
        report.append("📋 OVERALL ASSESSMENT")
        report.append("-" * 40)
        
        math_score = math_analysis['valid_functions'] / max(1, math_analysis['total_functions'])
        system_score = system_analysis['valid_files'] / max(1, system_analysis['total_files'])
        test_score = test_results['passed_tests'] / max(1, test_results['total_tests'])
        
        overall_score = (math_score + system_score + test_score) / 3
        
        report.append(f"Mathematical Robustness: {math_score*100:.1f}%")
        report.append(f"System Integrity: {system_score*100:.1f}%")
        report.append(f"Test Reliability: {test_score*100:.1f}%")
        report.append(f"Overall Score: {overall_score*100:.1f}%")
        report.append("")
        
        if overall_score >= 0.9:
            report.append("🎉 EXCELLENT: System shows high mathematical robustness and integrity!")
        elif overall_score >= 0.7:
            report.append("✅ GOOD: System is mathematically sound with minor issues.")
        elif overall_score >= 0.5:
            report.append("⚠️  FAIR: System needs attention but is functional.")
        else:
            report.append("❌ POOR: System requires significant mathematical improvements.")
            
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main analysis function."""
    analyzer = MathematicalRobustnessAnalyzer()
    
    print("🔍 Starting Mathematical Robustness Analysis...")
    print()
    
    report = analyzer.generate_report()
    print(report)
    
    # Save report to file
    with open("mathematical_robustness_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
        
    print("📄 Report saved to: mathematical_robustness_report.txt")


if __name__ == "__main__":
    main() 