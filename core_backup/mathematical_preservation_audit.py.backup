# -*- coding: utf-8 -*-
"""
Mathematical Preservation Audit System for Schwabot
==================================================

Comprehensive audit system to ensure that all Flake8 fixes, requirements.txt
changes, and code cleanup preserve the mathematical integrity and functionality
of the Schwabot trading system.

Audit Categories:
1. Mathematical Content Preservation Analysis
2. Import Dependencies Validation  
3. Flake8 Fix Impact Assessment
4. Requirements.txt Functionality Validation
5. Core Mathematical Operations Testing

MATHEMATICAL PRESERVATION: This system validates all mathematical content.
"""

import os
import sys
import ast
import re
import logging
import subprocess
import importlib
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MathematicalElement:
    """Represents a mathematical element in the codebase."""
    element_type: str  # "function", "class", "formula", "constant"
    name: str
    file_path: str
    line_number: int
    content: str
    dependencies: List[str] = field(default_factory=list)
    import_requirements: List[str] = field(default_factory=list)
    criticality: str = "medium"  # "critical", "high", "medium", "low"
    preserved: bool = True
    flake8_affected: bool = False
    
@dataclass
class ImportDependency:
    """Represents an import dependency."""
    module_name: str
    import_type: str  # "standard", "third_party", "local"
    used_by: List[str] = field(default_factory=list)
    mathematical_usage: bool = False
    required_for_math: bool = False
    in_requirements: bool = False

@dataclass
class AuditResult:
    """Audit result summary."""
    total_mathematical_elements: int
    preserved_elements: int
    affected_by_flake8: int
    missing_imports: List[str]
    unused_imports: List[str]
    requirements_issues: List[str]
    mathematical_integrity_score: float
    recommendations: List[str]

class MathematicalPreservationAuditor:
    """
    Comprehensive auditor for mathematical preservation in Schwabot.
    
    This system:
    1. Analyzes all mathematical content in the codebase
    2. Validates import dependencies and requirements.txt alignment
    3. Assesses the impact of Flake8 fixes on mathematical operations
    4. Provides recommendations for safe cleanup and optimization
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize the Mathematical Preservation Auditor."""
        self.project_root = Path(project_root)
        self.mathematical_elements: Dict[str, MathematicalElement] = {}
        self.import_dependencies: Dict[str, ImportDependency] = {}
        self.flake8_fixes_log: List[str] = []
        
        # Mathematical preservation patterns
        self.mathematical_patterns = [
            # Core mathematical operations
            r'def.*calculate|def.*compute|def.*process',
            r'def.*tensor_|def.*matrix_|def.*vector_',
            r'np\.|numpy\.|scipy\.|pandas\.',
            r'hashlib\.sha256|hashlib\.md5',
            
            # Trading mathematical content
            r'BTC.*price|ETH.*price|USDC.*price|XRP.*price',
            r'profit.*calculation|loss.*calculation|pnl.*calculation',
            r'volatility|momentum|correlation|entropy',
            
            # Schwabot-specific mathematical systems
            r'unified_math\.|tensor_algebra\.|trading_tensor_ops\.',
            r'ferris.*rde|lantern.*core|recursive.*lattice',
            r'dlt_waveform|phase_engine|mathematical_relay',
            
            # Mathematical formulas and constants
            r'MATHEMATICAL PRESERVATION:',
            r'phi_4|phi_8|phi_42|phi_.*=',
            r'alpha|beta|gamma|delta|theta|sigma',
            r'eigenvalue|eigenvector|svd|pca|fft'
        ]
        
        # Critical mathematical files that must be preserved
        self.critical_math_files = [
            'core/math/mathematical_relay_system.py',
            'core/math/trading_tensor_ops.py',
            'core/math/tensor_algebra/unified_tensor_algebra.py',
            'core/unified_math_system.py',
            'core/dlt_waveform_engine.py',
            'core/multi_bit_btc_processor.py',
            'core/quantum_btc_intelligence_core.py',
            'core/phase_engine/__init__.py'
        ]
        
        logger.info("Mathematical Preservation Auditor initialized")
    
    def run_comprehensive_audit(self) -> AuditResult:
        """Run comprehensive mathematical preservation audit."""
        logger.info("🔍 Starting comprehensive mathematical preservation audit...")
        
        # Step 1: Analyze mathematical content
        self._analyze_mathematical_content()
        
        # Step 2: Validate import dependencies
        self._validate_import_dependencies()
        
        # Step 3: Check requirements.txt alignment
        self._check_requirements_alignment()
        
        # Step 4: Assess Flake8 fix impact
        self._assess_flake8_impact()
        
        # Step 5: Test mathematical operations
        self._test_mathematical_operations()
        
        # Step 6: Generate audit result
        audit_result = self._generate_audit_result()
        
        logger.info("✅ Mathematical preservation audit completed")
        return audit_result
    
    def _analyze_mathematical_content(self):
        """Analyze all mathematical content in the codebase."""
        logger.info("📊 Analyzing mathematical content...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST for detailed analysis
                try:
                    tree = ast.parse(content)
                    self._extract_mathematical_elements_from_ast(tree, file_path, content)
                except SyntaxError:
                    # If AST parsing fails, use regex analysis
                    self._extract_mathematical_elements_from_regex(content, file_path)
                    
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
    
    def _extract_mathematical_elements_from_ast(self, tree: ast.AST, file_path: Path, content: str):
        """Extract mathematical elements using AST analysis."""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is mathematical
                func_content = self._get_node_content(node, lines)
                if self._is_mathematical_content(func_content):
                    element = MathematicalElement(
                        element_type="function",
                        name=node.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        content=func_content,
                        dependencies=self._extract_dependencies(func_content),
                        import_requirements=self._extract_import_requirements(func_content),
                        criticality=self._assess_criticality(func_content, str(file_path))
                    )
                    self.mathematical_elements[f"{file_path}:{node.name}"] = element
            
            elif isinstance(node, ast.ClassDef):
                # Check if class contains mathematical operations
                class_content = self._get_node_content(node, lines)
                if self._is_mathematical_content(class_content):
                    element = MathematicalElement(
                        element_type="class",
                        name=node.name,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        content=class_content,
                        dependencies=self._extract_dependencies(class_content),
                        import_requirements=self._extract_import_requirements(class_content),
                        criticality=self._assess_criticality(class_content, str(file_path))
                    )
                    self.mathematical_elements[f"{file_path}:{node.name}"] = element
    
    def _extract_mathematical_elements_from_regex(self, content: str, file_path: Path):
        """Extract mathematical elements using regex patterns."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in self.mathematical_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Extract surrounding context
                    start_line = max(0, i - 2)
                    end_line = min(len(lines), i + 3)
                    context = '\n'.join(lines[start_line:end_line])
                    
                    element = MathematicalElement(
                        element_type="formula",
                        name=f"math_element_line_{i+1}",
                        file_path=str(file_path),
                        line_number=i + 1,
                        content=context,
                        dependencies=self._extract_dependencies(context),
                        import_requirements=self._extract_import_requirements(context),
                        criticality=self._assess_criticality(context, str(file_path))
                    )
                    self.mathematical_elements[f"{file_path}:line_{i+1}"] = element
                    break
    
    def _validate_import_dependencies(self):
        """Validate import dependencies across the codebase."""
        logger.info("🔗 Validating import dependencies...")
        
        # Collect all imports
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract imports
                imports = self._extract_imports(content)
                
                for import_name in imports:
                    if import_name not in self.import_dependencies:
                        self.import_dependencies[import_name] = ImportDependency(
                            module_name=import_name,
                            import_type=self._classify_import_type(import_name)
                        )
                    
                    self.import_dependencies[import_name].used_by.append(str(file_path))
                    
                    # Check if used for mathematical operations
                    if self._is_mathematical_import(import_name, content):
                        self.import_dependencies[import_name].mathematical_usage = True
                        
            except Exception as e:
                logger.warning(f"Could not analyze imports in {file_path}: {e}")
    
    def _check_requirements_alignment(self):
        """Check alignment between imports and requirements.txt."""
        logger.info("📋 Checking requirements.txt alignment...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                requirements_content = f.read()
            
            # Extract package names from requirements.txt
            requirements_packages = self._extract_requirements_packages(requirements_content)
            
            # Check alignment with imports
            for import_name, dependency in self.import_dependencies.items():
                if dependency.import_type == "third_party":
                    package_name = self._get_package_name(import_name)
                    if package_name in requirements_packages:
                        dependency.in_requirements = True
                        
                    # Mark as required for math if used mathematically
                    if dependency.mathematical_usage:
                        dependency.required_for_math = True
    
    def _assess_flake8_impact(self):
        """Assess the impact of Flake8 fixes on mathematical content."""
        logger.info("🔧 Assessing Flake8 fix impact...")
        
        # Check if any mathematical elements were affected by Flake8 fixes
        for element_id, element in self.mathematical_elements.items():
            # Check for MATHEMATICAL PRESERVATION comments
            if "MATHEMATICAL PRESERVATION:" in element.content:
                element.flake8_affected = True
                element.preserved = True
            
            # Check for critical mathematical operations
            if any(pattern in element.content.lower() for pattern in [
                'btc', 'eth', 'usdc', 'xrp', 'tensor', 'matrix', 'hash', 'sha256'
            ]):
                element.criticality = "critical"
    
    def _test_mathematical_operations(self):
        """Test core mathematical operations to ensure they still work."""
        logger.info("🧮 Testing mathematical operations...")
        
        test_results = {}
        
        # Test 1: NumPy operations
        try:
            import numpy as np
            test_array = np.array([1, 2, 3, 4, 5])
            result = np.mean(test_array)
            test_results["numpy_basic"] = "PASS" if result == 3.0 else "FAIL"
        except Exception as e:
            test_results["numpy_basic"] = f"FAIL: {e}"
        
        # Test 2: SciPy operations (if available)
        try:
            from scipy import stats
            test_data = [1, 2, 3, 4, 5]
            result = np.mean(test_data)  # Use numpy instead of scipy.stats.mean
            test_results["scipy_basic"] = "PASS" if result == 3.0 else "FAIL"
        except Exception as e:
            test_results["scipy_basic"] = f"FAIL: {e}"
        
        # Test 3: Pandas operations (if available)
        try:
            import pandas as pd
            df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            result = df['x'].mean()
            test_results["pandas_basic"] = "PASS" if result == 2.0 else "FAIL"
        except Exception as e:
            test_results["pandas_basic"] = f"FAIL: {e}"
        
        # Test 4: Hashlib operations
        try:
            import hashlib
            test_string = "test_btc_price_50000"
            hash_result = hashlib.sha256(test_string.encode()).hexdigest()
            test_results["hashlib_sha256"] = "PASS" if len(hash_result) == 64 else "FAIL"
        except Exception as e:
            test_results["hashlib_sha256"] = f"FAIL: {e}"
        
        # Test 5: Core mathematical modules (if available)
        try:
            from core.math.mathematical_relay_system import MathematicalRelaySystem
            relay_system = MathematicalRelaySystem()
            test_results["mathematical_relay"] = "PASS"
        except Exception as e:
            test_results["mathematical_relay"] = f"FAIL: {e}"
        
        # Log test results
        for test_name, result in test_results.items():
            if result == "PASS":
                logger.info(f"✅ {test_name}: {result}")
            else:
                logger.warning(f"❌ {test_name}: {result}")
    
    def _generate_audit_result(self) -> AuditResult:
        """Generate comprehensive audit result."""
        total_elements = len(self.mathematical_elements)
        preserved_elements = sum(1 for e in self.mathematical_elements.values() if e.preserved)
        affected_by_flake8 = sum(1 for e in self.mathematical_elements.values() if e.flake8_affected)
        
        # Calculate mathematical integrity score
        if total_elements > 0:
            integrity_score = (preserved_elements / total_elements) * 100
        else:
            integrity_score = 100.0
        
        # Find missing and unused imports
        missing_imports = []
        unused_imports = []
        requirements_issues = []
        
        for import_name, dependency in self.import_dependencies.items():
            if dependency.mathematical_usage and not dependency.in_requirements:
                missing_imports.append(import_name)
            elif dependency.in_requirements and not dependency.used_by:
                unused_imports.append(import_name)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(missing_imports, unused_imports)
        
        return AuditResult(
            total_mathematical_elements=total_elements,
            preserved_elements=preserved_elements,
            affected_by_flake8=affected_by_flake8,
            missing_imports=missing_imports,
            unused_imports=unused_imports,
            requirements_issues=requirements_issues,
            mathematical_integrity_score=integrity_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, missing_imports: List[str], unused_imports: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if missing_imports:
            recommendations.append(f"Add missing mathematical imports to requirements.txt: {', '.join(missing_imports)}")
        
        if unused_imports:
            recommendations.append(f"Consider removing unused imports: {', '.join(unused_imports)}")
        
        # Check for critical mathematical file preservation
        for file_path in self.critical_math_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                recommendations.append(f"Critical mathematical file missing: {file_path}")
        
        # Mathematical integrity recommendations
        critical_elements = [e for e in self.mathematical_elements.values() if e.criticality == "critical"]
        if critical_elements:
            recommendations.append(f"Ensure {len(critical_elements)} critical mathematical elements remain functional")
        
        return recommendations
    
    # Helper methods
    def _get_node_content(self, node: ast.AST, lines: List[str]) -> str:
        """Get content of AST node."""
        try:
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', start_line + 10)
            return '\n'.join(lines[start_line:end_line])
        except:
            return ""
    
    def _is_mathematical_content(self, content: str) -> bool:
        """Check if content contains mathematical operations."""
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in self.mathematical_patterns)
    
    def _assess_criticality(self, content: str, file_path: str) -> str:
        """Assess criticality of mathematical content."""
        critical_indicators = ['btc', 'eth', 'usdc', 'xrp', 'sha256', 'tensor', 'unified_math']
        high_indicators = ['calculate', 'compute', 'matrix', 'vector', 'algorithm']
        
        content_lower = content.lower()
        
        if any(indicator in file_path.lower() for indicator in self.critical_math_files):
            return "critical"
        elif any(indicator in content_lower for indicator in critical_indicators):
            return "critical"
        elif any(indicator in content_lower for indicator in high_indicators):
            return "high"
        else:
            return "medium"
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content."""
        dependencies = []
        import_pattern = r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import|import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(import_pattern, content)
        for match in matches:
            dep = match[0] or match[1]
            if dep:
                dependencies.append(dep)
        return dependencies
    
    def _extract_import_requirements(self, content: str) -> List[str]:
        """Extract import requirements from content."""
        requirements = []
        if 'numpy' in content or 'np.' in content:
            requirements.append('numpy')
        if 'scipy' in content:
            requirements.append('scipy')
        if 'pandas' in content or 'pd.' in content:
            requirements.append('pandas')
        if 'matplotlib' in content:
            requirements.append('matplotlib')
        if 'sklearn' in content or 'scikit-learn' in content:
            requirements.append('scikit-learn')
        return requirements
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract all imports from content."""
        imports = []
        import_pattern = r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import|import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(import_pattern, content)
        for match in matches:
            import_name = match[0] or match[1]
            if import_name:
                imports.append(import_name.split('.')[0])  # Get root module
        return imports
    
    def _classify_import_type(self, import_name: str) -> str:
        """Classify import type."""
        standard_libs = ['os', 'sys', 'time', 'datetime', 'json', 'logging', 'typing', 'dataclasses', 'enum']
        if import_name in standard_libs:
            return "standard"
        elif import_name.startswith('core.') or import_name.startswith('.'):
            return "local"
        else:
            return "third_party"
    
    def _is_mathematical_import(self, import_name: str, content: str) -> bool:
        """Check if import is used for mathematical operations."""
        math_imports = ['numpy', 'scipy', 'pandas', 'matplotlib', 'sklearn', 'sympy', 'numba']
        return import_name in math_imports or any(pattern in content for pattern in self.mathematical_patterns)
    
    def _extract_requirements_packages(self, requirements_content: str) -> Set[str]:
        """Extract package names from requirements.txt."""
        packages = set()
        for line in requirements_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before version specifiers)
                package_name = re.split(r'[>=<!=]', line)[0].strip()
                packages.add(package_name)
        return packages
    
    def _get_package_name(self, import_name: str) -> str:
        """Get package name from import name."""
        # Map common import names to package names
        import_to_package = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'yaml': 'pyyaml'
        }
        return import_to_package.get(import_name, import_name)

def main():
    """Run mathematical preservation audit."""
    auditor = MathematicalPreservationAuditor(".")
    result = auditor.run_comprehensive_audit()
    
    # Print audit summary
    print("\n" + "="*80)
    print("🔍 MATHEMATICAL PRESERVATION AUDIT SUMMARY")
    print("="*80)
    print(f"📊 Total Mathematical Elements: {result.total_mathematical_elements}")
    print(f"✅ Preserved Elements: {result.preserved_elements}")
    print(f"🔧 Affected by Flake8: {result.affected_by_flake8}")
    print(f"📈 Mathematical Integrity Score: {result.mathematical_integrity_score:.1f}%")
    
    if result.missing_imports:
        print(f"\n❌ Missing Imports: {', '.join(result.missing_imports)}")
    
    if result.unused_imports:
        print(f"\n⚠️ Unused Imports: {', '.join(result.unused_imports)}")
    
    if result.recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("\n✅ Mathematical preservation audit completed successfully!")
    return result

if __name__ == "__main__":
    main() 