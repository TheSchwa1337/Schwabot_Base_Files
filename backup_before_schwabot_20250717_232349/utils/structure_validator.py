#!/usr/bin/env python3
"""
Structure Validator - FlakeAid equivalent for Schwabot system
Validates code structure, imports, and dependencies.
"""

import ast
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

class StructureValidator:
    """Validates system structure and dependencies."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.core_path = self.project_root / "core"
        self.utils_path = self.project_root / "utils"
        self.issues = []
        self.warnings = []

    def validate_imports(self, file_path: Path) -> List[str]:
        """Validate imports in a Python file."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check if module exists
                        try:
                            importlib.util.find_spec(alias.name)
                        except ImportError:
                            issues.append(f"Import not found: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Check relative imports
                        if node.module.startswith('.'):
                            module_path = self._resolve_relative_import()
                                file_path, node.module
                            )
                            if not module_path or not module_path.exists():
                                issues.append(f"Relative import not found: {node.module}")
                        else:
                            # Check absolute imports
                            try:
                                importlib.util.find_spec(node.module)
                            except ImportError:
                                issues.append(f"Module not found: {node.module}")

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Error parsing file: {e}")

        return issues

    def _resolve_relative_import(self, file_path: Path, module: str) -> Path:
        """Resolve relative import path."""
        current_dir = file_path.parent

        # Count leading dots
        level = 0
        for char in module:
            if char == '.':
                level += 1
            else:
                break

        # Go up directories
        target_dir = current_dir
        for _ in range(level - 1):
            target_dir = target_dir.parent

        # Add module path
        module_name = module[level:]
        if module_name:
            module_parts = module_name.split('.')
            for part in module_parts:
                target_dir = target_dir / part

        # Check for __init__.py or .py file
        if (target_dir / "__init__.py").exists():
            return target_dir / "__init__.py"
        elif (target_dir.parent / f"{target_dir.name}.py").exists():
            return target_dir.parent / f"{target_dir.name}.py"

        return None

    def validate_file_structure(self) -> Dict[str, List[str]]:
        """Validate overall file structure."""
        structure_issues = {}
            'missing_files': [],
            'empty_files': [],
            'large_files': [],
            'import_issues': {}
        }

        # Check for required files
        required_files = []
            'core/__init__.py',
            'core/antipole_router.py',
            'core/automated_trading_engine.py',
            'utils/__init__.py',
            'utils/logging_setup.py'
        ]

        for req_file in required_files:
            file_path = self.project_root / req_file
            if not file_path.exists():
                structure_issues['missing_files'].append(req_file)
            elif file_path.stat().st_size == 0:
                structure_issues['empty_files'].append(req_file)
            elif file_path.stat().st_size > 1024 * 1024:  # 1MB
                structure_issues['large_files'].append(req_file)

        # Check imports in all Python files
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name != "__pycache__":
                import_issues = self.validate_imports(py_file)
                if import_issues:
                    rel_path = py_file.relative_to(self.project_root)
                    structure_issues['import_issues'][str(rel_path)] = import_issues

        return structure_issues

    def check_circular_imports(self) -> List[str]:
        """Check for circular import dependencies."""
        # This is a simplified check - in a real system you'd need more sophisticated analysis'
        circular_issues = []

        # Build dependency graph
        dependencies = {}

        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            rel_path = str(py_file.relative_to(self.project_root))
            dependencies[rel_path] = []

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith('.'):
                            # Relative import
                            target = self._resolve_relative_import(py_file, node.module)
                            if target:
                                target_rel = str(target.relative_to(self.project_root))
                                dependencies[rel_path].append(target_rel)

            except Exception:
                continue

        # Simple circular dependency check
        for file_path, deps in dependencies.items():
            for dep in deps:
                if dep in dependencies and file_path in dependencies[dep]:
                    circular_issues.append(f"Circular import: {file_path} <-> {dep}")

        return circular_issues

    def run_validation(self) -> Dict:
        """Run complete validation."""
        logger.info("🔍 Running Structure Validation")

        results = {}
            'file_structure': self.validate_file_structure(),
            'circular_imports': self.check_circular_imports(),
            'project_stats': self._get_project_stats()
        }

        return results

    def _get_project_stats(self) -> Dict:
        """Get project statistics."""
        stats = {}
            'total_files': 0,
            'python_files': 0,
            'total_lines': 0,
            'core_files': 0,
            'utils_files': 0
        }

        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                stats['total_files'] += 1

                if file_path.suffix == '.py':
                    stats['python_files'] += 1

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = len(f.readlines())
                            stats['total_lines'] += lines
                    except Exception:
                        pass

                    if 'core' in str(file_path):
                        stats['core_files'] += 1
                    elif 'utils' in str(file_path):
                        stats['utils_files'] += 1

        return stats

def run_flakeaid_equivalent():
    """Run FlakeAid equivalent validation."""
    project_root = os.getcwd()
    validator = StructureValidator(project_root)

    results = validator.run_validation()

    # Print results
    print("=" * 60)
    print("🔍 STRUCTURE VALIDATION REPORT")
    print("=" * 60)

    # File structure issues
    structure = results['file_structure']
    if structure['missing_files']:
        print(f"❌ Missing files: {len(structure['missing_files'])}")
        for file in structure['missing_files']:
            print(f"   - {file}")

    if structure['empty_files']:
        print(f"⚠️  Empty files: {len(structure['empty_files'])}")
        for file in structure['empty_files']:
            print(f"   - {file}")

    if structure['import_issues']:
        print(f"❌ Import issues: {len(structure['import_issues'])}")
        for file, issues in structure['import_issues'].items():
            print(f"   {file}:")
            for issue in issues:
                print(f"     - {issue}")

    # Circular imports
    circular = results['circular_imports']
    if circular:
        print(f"🔄 Circular imports: {len(circular)}")
        for issue in circular:
            print(f"   - {issue}")

    # Project stats
    stats = results['project_stats']
    print(f"\n📊 Project Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Python files: {stats['python_files']}")
    print(f"   Total lines: {stats['total_lines']:,}")
    print(f"   Core files: {stats['core_files']}")
    print(f"   Utils files: {stats['utils_files']}")

    # Overall health
    total_issues = (len(structure['missing_files']) +)
                   len(structure['empty_files']) + 
                   len(structure['import_issues']) + 
                   len(circular))

    print("=" * 60)
    if total_issues == 0:
        print("✅ SYSTEM STATUS: HEALTHY - No issues found")
    elif total_issues <= 5:
        print("⚠️  SYSTEM STATUS: MINOR ISSUES - Attention needed")
    else:
        print("🚨 SYSTEM STATUS: MAJOR ISSUES - Immediate action required")

    print("=" * 60)

    return results

if __name__ == "__main__":
    run_flakeaid_equivalent() 