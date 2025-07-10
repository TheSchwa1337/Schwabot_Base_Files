#!/usr/bin/env python3
"""
🔍 CODEBASE VALIDATION AND REPAIR
=================================

Comprehensive validation and repair system for Schwabot codebase.
Checks for import issues, missing dependencies, syntax errors, and other problems.

Core Features:
- Import validation and dependency checking
- Syntax error detection and reporting
- Missing module identification
- Code quality assessment
- Automatic repair suggestions
"""

import ast
import importlib
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.append('.')

logger = logging.getLogger(__name__)


@dataclass
    class ValidationIssue:
    """Represents a validation issue found in the codebase"""
    file_path: str
    line_number: int
    issue_type: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggested_fix: Optional[str] = None


@dataclass
    class ModuleStatus:
    """Status of a module import"""
    module_name: str
    is_available: bool
    import_error: Optional[str] = None
    dependencies: List[str] = None
    fallback_available: bool = False


class CodebaseValidator:
    """
    Comprehensive codebase validation and repair system
    """

    def __init__(self, project_root: str = "."):
        """
        Initialize validator

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.issues: List[ValidationIssue] = []
        self.module_status: Dict[str, ModuleStatus] = {}
        self.import_graph: Dict[str, List[str]] = defaultdict(list)

        # Core dependencies that should be available
        self.core_dependencies = []
            'numpy', 'pandas', 'scipy', 'matplotlib', 'requests',
            'aiohttp', 'asyncio', 'logging', 'json', 'time', 'random'
        ]

        # Optional dependencies
        self.optional_dependencies = []
            'cupy', 'torch', 'tensorflow', 'sklearn', 'ccxt'
        ]

        logger.info(f"Codebase Validator initialized for {self.project_root}")

    def validate_entire_codebase(self) -> Dict[str, Any]:
        """
        Perform comprehensive validation of the entire codebase

        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Starting comprehensive codebase validation...")

        try:
            # Clear previous issues
            self.issues.clear()

            # Validate core dependencies
            self._validate_core_dependencies()

            # Validate all Python files
            self._validate_python_files()

            # Check for circular imports
            self._check_circular_imports()

            # Validate Layer 8 components specifically
            self._validate_layer8_components()

            # Generate summary
            summary = self._generate_validation_summary()

            logger.info(f"✅ Validation completed. Found {len(self.issues)} issues.")
            return summary

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {"error": str(e), "issues": []}

    def _validate_core_dependencies(self):
        """Validate core dependencies"""
        logger.info("📦 Validating core dependencies...")

        for dep in self.core_dependencies:
            try:
                importlib.import_module(dep)
                self.module_status[dep] = ModuleStatus()
                    module_name=dep,
                    is_available=True,
                    dependencies=[]
                )
                logger.debug(f"✅ {dep} - Available")
            except ImportError as e:
                self.module_status[dep] = ModuleStatus()
                    module_name=dep,
                    is_available=False,
                    import_error=str(e),
                    dependencies=[]
                )
                self.issues.append(ValidationIssue())
                    file_path="dependencies",
                    line_number=0,
                    issue_type="missing_dependency",
                    message=f"Core dependency '{dep}' not available: {e}",
                    severity="error",
                    suggested_fix=f"pip install {dep}"
                ))
                logger.warning(f"❌ {dep} - Not available: {e}")

        # Check optional dependencies
        for dep in self.optional_dependencies:
            try:
                importlib.import_module(dep)
                self.module_status[dep] = ModuleStatus()
                    module_name=dep,
                    is_available=True,
                    dependencies=[]
                )
                logger.debug(f"✅ {dep} - Available (optional)")
            except ImportError:
                self.module_status[dep] = ModuleStatus()
                    module_name=dep,
                    is_available=False,
                    import_error="Not installed",
                    dependencies=[],
                    fallback_available=True
                )
                logger.debug(f"⚠️ {dep} - Not available (optional, fallback, available)")

    def _validate_python_files(self):
        """Validate all Python files in the project"""
        logger.info("🐍 Validating Python files...")

        python_files = list(self.project_root.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        for py_file in python_files:
            try:
                self._validate_single_file(py_file)
            except Exception as e:
                logger.error(f"Error validating {py_file}: {e}")
                self.issues.append(ValidationIssue())
                    file_path=str(py_file),
                    line_number=0,
                    issue_type="validation_error",
                    message=f"Error during validation: {e}",
                    severity="error"
                ))

    def _validate_single_file(self, file_path: Path):
        """Validate a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check syntax
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(ValidationIssue())
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    issue_type="syntax_error",
                    message=f"Syntax error: {e.msg}",
                    severity="error",
                    suggested_fix="Fix syntax error in the specified line"
                ))
                return

            # Check imports
            self._check_imports_in_file(file_path, tree, content)

            # Check for common issues
            self._check_common_issues(file_path, tree, content)

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    def _check_imports_in_file(self, file_path: Path, tree: ast.AST, content: str):
        """Check imports in a file"""
        try:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._validate_import(file_path, alias.name, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        full_import = f"{module}.{alias.name}" if module else alias.name
                        self._validate_import(file_path, full_import, node.lineno)

                        # Check for relative imports
                        if node.module and node.module.startswith('.'):
                            self._check_relative_import(file_path, node, alias.name)

        except Exception as e:
            logger.error(f"Error checking imports in {file_path}: {e}")

    def _validate_import(self, file_path: Path, import_name: str, line_number: int):
        """Validate a specific import"""
        try:
            # Skip standard library imports
            if import_name in ['os', 'sys', 'time', 'json', 'logging', 'pathlib', 'typing', 'dataclasses']:
                return

            # Try to import
            try:
                importlib.import_module(import_name)
                return
            except ImportError:
                pass

            # Check if it's a local import'
            if import_name.startswith('core.') or import_name.startswith('utils.'):
                # Check if the module exists
                module_path = self._resolve_local_import(import_name)
                if not module_path.exists():
                    self.issues.append(ValidationIssue())
                        file_path=str(file_path),
                        line_number=line_number,
                        issue_type="missing_local_module",
                        message=f"Local module '{import_name}' not found",
                        severity="error",
                        suggested_fix=f"Create missing module: {import_name}"
                    ))
                return

            # Check if it's an optional dependency'
            if import_name in self.optional_dependencies:
                self.issues.append(ValidationIssue())
                    file_path=str(file_path),
                    line_number=line_number,
                    issue_type="optional_dependency_missing",
                    message=f"Optional dependency '{import_name}' not available",
                    severity="warning",
                    suggested_fix=f"pip install {import_name} (optional)"
                ))
                return

            # Unknown import
            self.issues.append(ValidationIssue())
                file_path=str(file_path),
                line_number=line_number,
                issue_type="unknown_import",
                message=f"Unknown import '{import_name}'",
                severity="warning",
                suggested_fix=f"Check if '{import_name}' should be installed or is a typo"
            ))

        except Exception as e:
            logger.error(f"Error validating import {import_name}: {e}")

    def _resolve_local_import(self, import_name: str) -> Path:
        """Resolve a local import to a file path"""
        try:
            # Convert import name to file path
            parts = import_name.split('.')
            if parts[0] in ['core', 'utils']:
                # Handle core and utils imports
                if len(parts) == 2:
                    # Simple import like 'core.module'
                    return self.project_root / parts[0] / f"{parts[1]}.py"
                elif len(parts) > 2:
                    # Nested import like 'core.submodule.module'
                    return self.project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"

            # Default fallback
            return self.project_root / f"{import_name.replace('.', '/')}.py"

        except Exception as e:
            logger.error(f"Error resolving local import {import_name}: {e}")
            return Path("/invalid/path")

    def _check_relative_import(self, file_path: Path, node: ast.ImportFrom, alias_name: str):
        """Check relative imports"""
        try:
            # Check if the relative import target exists
            target_module = node.module
            if target_module and target_module.startswith('.'):
                # Count dots to determine relative level
                level = len(target_module) - len(target_module.lstrip('.'))
                module_name = target_module[level:]

                # Resolve relative path
                current_dir = file_path.parent
                for _ in range(level - 1):
                    current_dir = current_dir.parent

                target_path = current_dir / f"{module_name}.py"
                if not target_path.exists():
                    self.issues.append(ValidationIssue())
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type="relative_import_error",
                        message=f"Relative import target '{target_module}' not found",
                        severity="error",
                        suggested_fix=f"Check if '{target_module}' exists at {target_path}"
                    ))

        except Exception as e:
            logger.error(f"Error checking relative import: {e}")

    def _check_common_issues(self, file_path: Path, tree: ast.AST, content: str):
        """Check for common code issues"""
        try:
            # Check for hardcoded paths
            if 'C:\\' in content or '/c/' in content:
                self.issues.append(ValidationIssue())
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="hardcoded_path",
                    message="Hardcoded Windows path detected",
                    severity="warning",
                    suggested_fix="Use Path objects or relative paths instead of hardcoded paths"
                ))

            # Check for print statements (should use, logger)
            if 'print(' in content and 'logger' not in content: )
                self.issues.append(ValidationIssue())
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="print_statement",
                    message="Print statements found, consider using logger",
                    severity="info",
                    suggested_fix="Replace print() with logger.info() or logger.debug()"
                ))

            # Check for bare except clauses
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        if handler.type is None:
                            self.issues.append(ValidationIssue())
                                file_path=str(file_path),
                                line_number=handler.lineno,
                                issue_type="bare_except",
                                message="Bare except clause detected",
                                severity="warning",
                                suggested_fix="Specify exception types to catch"
                            ))

        except Exception as e:
            logger.error(f"Error checking common issues in {file_path}: {e}")

    def _check_circular_imports(self):
        """Check for circular import dependencies"""
        logger.info("🔄 Checking for circular imports...")

        try:
            # Build import graph
            for file_path in self.project_root.rglob("*.py"):
                if file_path.name.startswith('__'):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    tree = ast.parse(content)
                    file_imports = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            if node.module.startswith('.'):
                                # Relative import
                                file_imports.append(node.module)

                    if file_imports:
                        rel_path = str(file_path.relative_to(self.project_root))
                        self.import_graph[rel_path] = file_imports

                except Exception as e:
                    logger.debug(f"Error parsing {file_path}: {e}")

            # Check for cycles (simplified)
            visited = set()
            temp_visited = set()

            def has_cycle(node: str) -> bool:
                if node in temp_visited:
                    return True
                if node in visited:
                    return False

                temp_visited.add(node)

                for neighbor in self.import_graph.get(node, []):
                    if has_cycle(neighbor):
                        return True

                temp_visited.remove(node)
                visited.add(node)
                return False

            # Check each file for cycles
            for file_path in self.import_graph:
                if has_cycle(file_path):
                    self.issues.append(ValidationIssue())
                        file_path=file_path,
                        line_number=0,
                        issue_type="circular_import",
                        message="Potential circular import detected",
                        severity="warning",
                        suggested_fix="Review import structure to break circular dependencies"
                    ))

        except Exception as e:
            logger.error(f"Error checking circular imports: {e}")

    def _validate_layer8_components(self):
        """Specifically validate Layer 8 components"""
        logger.info("🧬🔐🤖🔀 Validating Layer 8 components...")

        layer8_components = []
            'core.hash_glyph_compression',
            'core.ai_matrix_consensus', 
            'core.visual_decision_engine',
            'core.loop_strategy_switcher'
        ]

        for component in layer8_components:
            try:
                module = importlib.import_module(component)
                logger.info(f"✅ {component} - Valid")

                # Check for required classes
                if component == 'core.hash_glyph_compression':
                    if not hasattr(module, 'HashGlyphCompressor'):
                        self.issues.append(ValidationIssue())
                            file_path=f"{component}.py",
                            line_number=0,
                            issue_type="missing_class",
                            message="HashGlyphCompressor class not found",
                            severity="error"
                        ))

                elif component == 'core.ai_matrix_consensus':
                    if not hasattr(module, 'AIMatrixConsensus'):
                        self.issues.append(ValidationIssue())
                            file_path=f"{component}.py",
                            line_number=0,
                            issue_type="missing_class",
                            message="AIMatrixConsensus class not found",
                            severity="error"
                        ))

                elif component == 'core.visual_decision_engine':
                    if not hasattr(module, 'VisualDecisionEngine'):
                        self.issues.append(ValidationIssue())
                            file_path=f"{component}.py",
                            line_number=0,
                            issue_type="missing_class",
                            message="VisualDecisionEngine class not found",
                            severity="error"
                        ))

                elif component == 'core.loop_strategy_switcher':
                    if not hasattr(module, 'StrategyLoopSwitcher'):
                        self.issues.append(ValidationIssue())
                            file_path=f"{component}.py",
                            line_number=0,
                            issue_type="missing_class",
                            message="StrategyLoopSwitcher class not found",
                            severity="error"
                        ))

            except ImportError as e:
                self.issues.append(ValidationIssue())
                    file_path=f"{component}.py",
                    line_number=0,
                    issue_type="layer8_import_error",
                    message=f"Layer 8 component '{component}' import failed: {e}",
                    severity="error",
                    suggested_fix=f"Check dependencies and imports in {component}"
                ))
                logger.error(f"❌ {component} - Import failed: {e}")

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        try:
            # Categorize issues
            errors = [i for i in self.issues if i.severity == 'error']
            warnings = [i for i in self.issues if i.severity == 'warning']
            info = [i for i in self.issues if i.severity == 'info']

            # Group by issue type
            issue_types = defaultdict(int)
            for issue in self.issues:
                issue_types[issue.issue_type] += 1

            # Group by file
            file_issues = defaultdict(list)
            for issue in self.issues:
                file_issues[issue.file_path].append(issue)

            summary = {}
                "total_issues": len(self.issues),
                "errors": len(errors),
                "warnings": len(warnings),
                "info": len(info),
                "issue_types": dict(issue_types),
                "files_with_issues": len(file_issues),
                "module_status": {}
                    name: {}
                        "available": status.is_available,
                        "error": status.import_error,
                        "fallback_available": status.fallback_available
                    }
                    for name, status in self.module_status.items()
                },
                "issues": []
                    {}
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "message": issue.message,
                        "severity": issue.severity,
                        "suggested_fix": issue.suggested_fix
                    }
                    for issue in self.issues
                ],
                "file_issues": {}
                    file: len(issues) for file, issues in file_issues.items()
                }
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

    def generate_repair_script(self) -> str:
        """Generate a repair script based on found issues"""
        logger.info("🔧 Generating repair script...")

        repair_commands = []

        # Install missing dependencies
        missing_deps = []
        for issue in self.issues:
            if issue.issue_type == "missing_dependency" and issue.suggested_fix:
                missing_deps.append(issue.suggested_fix)

        if missing_deps:
            repair_commands.append("# Install missing dependencies")
            repair_commands.extend(missing_deps)
            repair_commands.append("")

        # Optional dependencies
        optional_deps = []
        for issue in self.issues:
            if issue.issue_type == "optional_dependency_missing" and issue.suggested_fix:
                optional_deps.append(issue.suggested_fix)

        if optional_deps:
            repair_commands.append("# Install optional dependencies (if, needed)")
            repair_commands.extend(optional_deps)
            repair_commands.append("")

        # File-specific fixes
        file_fixes = defaultdict(list)
        for issue in self.issues:
            if issue.suggested_fix and issue.file_path != "dependencies":
                file_fixes[issue.file_path].append(f"# Line {issue.line_number}: {issue.message}")
                file_fixes[issue.file_path].append(f"# Fix: {issue.suggested_fix}")

        if file_fixes:
            repair_commands.append("# File-specific fixes needed:")
            for file_path, fixes in file_fixes.items():
                repair_commands.append(f"# {file_path}:")
                repair_commands.extend(fixes)
                repair_commands.append("")

        return "\n".join(repair_commands)

    def save_validation_report(self, output_file: str = "validation_report.json"):
        """Save validation report to file"""
        try:
            summary = self._generate_validation_summary()

            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"📄 Validation report saved to {output_file}")

            # Also save repair script
            repair_script = self.generate_repair_script()
            if repair_script:
                with open("repair_commands.txt", 'w') as f:
                    f.write(repair_script)
                logger.info("🔧 Repair commands saved to repair_commands.txt")

        except Exception as e:
            logger.error(f"Error saving validation report: {e}")


def main():
    """Main validation function"""
    print("🔍 SCHWABOT CODEBASE VALIDATION AND REPAIR")
    print("=" * 60)

    # Setup logging
    logging.basicConfig()
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Create validator
        validator = CodebaseValidator()

        # Run validation
        results = validator.validate_entire_codebase()

        # Print summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"Total issues: {results.get('total_issues', 0)}")
        print(f"Errors: {results.get('errors', 0)}")
        print(f"Warnings: {results.get('warnings', 0)}")
        print(f"Info: {results.get('info', 0)}")

        # Print critical issues
        if results.get('errors', 0) > 0:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in results.get('issues', []):
                if issue['severity'] == 'error':
                    print(f"  {issue['file']}:{issue['line']} - {issue['message']}")

        # Print warnings
        if results.get('warnings', 0) > 0:
            print(f"\n⚠️ WARNINGS:")
            for issue in results.get('issues', []):
                if issue['severity'] == 'warning':
                    print(f"  {issue['file']}:{issue['line']} - {issue['message']}")

        # Save report
        validator.save_validation_report()

        # Print module status
        print(f"\n📦 MODULE STATUS:")
        for name, status in results.get('module_status', {}).items():
            if status['available']:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}: {status.get('error', 'Unknown error')}")

        # Overall assessment
        if results.get('errors', 0) == 0:
            print(f"\n🎉 CODEBASE STATUS: HEALTHY")
            print("All critical issues resolved. Codebase is ready for development.")
        else:
            print(f"\n⚠️ CODEBASE STATUS: NEEDS ATTENTION")
            print(f"Please address {results.get('errors', 0)} critical issues before proceeding.")

        return results.get('errors', 0) == 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 