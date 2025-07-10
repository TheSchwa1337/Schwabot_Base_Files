from __future__ import annotations

import ast
import logging
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dual_unicore_handler import DualUnicoreHandler
from utils.safe_print import debug, error, info, safe_print, success, warn

# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""Runtime Validator - Execution Integrity Checker.
This module validates all critical Schwabot files for syntax errors, improper formatting,
and compilation readiness to prevent runtime failures in trading logic.
Validation includes:
- Syntax error detection
- Flake8 compliance checking
- Import resolution verification
- Mathematical function integrity
- Type annotation validation
Windows CLI compatible with comprehensive error reporting."""

logger = logging.getLogger(__name__)


class ValidationResult:
    """Validation result container."""

    def __init__(self):
        """Initialize validation result."""
        self.syntax_errors: List[Tuple[str, str]] = []
        self.flake8_errors: List[Tuple[str, str]] = []
        self.import_errors: List[Tuple[str, str]] = []
        self.type_errors: List[Tuple[str, str]] = []
        self.critical_files_checked: int = 0
        self.total_errors: int = 0
        self.validation_time: float = 0.0
        self.passed: bool = False


class SchwabotruntimeValidator:
    """Comprehensive runtime validator for Schwabot execution integrity."""

    def __init__(self):
        """Initialize runtime validator."""
        self.critical_files = [
            "core/btc_investment_ratio_controller.py",
            "core/unified_signal_metrics.py",
            "core/entry_gate.py",
            "core/tick_resonance_engine.py",
            "core/drift_phase_monitor.py",
            "core/profit_router.py",
            "core/auto_scaler.py",
            "core/gan_anomaly_filter.py",
            "core/btc_data_processor.py",
            "core/altitude_adjustment_math.py",
        ]
        # Mathematical function signatures that must exist
        self.required_math_functions = {
            "core/entry_gate.py": ["execution_confidence", "entry_score"],
            "core/tick_resonance_engine.py": ["compute_harmony_vector"],
            "core/drift_phase_monitor.py": ["compute_phase_drift"],
            "core/altitude_adjustment_math.py": ["calculate_market_altitude"],
            "core/btc_data_processor.py": ["process_btc_data"],
        }

    def validate_all(self) -> ValidationResult:
        """
        Perform comprehensive validation of all critical files.
        Returns
        -------
        ValidationResult
            Complete validation results
        """
        start_time = time.time()
        result = ValidationResult()

        safe_print("\\u1f50d Starting Schwabot Runtime Validation...")
        safe_print("=" * 50)

        try:
            # 1. Check syntax errors
            safe_print("1. Checking syntax errors...")
            self._check_syntax_errors(result)

            # 2. Check Flake8 compliance
            safe_print("2. Checking Flake8 compliance...")
            self._check_flake8_compliance(result)

            # 3. Check import resolution
            safe_print("3. Checking import resolution...")
            self._check_import_resolution(result)

            # 4. Check mathematical function integrity
            safe_print("4. Checking mathematical function integrity...")
            self._check_math_function_integrity(result)

            # 5. Check type annotations
            safe_print("5. Checking type annotations...")
            self._check_type_annotations(result)

            # Calculate totals
            result.total_errors = (
                len(result.syntax_errors)
                + len(result.flake8_errors)
                + len(result.import_errors)
                + len(result.type_errors)
            )

            result.validation_time = time.time() - start_time
            result.passed = result.total_errors == 0

            # Print summary
            self._print_validation_summary(result)

            return result

        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            result.validation_time = time.time() - start_time
            result.passed = False
            return result

    def _check_syntax_errors(self, result: ValidationResult) -> None:
        """Check for syntax errors in critical files."""
        for file_path in self.critical_files:
            if not Path(file_path).exists():
                result.syntax_errors.append((file_path, "File does not exist"))
                continue

            try:
                # Try to compile the file
                py_compile.compile(file_path, doraise=True)
                result.critical_files_checked += 1

                # Also check AST parsing
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)

                safe_print(f"  \\u2705 {file_path}")

            except py_compile.PyCompileError as e:
                error_msg = str(e)
                result.syntax_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

            except SyntaxError as e:
                error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
                result.syntax_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

            except Exception as e:
                error_msg = f"Compilation error: {str(e)}"
                result.syntax_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

    def _check_flake8_compliance(self, result: ValidationResult) -> None:
        """Check Flake8 compliance for critical files."""
        try:
            # Run flake8 on core directory
            cmd = [
                "flake8",
                "core/",
                "--max-line-length=88",
                "--select=E,W,F,C,B",
                "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if process.returncode == 0:
                safe_print("  \\u2705 All files pass Flake8 compliance")
            else:
                # Parse flake8 output
                for line in process.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            file_path = parts[0]
                            error_details = ':'.join(parts[1:])
                            result.flake8_errors.append((file_path, error_details))
                            safe_print(f"  \\u274c {file_path}: {error_details}")

        except subprocess.TimeoutExpired:
            result.flake8_errors.append(("flake8", "Timeout during execution"))
            safe_print("  \\u274c Flake8 check timed out")

        except FileNotFoundError:
            result.flake8_errors.append(("flake8", "Flake8 not installed"))
            safe_print("  \\u26a0\\ufe0f  Flake8 not found - install with: pip install flake8")

        except Exception as e:
            result.flake8_errors.append(("flake8", str(e)))
            safe_print(f"  \\u274c Flake8 error: {e}")

    def _check_import_resolution(self, result: ValidationResult) -> None:
        """Check that all imports can be resolved."""
        for file_path in self.critical_files:
            if not Path(file_path).exists():
                continue

            try:
                # Parse the file and check imports
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._verify_import(alias.name, file_path, result)

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._verify_import(node.module, file_path, result)

                safe_print(f"  \\u2705 {file_path}")

            except Exception as e:
                error_msg = f"Import check failed: {str(e)}"
                result.import_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

    def _verify_import(self, module_name: str, file_path: str, result: ValidationResult) -> None:
        """Verify a single import can be resolved."""
        try:
            # Skip built-in modules and relative imports
            if module_name.startswith('.') or module_name in sys.builtin_module_names:
                return

            # Try to import the module
            __import__(module_name)

        except ImportError as e:
            # Only report if it's a core module (not external dependencies)
            if module_name.startswith('core.'):
                error_msg = f"Cannot import {module_name}: {str(e)}"
                result.import_errors.append((file_path, error_msg))

        except Exception:
            # Ignore other exceptions during import testing
            pass

    def _check_math_function_integrity(self, result: ValidationResult) -> None:
        """Check that required mathematical functions exist and are callable."""
        for file_path, required_functions in self.required_math_functions.items():
            if not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                # Find all function definitions
                defined_functions = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        defined_functions.add(node.name)

                # Check required functions
                missing_functions = []
                for func_name in required_functions:
                    if func_name not in defined_functions:
                        missing_functions.append(func_name)

                if missing_functions:
                    error_msg = f"Missing required functions: {', '.join(missing_functions)}"
                    result.import_errors.append((file_path, error_msg))
                    safe_print(f"  \\u274c {file_path}: {error_msg}")
                else:
                    safe_print(f"  \\u2705 {file_path}: All required functions present")

            except Exception as e:
                error_msg = f"Function integrity check failed: {str(e)}"
                result.import_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

    def _check_type_annotations(self, result: ValidationResult) -> None:
        """Check type annotation coverage."""
        for file_path in self.critical_files:
            if not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                # Count functions and their type annotations
                total_functions = 0
                annotated_functions = 0

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private and magic methods
                        if node.name.startswith('_'):
                            continue

                        total_functions += 1

                        # Check if function has return annotation
                        has_return_annotation = node.returns is not None

                        # Check if parameters have annotations
                        has_param_annotations = any(
                            arg.annotation is not None for arg in node.args.args if arg.arg != 'self'
                        )

                        if has_return_annotation and has_param_annotations:
                            annotated_functions += 1

                if total_functions > 0:
                    annotation_coverage = annotated_functions / total_functions
                    if annotation_coverage < 0.8:  # Require 80% coverage
                        error_msg = f"Low type annotation coverage: {annotation_coverage:.1%}"
                        result.type_errors.append((file_path, error_msg))
                        safe_print(f"  \\u26a0\\ufe0f  {file_path}: {error_msg}")
                    else:
                        safe_print(f"  \\u2705 {file_path}: {annotation_coverage:.1%} type coverage")

            except Exception as e:
                error_msg = f"Type annotation check failed: {str(e)}"
                result.type_errors.append((file_path, error_msg))
                safe_print(f"  \\u274c {file_path}: {error_msg}")

    def _print_validation_summary(self, result: ValidationResult) -> None:
        """Print comprehensive validation summary."""
        safe_print("\n" + "=" * 50)
        safe_print("\\u1f50d VALIDATION SUMMARY")
        safe_print("=" * 50)

        safe_print(f"Files Checked: {result.critical_files_checked}")
        safe_print(f"Validation Time: {result.validation_time:.2f}s")
        safe_print(f"Total Errors: {result.total_errors}")

        if result.syntax_errors:
            safe_print(f"\\n\\u274c SYNTAX ERRORS ({len(result.syntax_errors)}):")
            for file_path, error in result.syntax_errors:
                safe_print(f"  {file_path}: {error}")

        if result.flake8_errors:
            safe_print(f"\\n\\u274c FLAKE8 ERRORS ({len(result.flake8_errors)}):")
            for file_path, error in result.flake8_errors:
                safe_print(f"  {file_path}: {error}")

        if result.import_errors:
            safe_print(f"\\n\\u274c IMPORT ERRORS ({len(result.import_errors)}):")
            for file_path, error in result.import_errors:
                safe_print(f"  {file_path}: {error}")

        if result.type_errors:
            safe_print(f"\\n\\u26a0\\ufe0f  TYPE ANNOTATION WARNINGS ({len(result.type_errors)}):")
            for file_path, error in result.type_errors:
                safe_print(f"  {file_path}: {error}")

        safe_print("\n" + "=" * 50)
        if result.passed:
            safe_print("\\u2705 VALIDATION PASSED - SYSTEM READY FOR EXECUTION")
        else:
            safe_print("\\u274c VALIDATION FAILED - FIX ERRORS BEFORE RUNTIME")
            safe_print("\\n\\u1f6e0\\ufe0f  RECOMMENDED ACTIONS:")
            safe_print("1. Fix syntax errors first")
            safe_print("2. Run: black core/ && flake8 core/")
            safe_print("3. Check import paths and dependencies")
            safe_print("4. Add missing mathematical functions")
            safe_print("5. Improve type annotation coverage")
        safe_print("=" * 50)

    def check_flake8_runtime_halt(self) -> bool:
        """Check for Flake8 violations and halt if found.
        Returns
        -------
        bool
            True if system should halt, False if safe to proceed
        """
        try:
            result = subprocess.run(
                ["flake8", "core/", "--max-line-length=88"],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode != 0:
                safe_print("\\u26a0\\ufe0f  Flake8 errors detected. Fix before runtime.")
                print(result.stdout)
                return True

            return False

        except Exception as e:
            safe_print(f"\\u26a0\\ufe0f  Could not run Flake8 check: {e}")
            return False

    def create_validation_report(self, result: ValidationResult) -> str:
        """Create detailed validation report.
        Parameters
        ----------
        result : ValidationResult
            Validation results to report
        Returns
        -------
        str
            Formatted validation report
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Schwabot Runtime Validation Report
Generated: {timestamp}

## Summary
- Files Checked: {result.critical_files_checked}
- Validation Time: {result.validation_time:.2f}s
- Total Errors: {result.total_errors}
- Status: {'PASSED' if result.passed else 'FAILED'}

## Critical Files Status
"""

        for file_path in self.critical_files:
            if Path(file_path).exists():
                report += f"- \\u2705 {file_path}\n"
            else:
                report += f"- \\u274c {file_path} (missing)\n"

        if result.syntax_errors:
            report += f"\\n  ## Syntax Errors ({len(result.syntax_errors)})\n"
            for file_path, error in result.syntax_errors:
                report += f"- {file_path}: {error}\n"

        if result.flake8_errors:
            report += f"\\n  ## Flake8 Errors ({len(result.flake8_errors)})\n"
            for file_path, error in result.flake8_errors:
                report += f"- {file_path}: {error}\n"

        if result.import_errors:
            report += f"\\n  ## Import Errors ({len(result.import_errors)})\n"
            for file_path, error in result.import_errors:
                report += f"- {file_path}: {error}\n"

        report += "\\n  ## Recommendations\n"
        if result.passed:
            report += "- System is ready for production deployment\n"
            report += "- All critical files pass validation\n"
        else:
            report += "- Fix all syntax errors before deployment\n"
            report += "- Ensure Flake8 compliance with: black core/ && flake8 core/\n"
            report += "- Verify all imports are resolvable\n"
            report += "- Complete mathematical function implementations\n"

        return report


def main() -> None:
    """Main validation entry point."""
    validator = SchwabotruntimeValidator()

    # Check for runtime halt conditions
    if validator.check_flake8_runtime_halt():
        safe_print("\\u1f6d1 Runtime halted due to Flake8 violations")
        sys.exit(1)

    # Run full validation
    result = validator.validate_all()

    # Create validation report
    report = validator.create_validation_report(result)

    # Save report to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"logs/validation_report_{timestamp}.md"

    try:
        Path("logs").mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        safe_print(f"\\n\\u1f4c4 Validation report saved to: {report_file}")
    except Exception as e:
        safe_print(f"\\u26a0\\ufe0f  Could not save report: {e}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
