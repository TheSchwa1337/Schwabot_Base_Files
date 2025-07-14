import ast
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

#!/usr/bin/env python3
"""
Codebase Validation Script
==========================

Comprehensive validation of the entire codebase to ensure:
- No stubs or placeholder implementations
- All imports are properly resolved
- GPU/CPU functionality is properly implemented
- No missing definitions or incomplete code
- Consistent code quality across all modules
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodebaseValidator:
    """Comprehensive codebase validation system."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues = []
        self.stats = {}
            "files_checked": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "stubs_found": 0,
            "missing_imports": 0,
            "gpu_cpu_issues": 0,
            "syntax_errors": 0,
        }
        # Patterns to identify stubs and incomplete code
        self.stub_patterns = []
            "pass",  # Empty implementations
            "TODO",  # TODO comments
            "FIXME",  # FIXME comments
            "placeholder",  # Placeholder text
            "stub",  # Stub references
            "not implemented",  # Not implemented
            "raise NotImplementedError",  # Explicit not implemented
            "Function implementation pending",  # Pending implementations
            "Emergency placeholder docstring",  # Emergency placeholders
        ]
        # GPU/CPU related patterns
        self.gpu_cpu_patterns = []
            "GPU_AVAILABLE",
            "CUDA_AVAILABLE",
            "NUMBA_AVAILABLE",
            "cupy",
            "numba",
            "cuda",
            "gpu_available",
            "cpu_fallback",
            "hardware_optimization",
        ]

    def scan_directory(): -> List[Path]:
        """Scan directory for Python files."""
        if directory is None:
            directory = self.root_dir

        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip common directories that shouldn't contain source code'
            dirs[:] = []
                d
                for d in dirs
                if d
                not in {}
                    "__pycache__",
                    ".git",
                    ".vscode",
                    "node_modules",
                    "venv",
                    "env",
                    ".pytest_cache",
                    "build",
                    "dist",
                }
            ]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        return python_files

    def check_file_for_stubs(): -> List[Dict[str, Any]]:
        """Check a single file for stub patterns and issues."""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # Check for stub patterns
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()

                for pattern in self.stub_patterns:
                    if pattern.lower() in line_lower:
                        issues.append()
                            {}
                                "type": "stub_pattern",
                                "line": line_num,
                                "pattern": pattern,
                                "content": line.strip(),
                                "severity": "warning",
                            }
                        )

            # Check for GPU/CPU related issues
            gpu_cpu_issues = self._check_gpu_cpu_implementation(content, lines)
            issues.extend(gpu_cpu_issues)

            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append()
                    {}
                        "type": "syntax_error",
                        "line": e.lineno,
                        "message": str(e),
                        "severity": "error",
                    }
                )

            # Check for import issues
            import_issues = self._check_imports(content, lines, file_path)
            issues.extend(import_issues)

        except Exception as e:
            issues.append()
                {}
                    "type": "file_error",
                    "message": f"Error reading file: {e}",
                    "severity": "error",
                }
            )

        return issues

    def _check_gpu_cpu_implementation(): -> List[Dict[str, Any]]:
        """Check for proper GPU/CPU implementation patterns."""
        issues = []

        # Check for GPU/CPU detection patterns
        gpu_cpu_detected = any()
            pattern.lower() in content.lower() for pattern in self.gpu_cpu_patterns
        )

        if gpu_cpu_detected:
            # Check for proper fallback mechanisms
            if "try:" in content and "except ImportError:" in content:
                # Good - has proper import error handling
                pass
            elif "gpu_available" in content and "cpu" in content:
                # Good - has fallback logic
                pass
            else:
                # Potential issue - GPU/CPU code without proper fallbacks
                for line_num, line in enumerate(lines, 1):
                    if any()
                        pattern.lower() in line.lower()
                        for pattern in ["cupy", "numba", "cuda"]
                    ):
                        if "import" in line and "try:" not in content:
                            issues.append()
                                {}
                                    "type": "gpu_cpu_issue",
                                    "line": line_num,
                                    "message": "GPU import without proper fallback mechanism",
                                    "content": line.strip(),
                                    "severity": "warning",
                                }
                            )

        return issues

    def _check_imports(): -> List[Dict[str, Any]]:
        """Check for import issues."""
        issues = []

        # Parse imports
        try:
            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")

            # Check for common problematic imports
            for import_name in imports:
                if "unified_math" in import_name:
                    # Check if unified_math_system exists
                    if not self._check_module_exists("core.unified_math_system"):
                        issues.append()
                            {}
                                "type": "missing_import",
                                "message": f"Missing module: {import_name}",
                                "severity": "error",
                            }
                        )

                # Check for other core dependencies
                if import_name.startswith("core.") and not self._check_module_exists()
                    import_name
                ):
                    issues.append()
                        {}
                            "type": "missing_import",
                            "message": f"Missing core module: {import_name}",
                            "severity": "error",
                        }
                    )

        except SyntaxError:
            # Syntax errors are handled separately
            pass

        return issues

    def _check_module_exists(): -> bool:
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def validate_codebase(): -> Dict[str, Any]:
        """Run comprehensive validation of the entire codebase."""
        logger.info("Starting comprehensive codebase validation...")

        # Scan for Python files
        python_files = self.scan_directory()
        logger.info(f"Found {len(python_files)} Python files to validate")

        # Check each file
        for file_path in python_files:
            logger.info(f"Validating: {file_path}")
            issues = self.check_file_for_stubs(file_path)

            if issues:
                self.issues.append({"file": str(file_path), "issues": issues})
                self.stats["files_with_issues"] += 1
                self.stats["total_issues"] += len(issues)

                # Count issue types
                for issue in issues:
                    if issue["type"] == "stub_pattern":
                        self.stats["stubs_found"] += 1
                    elif issue["type"] == "missing_import":
                        self.stats["missing_imports"] += 1
                    elif issue["type"] == "gpu_cpu_issue":
                        self.stats["gpu_cpu_issues"] += 1
                    elif issue["type"] == "syntax_error":
                        self.stats["syntax_errors"] += 1

            self.stats["files_checked"] += 1

        # Run additional checks
        self._run_flake8_check()
        self._run_mypy_check()

        return self._generate_report()

    def _run_flake8_check(self):
        """Run flake8 to check for code quality issues."""
        try:
            result = subprocess.run()
                ["flake8", "--count", "--select=E,W,F", "--max-line-length=120"],
                capture_output=True,
                text=True,
                cwd=self.root_dir,
            )

            if result.stdout:
                logger.warning(f"Flake8 found {result.stdout.strip()} issues")
                self.stats["flake8_issues"] = int(result.stdout.strip())
            else:
                logger.info("Flake8 check passed")
                self.stats["flake8_issues"] = 0

        except FileNotFoundError:
            logger.warning("Flake8 not found, skipping flake8 check")
        except Exception as e:
            logger.error(f"Error running flake8: {e}")

    def _run_mypy_check(self):
        """Run mypy to check for type issues."""
        try:
            result = subprocess.run()
                ["mypy", "--ignore-missing-imports", "--no-strict-optional"],
                capture_output=True,
                text=True,
                cwd=self.root_dir,
            )

            if result.returncode != 0:
                logger.warning("Mypy found type issues")
                self.stats["mypy_issues"] = len(result.stdout.split("\n")) - 1
            else:
                logger.info("Mypy check passed")
                self.stats["mypy_issues"] = 0

        except FileNotFoundError:
            logger.warning("Mypy not found, skipping mypy check")
        except Exception as e:
            logger.error(f"Error running mypy: {e}")

    def _generate_report(): -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {}
            "summary": {}
                "total_files": self.stats["files_checked"],
                "files_with_issues": self.stats["files_with_issues"],
                "total_issues": self.stats["total_issues"],
                "validation_status": "PASS"
                if self.stats["total_issues"] == 0
                else "FAIL",
            },
            "issue_breakdown": {}
                "stubs_found": self.stats["stubs_found"],
                "missing_imports": self.stats["missing_imports"],
                "gpu_cpu_issues": self.stats["gpu_cpu_issues"],
                "syntax_errors": self.stats["syntax_errors"],
                "flake8_issues": self.stats.get("flake8_issues", 0),
                "mypy_issues": self.stats.get("mypy_issues", 0),
            },
            "detailed_issues": self.issues,
            "recommendations": self._generate_recommendations(),
        }
        return report

    def _generate_recommendations(): -> List[str]:
        """Generate recommendations based on found issues."""
        recommendations = []

        if self.stats["stubs_found"] > 0:
            recommendations.append()
                f"Found {self.stats['stubs_found']} stub patterns. "
                "Consider implementing proper functionality for production use."
            )

        if self.stats["missing_imports"] > 0:
            recommendations.append()
                f"Found {self.stats['missing_imports']} missing imports. "
                "Ensure all required modules are properly installed and available."
            )

        if self.stats["gpu_cpu_issues"] > 0:
            recommendations.append()
                f"Found {self.stats['gpu_cpu_issues']} GPU/CPU implementation issues. "
                "Ensure proper fallback mechanisms are in place for hardware compatibility."
            )

        if self.stats["syntax_errors"] > 0:
            recommendations.append()
                f"Found {self.stats['syntax_errors']} syntax errors. "
                "Fix these before proceeding with development."
            )

        if self.stats.get("flake8_issues", 0) > 0:
            recommendations.append()
                f"Found {self.stats['flake8_issues']} code quality issues. "
                "Run 'flake8' to see detailed issues and fix them."
            )

        if self.stats.get("mypy_issues", 0) > 0:
            recommendations.append()
                f"Found {self.stats['mypy_issues']} type checking issues. "
                "Run 'mypy' to see detailed type issues and fix them."
            )

        if not recommendations:
            recommendations.append()
                "Codebase validation passed! All checks completed successfully."
            )

        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print validation report in a readable format."""
        print("\n" + "=" * 80)
        print("CODEBASE VALIDATION REPORT")
        print("=" * 80)

        # Summary
        summary = report["summary"]
        print("\nSUMMARY:")
        print(f"  Total files checked: {summary['total_files']}")
        print(f"  Files with issues: {summary['files_with_issues']}")
        print(f"  Total issues found: {summary['total_issues']}")
        print(f"  Validation status: {summary['validation_status']}")

        # Issue breakdown
        breakdown = report["issue_breakdown"]
        print("\nISSUE BREAKDOWN:")
        print(f"  Stubs found: {breakdown['stubs_found']}")
        print(f"  Missing imports: {breakdown['missing_imports']}")
        print(f"  GPU/CPU issues: {breakdown['gpu_cpu_issues']}")
        print(f"  Syntax errors: {breakdown['syntax_errors']}")
        print(f"  Flake8 issues: {breakdown['flake8_issues']}")
        print(f"  Mypy issues: {breakdown['mypy_issues']}")

        # Detailed issues
        if report["detailed_issues"]:
            print("\nDETAILED ISSUES:")
            for file_issue in report["detailed_issues"]:
                print(f"\n  File: {file_issue['file']}")
                for issue in file_issue["issues"]:
                    print()
                        f"    [{issue['severity'].upper()}] Line {issue.get('line', 'N/A')}: {issue['message']}"
                    )
                    if "content" in issue:
                        print(f"      Content: {issue['content']}")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

        print("\n" + "=" * 80)

    def save_report()
        self, report: Dict[str, Any], filename: str = "codebase_validation_report.json"
    ):
        """Save validation report to JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main entry point for codebase validation."""
    validator = CodebaseValidator()

    try:
        # Run validation
        report = validator.validate_codebase()

        # Print report
        validator.print_report(report)

        # Save report
        validator.save_report(report)

        # Exit with appropriate code
        if report["summary"]["validation_status"] == "PASS":
            logger.info("Codebase validation completed successfully!")
            sys.exit(0)
        else:
            logger.warning("Codebase validation found issues that need attention.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
