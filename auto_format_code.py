#!/usr/bin/env python3
"""
Automated Code Formatting Script for Schwabot

This script automatically formats all Python files in the Schwabot codebase
using Black, autopep8, isort, and flake8 to ensure PEP 8 compliance.

Usage:
    python auto_format_code.py [--check] [--fix] [--verbose]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class CodeFormatter:
    """Automated code formatter for Schwabot codebase."""

    def __init__(self, verbose: bool = False):
        """Initialize the code formatter."""
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.python_files = []
        self.formatting_results = {}
            "black": {"success": 0, "failed": 0, "errors": []},
            "autopep8": {"success": 0, "failed": 0, "errors": []},
            "isort": {"success": 0, "failed": 0, "errors": []},
            "flake8": {"success": 0, "failed": 0, "errors": []},
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []

        # Directories to include
        include_dirs = ["core", "utils", "schwabot", "tests", "scripts"]

        # Directories to exclude
        exclude_dirs = ["__pycache__", ".git", ".venv", "venv", "env", "node_modules"]

        for include_dir in include_dirs:
            dir_path = self.project_root / include_dir
            if dir_path.exists():
                for file_path in dir_path.rglob("*.py"):
                    # Skip excluded directories
                    if any(exclude in str(file_path) for exclude in exclude_dirs):
                        continue
                    python_files.append(file_path)

        # Also include Python files in root directory
        for file_path in self.project_root.glob("*.py"):
            python_files.append(file_path)

        return sorted(python_files)

    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            if self.verbose:
                print(f"Running {description}: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, cwd=self.project_root)

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if self.verbose and output.strip():
                print(f"Output: {output}")

            return success, output

        except Exception as e:
            error_msg = f"Error running {description}: {e}"
            if self.verbose:
                print(error_msg)
            return False, error_msg

    def format_with_black(self, check_only: bool = False) -> None:
        """Format code using Black."""
        print("🎨 Running Black code formatter...")

        command = ["black"]
        if check_only:
            command.append("--check")
        command.extend(["--line-length=100", "--target-version=py39"])

        # Add all Python files
        python_files = self.find_python_files()
        if python_files:
            command.extend([str(f) for f in python_files])

            success, output = self.run_command(command, "Black")

            if success:
                self.formatting_results["black"]["success"] += 1
                print("✅ Black formatting completed successfully")
            else:
                self.formatting_results["black"]["failed"] += 1
                self.formatting_results["black"]["errors"].append(output)
                print("❌ Black formatting failed")
                if output.strip():
                    print(f"Error: {output}")
        else:
            print("⚠️ No Python files found for Black formatting")

    def format_with_autopep8(self, check_only: bool = False) -> None:
        """Format code using autopep8."""
        print("🔧 Running autopep8 for PEP 8 compliance...")

        python_files = self.find_python_files()
        total_files = len(python_files)

        for i, file_path in enumerate(python_files, 1):
            if self.verbose:
                print(f"Processing {i}/{total_files}: {file_path.name}")

            command = ["autopep8"]
            if check_only:
                command.append("--diff")"
            else:
                command.append("--in-place")
            command.extend()
                ["--max-line-length=100", "--aggressive", "--aggressive", str(file_path)]
            )

            success, output = self.run_command(command, f"autopep8 on {file_path.name}")

            if success:
                self.formatting_results["autopep8"]["success"] += 1
            else:
                self.formatting_results["autopep8"]["failed"] += 1
                self.formatting_results["autopep8"]["errors"].append(f"{file_path}: {output}")

        print()
            f"✅ autopep8 processing completed: {"}
                self.formatting_results['autopep8']['success']} files"
        )

    def sort_imports_with_isort(self, check_only: bool = False) -> None:
        """Sort imports using isort."""
        print("📦 Running isort for import sorting...")

        command = ["isort"]
        if check_only:
            command.append("--check-only")
        else:
            command.append("--atomic")
        command.extend()
            []
                "--profile=black",
                "--line-length=100",
                "--multi-line=3",
                "--trailing-comma",
                "--force-grid-wrap=0",
                "--use-parentheses",
                "--ensure-newline-before-comments",
            ]
        )

        # Add all Python files
        python_files = self.find_python_files()
        if python_files:
            command.extend([str(f) for f in python_files])

            success, output = self.run_command(command, "isort")

            if success:
                self.formatting_results["isort"]["success"] += 1
                print("✅ Import sorting completed successfully")
            else:
                self.formatting_results["isort"]["failed"] += 1
                self.formatting_results["isort"]["errors"].append(output)
                print("❌ Import sorting failed")
                if output.strip():
                    print(f"Error: {output}")
        else:
            print("⚠️ No Python files found for import sorting")

    def lint_with_flake8(self) -> None:
        """Lint code using flake8."""
        print("🔍 Running flake8 for code linting...")

        command = []
            "flake8",
            "--max-line-length=100",
            "--extend-ignore=E203,W503",
            "--exclude=__pycache__,.git,venv,env,.venv",
            "--count",
            "--statistics",
        ]

        # Add all Python files
        python_files = self.find_python_files()
        if python_files:
            command.extend([str(f) for f in python_files])

            success, output = self.run_command(command, "flake8")

            if success:
                self.formatting_results["flake8"]["success"] += 1
                print("✅ Code linting completed successfully")
                if output.strip():
                    print(f"Linting results:\n{output}")
            else:
                self.formatting_results["flake8"]["failed"] += 1
                self.formatting_results["flake8"]["errors"].append(output)
                print("❌ Code linting found issues")
                if output.strip():
                    print(f"Linting issues:\n{output}")
        else:
            print("⚠️ No Python files found for linting")

    def create_pyproject_toml(self) -> None:
        """Create pyproject.toml configuration file."""
        pyproject_content = """[tool.black]"
line-length = 100
target-version = ['py39']
include = '\\.pyi?$'
extend-exclude = '''
/()
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
known_first_party = ["core", "utils", "schwabot"]

[tool.flake8]
max-line-length = 100
extend-ignore = ["E203", "W503"]
exclude = []
    "__pycache__",
    ".git",
    "venv",
    "env",
    ".venv"
]
"""

        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            with open(pyproject_path, "w") as f:
                f.write(pyproject_content)
            print("✅ Created pyproject.toml configuration file")
        else:
            print("ℹ️ pyproject.toml already exists")

    def format_all(self, check_only: bool = False) -> None:
        """Run all formatting tools."""
        print("🚀 Starting automated code formatting for Schwabot...")
        print("=" * 60)

        # Create configuration file
        self.create_pyproject_toml()

        # Find Python files
        self.python_files = self.find_python_files()
        print(f"📁 Found {len(self.python_files)} Python files to format")

        if not self.python_files:
            print("⚠️ No Python files found in the project")
            return

        # Run formatting tools
        if not check_only:
            # Format with Black first
            self.format_with_black(check_only)

            # Sort imports with isort
            self.sort_imports_with_isort(check_only)

            # Apply PEP 8 fixes with autopep8
            self.format_with_autopep8(check_only)
        else:
            # Check mode - run all tools in check mode
            self.format_with_black(check_only)
            self.sort_imports_with_isort(check_only)
            self.format_with_autopep8(check_only)

        # Always run flake8 for linting
        self.lint_with_flake8()

        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print formatting summary."""
        print("\n" + "=" * 60)
        print("📊 FORMATTING SUMMARY")
        print("=" * 60)

        total_files = len(self.python_files)

        for tool, results in self.formatting_results.items():
            success = results["success"]
            failed = results["failed"]

            if tool == "flake8":
                if failed == 0:
                    print(f"✅ {tool.upper()}: All files passed linting")
                else:
                    print(f"⚠️ {tool.upper()}: {failed} linting issues found")
            else:
                if failed == 0:
                    print(f"✅ {tool.upper()}: {success} files processed successfully")
                else:
                    print(f"❌ {tool.upper()}: {failed} files failed")

        print(f"\n📁 Total Python files: {total_files}")

        # Show any errors
        has_errors = any(results["errors"] for results in self.formatting_results.values())
        if has_errors and self.verbose:
            print("\n❌ ERRORS DETAILS:")
            for tool, results in self.formatting_results.items():
                if results["errors"]:
                    print(f"\n{tool.upper()} Errors:")
                    for error in results["errors"][:3]:  # Show first 3 errors
                        print(f"  - {error[:100]}...")

        print("\n🎉 Code formatting completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Automated code formatting for Schwabot")
    parser.add_argument()
        "--check", action="store_true", help="Check formatting without making changes"
    )
    parser.add_argument("--fix", action="store_true", help="Fix formatting issues (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Default to fix mode if no mode specified
    check_only = args.check and not args.fix

    formatter = CodeFormatter(verbose=args.verbose)
    formatter.format_all(check_only=check_only)


if __name__ == "__main__":
    main()
