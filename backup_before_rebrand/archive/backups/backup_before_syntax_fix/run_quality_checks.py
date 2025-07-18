#!/usr/bin/env python3
"""
Comprehensive Quality Check Runner
=================================

This script runs all quality checks and provides a detailed status report
    for the Schwabot project.
"""

import os
import subprocess
import sys
import time
from typing import List, Tuple


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def run_check(command: str, description: str) -> Tuple[bool, str, str, float]:
    """Run a quality check and return results."""
    print(f"\n[*] {description}")
    print(f"Command: {command}")

    start_time = time.time()
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            return True, result.stdout, result.stderr, duration
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
            return False, result.stdout, result.stderr, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ ERROR ({duration:.2f}s): {e}")
        return False, "", str(e), duration


def check_python_files() -> List[str]:
    """Count Python files in the project."""
    python_files = []
    skipped_dirs = ['.git', '__pycache__', '.venv', 'venv', 'build', 'dist']
    for root, _, files in os.walk('.'):
        if any(skip in root for skip in, skipped_dirs):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def main() -> bool:
    """Main function to run all quality checks."""
    print_header("SCHWABOT COMPREHENSIVE QUALITY CHECK")

    # Project analysis
    print_section("Project Analysis")
    python_files = check_python_files()
    print(f"Total Python files: {len(python_files)}")

    # Core directories
    core_dirs = ['core', 'utils', 'config']
    valid_dirs = [d for d in core_dirs if os.path.exists(d)]
    print(f"Core directories: {', '.join(valid_dirs)}")

    # Quality checks
    print_section("Quality Checks")

    dirs_str = " ".join(valid_dirs)
    checks = []
        # Basic compilation
        ('python -m py_compile core/__init__.py', 'Core module compilation'),

        # Null byte check
        ('python find_null_byte_files.py', 'Null byte detection'),

        # Linting
        (f'flake8 {dirs_str} --max-line-length=100 --count', 'Flake8 style check'),

        # Import sorting
        (f'isort {dirs_str} --check-only --profile black', 'Import sorting check'),

        # Type checking
        (f'mypy {dirs_str} --ignore-missing-imports', 'MyPy type checking'),

        # Security
        (f'bandit -r {dirs_str}', 'Bandit security check'),
    ]

    results = []
    for command, description in checks:
        success, stdout, stderr, duration = run_check(command, description)
        results.append({)}
            'description': description,
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'duration': duration
        })

    # Summary
    print_section("Summary")

    passed = sum(1 for r in results if r['success'])
    total = len(results)
    total_duration = sum(r['duration'] for r in, results)

    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed / total * 100:.1f}%")
    print(f"Total duration: {total_duration:.2f}s")

    # Failed checks details
    failed_checks = [r for r in results if not r['success']]
    if failed_checks:
        print_section("Failed Checks Details")
        for check in failed_checks:
            print(f"\n❌ {check['description']}")
            if check['stderr']:
                print(f"Error: {check['stderr'][:300]}...")

    # Recommendations
    print_section("Recommendations")

    if passed == total:
        print("🎉 All quality checks passed!")
        print("The codebase is in excellent condition.")
    elif passed >= total * 0.7:
        print("✅ Most quality checks passed.")
        print("Focus on fixing the remaining issues:")
        for check in failed_checks:
            print(f"  - {check['description']}")
    else:
        print("⚠️  Multiple quality issues detected.")
        print("Priority fixes needed:")
        for check in failed_checks:
            print(f"  - {check['description']}")

    # Next steps
    print_section("Next Steps")
    print("1. Run 'python mathematical_fixes.py' to fix mathematical files")
    print("2. Run 'python advanced_syntax_fixes.py' for syntax errors")
    print("3. Run 'python quick_fixes.py' for general fixes")
    print("4. Run 'python final_comprehensive_check.py' for full assessment")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
