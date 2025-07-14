#!/usr/bin/env python3
"""
Final Comprehensive Code Quality Check for Schwabot
===================================================

This script runs all code quality checks and provides a detailed report
of the current state of the codebase.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def run_command(command, description, capture_output=True):
    """Run a shell command and capture its output."""
    print(f"\n[*] {description}")
    print(f"Command: {command}")

    start_time = time.time()
    try:
        if capture_output:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(command, shell=True)
            return {}
                'command': command,
                'returncode': result.returncode,
                'stdout': '',
                'stderr': '',
                'duration': time.time() - start_time
            }

        return {}
            'command': command,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': time.time() - start_time
        }
    except Exception as e:
        return {}
            'command': command,
            'returncode': 1,
            'stdout': '',
            'stderr': str(e),
            'duration': time.time() - start_time
        }


def check_python_files():
    """Count Python files in the project."""
    python_files = []
    for root, _, files in os.walk('.'):
        if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv', 'build', 'dist']):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def generate_comprehensive_report(checks, python_files):
    """Generate a comprehensive code quality report."""
    report = {}
        'timestamp': datetime.now().isoformat(),
        'project_info': {}
            'total_python_files': len(python_files),
            'python_version': sys.version,
            'platform': sys.platform
        },
        'total_checks': len(checks),
        'passed_checks': sum(1 for check in checks if check['returncode'] == 0),
        'failed_checks': sum(1 for check in checks if check['returncode'] != 0),
        'total_duration': sum(check['duration'] for check in, checks),
        'checks': checks
    }

    # Write JSON report
    with open('comprehensive_quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Generate human-readable report
    with open('comprehensive_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write("Comprehensive Code Quality Report\n")
        f.write(f"Generated: {report['timestamp']}\n\n")

        f.write("Project Information:\n")
        f.write(f"  Total Python Files: {report['project_info']['total_python_files']}\n")
        f.write(f"  Python Version: {report['project_info']['python_version']}\n")
        f.write(f"  Platform: {report['project_info']['platform']}\n\n")

        f.write("Check Summary:\n")
        f.write(f"  Total Checks: {report['total_checks']}\n")
        f.write(f"  Passed Checks: {report['passed_checks']}\n")
        f.write(f"  Failed Checks: {report['failed_checks']}\n")
        f.write(f"  Total Duration: {report['total_duration']:.2f} seconds\n\n")

        f.write("Detailed Check Results:\n")
        for i, check in enumerate(report['checks'], 1):
            status = "PASSED" if check['returncode'] == 0 else "FAILED"
            f.write(f"\n{i}. {status} - {check['command']}\n")
            f.write(f"   Duration: {check['duration']:.2f} seconds\n")
            if check['stdout']:
                f.write(f"   STDOUT:\n{check['stdout']}\n")
            if check['stderr']:
                f.write(f"   STDERR:\n{check['stderr']}\n")

    return report


def main():
    """Main function to run all code quality checks."""
    print_header("SCHWABOT COMPREHENSIVE CODE QUALITY CHECK")

    # Check Python files
    print_section("Project Analysis")
    python_files = check_python_files()
    print(f"Found {len(python_files)} Python files in the project")

    # Install required tools
    print_section("Tool Installation")
    install_checks = []
        run_command()
            'pip install --upgrade pip',
            'Upgrading pip'),
        run_command()
            'pip install black isort flake8 mypy bandit autoflake chardet',
            'Installing code quality tools'),
    ]

    # Run comprehensive checks
    print_section("Code Quality Checks")

    # Directories to check
    directories = ['core', 'schwabot', 'utils', 'config']
    valid_dirs = [d for d in directories if os.path.exists(d)]

    checks = []
        # File preparation
        run_command('python find_null_byte_files.py', 'Checking for null byte files'),
        run_command('python comprehensive_code_check.py', 'Running comprehensive code check'),

        # Linting and formatting
        run_command()
            f'flake8 {'}
                " ".join(valid_dirs)} --max-line-length=100 --count','
            'Running Flake8 style check'),
        run_command()
            f'black {'}
                " ".join(valid_dirs)} --check --line-length=100 --target-version py311','
            'Checking code formatting with Black'),
        run_command()
            f'isort {'}
                " ".join(valid_dirs)} --check-only --profile black','
            'Checking import sorting'),

        # Type checking
        run_command()
            f'mypy {'}
                " ".join(valid_dirs)} --ignore-missing-imports','
            'Running MyPy type checking'),

        # Security checks
        run_command(f'bandit -r {" ".join(valid_dirs)}', 'Running Bandit security linting'),

        # Additional checks
        run_command('python -m py_compile core/__init__.py', 'Testing core module compilation'),
        run_command('python -m py_compile schwabot/__init__.py',)
                    'Testing schwabot module compilation'),
    ]

    # Combine all checks
    all_checks = install_checks + checks

    # Generate report
    print_section("Report Generation")
    report = generate_comprehensive_report(all_checks, python_files)

    # Print summary
    print_section("Final Summary")
    print(f"Total Python Files: {report['project_info']['total_python_files']}")
    print(f"Total Checks: {report['total_checks']}")
    print(f"Passed Checks: {report['passed_checks']}")
    print(f"Failed Checks: {report['failed_checks']}")
    print(f"Total Duration: {report['total_duration']:.2f} seconds")

    # Print failed checks
    failed_checks = [check for check in all_checks if check['returncode'] != 0]
    if failed_checks:
        print_section("Failed Checks")
        for check in failed_checks:
            print(f"\nFailed: {check['command']}")
            if check['stderr']:
                print(f"Error: {check['stderr'][:200]}...")

    # Final status
    if report['failed_checks'] > 0:
        print_section("Status: FAILED")
        print("Some code quality checks failed. Please review the detailed report.")
        print("Reports saved to:")
        print("  - comprehensive_quality_report.json")
        print("  - comprehensive_quality_report.txt")
        sys.exit(1)
    else:
        print_section("Status: PASSED")
        print("All code quality checks passed successfully!")
        print("Reports saved to:")
        print("  - comprehensive_quality_report.json")
        print("  - comprehensive_quality_report.txt")
        sys.exit(0)


if __name__ == "__main__":
    main()
