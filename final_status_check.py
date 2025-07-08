#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Status Check Script.

Comprehensive status check for Schwabot's flake8 compliance'
and syntax error resolution progress.
"""

import subprocess
import sys
from pathlib import Path


def run_flake8_check(directory: str, max_length: int = 100) -> int:
    """Run flake8 check and return violation count."""
    try:
        result = subprocess.run()
            []
                sys.executable,
                "-m",
                "flake8",
                directory,
                f"--max-line-length={max_length}",
                "--count",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout.strip():
            return int(result.stdout.strip())
        return 0
    except Exception as e:
        print(f"Error running flake8: {e}")
        return -1


def check_syntax_errors(directory: str) -> int:
    """Check for syntax errors in Python files."""
    try:
        python_files = list(Path(directory).rglob("*.py"))
        syntax_errors = 0

        for file_path in python_files:
            if file_path.is_file():
                result = subprocess.run()
                    [sys.executable, "-m", "py_compile", str(file_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode != 0:
                    syntax_errors += 1

        return syntax_errors
    except Exception as e:
        print(f"Error checking syntax: {e}")
        return -1


def main():
    """Main function to check final status."""
    print("🔍 FINAL STATUS CHECK")
    print("=" * 60)

    # Check core directory
    print("\n📁 Checking core/ directory...")
    core_violations = run_flake8_check("core", 100)
    core_syntax_errors = check_syntax_errors("core")

    print(f"  Flake8 violations: {core_violations}")
    print(f"  Syntax errors: {core_syntax_errors}")

    # Check schwabot directory if it exists
    if Path("schwabot").exists():
        print("\n📁 Checking schwabot/ directory...")
        schwabot_violations = run_flake8_check("schwabot", 100)
        schwabot_syntax_errors = check_syntax_errors("schwabot")

        print(f"  Flake8 violations: {schwabot_violations}")
        print(f"  Syntax errors: {schwabot_syntax_errors}")
    else:
        schwabot_violations = 0
        schwabot_syntax_errors = 0

    # Check apply_enhanced_cli_compatibility.py if it exists
    if Path("apply_enhanced_cli_compatibility.py").exists():
        print("\n📁 Checking apply_enhanced_cli_compatibility.py...")
        cli_violations = run_flake8_check("apply_enhanced_cli_compatibility.py", 100)
        print(f"  Flake8 violations: {cli_violations}")
    else:
        cli_violations = 0

    # Calculate totals
    total_violations = core_violations + schwabot_violations + cli_violations
    total_syntax_errors = core_syntax_errors + schwabot_syntax_errors

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL STATUS SUMMARY")
    print("=" * 60)
    print(f"🎯 Total Flake8 Violations: {total_violations}")
    print(f"🎯 Total Syntax Errors: {total_syntax_errors}")

    if total_violations == 0 and total_syntax_errors == 0:
        print("🎉 PERFECT! 100% flake8 compliance achieved!")
        print("✅ All syntax errors resolved!")
        print("🚀 Schwabot is ready for production deployment!")
        return 0
    elif total_violations < 50 and total_syntax_errors == 0:
        print("✅ Excellent progress! Most violations resolved!")
        print("🔧 Only minor style issues remain.")
        return 1
    elif total_syntax_errors == 0:
        print("✅ Good progress! All syntax errors resolved!")
        print(f"🔧 {total_violations} style violations remain.")
        return 2
    else:
        print("⚠️  Some issues remain:")
        print(f"   - {total_syntax_errors} syntax errors")
        print(f"   - {total_violations} style violations")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
