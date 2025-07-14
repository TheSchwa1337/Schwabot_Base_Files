#!/usr/bin/env python3
"""
Quick Fixes for Code Quality Issues
===================================

This script addresses the most critical code quality issues found in the project.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def fix_mypy_config():
    """Fix MyPy configuration by removing duplicate sections."""
    mypy_file = Path('mypy.ini')
    if not mypy_file.exists():
        print("mypy.ini not found, skipping MyPy config fix")
        return True

    print("Fixing MyPy configuration...")

    try:
        with open(mypy_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove duplicate sections
        lines = content.split('\n')
        seen_sections = set()
        cleaned_lines = []

        for line in lines:
            if line.strip().startswith('[mypy-'):]
                section = line.strip()
                if section in seen_sections:
                    print(f"  Removing duplicate section: {section}")
                    continue
                seen_sections.add(section)
            cleaned_lines.append(line)

        # Write cleaned content
        with open(mypy_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))

        print("  MyPy configuration fixed successfully")
        return True
    except Exception as e:
        print(f"  Error fixing MyPy config: {e}")
        return False


def fix_import_sorting():
    """Fix import sorting issues."""
    print("Fixing import sorting...")

    try:
        result = subprocess.run()
            ['isort', 'core', 'schwabot', 'utils', 'config', '--profile', 'black'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("  Import sorting fixed successfully")
            return True
        else:
            print(f"  Import sorting failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error fixing import sorting: {e}")
        return False


def fix_basic_formatting():
    """Fix basic formatting issues where possible."""
    print("Fixing basic formatting...")

    try:
        result = subprocess.run(['black',)]
                                 'core',
                                 'schwabot',
                                 'utils',
                                 'config',
                                 '--line-length',
                                 '100',
                                 '--target-version',
                                 'py311'],
                                capture_output=True,
                                text=True)

        if result.returncode == 0:
            print("  Basic formatting fixed successfully")
            return True
        else:
            print(f"  Basic formatting failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error fixing basic formatting: {e}")
        return False


def remove_unused_imports():
    """Remove unused imports using autoflake."""
    print("Removing unused imports...")

    try:
        result = subprocess.run(['autoflake',)]
                                 '--in-place',
                                 '--remove-all-unused-imports',
                                 '--remove-unused-variables',
                                 'core',
                                 'schwabot',
                                 'utils',
                                 'config'],
                                capture_output=True,
                                text=True)

        if result.returncode == 0:
            print("  Unused imports removed successfully")
            return True
        else:
            print(f"  Unused imports removal failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error removing unused imports: {e}")
        return False


def check_syntax_errors():
    """Check for syntax errors in Python files."""
    print("Checking for syntax errors...")

    syntax_errors = []
    python_files = []

    # Collect Python files
    for root, _, files in os.walk('.'):
        if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv', 'build', 'dist']):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    # Check syntax
    for file_path in python_files:
        try:
            result = subprocess.run()
                ['python', '-m', 'py_compile', file_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                syntax_errors.append((file_path, result.stderr))
        except Exception as e:
            syntax_errors.append((file_path, str(e)))

    if syntax_errors:
        print(f"  Found {len(syntax_errors)} files with syntax errors:")
        for file_path, error in syntax_errors[:5]:  # Show first 5
            print(f"    {file_path}: {error[:100]}...")
        if len(syntax_errors) > 5:
            print(f"    ... and {len(syntax_errors) - 5} more files")
        return False
    else:
        print("  No syntax errors found")
        return True


def main():
    """Main function to run quick fixes."""
    print("=" * 60)
    print(" QUICK FIXES FOR CODE QUALITY ISSUES")
    print("=" * 60)

    fixes = []
        ("MyPy Configuration", fix_mypy_config),
        ("Import Sorting", fix_import_sorting),
        ("Unused Imports", remove_unused_imports),
        ("Basic Formatting", fix_basic_formatting),
        ("Syntax Check", check_syntax_errors),
    ]

    results = []

    for fix_name, fix_func in fixes:
        print(f"\n[*] Running: {fix_name}")
        try:
            success = fix_func()
            results.append((fix_name, success))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((fix_name, False))

    # Summary
    print("\n" + "=" * 60)
    print(" QUICK FIXES SUMMARY")
    print("=" * 60)

    successful_fixes = sum(1 for _, success in results if, success)
    total_fixes = len(results)

    for fix_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {fix_name}")

    print(f"\nOverall: {successful_fixes}/{total_fixes} fixes successful")

    if successful_fixes == total_fixes:
        print("\n🎉 All quick fixes completed successfully!")
        print("You can now run the comprehensive check again.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_fixes - successful_fixes} fixes failed.")
        print("Some issues may require manual intervention.")
        sys.exit(1)


if __name__ == "__main__":
    main()
