#!/usr/bin/env python3
"""
Targeted Fixes for Critical Issues
==================================

This script addresses the specific critical issues found in the quality check.
"""

import os
import subprocess
import sys
from pathlib import Path


def fix_core_init():
    """Fix the core/__init__.py file."""
    print("Fixing core/__init__.py...")

    core_init = Path('core/__init__.py')
    if not core_init.exists():
        print("  core/__init__.py not found, creating...")
        with open(core_init, 'w', encoding='utf-8') as f:
            f.write('"""Core module for Schwabot trading system."""\n\n')
            f.write('# Core module initialization\n')
            f.write('__version__ = "1.0.0"\n')
        return True

    try:
        with open(core_init, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix indentation issues
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Remove leading tabs and replace with spaces
            if line.startswith('\t'):
                line = '    ' + line[1:]
            fixed_lines.append(line)

        # Ensure proper module structure
        if not content.strip():
            fixed_lines = []
                '"""Core module for Schwabot trading system."""',
                '',
                '# Core module initialization',
                '__version__ = "1.0.0"',
                ''
            ]

        new_content = '\n'.join(fixed_lines)

        if new_content != content:
            with open(core_init, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  Fixed core/__init__.py")
            return True
        else:
            print("  core/__init__.py already correct")
            return True

    except Exception as e:
        print(f"  Error fixing core/__init__.py: {e}")
        return False

def fix_schwabot_init():
    """Fix the schwabot/__init__.py file."""
    print("Fixing schwabot/__init__.py...")

    schwabot_init = Path('schwabot/__init__.py')
    if not schwabot_init.exists():
        print("  schwabot/__init__.py not found, creating...")
        with open(schwabot_init, 'w', encoding='utf-8') as f:
            f.write('"""Schwabot trading system main module."""\n\n')
            f.write('# Schwabot module initialization\n')
            f.write('__version__ = "1.0.0"\n')
        return True

    try:
        with open(schwabot_init, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix syntax errors
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            # Fix common syntax issues
            if line.strip().startswith('from datetime import datetime'):
                # Ensure proper indentation
                if not line.startswith('    '):
                    line = '    ' + line.strip()
            fixed_lines.append(line)

        # Ensure proper module structure
        if not content.strip():
            fixed_lines = []
                '"""Schwabot trading system main module."""',
                '',
                '# Schwabot module initialization',
                '__version__ = "1.0.0"',
                ''
            ]

        new_content = '\n'.join(fixed_lines)

        if new_content != content:
            with open(schwabot_init, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  Fixed schwabot/__init__.py")
            return True
        else:
            print("  schwabot/__init__.py already correct")
            return True

    except Exception as e:
        print(f"  Error fixing schwabot/__init__.py: {e}")
        return False

def fix_mypy_config():
    """Fix MyPy configuration conflicts."""
    print("Fixing MyPy configuration...")

    mypy_file = Path('mypy.ini')
    if not mypy_file.exists():
        print("  mypy.ini not found, skipping")
        return True

    try:
        with open(mypy_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove duplicate sections
        lines = content.split('\n')
        cleaned_lines = []
        seen_sections = set()

        for line in lines:
            if line.strip().startswith('[mypy-'):]
                section = line.strip()
                if section in seen_sections:
                    print(f"    Removing duplicate section: {section}")
                    continue
                seen_sections.add(section)
            cleaned_lines.append(line)

        new_content = '\n'.join(cleaned_lines)

        if new_content != content:
            with open(mypy_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  Fixed MyPy configuration")
            return True
        else:
            print("  MyPy configuration already correct")
            return True

    except Exception as e:
        print(f"  Error fixing MyPy config: {e}")
        return False

def fix_import_sorting():
    """Fix import sorting issues."""
    print("Fixing import sorting...")

    try:
        # Run isort to fix imports
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

def fix_syntax_errors():
    """Fix basic syntax errors in Python files."""
    print("Fixing basic syntax errors...")

    # Focus on core directories
    directories = ['core', 'schwabot', 'utils', 'config']
    fixed_count = 0

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        original_content = content

                        # Fix common syntax issues
                        # Fix unterminated strings
                        if content.count('"') % 2 != 0:"
                            content = content.replace('"', '"', content.count('"') - 1)"

                        # Fix unterminated parentheses
                        if content.count('(') > content.count(')'):
                            content += ')' * (content.count('(') - content.count(')'))

                        # Fix mixed tabs and spaces
                        lines = content.split('\n')
                        fixed_lines = []
                        for line in lines:
                            if '\t' in line:
                                line = line.replace('\t', '    ')
                            fixed_lines.append(line)
                        content = '\n'.join(fixed_lines)

                        # Only write if changes were made
                        if content != original_content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            fixed_count += 1

                    except Exception:
                        continue

    print(f"  Fixed syntax errors in {fixed_count} files")
    return fixed_count > 0

def validate_fixes():
    """Validate that the fixes worked."""
    print("Validating fixes...")

    validation_checks = []
        ('python -m py_compile core/__init__.py', 'Core module compilation'),
        ('python -m py_compile schwabot/__init__.py', 'Schwabot module compilation'),
        ('mypy core --ignore-missing-imports', 'MyPy configuration'),
    ]

    passed = 0
    total = len(validation_checks)

    for command, description in validation_checks:
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"  ✅ {description}")
                passed += 1
            else:
                print(f"  ❌ {description}")
        except Exception as e:
            print(f"  ❌ {description}: {e}")

    print(f"Validation: {passed}/{total} checks passed")
    return passed == total

def main():
    """Main function to run targeted fixes."""
    print("=" * 60)
    print(" TARGETED FIXES FOR CRITICAL ISSUES")
    print("=" * 60)

    fixes = []
        ("Core module initialization", fix_core_init),
        ("Schwabot module initialization", fix_schwabot_init),
        ("MyPy configuration", fix_mypy_config),
        ("Import sorting", fix_import_sorting),
        ("Basic syntax errors", fix_syntax_errors),
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
    print(" TARGETED FIXES SUMMARY")
    print("=" * 60)

    successful_fixes = sum(1 for _, success in results if, success)
    total_fixes = len(results)

    for fix_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {fix_name}")

    print(f"\nOverall: {successful_fixes}/{total_fixes} fixes successful")

    # Validate fixes
    print("\n[*] Validating fixes...")
    validation_success = validate_fixes()

    if validation_success:
        print("\n🎉 All critical issues have been addressed!")
        print("You can now run the comprehensive quality check again.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some issues remain. Manual intervention may be needed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 