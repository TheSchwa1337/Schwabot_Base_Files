#!/usr/bin/env python3
"""Comprehensive Flake8 error analyzer and fixer for Schwabot.

This script identifies and fixes all remaining flake8 errors to ensure
the codebase is completely lint-clean and ready for production trading.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List


def get_flake8_errors() -> List[str]:
    """Get all flake8 errors from the codebase."""
    try:
        result = subprocess.run(
            ["flake8", "core", "tools"],
            capture_output=True,
            text=True,
            check=False
        )
        return [line.strip() for line in result.stdout.split('\n') if line.strip()]
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []


def categorize_errors(errors: List[str]) -> Dict[str, List[str]]:
    """Categorize errors by type for systematic fixing."""
    categories = {
        'critical': [],       # E999, F821 - runtime blocking
        'whitespace': [],     # W291, W292, W293
        'imports': [],        # I201, F401, F405
        'style': [],         # E203, E501, N806
        'docstrings': [],    # D400, D401, D205
        'annotations': [],   # ANN101, ANN
        'complexity': [],    # C901
        'other': []
    }
    
    for error in errors:
        if any(code in error for code in ['E999', 'F821']):
            categories['critical'].append(error)
        elif any(code in error for code in ['W291', 'W292', 'W293']):
            categories['whitespace'].append(error)
        elif any(code in error for code in ['I201', 'F401', 'F405']):
            categories['imports'].append(error)
        elif any(code in error for code in ['E203', 'E501', 'N806']):
            categories['style'].append(error)
        elif any(code in error for code in ['D400', 'D401', 'D205']):
            categories['docstrings'].append(error)
        elif 'ANN' in error:
            categories['annotations'].append(error)
        elif 'C901' in error:
            categories['complexity'].append(error)
        else:
            categories['other'].append(error)
    
    return categories


def fix_whitespace_errors(file_path: str) -> bool:
    """Fix trailing whitespace and newline issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        
        # Fix trailing whitespace (W291, W293)
        for i, line in enumerate(lines):
            if line.rstrip() != line.rstrip('\n'):
                lines[i] = line.rstrip() + '\n'
                modified = True
        
        # Ensure file ends with newline (W292)
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
            modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
            
    except Exception as e:
        print(f"Error fixing whitespace in {file_path}: {e}")
    
    return False


def main() -> None:
    """Main function to systematically fix all flake8 errors."""
    print("🔍 COMPREHENSIVE FLAKE8 ERROR ANALYSIS")
    print("=" * 50)
    
    # Get all errors
    errors = get_flake8_errors()
    if not errors:
        print("✅ No flake8 errors found!")
        return
    
    print(f"Found {len(errors)} flake8 errors")
    
    # Categorize errors
    categories = categorize_errors(errors)
    
    print("\n📊 ERROR BREAKDOWN:")
    for category, error_list in categories.items():
        if error_list:
            print(f"  {category.upper()}: {len(error_list)} errors")
    
    # Extract unique files with whitespace issues
    whitespace_files = set()
    for error in categories['whitespace']:
        if ':' in error:
            file_part = error.split(':')[0]
            if os.path.exists(file_part):
                whitespace_files.add(file_part)
    
    print(f"\n🎯 FILES WITH WHITESPACE ISSUES: {len(whitespace_files)}")
    
    # Fix whitespace issues first
    fixed_count = 0
    for file_path in sorted(whitespace_files):
        if fix_whitespace_errors(file_path):
            print(f"  ✅ Fixed whitespace: {file_path}")
            fixed_count += 1
    
    print(f"\n🎉 SUMMARY:")
    print(f"  Fixed whitespace in {fixed_count} files")
    
    # Check remaining critical issues
    critical_count = len(categories['critical'])
    if critical_count > 0:
        print(f"  🚨 CRITICAL: {critical_count} syntax/import errors need manual fixing")
    
    print("\n🔒 RECOMMENDATIONS:")
    if categories['critical']:
        print("  🚨 PRIORITY 1: Fix E999/F821 syntax errors immediately")
    if categories['imports']:
        print("  📦 PRIORITY 2: Run isort and autoflake for import cleanup")
    if categories['style']:
        print("  🎨 PRIORITY 3: Run black for code formatting")
    if categories['docstrings']:
        print("  📖 PRIORITY 4: Fix docstring formatting")


if __name__ == "__main__":
    main() 