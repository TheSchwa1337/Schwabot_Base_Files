#!/usr/bin/env python3
"""Fix remaining syntax errors in the Schwabot codebase.

This script addresses the E999 syntax errors that are still present
in various files, particularly malformed docstrings and unterminated
triple-quoted strings.
"""

import os
import re
from pathlib import Path


def fix_malformed_docstrings(file_path: str) -> bool:
    """Fix malformed docstrings in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix pattern: """Stub main function."""."""
        content = re.sub(
            r'"""Stub main function\."""\."""',
            '"""Stub main function."""\n    pass\n',
            content
        )
        
        # Fix pattern: """Some text."""."""
        content = re.sub(
            r'"""([^"]*)\."""\."""',
            r'"""\1."""',
            content
        )
        
        # Fix unterminated triple-quoted strings
        # Look for patterns like: """text without closing
        content = re.sub(
            r'"""([^"]*)\n\s*"""\s*def\s+',
            r'"""\1"""\n\ndef ',
            content
        )
        
        # Fix stray periods after function definitions
        content = re.sub(
            r'def\s+(\w+)\s*\([^)]*\)\s*:\s*\.',
            r'def \1(\2):',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed syntax errors in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_specific_files():
    """Fix specific files known to have syntax errors."""
    files_to_fix = [
        'core/fractal_command_dispatcher.py',
        'core/fractal_containment_lock.py',
        'core/fractal_controller.py',
        'core/fractal_weights.py',
        'core/function_patterns.py',
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_malformed_docstrings(file_path)


def scan_and_fix_all_python_files():
    """Scan all Python files and fix syntax errors."""
    fixed_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_malformed_docstrings(file_path):
                    fixed_count += 1
    
    print(f"Fixed syntax errors in {fixed_count} files")


def main():
    """Main function to fix syntax errors."""
    print("Fixing remaining syntax errors in Schwabot codebase...")
    
    # Fix specific known files first
    fix_specific_files()
    
    # Then scan all Python files
    scan_and_fix_all_python_files()
    
    print("Syntax error fixing completed!")


if __name__ == "__main__":
    main() 