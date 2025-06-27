#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for remaining flake8 E999 errors.
Addresses all the major error patterns we've identified.
"""

import os
import re
import glob
from pathlib import Path


def fix_import_after_try_pattern(content):
    """Fix imports that appear after try statements without except/finally."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a try statement
        if line.startswith('try:'):
            fixed_lines.append(lines[i])
            i += 1

            # Look for imports after try
            while i < len(lines) and lines[i].strip().startswith('from ') or lines[i].strip().startswith('import '):
                # Move the import before the try
                import_line = lines[i]
                fixed_lines.insert(-1, import_line)  # Insert before the try line
                i += 1

            # Add pass if no except/finally found
            if i < len(lines) and not (lines[i].strip().startswith('except') or lines[i].strip().startswith('finally')):
                fixed_lines.append('    pass')
        else:
            fixed_lines.append(lines[i])
            i += 1

    return '\n'.join(fixed_lines)


def fix_unmatched_parentheses(content):
    """Fix unmatched parentheses and brackets."""
    # Fix common patterns
    content = re.sub(r'\(\\s*\]', '()', content)  # (] -> ()
    content = re.sub(r'\[\\s*\)', '[]', content)  # [) -> []
    content = re.sub(r'{\\s*\]', '{}', content)   # {] -> {}
    content = re.sub(r'\[\\s*}', '[]', content)   # [} -> []

    # Fix specific patterns we've seen
    content = re.sub(r'\[\\s*\]\\s*\)', '[]', content)  # []) -> []
    content = re.sub(r'\(\\s*\[\\s*\]', '()', content)  # ([]) -> ()

    return content


def fix_missing_indented_blocks(content):
    """Fix missing indented blocks after try, if, def statements."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for statements that need indented blocks
        if (stripped.endswith(':') and
            (stripped.startswith('try:') or
             stripped.startswith('if ') or
             stripped.startswith('def ') or
             stripped.startswith('except') or
             stripped.startswith('finally') or
             stripped.startswith('else:') or
             stripped.startswith('elif '))):

            fixed_lines.append(line)
            i += 1

            # Check if next line is not indented or is empty
            if i < len(lines):
                next_line = lines[i]
                if not next_line.strip() or not next_line.startswith('    ') and not next_line.startswith('\t'):
                    fixed_lines.append('    pass')
        else:
            fixed_lines.append(line)
            i += 1

    return '\n'.join(fixed_lines)


def fix_unterminated_strings(content):
    """Fix unterminated triple-quoted strings."""
    # Fix the specific pattern we've seen: """Stub main function."""."""
    content = re.sub(r'"""Stub main function\."""\."""', '"""Stub main function."""', content)

    # Fix other unterminated patterns
    content = re.sub(r'"""([^"]*?)"""\."""', r'"""\1"""', content)

    return content


def fix_invalid_syntax(content):
    """Fix various invalid syntax patterns."""
    # Fix "from mathlib from" -> "from mathlib import"
    content = re.sub(r'from mathlib from', 'from mathlib import', content)

    # Fix unclosed parentheses in imports
    content = re.sub(r'from \.([^)]+?) import \($', r'from .\1 import (', content)

    return content


def fix_specific_file_patterns(filepath, content):
    """Apply file-specific fixes based on the file path."""
    filename = os.path.basename(filepath)

    # Fix specific files with known issues
    if 'typing_schemas.py' in filepath:
        # Fix the import hashlib issue
        content = re.sub(r'^\\s*import hashlib\\s*$', '', content, flags=re.MULTILINE)

    if 'memory_key_allocator.py' in filepath:
        # Fix the logger issue
        content = re.sub(r'^\\s*logger = logging\.getLogger\(__name__\)\\s*$', '', content, flags=re.MULTILINE)

    return content


def fix_file(filepath):
    """Fix all syntax errors in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply all fixes
        content = fix_import_after_try_pattern(content)
        content = fix_unmatched_parentheses(content)
        content = fix_missing_indented_blocks(content)
        content = fix_unterminated_strings(content)
        content = fix_invalid_syntax(content)
        content = fix_specific_file_patterns(filepath, content)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Main function to fix all remaining syntax errors."""
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'build', 'dist', 'venv', 'env']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files")

    fixed_count = 0
    for filepath in python_files:
        if fix_file(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1

    print(f"\\nFixed {fixed_count} files")
    print("Syntax error fixing complete!")


if __name__ == "__main__":
    main()

"""