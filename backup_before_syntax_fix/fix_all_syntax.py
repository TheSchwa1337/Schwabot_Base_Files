#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for Schwabot trading system.

This script fixes common syntax errors that prevent the codebase from running:
1. Unterminated string literals
2. Unmatched parentheses/brackets
3. Invalid f-string syntax
4. Leading zeros in decimal integers
5. Indentation errors
6. Missing commas in function calls
"""

import glob
import os
import re
from pathlib import Path


def fix_unterminated_strings(content):
    """Fix unterminated string literals."""
    lines = content.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        # Check for unterminated strings
        if '"' in line or "'" in line: '
            # Count quotes
            single_quotes = line.count("'")'
            double_quotes = line.count('"')"

            # If odd number of quotes, likely unterminated
            if single_quotes % 2 == 1:
                # Add closing single quote
                if not line.rstrip().endswith("'"):'
                    line = line.rstrip() + "'"'
            elif double_quotes % 2 == 1:
                # Add closing double quote
                if not line.rstrip().endswith('"'):"
                    line = line.rstrip() + '"'"

        # Fix common unterminated patterns
        line = re.sub(r'print\("([^"]*)$', r'print("\1")', line))
        line = re.sub(r'print\(\'([^\']*)$', r"print('\1')", line))
        line = re.sub(r'info\("([^"]*)$', r'info("\1")', line))
        line = re.sub(r'error\("([^"]*)$', r'error("\1")', line))
        line = re.sub(r'warn\("([^"]*)$', r'warn("\1")', line))
        line = re.sub(r'success\("([^"]*)$', r'success("\1")', line))

        # Fix f-string issues
        line = re.sub(r'f"([^"]*)$', r'f"\1"', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_unmatched_parentheses(content):
    """Fix unmatched parentheses and brackets."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Count parentheses
        open_paren = line.count('('))
        close_paren = line.count(')')
        open_brace = line.count('{')}
        close_brace = line.count('}')
        open_bracket = line.count('[')]
        close_bracket = line.count(']')

        # Add missing closing parentheses
        if open_paren > close_paren:
            line = line.rstrip() + ')' * (open_paren - close_paren)
        if open_brace > close_brace:
            line = line.rstrip() + '}' * (open_brace - close_brace)
        if open_bracket > close_bracket:
            line = line.rstrip() + ']' * (open_bracket - close_bracket)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_leading_zeros(content):
    """Fix leading zeros in decimal integers."""
    # Fix octal-like numbers that should be decimal
    content = re.sub(r'\b0(\d+)\b', r'\1', content)
    return content

def fix_missing_commas(content):
    """Fix missing commas in function calls and data structures."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Fix missing commas in function calls
        if re.search(r'\([^)]*[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\)', line):
            # Add comma between arguments
            line = re.sub(r'(\w+)\s+(\w+)(\s*\))', r'\1, \2\3', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_indentation_errors(content):
    """Fix indentation errors."""
    lines = content.split('\n')
    fixed_lines = []
    indent_stack = [0]  # Track indentation levels

    for line in lines:
        stripped = line.lstrip()
        if not stripped:  # Empty line
            fixed_lines.append('')
            continue

        # Calculate proper indentation
        current_indent = len(line) - len(stripped)

        # If line starts with a keyword that should be indented
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'else:', 'elif ')):
            if current_indent == 0 and len(fixed_lines) > 0 and fixed_lines[-1].strip():
                # This should be indented
                line = '    ' + stripped

        # If line starts with 'return', 'break', 'continue', 'pass' and is not properly indented
        elif stripped.startswith(('return', 'break', 'continue', 'pass')):
            if current_indent == 0 and len(fixed_lines) > 0 and fixed_lines[-1].strip():
                line = '    ' + stripped

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_specific_patterns(content):
    """Fix specific problematic patterns."""
    # Fix common patterns that cause syntax errors

    # Fix unterminated print statements
    content = re.sub(r'print\("([^"]*)$', r'print("\1")', content, flags=re.MULTILINE))
    content = re.sub(r"print\('([^']*)$", r"print('\1')", content, flags=re.MULTILINE))

    # Fix unterminated info/error/warn/success calls
    content = re.sub(r'info\("([^"]*)$', r'info("\1")', content, flags=re.MULTILINE))
    content = re.sub(r'error\("([^"]*)$', r'error("\1")', content, flags=re.MULTILINE))
    content = re.sub(r'warn\("([^"]*)$', r'warn("\1")', content, flags=re.MULTILINE))
    content = re.sub(r'success\("([^"]*)$', r'success("\1")', content, flags=re.MULTILINE))

    # Fix unterminated f-strings
    content = re.sub(r'f"([^"]*)$', r'f"\1"', content, flags=re.MULTILINE)

    # Fix empty try blocks
    content = re.sub(r'try:\s*\nexcept', r'try:\n    pass\nexcept', content)

    return content

def fix_file(file_path):
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply all fixes
        content = fix_unterminated_strings(content)
        content = fix_unmatched_parentheses(content)
        content = fix_leading_zeros(content)
        content = fix_missing_commas(content)
        content = fix_indentation_errors(content)
        content = fix_specific_patterns(content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"✓ No changes needed: {file_path}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all Python files."""
    print("🔧 Comprehensive Syntax Error Fixer")
    print("=" * 40)

    # Find all Python files
    python_files = []
    for pattern in ['core/**/*.py', 'utils/**/*.py', 'tests/**/*.py', '*.py']:
        python_files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates and sort
    python_files = sorted(list(set(python_files)))

    print(f"Found {len(python_files)} Python files to check")

    fixed_count = 0
    for file_path in python_files:
        if os.path.isfile(file_path):
            if fix_file(file_path):
                fixed_count += 1

    print(f"\n🎉 Fixed {fixed_count} files out of {len(python_files)} total files")

    # Test import
    print("\n🧪 Testing core imports...")
    try:
        import core.zbe_core
        print("✅ ZBE Core imports successfully")
    except Exception as e:
        print(f"❌ ZBE Core import failed: {e}")

    try:
        import core.zpe_core
        print("✅ ZPE Core imports successfully")
    except Exception as e:
        print(f"❌ ZPE Core import failed: {e}")

if __name__ == "__main__":
    main() 