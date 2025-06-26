from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""Comprehensive Syntax Error Fixer for Schwabot Codebase.

This script addresses all remaining E999 syntax errors:
1. Malformed stub docstrings
2. Invalid Unicode characters
3. Unterminated triple-quoted strings
4. Invalid syntax patterns
"""

import os
import re
from pathlib import Path


def fix_malformed_stub_docstrings(content: str) -> str:
    """Fix malformed stub docstrings."""
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

    return content


def fix_unicode_characters(content: str) -> str:
    """Replace Unicode characters with ASCII equivalents."""
    unicode_replacements = {
        '∇': 'del',  # nabla
        '∈': 'in',   # element of
        '≤': '<=',   # less than or equal
        '≥': '>=',   # greater than or equal
        '⇒': '=>',   # implies
        '∫': 'int',  # integral
        '∂': 'd',    # partial derivative
        '·': '.',    # middle dot
        '–': '-',    # en dash
        '₍': '(',    # subscript left parenthesis
        '₎': ')',    # subscript right parenthesis
    }

    for unicode_char, ascii_replacement in unicode_replacements.items():
        content = content.replace(unicode_char, ascii_replacement)

    return content


def fix_unterminated_strings(content: str) -> str:
    """Fix unterminated triple-quoted strings."""
    # Fix pattern: """text without closing
    content = re.sub(
        r'"""([^"]*)\n\s*"""\s*def\s+',
        r'"""\1"""\n\ndef ',
        content
    )

    # Fix pattern: """text at end of line
    content = re.sub(
        r'"""([^"]*)\n\s*def\s+',
        r'"""\1"""\n\ndef ',
        content
    )

    return content


def fix_invalid_syntax(content: str) -> str:
    """Fix various invalid syntax patterns."""
    # Fix stray periods after function definitions
    content = re.sub(
        r'def\s+(\w+)\s*\([^)]*\)\s*:\s*\.',
        r'def \1(\2):',
        content
    )

    # Fix invalid decimal literals
    content = re.sub(
        r'(\d+)\.(\d+)\.(\d+)',
        r'\1.\2_\3',  # Replace with underscore
        content
    )

    # Fix unterminated string literals
    content = re.sub(
        r'(["\'])([^"\']*)\n',
        r'\1\2\1\n',
        content
    )

    return content


def fix_file_syntax(file_path: str) -> bool:
    """Fix all syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply all fixes
        content = fix_malformed_stub_docstrings(content)
        content = fix_unicode_characters(content)
        content = fix_unterminated_strings(content)
        content = fix_invalid_syntax(content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            safe_print(f"Fixed syntax errors in {file_path}")
            return True

        return False

    except Exception as e:
        safe_print(f"Error processing {file_path}: {e}")
        return False


def find_and_fix_all_files():
    """Find and fix all Python files with syntax errors."""
    fixed_count = 0

    # Search in all directories
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file_syntax(file_path):
                    fixed_count += 1

    safe_print(f"Fixed syntax errors in {fixed_count} files")
    return fixed_count


def verify_fixes():
    """Verify that the fixes worked."""
    safe_print("\nVerifying fixes...")

    # Run a quick Flake8 check to see remaining errors
    import subprocess

    try:
        result = subprocess.run(
            ['flake8', '.', '--count', '--select=E9', '--max-line-length=79'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            safe_print("✅ No E999 syntax errors found!")
        else:
            error_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            safe_print(f"⚠️  Still found {error_count} E999 syntax errors")
            safe_print("First few errors:")
            for line in result.stdout.strip().split('\n')[:5]:
                safe_print(f"  {line}")

    except Exception as e:
        safe_print(f"Could not run Flake8 verification: {e}")


def main():
    """Main function to fix all syntax errors."""
    safe_print("Comprehensive Syntax Error Fixer for Schwabot Codebase")
    safe_print("=" * 60)

    # Fix all files
    fixed_count = find_and_fix_all_files()

    safe_print(f"\nFixed {fixed_count} files with syntax errors")

    # Verify the fixes
    verify_fixes()

    safe_print("\nComprehensive syntax fixing completed!")


if __name__ == "__main__":
    main()
