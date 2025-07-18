#!/usr/bin/env python3
"""
Flake8 Issue Fixer for Schwabot Trading System.

This script automatically fixes common Flake8 violations:
- W291: Trailing whitespace
- W292: No newline at end of file
- W293: Blank line contains whitespace
- E261: At least two spaces before inline comment
- E305: Expected 2 blank lines after class or function definition
- F401: Unused imports (with safety checks)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from lines."""
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]
    return '\n'.join(fixed_lines)


def fix_end_of_file_newline(content: str) -> str:
    """Ensure file ends with exactly one newline."""
    content = content.rstrip('\n')
    return content + '\n'


def fix_blank_line_whitespace(content: str) -> str:
    """Remove whitespace from blank lines."""
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if line.strip() == '':
            fixed_lines.append('')
        else:
            fixed_lines.append(line)
    return '\n'.join(fixed_lines)


def fix_inline_comments(content: str) -> str:
    """Fix inline comment spacing (E261)."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Find comments that don't have proper spacing
        if '#' in line and not line.strip().startswith('#'):
            # Split on comment
            code_part, comment_part = line.split('#', 1)
            if code_part.strip() and comment_part.strip():
                # Ensure at least 2 spaces before comment
                if not code_part.endswith('  '):
                    code_part = code_part.rstrip() + '  '
                line = code_part + '#' + comment_part
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_function_class_spacing(content: str) -> str:
    """Fix spacing after function/class definitions (E305)."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)

        # Check if this is a function or class definition
        if re.match(r'^\s*(def|class)\s+\w+', line) and not line.strip().endswith(':'):
            # Look for the colon on the next line
            if i + 1 < len(lines) and ':' in lines[i + 1]:
                i += 1
                fixed_lines.append(lines[i])

        # Check if we need to add blank lines after function/class
        if (
            re.match(r'^\s*(def|class)\s+\w+.*:$', line)
            and i + 1 < len(lines)
            and lines[i + 1].strip() != ''
            and not lines[i + 1].startswith('    ')
            and not lines[i + 1].startswith('\t')
        ):
            # Add two blank lines after function/class definition
            fixed_lines.append('')
            fixed_lines.append('')

        i += 1

    return '\n'.join(fixed_lines)


def remove_unused_imports(content: str, file_path: str) -> str:
    """Remove unused imports (F401) with safety checks."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0

    # Common imports that should be preserved
    preserved_imports = {
        'typing',
        'numpy',
        'pandas',
        'logging',
        'time',
        'json',
        'asyncio',
        'threading',
        'decimal',
        'enum',
        'dataclasses',
        'pathlib',
        'os',
        'sys',
        're',
        'hashlib',
        'math',
        'random',
    }

    while i < len(lines):
        line = lines[i]

        # Check for import statements
        if line.strip().startswith(('import ', 'from ')):
            # Skip if it's a preserved import or has a comment indicating it's needed
            if any(preserved in line for preserved in preserved_imports) or 'noqa' in line:
                fixed_lines.append(line)
            else:
                # Check if the imported module is actually used
                import_name = extract_import_name(line)
                if import_name and not is_import_used(content, import_name):
                    # Skip this import line
                    pass
                else:
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    return '\n'.join(fixed_lines)


def extract_import_name(import_line: str) -> str:
    """Extract the main import name from an import statement."""
    line = import_line.strip()

    if line.startswith('from '):
        # from module import name
        parts = line.split(' import ')
        if len(parts) == 2:
            return parts[0].replace('from ', '').split('.')[0]
    elif line.startswith('import '):
        # import module or import module as alias
        module_part = line.replace('import ', '').split(' as ')[0]
        return module_part.split('.')[0]

    return ''


def is_import_used(content: str, import_name: str) -> bool:
    """Check if an import is actually used in the content."""
    # Simple heuristic - look for the import name in the code
    # This is a basic check and might need refinement
    code_lines = [line for line in content.split('\n') if not line.strip().startswith(('import ', 'from '))]
    code_content = '\n'.join(code_lines)

    # Look for the import name as a standalone word
    pattern = r'\b' + re.escape(import_name) + r'\b'
    return bool(re.search(pattern, code_content))


def fix_file(file_path: str) -> bool:
    """Fix Flake8 issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_trailing_whitespace(content)
        content = fix_blank_line_whitespace(content)
        content = fix_inline_comments(content)
        content = fix_function_class_spacing(content)
        content = fix_end_of_file_newline(content)

        # Only remove unused imports for Python files
        if file_path.endswith('.py'):
            content = remove_unused_imports(content, file_path)

        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and other common directories
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'venv', 'env')]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files


def main():
    """Main function to fix Flake8 issues."""
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = 'core'

    if os.path.isfile(target):
        # Fix single file
        fix_file(target)
    elif os.path.isdir(target):
        # Fix all Python files in directory
        python_files = find_python_files(target)
        print(f"Found {len(python_files)} Python files to check")

        fixed_count = 0
        for file_path in python_files:
            if fix_file(file_path):
                fixed_count += 1

        print(f"\nFixed {fixed_count} out of {len(python_files)} files")
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
